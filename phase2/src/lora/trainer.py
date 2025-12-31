"""
LoRAトレーナー

LoRAパラメータのみを効率的に学習するトレーナークラス
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LoRAModel
from .dataset import InstructionDataset


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    ウォームアップ付き線形スケジューラ
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LoRATrainer:
    """LoRAファインチューニング用トレーナー"""
    
    def __init__(
        self,
        model: LoRAModel,
        tokenizer: Any,
        train_dataset: InstructionDataset,
        val_dataset: Optional[InstructionDataset] = None,
        output_dir: str = "checkpoints/lora",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 10,
        save_interval: int = 500,
        fp16: bool = True,
        gradient_checkpointing: bool = True
    ):
        """
        Args:
            model: LoRAモデル
            tokenizer: トークナイザー
            train_dataset: 訓練データセット
            val_dataset: 検証データセット（オプション）
            output_dir: チェックポイント保存先
            learning_rate: 学習率
            batch_size: バッチサイズ
            gradient_accumulation_steps: 勾配累積ステップ数
            num_epochs: エポック数
            warmup_steps: ウォームアップステップ数
            log_interval: ログ出力間隔
            save_interval: チェックポイント保存間隔
            fp16: 混合精度学習を使用するか
            gradient_checkpointing: 勾配チェックポイントを使用するか
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        
        # ハイパーパラメータ
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.fp16 = fp16 and torch.cuda.is_available()
        self.gradient_checkpointing = gradient_checkpointing
        
        # 初期化
        self._setup()
    
    def _setup(self) -> None:
        """トレーニング環境のセットアップ"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Optimizer（LoRAパラメータのみ）
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = (
            len(self.train_loader) * self.num_epochs 
            // self.gradient_accumulation_steps
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Gradient checkpointing
        if self.gradient_checkpointing:
            if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
                self.model.base_model.gradient_checkpointing_enable()
    
    def train(self) -> Dict[str, List[float]]:
        """
        トレーニングループ
        
        Returns:
            {"train_loss": [...], "val_loss": [...]}
        """
        self.model.train()
        device = self.model.device
        
        history = {"train_loss": [], "val_loss": []}
        global_step = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                # バッチをデバイスに移動
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                if self.fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # ログ出力
                    if global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        history["train_loss"].append(avg_loss)
                    
                    # チェックポイント保存
                    if global_step % self.save_interval == 0:
                        self.save_checkpoint(f"step_{global_step}")
            
            # エポック終了時の処理
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}")
            
            # 検証
            if self.val_loader:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
            
            # エポック終了時のチェックポイント
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        # 最終チェックポイント
        self.save_checkpoint("final")
        
        return history
    
    def evaluate(self) -> float:
        """
        検証データでの評価
        
        Returns:
            平均損失
        """
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        device = self.model.device
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if self.fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, name: str) -> None:
        """
        チェックポイントを保存
        
        Args:
            name: チェックポイント名
        """
        checkpoint_path = self.output_dir / name
        
        # LoRA重みを保存
        self.model.save_lora_weights(str(checkpoint_path))
        
        # トレーニング状態も保存
        trainer_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.scaler:
            trainer_state["scaler"] = self.scaler.state_dict()
        
        torch.save(trainer_state, checkpoint_path / "trainer_state.pt")
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, name: str) -> None:
        """
        チェックポイントから復元
        
        Args:
            name: チェックポイント名
        """
        checkpoint_path = self.output_dir / name
        
        # LoRA重みを読み込み
        self.model.load_lora_weights(str(checkpoint_path))
        
        # トレーニング状態を復元
        state_path = checkpoint_path / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if self.scaler and "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
