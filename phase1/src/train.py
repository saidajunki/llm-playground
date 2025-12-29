"""
Training loop for Mini Transformer.

学習ループ、学習率スケジューラ、チェックポイント管理を実装。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .model import MiniGPT, ModelConfig
from .tokenizer import CharTokenizer
from .dataset import load_text_file, create_dataloader


@dataclass
class TrainConfig:
    """学習のハイパーパラメータ設定"""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    grad_clip: float = 1.0
    seq_len: int = 128


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int
) -> LambdaLR:
    """
    Warmup + Cosine Decay学習率スケジューラを作成。
    
    Args:
        optimizer: オプティマイザ
        warmup_steps: ウォームアップステップ数
        max_steps: 最大ステップ数
        
    Returns:
        学習率スケジューラ
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    loss: float,
    path: Union[str, Path]
) -> None:
    """
    チェックポイントを保存。
    
    Args:
        model: モデル
        optimizer: オプティマイザ
        scheduler: スケジューラ
        step: 現在のステップ
        loss: 現在の損失
        path: 保存先パス
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': model.config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: MiniGPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None
) -> Dict:
    """
    チェックポイントを読み込み。
    
    Args:
        path: チェックポイントのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: スケジューラ（オプション）
        
    Returns:
        チェックポイントの情報
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def train(
    model: MiniGPT,
    dataloader: torch.utils.data.DataLoader,
    train_config: TrainConfig,
    device: torch.device,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    resume_from: Optional[Union[str, Path]] = None
) -> List[float]:
    """
    学習ループを実行。
    
    Args:
        model: モデル
        dataloader: データローダー
        train_config: 学習設定
        device: デバイス
        checkpoint_dir: チェックポイント保存先
        resume_from: 再開するチェックポイントのパス（オプション）
        
    Returns:
        損失の履歴
    """
    model = model.to(device)
    model.train()
    
    # オプティマイザとスケジューラ
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    scheduler = get_lr_scheduler(
        optimizer,
        train_config.warmup_steps,
        train_config.max_steps
    )
    
    # チェックポイントから再開
    start_step = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_step = checkpoint.get('step', 0)
        print(f"Resuming from step {start_step}")
        # オプティマイザの状態をデバイスに移動
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()
    
    # 学習ループ
    losses = []
    step = start_step
    data_iter = iter(dataloader)
    
    pbar = tqdm(total=train_config.max_steps, initial=start_step, desc="Training")
    
    while step < train_config.max_steps:
        # データを取得（エポックをまたぐ）
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # 損失計算: (batch, seq, vocab) -> (batch * seq, vocab)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip
        )
        
        optimizer.step()
        scheduler.step()
        
        # ログ
        losses.append(loss.item())
        step += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        pbar.update(1)
        
        # 評価
        if step % train_config.eval_interval == 0:
            avg_loss = sum(losses[-train_config.eval_interval:]) / train_config.eval_interval
            print(f"\nStep {step}: avg_loss = {avg_loss:.4f}")
        
        # チェックポイント保存
        if step % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, step, loss.item(),
                Path(checkpoint_dir) / f"checkpoint_step{step}.pt"
            )
    
    pbar.close()
    
    # 最終チェックポイント
    save_checkpoint(
        model, optimizer, scheduler, step, losses[-1],
        Path(checkpoint_dir) / "checkpoint_final.pt"
    )
    
    return losses


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    最新のチェックポイントを探す。
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
        
    Returns:
        最新のチェックポイントのパス、なければNone
    """
    import re
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_step*.pt"))
    if not checkpoints:
        return None
    
    # ステップ番号で数値ソート
    def get_step(path: Path) -> int:
        match = re.search(r'checkpoint_step(\d+)\.pt', path.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step)
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Train Mini Transformer")
    parser.add_argument("--data", type=str, default="data/sample.txt", help="Training data file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (converted to steps)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (path or 'latest')")
    args = parser.parse_args()
    
    # デバイス設定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon M3 Pro)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # データ読み込み
    print(f"Loading data from {args.data}")
    text = load_text_file(args.data)
    
    # トークナイザー
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # モデル設定
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        max_len=args.seq_len,
    )
    
    # モデル作成
    model = MiniGPT(model_config)
    print(model)
    
    # データローダー
    dataloader = create_dataloader(
        [text],
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    
    # 学習設定
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        max_steps=args.epochs * len(dataloader),
    )
    
    # チェックポイントから再開するか確認
    resume_from = None
    if args.resume:
        if args.resume == 'latest':
            resume_from = find_latest_checkpoint(args.checkpoint_dir)
            if resume_from:
                print(f"Found latest checkpoint: {resume_from}")
            else:
                print("No checkpoint found, starting from scratch")
        else:
            resume_from = Path(args.resume)
            if not resume_from.exists():
                print(f"Checkpoint not found: {resume_from}, starting from scratch")
                resume_from = None
    
    # 学習実行
    losses = train(model, dataloader, train_config, device, args.checkpoint_dir, resume_from)
    
    # トークナイザー保存
    tokenizer.save(Path(args.checkpoint_dir) / "tokenizer.json")
    
    print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
