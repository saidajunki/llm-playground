"""
ファインチューニング用スクリプト

使用方法:
    python -m src.finetune \
        --base-checkpoint checkpoints/checkpoint_step144000.pt \
        --data data/finetune_data.txt \
        --output-dir checkpoints/finetuned \
        --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MiniGPT, ModelConfig
from .tokenizer import CharTokenizer
from .dataset import TextDataset


def load_base_model(checkpoint_path: str, device: str) -> tuple[MiniGPT, CharTokenizer, dict]:
    """ベースモデルとトークナイザーを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 設定を復元（ModelConfigオブジェクトまたは辞書に対応）
    config_or_dict = checkpoint.get('config', {})
    
    if isinstance(config_or_dict, ModelConfig):
        config = config_or_dict
    else:
        config = ModelConfig(
            vocab_size=config_or_dict.get('vocab_size', 5000),
            embed_dim=config_or_dict.get('embed_dim', 384),
            num_heads=config_or_dict.get('num_heads', 6),
            num_layers=config_or_dict.get('num_layers', 6),
            ff_dim=config_or_dict.get('ff_dim', 1536),
            max_len=config_or_dict.get('max_len', 128),
            dropout=config_or_dict.get('dropout', 0.1),
        )
    
    # モデルを作成して重みをロード
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # トークナイザーを読み込む
    tokenizer_path = Path(checkpoint_path).parent / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = CharTokenizer()
        tokenizer.load(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    return model, tokenizer, checkpoint


def create_finetune_dataset(
    data_path: str,
    tokenizer: CharTokenizer,
    max_len: int = 128,
    format_type: str = "plain"
) -> TextDataset:
    """
    ファインチューニング用データセットを作成
    
    format_type:
        - "plain": 通常のテキスト（継続学習）
        - "qa": 質問\\t回答 形式
        - "instruction": [INST]指示[/INST]回答 形式
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if format_type == "qa":
        # Q&A形式を特殊トークンで区切る
        lines = text.strip().split('\n')
        processed = []
        for line in lines:
            if '\t' in line:
                q, a = line.split('\t', 1)
                processed.append(f"質問: {q}\n回答: {a}\n")
        text = '\n'.join(processed)
    
    elif format_type == "instruction":
        # 指示形式
        lines = text.strip().split('\n')
        processed = []
        for line in lines:
            if '\t' in line:
                inst, resp = line.split('\t', 1)
                processed.append(f"[指示] {inst} [回答] {resp}\n")
        text = '\n'.join(processed)
    
    return TextDataset(text, tokenizer, max_len)


def finetune(
    model: MiniGPT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    save_dir: Optional[str] = None,
    tokenizer: Optional[CharTokenizer] = None,
) -> list[float]:
    """ファインチューニングを実行"""
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item() * gradient_accumulation_steps:.4f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")
        
        # エポックごとに保存
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'vocab_size': model.config.vocab_size,
                    'embed_dim': model.config.embed_dim,
                    'num_heads': model.config.num_heads,
                    'num_layers': model.config.num_layers,
                    'ff_dim': model.config.ff_dim,
                    'max_len': model.config.max_len,
                    'dropout': model.config.dropout,
                }
            }
            torch.save(checkpoint, save_path / f"finetuned_epoch{epoch + 1}.pt")
            
            # トークナイザーもコピー
            if tokenizer:
                tokenizer.save(str(save_path / "tokenizer.json"))
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="ファインチューニング")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="ベースモデルのチェックポイント")
    parser.add_argument("--data", type=str, required=True,
                        help="ファインチューニング用データ")
    parser.add_argument("--output-dir", type=str, default="checkpoints/finetuned",
                        help="出力ディレクトリ")
    parser.add_argument("--format", type=str, default="plain",
                        choices=["plain", "qa", "instruction"],
                        help="データ形式")
    parser.add_argument("--epochs", type=int, default=3,
                        help="エポック数")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="学習率（ファインチューニングは小さめ）")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="凍結する下位層の数（0=全層学習）")
    args = parser.parse_args()
    
    # デバイス設定
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # ベースモデルを読み込む
    print(f"Loading base model: {args.base_checkpoint}")
    model, tokenizer, _ = load_base_model(args.base_checkpoint, device)
    print(f"Model: {model.count_parameters():,} parameters")
    
    # 層の凍結（オプション）
    if args.freeze_layers > 0:
        print(f"Freezing first {args.freeze_layers} layers")
        # Embeddingを凍結
        for param in model.token_embedding.parameters():
            param.requires_grad = False
        # 指定した数の層を凍結
        for i in range(min(args.freeze_layers, len(model.blocks))):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
    
    # データセット作成
    print(f"Loading data: {args.data}")
    dataset = create_finetune_dataset(
        args.data, tokenizer, model.config.max_len, args.format
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # オプティマイザ（ファインチューニングは小さい学習率）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # ファインチューニング実行
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    losses = finetune(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        save_dir=args.output_dir,
        tokenizer=tokenizer,
    )
    
    print(f"\nFine-tuning complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
