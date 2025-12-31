"""
LoRAファインチューニング実行スクリプト

使用方法:
    python -m src.finetune_lora --data data/instruction_sample.txt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import LoRAConfig, LoRAModel, InstructionDataset, LoRATrainer


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Qwen2-0.5B")
    
    # モデル設定
    parser.add_argument("--model", type=str, default="models/qwen2-0.5b",
                        help="Base model path or Hugging Face model name")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data file (tab-separated)")
    parser.add_argument("--output", type=str, default="checkpoints/lora",
                        help="Output directory for checkpoints")
    
    # LoRA設定
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target-modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Target modules for LoRA")
    
    # トレーニング設定
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation data ratio")
    
    # その他
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                        help="Disable mixed precision training")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Use gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log interval")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save interval")
    
    args = parser.parse_args()
    
    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # モデル読み込み
    print(f"Loading model from {args.model}...")
    model_path = args.model
    if not Path(model_path).exists():
        print("Local model not found. Loading from Hugging Face...")
        model_path = "Qwen/Qwen2-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if args.fp16 and device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        base_model = base_model.to(device)
    
    # LoRA設定
    lora_config = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules
    )
    print(f"LoRA config: {lora_config}")
    
    # LoRAモデル作成
    lora_model = LoRAModel(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    # データセット読み込み
    print(f"Loading dataset from {args.data}...")
    dataset = InstructionDataset(
        args.data,
        tokenizer,
        max_length=args.max_length
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 訓練/検証分割
    if args.val_ratio > 0:
        train_dataset, val_dataset = dataset.split(args.val_ratio)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None
    
    # トレーナー作成
    trainer = LoRATrainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    # トレーニング実行
    print("\nStarting training...")
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Final checkpoint saved to {args.output}/final")
    
    # 学習曲線を表示
    if history["train_loss"]:
        print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
