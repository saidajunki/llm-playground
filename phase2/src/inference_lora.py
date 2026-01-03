"""
LoRAモデル推論スクリプト

使用方法:
    python -m src.inference_lora --lora checkpoints/lora/final --prompt "こんにちは"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import LoRAConfig, LoRAModel, LoRAInference


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Inference")
    
    # モデル設定
    parser.add_argument("--model", type=str, default="models/qwen2-0.5b",
                        help="Base model path or Hugging Face model name")
    parser.add_argument("--lora", type=str, required=True,
                        help="LoRA checkpoint path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights into base model")
    
    # 生成設定
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    
    # その他
    parser.add_argument("--save-merged", type=str, default=None,
                        help="Save merged model to this path")
    
    args = parser.parse_args()
    
    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # モデル読み込み
    print(f"Loading base model from {args.model}...")
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
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        base_model = base_model.to(device)
    
    # LoRA設定読み込み
    print(f"Loading LoRA weights from {args.lora}...")
    lora_config = LoRAConfig.load(f"{args.lora}/lora_config.json")
    print(f"LoRA config: {lora_config}")
    
    # LoRAモデル作成
    lora_model = LoRAModel(base_model, lora_config)
    lora_model.load_lora_weights(args.lora)
    lora_model.print_trainable_parameters()
    
    # マージオプション
    if args.merge:
        print("Merging LoRA weights into base model...")
        model = lora_model.merge_and_unload()
        
        # マージしたモデルを保存
        if args.save_merged:
            print(f"Saving merged model to {args.save_merged}...")
            model.save_pretrained(args.save_merged)
            tokenizer.save_pretrained(args.save_merged)
            print("Merged model saved!")
    else:
        model = lora_model
    
    # 推論インスタンス作成
    inference = LoRAInference(model, tokenizer)
    
    # パラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")
    
    if args.prompt:
        # 単発の質問
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)
        response = inference.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"Response: {response}")
    else:
        # 対話モード
        inference.interactive_chat()


if __name__ == "__main__":
    main()
