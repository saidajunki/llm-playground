"""
Qwen2-0.5B 推論スクリプト

使用方法:
    python -m src.inference --prompt "バナナは何の種類ですか？"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str = "models/qwen2-0.5b"):
    """モデルとトークナイザーを読み込む"""
    
    path = Path(model_path)
    
    # ローカルにあればローカルから、なければHugging Faceから
    if path.exists():
        print(f"Loading model from {model_path}...")
        model_name = model_path
    else:
        print("Local model not found. Loading from Hugging Face...")
        model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """プロンプトに対する応答を生成"""
    
    # Qwen2のチャットテンプレートを使用
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 入力部分を除いて出力をデコード
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def interactive_chat(model, tokenizer):
    """対話モード"""
    print("\n=== Qwen2-0.5B Interactive Chat ===")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue
            
            response = generate_response(model, tokenizer, user_input)
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Qwen2-0.5B Inference")
    parser.add_argument("--model", type=str, default="models/qwen2-0.5b",
                        help="Model path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()
    
    # モデル読み込み
    model, tokenizer = load_model(args.model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")
    
    if args.prompt:
        # 単発の質問
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)
        response = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"Response: {response}")
    else:
        # 対話モード
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
