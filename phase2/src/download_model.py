"""
Qwen2-0.5B モデルのダウンロードスクリプト

使用方法:
    python -m src.download_model
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_qwen2_05b(save_dir: str = "models/qwen2-0.5b"):
    """Qwen2-0.5B-Instruct モデルをダウンロード"""
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    print("This may take a few minutes...")
    
    # トークナイザーをダウンロード
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")
    
    # モデルをダウンロード
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # モデル情報を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    
    return model, tokenizer


def main():
    download_qwen2_05b()
    print("\nDownload complete!")
    print("You can now run inference with: python -m src.inference")


if __name__ == "__main__":
    main()
