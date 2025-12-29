"""
Hugging Face Datasetsを使って日本語コーパスをダウンロード

使用方法:
    pip install datasets
    python download_hf.py --size 1gb
"""

import argparse
from pathlib import Path


def download_wikipedia_ja(output_path: str, target_size_gb: float = 1.0):
    """Wikipedia日本語版をダウンロード"""
    from datasets import load_dataset
    
    print("Wikipedia日本語版をダウンロード中...")
    # 新しい形式のWikipediaデータセットを使用
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", trust_remote_code=True)
    
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item["text"]
            f.write(text + "\n\n")
            current_bytes += len(text.encode("utf-8"))
            
            if current_bytes >= target_bytes:
                break
            
            if current_bytes % (100 * 1024 * 1024) < len(text.encode("utf-8")):
                print(f"  {current_bytes / 1024 / 1024 / 1024:.2f} GB 完了...")
    
    final_size = Path(output_path).stat().st_size / 1024 / 1024 / 1024
    print(f"完了: {output_path} ({final_size:.2f} GB)")


def download_cc100_ja(output_path: str, target_size_gb: float = 1.0):
    """CC-100日本語コーパスをダウンロード（ストリーミング）"""
    from datasets import load_dataset
    
    print("CC-100日本語コーパスをダウンロード中...")
    dataset = load_dataset("cc100", lang="ja", split="train", streaming=True)
    
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item["text"]
            f.write(text + "\n")
            current_bytes += len(text.encode("utf-8"))
            
            if current_bytes >= target_bytes:
                break
            
            if current_bytes % (100 * 1024 * 1024) < len(text.encode("utf-8")):
                print(f"  {current_bytes / 1024 / 1024 / 1024:.2f} GB 完了...")
    
    final_size = Path(output_path).stat().st_size / 1024 / 1024 / 1024
    print(f"完了: {output_path} ({final_size:.2f} GB)")


def download_oscar_ja(output_path: str, target_size_gb: float = 1.0):
    """OSCAR日本語コーパスをダウンロード"""
    from datasets import load_dataset
    
    print("OSCAR日本語コーパスをダウンロード中...")
    print("注意: 初回はHugging Faceへのログインが必要な場合があります")
    print("  huggingface-cli login")
    
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301", 
        language="ja",
        split="train",
        streaming=True
    )
    
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item["text"]
            f.write(text + "\n")
            current_bytes += len(text.encode("utf-8"))
            
            if current_bytes >= target_bytes:
                break
            
            if current_bytes % (100 * 1024 * 1024) < len(text.encode("utf-8")):
                print(f"  {current_bytes / 1024 / 1024 / 1024:.2f} GB 完了...")
    
    final_size = Path(output_path).stat().st_size / 1024 / 1024 / 1024
    print(f"完了: {output_path} ({final_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="日本語コーパスをダウンロード")
    parser.add_argument(
        "--source", 
        type=str, 
        default="wikipedia",
        choices=["wikipedia", "cc100", "oscar"],
        help="データソース (default: wikipedia)"
    )
    parser.add_argument(
        "--size", 
        type=str, 
        default="1gb",
        help="目標サイズ (例: 500mb, 1gb, 5gb)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="出力ファイルパス"
    )
    args = parser.parse_args()
    
    # サイズをパース
    size_str = args.size.lower()
    if size_str.endswith("gb"):
        target_size = float(size_str[:-2])
    elif size_str.endswith("mb"):
        target_size = float(size_str[:-2]) / 1024
    else:
        target_size = float(size_str)
    
    # 出力パス
    output_path = args.output or f"data/{args.source}_ja_{args.size}.txt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"データソース: {args.source}")
    print(f"目標サイズ: {target_size:.2f} GB")
    print(f"出力先: {output_path}")
    print("-" * 50)
    
    if args.source == "wikipedia":
        download_wikipedia_ja(output_path, target_size)
    elif args.source == "cc100":
        download_cc100_ja(output_path, target_size)
    elif args.source == "oscar":
        download_oscar_ja(output_path, target_size)


if __name__ == "__main__":
    main()
