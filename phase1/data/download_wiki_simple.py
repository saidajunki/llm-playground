"""
Wikipedia日本語版をシンプルにダウンロードするスクリプト
依存関係: requests のみ

使用方法:
    python download_wiki_simple.py --size 1gb
"""

import argparse
import bz2
import re
import os
from pathlib import Path
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET


def download_with_progress(url: str, output_path: str):
    """プログレス表示付きダウンロード"""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded = count * block_size / 1024 / 1024
        total = total_size / 1024 / 1024
        print(f"\rダウンロード中: {downloaded:.1f}MB / {total:.1f}MB ({percent}%)", end="")
    
    print(f"ダウンロード開始: {url}")
    urlretrieve(url, output_path, progress_hook)
    print("\nダウンロード完了!")


def clean_wiki_text(text: str) -> str:
    """Wikipediaのマークアップを除去"""
    # テンプレートを除去
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # リンクを除去
    text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)
    # 外部リンクを除去
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    # HTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)
    # 見出しを除去
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    # 箇条書きを除去
    text = re.sub(r'^\*+', '', text, flags=re.MULTILINE)
    # 余分な空白を除去
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_articles_from_dump(dump_path: str, output_path: str, target_size_gb: float):
    """Wikipediaダンプから記事を抽出"""
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0
    article_count = 0
    
    print(f"記事を抽出中... 目標: {target_size_gb}GB")
    
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        in_text = False
        current_text = []
        
        for line in f_in:
            if '<text' in line:
                in_text = True
                # テキスト開始タグの後の内容を取得
                match = re.search(r'<text[^>]*>(.*)', line)
                if match:
                    current_text.append(match.group(1))
            elif '</text>' in line:
                in_text = False
                # テキスト終了タグの前の内容を取得
                match = re.search(r'(.*)</text>', line)
                if match:
                    current_text.append(match.group(1))
                
                # 記事を処理
                full_text = ''.join(current_text)
                cleaned = clean_wiki_text(full_text)
                
                if len(cleaned) > 100:  # 短すぎる記事は除外
                    f_out.write(cleaned + '\n\n')
                    current_bytes += len(cleaned.encode('utf-8'))
                    article_count += 1
                    
                    if article_count % 1000 == 0:
                        print(f"\r  {article_count}記事処理, {current_bytes / 1024 / 1024 / 1024:.2f}GB", end="")
                    
                    if current_bytes >= target_bytes:
                        break
                
                current_text = []
            elif in_text:
                current_text.append(line)
    
    print(f"\n完了: {article_count}記事, {current_bytes / 1024 / 1024 / 1024:.2f}GB")


def download_sample_articles():
    """サンプル記事をダウンロード（API経由）"""
    import urllib.request
    import json
    
    # Wikipedia APIで人気記事を取得
    api_url = "https://ja.wikipedia.org/w/api.php"
    
    # カテゴリ別に記事を取得
    categories = [
        "日本の歴史",
        "日本の地理",
        "日本の文化",
        "科学",
        "技術",
        "文学",
        "音楽",
        "スポーツ",
        "政治",
        "経済",
    ]
    
    all_text = []
    
    for category in categories:
        print(f"カテゴリ '{category}' から記事を取得中...")
        
        # カテゴリ内の記事一覧を取得
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "50",
            "format": "json",
        }
        
        url = f"{api_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                for member in data.get("query", {}).get("categorymembers", []):
                    title = member["title"]
                    
                    # 記事本文を取得
                    content_params = {
                        "action": "query",
                        "titles": title.replace(" ", "_"),
                        "prop": "extracts",
                        "explaintext": "true",
                        "format": "json",
                    }
                    
                    content_url = f"{api_url}?" + "&".join(f"{k}={v}" for k, v in content_params.items())
                    
                    with urllib.request.urlopen(content_url) as content_response:
                        content_data = json.loads(content_response.read().decode('utf-8'))
                        pages = content_data.get("query", {}).get("pages", {})
                        
                        for page in pages.values():
                            extract = page.get("extract", "")
                            if extract and len(extract) > 200:
                                all_text.append(f"# {title}\n\n{extract}")
                                print(f"  取得: {title} ({len(extract)}文字)")
        except Exception as e:
            print(f"  エラー: {e}")
            continue
    
    return "\n\n".join(all_text)


def main():
    parser = argparse.ArgumentParser(description="Wikipedia日本語版をダウンロード")
    parser.add_argument(
        "--size",
        type=str,
        default="100mb",
        help="目標サイズ (例: 100mb, 500mb, 1gb)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase1/data/wikipedia_ja.txt",
        help="出力ファイルパス"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="api",
        choices=["api", "dump"],
        help="取得方法: api (小規模), dump (大規模)"
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
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.method == "api":
        print("Wikipedia APIから記事を取得します...")
        print("注意: APIは大量取得に制限があるため、100MB程度が上限です")
        
        text = download_sample_articles()
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        size = output_path.stat().st_size / 1024 / 1024
        print(f"\n保存完了: {output_path} ({size:.2f}MB)")
        
    else:
        # ダンプファイルをダウンロード
        dump_url = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2"
        dump_path = "jawiki-latest-pages-articles.xml.bz2"
        
        if not os.path.exists(dump_path):
            print("Wikipediaダンプをダウンロードします（約3GB）...")
            download_with_progress(dump_url, dump_path)
        
        extract_articles_from_dump(dump_path, str(output_path), target_size)


if __name__ == "__main__":
    main()
