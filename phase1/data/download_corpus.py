"""
日本語学習データをダウンロードするスクリプト

以下のソースから選択できます：
1. 青空文庫 (著作権切れの日本文学)
2. Wikipedia日本語版
3. CC-100 日本語コーパス
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path


def download_aozora_bunko():
    """
    青空文庫から複数の作品をダウンロード
    約50-100MB程度のテキストが取得可能
    """
    # 青空文庫の人気作品リスト（作品ID）
    works = [
        # 夏目漱石
        ("789", "wagahaiwa_nekodearu"),  # 吾輩は猫である
        ("752", "botchan"),  # 坊っちゃん
        ("1567", "kokoro"),  # こころ
        ("776", "sanshiro"),  # 三四郎
        # 芥川龍之介
        ("127", "rashomon"),  # 羅生門
        ("128", "hana"),  # 鼻
        ("33", "kumonoito"),  # 蜘蛛の糸
        # 太宰治
        ("301", "ningen_shikkaku"),  # 人間失格
        ("31", "hashire_melos"),  # 走れメロス
        # 宮沢賢治
        ("1921", "ginga_tetsudo"),  # 銀河鉄道の夜
        ("81", "chumon"),  # 注文の多い料理店
    ]
    
    print("青空文庫からのダウンロードは手動で行ってください：")
    print("https://www.aozora.gr.jp/")
    print("\nまたは、以下のコマンドでaozora-cliを使用：")
    print("pip install aozorabunko")
    print("aozora download --all")


def download_wikipedia_ja():
    """
    Wikipedia日本語版のダンプをダウンロード
    注意: 圧縮状態で約3GB、展開後は約15GB
    """
    print("Wikipedia日本語版ダンプのダウンロード方法：")
    print("\n1. 以下のURLからダウンロード：")
    print("   https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2")
    print("\n2. WikiExtractorで抽出：")
    print("   pip install wikiextractor")
    print("   python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml.bz2 -o extracted")
    print("\n3. テキストファイルに結合：")
    print("   find extracted -name 'wiki_*' -exec cat {} \\; > wikipedia_ja.txt")


def download_cc100_sample():
    """
    CC-100日本語コーパスのサンプルをダウンロード
    フルデータは約70GB
    """
    print("CC-100 日本語コーパス：")
    print("https://data.statmt.org/cc-100/")
    print("\nダウンロードコマンド：")
    print("wget https://data.statmt.org/cc-100/ja.txt.xz")
    print("xz -d ja.txt.xz")


def create_sample_from_web():
    """
    小規模なサンプルデータを作成（テスト用）
    """
    sample_texts = []
    
    # 日本語の様々なジャンルのテキストを生成
    genres = {
        "技術": generate_tech_text,
        "日常": generate_daily_text,
        "物語": generate_story_text,
        "説明": generate_explanation_text,
    }
    
    output_path = Path("sample_large.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for genre, generator in genres.items():
            print(f"Generating {genre} texts...")
            for i in range(100):
                text = generator(i)
                f.write(text + "\n\n")
    
    print(f"Sample data saved to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def generate_tech_text(seed):
    """技術系テキストのテンプレート"""
    topics = ["機械学習", "深層学習", "自然言語処理", "コンピュータビジョン", "強化学習"]
    topic = topics[seed % len(topics)]
    return f"""
{topic}について解説します。{topic}は人工知能の重要な分野の一つです。
この技術は様々な応用があり、日々進化を続けています。
研究者たちは新しいアルゴリズムを開発し、性能を向上させています。
実用化も進んでおり、多くの企業がこの技術を活用しています。
"""


def generate_daily_text(seed):
    """日常系テキストのテンプレート"""
    seasons = ["春", "夏", "秋", "冬"]
    season = seasons[seed % len(seasons)]
    return f"""
{season}になりました。{season}は一年の中でも特別な季節です。
今日は天気が良いので、外出することにしました。
街を歩いていると、様々な発見があります。
人々は忙しそうに行き交い、それぞれの生活を送っています。
"""


def generate_story_text(seed):
    """物語系テキストのテンプレート"""
    names = ["太郎", "花子", "健一", "美咲", "翔太"]
    name = names[seed % len(names)]
    return f"""
昔々、ある村に{name}という若者が住んでいました。
{name}は毎日一生懸命働き、村人たちから信頼されていました。
ある日、{name}は不思議な出来事に遭遇します。
それは{name}の人生を大きく変える出来事でした。
"""


def generate_explanation_text(seed):
    """説明系テキストのテンプレート"""
    subjects = ["歴史", "科学", "文化", "経済", "社会"]
    subject = subjects[seed % len(subjects)]
    return f"""
{subject}について考えてみましょう。{subject}は私たちの生活に深く関わっています。
{subject}を理解することで、世界をより深く知ることができます。
多くの専門家が{subject}について研究を行っています。
{subject}の知識は、現代社会を生きる上で重要です。
"""


if __name__ == "__main__":
    print("=" * 60)
    print("日本語学習データ取得ガイド")
    print("=" * 60)
    
    print("\n【推奨】大規模データセットの取得方法：\n")
    
    print("1. Hugging Face Datasets（最も簡単）")
    print("-" * 40)
    print("pip install datasets")
    print("""
from datasets import load_dataset

# Wikipedia日本語（約1GB）
dataset = load_dataset("wikipedia", "20220301.ja")
with open("data/wikipedia_ja.txt", "w") as f:
    for item in dataset["train"]:
        f.write(item["text"] + "\\n\\n")

# CC-100日本語（約70GB、サブセット取得可能）
dataset = load_dataset("cc100", lang="ja", streaming=True)
with open("data/cc100_ja.txt", "w") as f:
    for i, item in enumerate(dataset["train"]):
        if i >= 1000000:  # 100万件で停止
            break
        f.write(item["text"] + "\\n")
""")
    
    print("\n2. 青空文庫（日本文学、約100MB）")
    print("-" * 40)
    download_aozora_bunko()
    
    print("\n3. Wikipedia日本語版（約15GB展開後）")
    print("-" * 40)
    download_wikipedia_ja()
    
    print("\n" + "=" * 60)
    print("小規模サンプルを生成する場合は以下を実行：")
    print("python download_corpus.py --sample")
