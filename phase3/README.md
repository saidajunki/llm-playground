# Phase 3: RAG (Retrieval-Augmented Generation) システム

ローカルのテキストファイルやデータベースから関連情報を検索し、LLMの回答生成に活用するRAGシステム。

## 目的

- RAGの仕組みを理解できるスクラッチ実装
- ベクトル検索の基礎を学ぶ
- 実用的なドキュメント検索システムを構築

## 機能

- **ドキュメント読み込み**: テキストファイル（.txt, .md, .py, .json等）の読み込み
- **テキスト分割**: 設定可能なチャンクサイズとオーバーラップ
- **エンベディング**: sentence-transformers（日本語対応）
- **ベクトルストア**: FAISS / インメモリ
- **検索**: セマンティック検索、リランキング
- **データベース連携**: SQLite対応

## セットアップ

```bash
cd phase3
pip install -r requirements.txt
```

## 使い方

### 1. インデックス作成

```bash
# ディレクトリをインデックス
python -m src.index --source ./data --output ./index

# 特定の拡張子のみ
python -m src.index --source ./data --output ./index --extensions .txt .md

# チャンクサイズを指定
python -m src.index --source ./data --output ./index --chunk-size 1000
```

### 2. クエリ実行

```bash
# 単発クエリ
python -m src.query --index ./index --query "Transformerとは何ですか？"

# リランキング付き
python -m src.query --index ./index --query "質問" --rerank

# 結果数を指定
python -m src.query --index ./index --query "質問" --top-k 10
```

### 3. 対話モード

```bash
python -m src.chat --index ./index
```

対話モードのコマンド:
- `quit` / `exit`: 終了
- `history`: 履歴表示
- `clear`: 履歴クリア

## Pythonから使用

```python
from src.rag import RAGPipeline

# パイプライン作成
pipeline = RAGPipeline()

# ディレクトリをインデックス
pipeline.index_directory("./data")

# クエリ実行
response, sources = pipeline.query("質問", with_citations=True)
print(response)
print("Sources:", sources)

# 保存
pipeline.save("./index")

# 読み込み
pipeline = RAGPipeline.from_index("./index")
```

## モジュール構成

```
src/rag/
├── __init__.py          # モジュール初期化
├── document.py          # Documentデータクラス
├── loader.py            # DocumentLoader
├── splitter.py          # TextSplitter, CodeSplitter
├── embedding.py         # EmbeddingModel
├── vector_store.py      # VectorStore (InMemory, FAISS)
├── retriever.py         # Retriever
├── generator.py         # Generator
├── pipeline.py          # RAGPipeline
├── database.py          # DatabaseLoader
└── errors.py            # カスタム例外
```

## 設定オプション

### エンベディングモデル

デフォルト: `intfloat/multilingual-e5-small`（日本語対応、384次元）

```python
from src.rag.embedding import SentenceTransformerEmbedding

# 別のモデルを使用
embedding = SentenceTransformerEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### ベクトルストア

```python
from src.rag.vector_store import FAISSVectorStore, InMemoryVectorStore

# FAISS（高速、推奨）
store = FAISSVectorStore(dimension=384)

# インメモリ（教育用）
store = InMemoryVectorStore()
```

### テキスト分割

```python
from src.rag.splitter import TextSplitter, CodeSplitter

# 通常のテキスト
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

# コード用
code_splitter = CodeSplitter(language="python", chunk_size=1000)
```

## データベース連携

```python
from src.rag.database import SQLiteLoader

# SQLiteから読み込み
loader = SQLiteLoader("./data.db")
docs = loader.load_table(
    table_name="articles",
    content_column="body",
    metadata_columns=["title", "author"]
)

# パイプラインにインデックス
pipeline.index_documents(docs)
```

## Phase 1, 2 との関係

| Phase | 内容 | 目的 |
|-------|------|------|
| Phase 1 | Transformerスクラッチ実装 | 仕組みの理解 |
| Phase 2 | LoRAファインチューニング | 効率的な学習 |
| Phase 3 | RAGシステム + Agent | 外部知識の活用 + ツール実行 |

Phase 3では、Phase 2で学習したモデルをGeneratorとして使用することも可能です。

## Agent機能（Bedrock Agents風）

エージェントはツールを選択・実行して回答を生成します。

```python
from src.rag.agent import Agent, AgentExecutor
from src.rag.pipeline import RAGPipeline

# RAGパイプライン読み込み
pipeline = RAGPipeline.from_index("./index")

# エージェント作成
agent = Agent()
executor = AgentExecutor(agent, rag_pipeline=pipeline)

# 実行
answer = executor.run("Transformerについて検索して")
print(answer)

# ステップ表示付き
answer = executor.chat("2 + 3 * 4 を計算して", show_steps=True)
print(answer)
```

### 組み込みツール

| ツール | 説明 |
|--------|------|
| search_documents | ドキュメント検索（RAG） |
| query_database | SQLクエリ実行 |
| calculate | 数式計算 |
| get_datetime | 日時取得 |
| read_file | ファイル読み込み |

### カスタムツールの追加

```python
from src.rag.tools import Tool, ToolParameter

# カスタムツール定義
weather_tool = Tool(
    name="get_weather",
    description="天気を取得します",
    parameters=[
        ToolParameter("location", "場所", "string"),
    ],
    function=lambda location: f"{location}の天気は晴れです"
)

# エージェントに登録
agent.register_tool(weather_tool)
```
