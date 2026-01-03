# Design Document: RAG System for Phase 3

## Overview

本設計では、RAG（Retrieval-Augmented Generation）システムを構築する。このシステムは、ローカルのテキストファイルやデータベースから関連情報を検索し、LLMの回答生成に活用することで、より正確で文脈に基づいた応答を実現する。

### 設計方針

1. **教育的実装**: RAGの仕組みを理解できるスクラッチ実装
2. **モジュラー設計**: 各コンポーネントを独立して使用・テスト可能
3. **拡張性**: 複数のエンベディングモデル、ベクトルストア、LLMをサポート
4. **実用性**: 実際のユースケースで使える完全な実装

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG System                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │DocumentLoader│───▶│ TextSplitter │───▶│EmbeddingModel│              │
│  │              │    │              │    │              │              │
│  │ - load_dir() │    │ - split()    │    │ - embed()    │              │
│  │ - load_file()│    │ - overlap    │    │ - batch()    │              │
│  │ - load_db()  │    │ - chunk_size │    │ - cache      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│  ┌──────────────────────────────────────────────────────┐              │
│  │                    VectorStore                        │              │
│  │                                                       │              │
│  │  - add_documents()    - search()    - persist()      │              │
│  │  - FAISS / ChromaDB / InMemory                       │              │
│  └──────────────────────────────────────────────────────┘              │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Retriever  │───▶│  Generator   │───▶│     CLI      │              │
│  │              │    │              │    │              │              │
│  │ - retrieve() │    │ - generate() │    │ - index      │              │
│  │ - rerank()   │    │ - cite()     │    │ - query      │              │
│  │ - hybrid     │    │ - template   │    │ - chat       │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### RAGの処理フロー

1. **インデックス作成フェーズ**
   - ドキュメント読み込み → チャンク分割 → エンベディング生成 → ベクトルストア保存

2. **クエリフェーズ**
   - クエリエンベディング → 類似検索 → リランキング → コンテキスト構築 → LLM生成

## Components and Interfaces

### 1. Document

```python
@dataclass
class Document:
    """ドキュメントを表すデータクラス"""
    content: str                    # テキスト内容
    metadata: Dict[str, Any]        # メタデータ
    id: Optional[str] = None        # ユニークID
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> dict: ...
    
    @classmethod
    def from_dict(cls, d: dict) -> "Document": ...
```

### 2. DocumentLoader

```python
class DocumentLoader:
    """ドキュメント読み込みクラス"""
    
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".py", ".json", ".yaml", ".yml"]
    
    def load_file(self, path: str) -> Document:
        """単一ファイルを読み込み"""
        ...
    
    def load_directory(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """ディレクトリからファイルを読み込み"""
        ...
    
    def load_database(
        self,
        connection_string: str,
        query: str,
        content_column: str,
        metadata_columns: Optional[List[str]] = None
    ) -> List[Document]:
        """データベースからドキュメントを読み込み"""
        ...
```

### 3. TextSplitter

```python
class TextSplitter:
    """テキスト分割クラス"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """テキストをチャンクに分割"""
        ...
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """ドキュメントリストを分割"""
        ...


class CodeSplitter(TextSplitter):
    """コード用テキスト分割クラス"""
    
    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
    
    def split_text(self, text: str) -> List[str]:
        """コード構造を考慮して分割"""
        # 関数、クラス単位で分割
        ...
```

### 4. EmbeddingModel

```python
class EmbeddingModel(ABC):
    """エンベディングモデルの抽象基底クラス"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """単一テキストをエンベディング"""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """バッチでエンベディング"""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """エンベディングの次元数"""
        ...


class SentenceTransformerEmbedding(EmbeddingModel):
    """sentence-transformersを使用したエンベディング"""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = cache_dir
        self._cache: Dict[str, List[float]] = {}
    
    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        embedding = self.model.encode(text).tolist()
        self._cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI APIを使用したエンベディング"""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [d.embedding for d in response.data]
    
    @property
    def dimension(self) -> int:
        return 1536  # text-embedding-3-small
```

### 5. VectorStore

```python
class VectorStore(ABC):
    """ベクトルストアの抽象基底クラス"""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        """ドキュメントとエンベディングを追加"""
        ...
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """類似検索"""
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        """永続化"""
        ...
    
    @abstractmethod
    def load(self, path: str) -> None:
        """読み込み"""
        ...


class InMemoryVectorStore(VectorStore):
    """インメモリベクトルストア（教育用）"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        # コサイン類似度で検索
        scores = []
        for i, emb in enumerate(self.embeddings):
            if filter and not self._match_filter(self.documents[i], filter):
                continue
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((self.documents[i], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
    
    def _match_filter(self, doc: Document, filter: Dict[str, Any]) -> bool:
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True
    
    def save(self, path: str) -> None:
        data = {
            "documents": [d.to_dict() for d in self.documents],
            "embeddings": self.embeddings
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.documents = [Document.from_dict(d) for d in data["documents"]]
        self.embeddings = data["embeddings"]


class FAISSVectorStore(VectorStore):
    """FAISSを使用したベクトルストア"""
    
    def __init__(self, dimension: int):
        import faiss
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine with normalized vectors)
        self.documents: List[Document] = []
        self.dimension = dimension
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        import numpy as np
        vectors = np.array(embeddings, dtype=np.float32)
        # L2正規化してコサイン類似度として使用
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.documents.extend(documents)
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        import numpy as np
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, k * 2 if filter else k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            if filter and not self._match_filter(doc, filter):
                continue
            results.append((doc, float(score)))
            if len(results) >= k:
                break
        
        return results
    
    def save(self, path: str) -> None:
        import faiss
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "documents.json", "w") as f:
            json.dump([d.to_dict() for d in self.documents], f)
    
    def load(self, path: str) -> None:
        import faiss
        self.index = faiss.read_index(str(Path(path) / "index.faiss"))
        with open(Path(path) / "documents.json", "r") as f:
            self.documents = [Document.from_dict(d) for d in json.load(f)]
```

### 6. Retriever

```python
class Retriever:
    """検索クラス"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """クエリに関連するドキュメントを検索"""
        k = k or self.top_k
        query_embedding = self.embedding_model.embed(query)
        return self.vector_store.search(query_embedding, k, filter)
    
    def retrieve_with_rerank(
        self,
        query: str,
        k: int = 5,
        initial_k: int = 20
    ) -> List[Tuple[Document, float]]:
        """リランキング付き検索"""
        # 初期検索
        initial_results = self.retrieve(query, k=initial_k)
        
        # リランキング（簡易版：キーワードマッチでスコア調整）
        query_terms = set(query.lower().split())
        reranked = []
        for doc, score in initial_results:
            doc_terms = set(doc.content.lower().split())
            keyword_score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            combined_score = 0.7 * score + 0.3 * keyword_score
            reranked.append((doc, combined_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]
```

### 7. Generator

```python
class Generator:
    """回答生成クラス"""
    
    DEFAULT_TEMPLATE = """以下のコンテキストを参考に、質問に回答してください。

コンテキスト:
{context}

質問: {query}

回答:"""
    
    def __init__(
        self,
        model: Any,  # LLMモデル
        tokenizer: Any,
        template: Optional[str] = None,
        max_context_length: int = 2000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or self.DEFAULT_TEMPLATE
        self.max_context_length = max_context_length
    
    def generate(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """検索結果を基に回答を生成"""
        # コンテキスト構築
        context = self._build_context(retrieved_docs)
        
        # プロンプト作成
        prompt = self.template.format(context=context, query=query)
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _build_context(self, docs: List[Tuple[Document, float]]) -> str:
        """検索結果からコンテキストを構築"""
        context_parts = []
        total_length = 0
        
        for doc, score in docs:
            if total_length + len(doc.content) > self.max_context_length:
                break
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{doc.content}")
            total_length += len(doc.content)
        
        return "\n\n".join(context_parts)
    
    def generate_with_citations(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """引用付きで回答を生成"""
        response = self.generate(query, retrieved_docs, **kwargs)
        sources = [doc.metadata.get("source", "unknown") for doc, _ in retrieved_docs]
        return response, sources
```

### 8. RAGPipeline

```python
class RAGPipeline:
    """RAGパイプライン全体を管理"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        generator: Generator,
        text_splitter: Optional[TextSplitter] = None
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generator = generator
        self.text_splitter = text_splitter or TextSplitter()
        self.retriever = Retriever(vector_store, embedding_model)
    
    def index_documents(self, documents: List[Document]) -> int:
        """ドキュメントをインデックス"""
        # 分割
        chunks = self.text_splitter.split_documents(documents)
        
        # エンベディング
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_batch(texts)
        
        # 保存
        self.vector_store.add_documents(chunks, embeddings)
        
        return len(chunks)
    
    def query(
        self,
        query: str,
        k: int = 5,
        with_citations: bool = True
    ) -> Union[str, Tuple[str, List[str]]]:
        """クエリを実行"""
        # 検索
        results = self.retriever.retrieve(query, k=k)
        
        if not results:
            return "関連する情報が見つかりませんでした。"
        
        # 生成
        if with_citations:
            return self.generator.generate_with_citations(query, results)
        else:
            return self.generator.generate(query, results)
    
    def save(self, path: str) -> None:
        """パイプラインを保存"""
        self.vector_store.save(path)
    
    def load(self, path: str) -> None:
        """パイプラインを読み込み"""
        self.vector_store.load(path)
```

## Data Models

### Document JSON Format

```json
{
    "id": "uuid-string",
    "content": "ドキュメントの内容",
    "metadata": {
        "source": "path/to/file.txt",
        "type": "text",
        "timestamp": "2024-01-01T00:00:00Z",
        "chunk_index": 0
    }
}
```

### Vector Store Index Structure

```
index/
├── index.faiss          # FAISSインデックス
├── documents.json       # ドキュメントメタデータ
└── config.json          # 設定情報
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Document Loading Preserves Content

*For any* valid text file, loading it through Document_Loader should produce a Document with content equal to the file's content and metadata containing the source path.

**Validates: Requirements 1.2, 1.4**

### Property 2: Text Splitter Chunk Size Constraint

*For any* text and configured chunk_size, all resulting chunks from Text_Splitter should have length not exceeding chunk_size.

**Validates: Requirements 2.1, 2.5**

### Property 3: Text Splitter Overlap Preservation

*For any* text split with overlap > 0, consecutive chunks should share exactly overlap characters at their boundaries (end of chunk N matches start of chunk N+1).

**Validates: Requirements 2.2**

### Property 4: Chunk Metadata Preservation

*For any* document split into chunks, all resulting chunks should have metadata containing the original document's source and their position index.

**Validates: Requirements 2.4**

### Property 5: Embedding Dimension Consistency

*For any* set of texts embedded by the same EmbeddingModel, all resulting vectors should have the same dimension equal to model.dimension.

**Validates: Requirements 3.1, 3.5**

### Property 6: Embedding Batch Equivalence

*For any* list of texts, embedding them as a batch should produce the same vectors as embedding them individually.

**Validates: Requirements 3.3**

### Property 7: Embedding Cache Consistency

*For any* text embedded twice through the same EmbeddingModel with caching enabled, both calls should return identical vectors.

**Validates: Requirements 3.4**

### Property 8: Vector Store Add-Search Round Trip

*For any* document added to VectorStore with its embedding, searching with the same embedding should return that document with a high similarity score.

**Validates: Requirements 4.1, 4.2**

### Property 9: Vector Store Persistence Round Trip

*For any* VectorStore with documents, saving to disk and loading into a new VectorStore should produce identical search results for any query.

**Validates: Requirements 4.4, 4.5**

### Property 10: Vector Store Filter Correctness

*For any* search with a metadata filter, all returned documents should match the filter criteria.

**Validates: Requirements 4.3**

### Property 11: Retriever Top-K Constraint

*For any* retrieval query with parameter k, the number of returned results should be at most k, and results should be ordered by descending relevance score.

**Validates: Requirements 5.1, 5.3**

### Property 12: Generator Template Application

*For any* query and context, the Generator should apply the configured template correctly, substituting {query} and {context} placeholders.

**Validates: Requirements 6.3**

## Error Handling

### File Loading Errors

```python
class DocumentLoadError(Exception):
    """ドキュメント読み込みエラー"""
    
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load document '{path}': {reason}")
```

### Embedding Errors

```python
class EmbeddingError(Exception):
    """エンベディング生成エラー"""
    
    def __init__(self, text_preview: str, reason: str):
        self.text_preview = text_preview[:50]
        self.reason = reason
        super().__init__(f"Failed to embed text '{self.text_preview}...': {reason}")
```

### Database Connection Errors

```python
class DatabaseConnectionError(Exception):
    """データベース接続エラー"""
    
    def __init__(self, connection_string: str, reason: str):
        # 接続文字列からパスワードを隠す
        safe_string = re.sub(r':([^:@]+)@', ':***@', connection_string)
        self.connection_string = safe_string
        self.reason = reason
        super().__init__(f"Failed to connect to database '{safe_string}': {reason}")
```

## Testing Strategy

### Unit Tests

単体テストは具体的な例とエッジケースを検証する：

1. **DocumentLoader Tests**
   - 各ファイル形式の読み込み確認
   - 存在しないファイルのエラー処理
   - ディレクトリの再帰的読み込み

2. **TextSplitter Tests**
   - 様々なチャンクサイズでの分割
   - オーバーラップの動作確認
   - 空テキストや短いテキストの処理

3. **EmbeddingModel Tests**
   - 各バックエンドの動作確認
   - キャッシュの動作確認
   - バッチ処理の動作確認

4. **VectorStore Tests**
   - 追加と検索の動作確認
   - フィルタリングの動作確認
   - 永続化と読み込みの動作確認

5. **Retriever Tests**
   - 検索結果の順序確認
   - リランキングの動作確認

6. **Generator Tests**
   - テンプレート適用の確認
   - 引用生成の確認

### Property-Based Tests

プロパティベーステストは普遍的な性質を検証する。Hypothesisライブラリを使用：

```python
from hypothesis import given, strategies as st

# Property 2: Text Splitter Chunk Size Constraint
@given(
    text=st.text(min_size=1, max_size=10000),
    chunk_size=st.integers(min_value=10, max_value=1000)
)
def test_chunk_size_constraint(text, chunk_size):
    """Feature: rag-system, Property 2: Text Splitter Chunk Size Constraint"""
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert len(chunk) <= chunk_size
```

### Test Configuration

- Property-based tests: minimum 100 iterations per property
- Use Hypothesis for Python property-based testing
- Each property test references its design document property number
- Tag format: `Feature: rag-system, Property {number}: {property_text}`

### Test File Structure

```
phase3/tests/
├── __init__.py
├── test_document.py         # Document, DocumentLoader tests
├── test_splitter.py         # TextSplitter tests
├── test_embedding.py        # EmbeddingModel tests
├── test_vector_store.py     # VectorStore tests
├── test_retriever.py        # Retriever tests
├── test_generator.py        # Generator tests
├── test_pipeline.py         # RAGPipeline integration tests
├── test_properties.py       # Property-based tests
└── test_cli.py              # CLI tests
```
