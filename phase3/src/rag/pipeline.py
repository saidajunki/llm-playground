"""
RAGPipeline - RAGパイプライン

RAGシステム全体を統合するパイプライン
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from .document import Document
from .embedding import EmbeddingModel, SentenceTransformerEmbedding
from .generator import Generator
from .loader import DocumentLoader
from .retriever import Retriever
from .splitter import TextSplitter
from .vector_store import FAISSVectorStore, InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAGパイプライン全体を管理"""
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        generator: Optional[Generator] = None,
        text_splitter: Optional[TextSplitter] = None,
        use_faiss: bool = True
    ):
        """
        Args:
            embedding_model: エンベディングモデル（Noneの場合はデフォルト）
            vector_store: ベクトルストア（Noneの場合はデフォルト）
            generator: ジェネレーター（Noneの場合はデフォルト）
            text_splitter: テキストスプリッター（Noneの場合はデフォルト）
            use_faiss: FAISSを使用するかどうか
        """
        # エンベディングモデル
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        
        # ベクトルストア
        if vector_store:
            self.vector_store = vector_store
        elif use_faiss:
            self.vector_store = FAISSVectorStore(self.embedding_model.dimension)
        else:
            self.vector_store = InMemoryVectorStore()
        
        # ジェネレーター
        self.generator = generator or Generator()
        
        # テキストスプリッター
        self.text_splitter = text_splitter or TextSplitter()
        
        # リトリーバー
        self.retriever = Retriever(
            self.vector_store,
            self.embedding_model
        )
        
        # ドキュメントローダー
        self.loader = DocumentLoader()
    
    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> int:
        """
        ドキュメントをインデックス
        
        Args:
            documents: インデックスするドキュメントのリスト
            show_progress: 進捗を表示するかどうか
            
        Returns:
            int: インデックスしたチャンク数
        """
        if not documents:
            logger.warning("No documents to index")
            return 0
        
        # 分割
        logger.info(f"Splitting {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # エンベディング
        logger.info("Generating embeddings...")
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_batch(texts)
        
        # ベクトルストアに追加
        logger.info("Adding to vector store...")
        self.vector_store.add_documents(chunks, embeddings)
        
        logger.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)
    
    def index_directory(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> int:
        """
        ディレクトリをインデックス
        
        Args:
            path: ディレクトリパス
            recursive: 再帰的に読み込むかどうか
            extensions: 読み込む拡張子のリスト
            
        Returns:
            int: インデックスしたチャンク数
        """
        documents = self.loader.load_directory(path, recursive, extensions)
        return self.index_documents(documents)
    
    def index_file(self, path: str) -> int:
        """
        単一ファイルをインデックス
        
        Args:
            path: ファイルパス
            
        Returns:
            int: インデックスしたチャンク数
        """
        document = self.loader.load_file(path)
        return self.index_documents([document])
    
    def query(
        self,
        query: str,
        k: int = 5,
        with_citations: bool = True,
        rerank: bool = False
    ) -> Union[str, Tuple[str, List[str]]]:
        """
        クエリを実行
        
        Args:
            query: 質問
            k: 検索結果数
            with_citations: 引用を含めるかどうか
            rerank: リランキングを使用するかどうか
            
        Returns:
            str または Tuple[str, List[str]]: 回答（引用付きの場合はタプル）
        """
        # 検索
        if rerank:
            results = self.retriever.retrieve_with_rerank(query, k=k)
        else:
            results = self.retriever.retrieve(query, k=k)
        
        if not results:
            no_result_msg = "関連する情報が見つかりませんでした。"
            if with_citations:
                return no_result_msg, []
            return no_result_msg
        
        # 生成
        if with_citations:
            response, sources = self.generator.generate_with_citations(query, results)
            return response, sources
        else:
            return self.generator.generate(query, results)
    
    def chat(
        self,
        query: str,
        k: int = 5,
        show_sources: bool = True
    ) -> str:
        """
        チャット形式でクエリを実行（フォーマット済み）
        
        Args:
            query: 質問
            k: 検索結果数
            show_sources: ソースを表示するかどうか
            
        Returns:
            str: フォーマットされた回答
        """
        response, sources = self.query(query, k=k, with_citations=True)
        
        if show_sources and sources:
            return self.generator.format_response_with_sources(response, sources)
        return response
    
    def save(self, path: str) -> None:
        """
        パイプラインを保存
        
        Args:
            path: 保存先パス
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save(str(save_path))
        
        # エンベディングキャッシュも保存
        if hasattr(self.embedding_model, 'save_cache'):
            self.embedding_model.cache_dir = save_path
            self.embedding_model.save_cache()
        
        logger.info(f"Saved pipeline to {path}")
    
    def load(self, path: str) -> None:
        """
        パイプラインを読み込み
        
        Args:
            path: 読み込み元パス
        """
        self.vector_store.load(path)
        logger.info(f"Loaded pipeline from {path}")
    
    @classmethod
    def from_index(
        cls,
        index_path: str,
        embedding_model: Optional[EmbeddingModel] = None,
        generator: Optional[Generator] = None,
        use_faiss: bool = True
    ) -> "RAGPipeline":
        """
        保存済みインデックスからパイプラインを作成
        
        Args:
            index_path: インデックスのパス
            embedding_model: エンベディングモデル
            generator: ジェネレーター
            use_faiss: FAISSを使用するかどうか
            
        Returns:
            RAGPipeline: 読み込んだパイプライン
        """
        pipeline = cls(
            embedding_model=embedding_model,
            generator=generator,
            use_faiss=use_faiss
        )
        pipeline.load(index_path)
        return pipeline
