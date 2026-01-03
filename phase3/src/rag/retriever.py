"""
Retriever - 検索クラス

クエリに関連するドキュメントを検索する
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .document import Document
from .embedding import EmbeddingModel
from .query_rewriter import QueryRewriter
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """検索クラス"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        query_rewriter: Optional[QueryRewriter] = None,
        enable_rewrite: bool = True
    ):
        """
        Args:
            vector_store: ベクトルストア
            embedding_model: エンベディングモデル
            top_k: デフォルトの検索結果数
            query_rewriter: クエリ書き換えクラス
            enable_rewrite: クエリ書き換えを有効にするか
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.enable_rewrite = enable_rewrite
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        rewrite: Optional[bool] = None
    ) -> List[Tuple[Document, float]]:
        """
        クエリに関連するドキュメントを検索
        
        Args:
            query: 検索クエリ
            k: 返す結果の数
            filter: メタデータフィルタ
            rewrite: クエリ書き換えを行うか（Noneの場合はインスタンス設定に従う）
            
        Returns:
            List[Tuple[Document, float]]: (ドキュメント, スコア)のリスト
        """
        k = k or self.top_k
        
        # クエリ書き換え
        should_rewrite = rewrite if rewrite is not None else self.enable_rewrite
        if should_rewrite:
            search_query = self.query_rewriter.rewrite(query)
            # 履歴に追加
            self.query_rewriter.add_to_history(query)
        else:
            search_query = query
        
        # クエリをエンベディング
        query_embedding = self.embedding_model.embed(search_query)
        
        # 検索
        results = self.vector_store.search(query_embedding, k, filter)
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        if search_query != query:
            logger.debug(f"  (rewritten to: {search_query[:50]}...)")
        
        return results
    
    def retrieve_with_rerank(
        self,
        query: str,
        k: int = 5,
        initial_k: int = 20,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        リランキング付き検索
        
        初期検索で多めに取得し、キーワードマッチでリランキング
        
        Args:
            query: 検索クエリ
            k: 最終的に返す結果の数
            initial_k: 初期検索で取得する数
            filter: メタデータフィルタ
            
        Returns:
            List[Tuple[Document, float]]: (ドキュメント, スコア)のリスト
        """
        # 初期検索
        initial_results = self.retrieve(query, k=initial_k, filter=filter)
        
        if not initial_results:
            return []
        
        # クエリのキーワードを抽出
        query_terms = set(self._tokenize(query.lower()))
        
        # リランキング
        reranked = []
        for doc, semantic_score in initial_results:
            # ドキュメントのキーワードを抽出
            doc_terms = set(self._tokenize(doc.content.lower()))
            
            # キーワードマッチスコア
            if query_terms:
                keyword_score = len(query_terms & doc_terms) / len(query_terms)
            else:
                keyword_score = 0.0
            
            # 複合スコア（セマンティック70% + キーワード30%）
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            reranked.append((doc, combined_score))
        
        # スコアで降順ソート
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Reranked {len(initial_results)} -> {k} documents")
        return reranked[:k]
    
    def _tokenize(self, text: str) -> List[str]:
        """簡易トークナイズ"""
        # 日本語と英語の両方に対応する簡易トークナイズ
        import re
        
        # 英単語と日本語文字を抽出
        tokens = re.findall(r'\w+', text)
        
        # 日本語の場合は文字単位でも分割
        result = []
        for token in tokens:
            if any('\u3040' <= c <= '\u9fff' for c in token):
                # 日本語を含む場合は2-gramも追加
                result.append(token)
                for i in range(len(token) - 1):
                    result.append(token[i:i+2])
            else:
                result.append(token)
        
        return result
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        ハイブリッド検索（セマンティック + キーワード）
        
        Args:
            query: 検索クエリ
            k: 返す結果の数
            semantic_weight: セマンティック検索の重み（0-1）
            filter: メタデータフィルタ
            
        Returns:
            List[Tuple[Document, float]]: (ドキュメント, スコア)のリスト
        """
        return self.retrieve_with_rerank(
            query,
            k=k,
            initial_k=k * 3,
            filter=filter
        )
