"""
VectorStore - ベクトルストア

ベクトル化されたドキュメントを保存・検索する
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .document import Document

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """ベクトルストアの抽象基底クラス"""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        """
        ドキュメントとエンベディングを追加
        
        Args:
            documents: ドキュメントのリスト
            embeddings: エンベディングのリスト
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        類似検索
        
        Args:
            query_embedding: クエリのエンベディング
            k: 返す結果の数
            filter: メタデータフィルタ
            
        Returns:
            List[Tuple[Document, float]]: (ドキュメント, スコア)のリスト
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        永続化
        
        Args:
            path: 保存先パス
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        読み込み
        
        Args:
            path: 読み込み元パス
        """
        pass
    
    @property
    @abstractmethod
    def count(self) -> int:
        """保存されているドキュメント数"""
        pass


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
        """ドキュメントとエンベディングを追加"""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        logger.info(f"Added {len(documents)} documents (total: {len(self.documents)})")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """コサイン類似度で検索"""
        if not self.documents:
            return []
        
        scores = []
        for i, emb in enumerate(self.embeddings):
            doc = self.documents[i]
            
            # フィルタチェック
            if filter and not self._match_filter(doc, filter):
                continue
            
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((doc, score))
        
        # スコアで降順ソート
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """コサイン類似度を計算"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
    
    def _match_filter(self, doc: Document, filter: Dict[str, Any]) -> bool:
        """フィルタ条件にマッチするかチェック"""
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True
    
    def save(self, path: str) -> None:
        """JSON形式で保存"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": [d.to_dict() for d in self.documents],
            "embeddings": self.embeddings
        }
        
        with open(save_path / "index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.documents)} documents to {path}")
    
    def load(self, path: str) -> None:
        """JSON形式から読み込み"""
        load_path = Path(path)
        
        with open(load_path / "index.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.documents = [Document.from_dict(d) for d in data["documents"]]
        self.embeddings = data["embeddings"]
        
        logger.info(f"Loaded {len(self.documents)} documents from {path}")
    
    @property
    def count(self) -> int:
        """保存されているドキュメント数"""
        return len(self.documents)


class FAISSVectorStore(VectorStore):
    """FAISSを使用したベクトルストア"""
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: エンベディングの次元数
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required. "
                "Install it with: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        # Inner Product（正規化ベクトルでコサイン類似度として使用）
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        """ドキュメントとエンベディングを追加"""
        import faiss
        import numpy as np
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        vectors = np.array(embeddings, dtype=np.float32)
        
        # L2正規化してコサイン類似度として使用
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents (total: {len(self.documents)})")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """FAISS検索"""
        import faiss
        import numpy as np
        
        if not self.documents:
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # フィルタがある場合は多めに取得
        search_k = k * 3 if filter else k
        search_k = min(search_k, len(self.documents))
        
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            doc = self.documents[idx]
            
            # フィルタチェック
            if filter and not self._match_filter(doc, filter):
                continue
            
            results.append((doc, float(score)))
            
            if len(results) >= k:
                break
        
        return results
    
    def _match_filter(self, doc: Document, filter: Dict[str, Any]) -> bool:
        """フィルタ条件にマッチするかチェック"""
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True
    
    def save(self, path: str) -> None:
        """FAISSインデックスとドキュメントを保存"""
        import faiss
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # FAISSインデックスを保存
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # ドキュメントを保存
        with open(save_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(
                [d.to_dict() for d in self.documents],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 設定を保存
        config = {"dimension": self.dimension}
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved {len(self.documents)} documents to {path}")
    
    def load(self, path: str) -> None:
        """FAISSインデックスとドキュメントを読み込み"""
        import faiss
        
        load_path = Path(path)
        
        # FAISSインデックスを読み込み
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # ドキュメントを読み込み
        with open(load_path / "documents.json", "r", encoding="utf-8") as f:
            self.documents = [Document.from_dict(d) for d in json.load(f)]
        
        # 設定を読み込み
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)
            self.dimension = config["dimension"]
        
        logger.info(f"Loaded {len(self.documents)} documents from {path}")
    
    @property
    def count(self) -> int:
        """保存されているドキュメント数"""
        return len(self.documents)
