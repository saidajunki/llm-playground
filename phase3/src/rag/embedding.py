"""
EmbeddingModel - エンベディングモデル

テキストをベクトル表現に変換する
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """エンベディングモデルの抽象基底クラス"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        単一テキストをエンベディング
        
        Args:
            text: エンベディングするテキスト
            
        Returns:
            List[float]: エンベディングベクトル
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        バッチでエンベディング
        
        Args:
            texts: エンベディングするテキストのリスト
            
        Returns:
            List[List[float]]: エンベディングベクトルのリスト
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """エンベディングの次元数"""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """sentence-transformersを使用したエンベディング"""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Args:
            model_name: sentence-transformersのモデル名
            device: 使用するデバイス（cpu/cuda）
            cache_dir: キャッシュディレクトリ
            use_cache: キャッシュを使用するかどうか
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        
        # キャッシュ設定
        self.use_cache = use_cache
        self._cache: Dict[str, List[float]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir and self.cache_dir.exists():
            self._load_cache()
        
        logger.info(f"Loaded embedding model: {model_name} (dim={self._dimension})")
    
    def embed(self, text: str) -> List[float]:
        """単一テキストをエンベディング"""
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # E5モデルの場合はプレフィックスを追加
        if "e5" in self.model_name.lower():
            text = f"query: {text}"
        
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        
        if self.use_cache:
            self._cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """バッチでエンベディング"""
        if not texts:
            return []
        
        # キャッシュチェック
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        if self.use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
        
        # 未キャッシュのテキストをエンベディング
        if texts_to_embed:
            # E5モデルの場合はプレフィックスを追加
            if "e5" in self.model_name.lower():
                texts_to_embed = [f"passage: {t}" for t in texts_to_embed]
            
            embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=len(texts_to_embed) > 10
            ).tolist()
            
            for idx, embedding in zip(indices_to_embed, embeddings):
                results[idx] = embedding
                if self.use_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = embedding
        
        return results
    
    @property
    def dimension(self) -> int:
        """エンベディングの次元数"""
        return self._dimension
    
    def _get_cache_key(self, text: str) -> str:
        """テキストからキャッシュキーを生成"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def save_cache(self) -> None:
        """キャッシュを保存"""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "embedding_cache.json"
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} embeddings to cache")
    
    def _load_cache(self) -> None:
        """キャッシュを読み込み"""
        if self.cache_dir:
            cache_file = self.cache_dir / "embedding_cache.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} embeddings from cache")
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
