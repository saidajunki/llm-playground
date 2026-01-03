"""
Document - ドキュメントを表すデータクラス

RAGシステムで扱うドキュメントの基本単位を定義
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid


@dataclass
class Document:
    """ドキュメントを表すデータクラス"""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        """初期化後処理：IDが未設定の場合はUUIDを生成"""
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        """辞書から復元"""
        return cls(
            content=d["content"],
            metadata=d.get("metadata", {}),
            id=d.get("id")
        )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id[:8]}..., content='{content_preview}')"
    
    def __len__(self) -> int:
        """コンテンツの長さを返す"""
        return len(self.content)
