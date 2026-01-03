"""
エラーハンドリング

RAGシステムで使用するカスタム例外クラス
"""

import re


class RAGError(Exception):
    """RAGシステムの基底例外クラス"""
    pass


class DocumentLoadError(RAGError):
    """ドキュメント読み込みエラー"""
    
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load document '{path}': {reason}")


class EmbeddingError(RAGError):
    """エンベディング生成エラー"""
    
    def __init__(self, text_preview: str, reason: str):
        self.text_preview = text_preview[:50] if len(text_preview) > 50 else text_preview
        self.reason = reason
        super().__init__(f"Failed to embed text '{self.text_preview}...': {reason}")


class VectorStoreError(RAGError):
    """ベクトルストアエラー"""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store {operation} failed: {reason}")


class DatabaseConnectionError(RAGError):
    """データベース接続エラー"""
    
    def __init__(self, connection_string: str, reason: str):
        # 接続文字列からパスワードを隠す
        safe_string = re.sub(r':([^:@]+)@', ':***@', connection_string)
        self.connection_string = safe_string
        self.reason = reason
        super().__init__(f"Failed to connect to database '{safe_string}': {reason}")


class IndexNotFoundError(RAGError):
    """インデックスが見つからないエラー"""
    
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Index not found at '{path}'")


class ConfigurationError(RAGError):
    """設定エラー"""
    
    def __init__(self, parameter: str, reason: str):
        self.parameter = parameter
        self.reason = reason
        super().__init__(f"Invalid configuration for '{parameter}': {reason}")
