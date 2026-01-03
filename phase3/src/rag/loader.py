"""
DocumentLoader - ドキュメント読み込みクラス

テキストファイルやディレクトリからドキュメントを読み込む
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .document import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """ドキュメント読み込みクラス"""
    
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".rst", ".csv"]
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Args:
            encoding: ファイル読み込み時のエンコーディング
        """
        self.encoding = encoding
    
    def load_file(self, path: str) -> Document:
        """
        単一ファイルを読み込み
        
        Args:
            path: ファイルパス
            
        Returns:
            Document: 読み込んだドキュメント
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            IOError: ファイル読み込みに失敗した場合
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode {path} with {self.encoding}, trying latin-1")
            content = file_path.read_text(encoding="latin-1")
        
        metadata = {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "extension": file_path.suffix,
            "type": self._get_file_type(file_path.suffix),
            "timestamp": datetime.now().isoformat(),
            "size": file_path.stat().st_size
        }
        
        return Document(content=content, metadata=metadata)
    
    def load_directory(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        ディレクトリからファイルを読み込み
        
        Args:
            path: ディレクトリパス
            recursive: 再帰的に読み込むかどうか
            extensions: 読み込む拡張子のリスト（Noneの場合はSUPPORTED_EXTENSIONS）
            
        Returns:
            List[Document]: 読み込んだドキュメントのリスト
        """
        dir_path = Path(path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        extensions = extensions or self.SUPPORTED_EXTENSIONS
        # 拡張子を正規化（.を付ける）
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        
        documents = []
        
        if recursive:
            files = dir_path.rglob("*")
        else:
            files = dir_path.glob("*")
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in extensions:
                continue
            
            # 隠しファイルをスキップ
            if file_path.name.startswith("."):
                continue
            
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
                logger.debug(f"Loaded: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {path}")
        return documents
    
    def _get_file_type(self, extension: str) -> str:
        """拡張子からファイルタイプを判定"""
        type_map = {
            ".txt": "text",
            ".md": "markdown",
            ".py": "python",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".rst": "restructuredtext",
            ".csv": "csv"
        }
        return type_map.get(extension.lower(), "unknown")
