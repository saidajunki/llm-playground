"""
DatabaseLoader - データベース連携

データベースからドキュメントを読み込む
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .document import Document

logger = logging.getLogger(__name__)


class DatabaseLoader:
    """データベースからドキュメントを読み込むクラス"""
    
    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: データベース接続文字列
                SQLite: "sqlite:///path/to/db.sqlite"
                PostgreSQL: "postgresql://user:pass@host:port/dbname"
        """
        self.connection_string = connection_string
        self._engine = None
    
    @property
    def engine(self):
        """SQLAlchemyエンジンを取得（遅延初期化）"""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError:
                raise ImportError(
                    "sqlalchemy is required. "
                    "Install it with: pip install sqlalchemy"
                )
            
            self._engine = create_engine(self.connection_string)
            logger.info(f"Connected to database: {self._safe_connection_string()}")
        
        return self._engine
    
    def _safe_connection_string(self) -> str:
        """パスワードを隠した接続文字列を返す"""
        import re
        return re.sub(r':([^:@]+)@', ':***@', self.connection_string)
    
    def load_from_query(
        self,
        query: str,
        content_column: str,
        metadata_columns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        SQLクエリからドキュメントを読み込み
        
        Args:
            query: SQLクエリ
            content_column: コンテンツとして使用するカラム名
            metadata_columns: メタデータとして使用するカラム名のリスト
            
        Returns:
            List[Document]: ドキュメントのリスト
        """
        from sqlalchemy import text
        
        metadata_columns = metadata_columns or []
        documents = []
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                
                # コンテンツを取得
                content = str(row_dict.get(content_column, ""))
                if not content:
                    continue
                
                # メタデータを構築
                metadata = {
                    "source": f"database:{self._safe_connection_string()}",
                    "type": "database",
                    "timestamp": datetime.now().isoformat(),
                    "query": query[:100]  # クエリの一部を保存
                }
                
                # 指定されたメタデータカラムを追加
                for col in metadata_columns:
                    if col in row_dict:
                        metadata[col] = row_dict[col]
                
                documents.append(Document(content=content, metadata=metadata))
        
        logger.info(f"Loaded {len(documents)} documents from database")
        return documents
    
    def load_table(
        self,
        table_name: str,
        content_column: str,
        metadata_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Document]:
        """
        テーブルからドキュメントを読み込み
        
        Args:
            table_name: テーブル名
            content_column: コンテンツとして使用するカラム名
            metadata_columns: メタデータとして使用するカラム名のリスト
            where_clause: WHERE句（オプション）
            limit: 取得する最大行数（オプション）
            
        Returns:
            List[Document]: ドキュメントのリスト
        """
        # カラムリストを構築
        columns = [content_column]
        if metadata_columns:
            columns.extend(metadata_columns)
        
        columns_str = ", ".join(columns)
        
        # クエリを構築
        query = f"SELECT {columns_str} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.load_from_query(query, content_column, metadata_columns)
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        テーブルスキーマを取得
        
        Args:
            table_name: テーブル名
            
        Returns:
            Dict[str, str]: カラム名と型のマッピング
        """
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        
        return {col["name"]: str(col["type"]) for col in columns}
    
    def list_tables(self) -> List[str]:
        """
        データベース内のテーブル一覧を取得
        
        Returns:
            List[str]: テーブル名のリスト
        """
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def close(self) -> None:
        """接続を閉じる"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")


class SQLiteLoader(DatabaseLoader):
    """SQLite専用ローダー"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLiteデータベースファイルのパス
        """
        super().__init__(f"sqlite:///{db_path}")
        self.db_path = db_path
