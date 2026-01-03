"""
Tools - エージェントが使用できるツール定義

Bedrock Agents の Action Group に相当する機能
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """ツールのパラメータ定義"""
    name: str
    description: str
    type: str  # "string", "number", "boolean", "array", "object"
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """ツール定義"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    function: Optional[Callable] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """OpenAI Function Calling形式のスキーマを生成"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def execute(self, **kwargs) -> Any:
        """ツールを実行"""
        if self.function is None:
            raise ValueError(f"Tool '{self.name}' has no function defined")
        
        # デフォルト値を適用
        for param in self.parameters:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default
        
        logger.info(f"Executing tool: {self.name} with args: {kwargs}")
        return self.function(**kwargs)


class ToolRegistry:
    """ツールレジストリ"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """ツールを登録"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable
    ) -> Tool:
        """関数からツールを登録"""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        self.register(tool)
        return tool
    
    def get(self, name: str) -> Optional[Tool]:
        """ツールを取得"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """登録されているツール名一覧"""
        return list(self.tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """全ツールのスキーマを取得"""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """ツール一覧の説明文を生成"""
        lines = ["利用可能なツール:"]
        for name, tool in self.tools.items():
            params = ", ".join(
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in tool.parameters
            )
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines)


# ========== 組み込みツール ==========

def create_search_tool(rag_pipeline) -> Tool:
    """ドキュメント検索ツールを作成"""
    def search_documents(query: str, top_k: int = 5) -> str:
        results = rag_pipeline.retriever.retrieve(query, k=top_k)
        if not results:
            return "関連するドキュメントが見つかりませんでした。"
        
        output = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            output.append(f"[{source}] (score: {score:.3f})\n{doc.content[:200]}...")
        
        return "\n\n".join(output)
    
    return Tool(
        name="search_documents",
        description="ドキュメントを検索して関連情報を取得します",
        parameters=[
            ToolParameter("query", "検索クエリ", "string"),
            ToolParameter("top_k", "取得する結果数", "number", required=False, default=5)
        ],
        function=search_documents
    )


def create_database_tool(db_loader) -> Tool:
    """データベースクエリツールを作成"""
    def query_database(sql: str, limit: int = 10) -> str:
        try:
            from sqlalchemy import text
            with db_loader.engine.connect() as conn:
                result = conn.execute(text(sql + f" LIMIT {limit}"))
                rows = result.fetchall()
                columns = result.keys()
                
                if not rows:
                    return "結果が見つかりませんでした。"
                
                # テーブル形式で出力
                output = [" | ".join(str(c) for c in columns)]
                output.append("-" * 50)
                for row in rows:
                    output.append(" | ".join(str(v) for v in row))
                
                return "\n".join(output)
        except Exception as e:
            return f"クエリエラー: {e}"
    
    return Tool(
        name="query_database",
        description="SQLクエリを実行してデータベースから情報を取得します",
        parameters=[
            ToolParameter("sql", "実行するSQLクエリ（SELECT文のみ）", "string"),
            ToolParameter("limit", "取得する最大行数", "number", required=False, default=10)
        ],
        function=query_database
    )


def create_calculator_tool() -> Tool:
    """計算ツールを作成"""
    import math
    
    def calculate(expression: str) -> str:
        try:
            # 安全な評価のため、許可する関数を制限
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len,
                "sqrt": math.sqrt, "pow": pow,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"計算結果: {result}"
        except Exception as e:
            return f"計算エラー: {e}"
    
    return Tool(
        name="calculate",
        description="数式を計算します（四則演算、三角関数、対数など）",
        parameters=[
            ToolParameter("expression", "計算する数式（例: 2 + 3 * 4, sqrt(16)）", "string")
        ],
        function=calculate
    )


def create_datetime_tool() -> Tool:
    """日時ツールを作成"""
    from datetime import datetime, timedelta
    
    def get_datetime(format: str = "%Y-%m-%d %H:%M:%S", offset_days: int = 0) -> str:
        dt = datetime.now() + timedelta(days=offset_days)
        return dt.strftime(format)
    
    return Tool(
        name="get_datetime",
        description="現在の日時を取得します。offset_daysで日数をずらせます",
        parameters=[
            ToolParameter("format", "日時フォーマット", "string", required=False, default="%Y-%m-%d %H:%M:%S"),
            ToolParameter("offset_days", "日数オフセット（-1で昨日、1で明日）", "number", required=False, default=0)
        ],
        function=get_datetime
    )


def create_file_reader_tool() -> Tool:
    """ファイル読み込みツールを作成"""
    def read_file(path: str, max_chars: int = 1000) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(max_chars)
                if len(content) == max_chars:
                    content += "...(truncated)"
                return content
        except FileNotFoundError:
            return f"ファイルが見つかりません: {path}"
        except Exception as e:
            return f"ファイル読み込みエラー: {e}"
    
    return Tool(
        name="read_file",
        description="ファイルの内容を読み込みます",
        parameters=[
            ToolParameter("path", "ファイルパス", "string"),
            ToolParameter("max_chars", "読み込む最大文字数", "number", required=False, default=1000)
        ],
        function=read_file
    )


def create_web_search_tool() -> Tool:
    """Web検索ツール（モック）を作成"""
    def web_search(query: str) -> str:
        # 実際の実装ではAPIを呼び出す
        return f"[Web検索結果] クエリ '{query}' の検索結果はモックです。実際のAPIを接続してください。"
    
    return Tool(
        name="web_search",
        description="Webを検索して最新情報を取得します",
        parameters=[
            ToolParameter("query", "検索クエリ", "string")
        ],
        function=web_search
    )
