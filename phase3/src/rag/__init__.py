"""
RAG (Retrieval-Augmented Generation) モジュール

ローカルのテキストファイルやデータベースから関連情報を検索し、
LLMの回答生成に活用するRAGシステムを実装
"""

__all__ = []

# 遅延インポート用（モジュールが実装されたら追加）
def __getattr__(name):
    if name == "Document":
        from .document import Document
        return Document
    elif name == "DocumentLoader":
        from .loader import DocumentLoader
        return DocumentLoader
    elif name == "TextSplitter":
        from .splitter import TextSplitter
        return TextSplitter
    elif name == "CodeSplitter":
        from .splitter import CodeSplitter
        return CodeSplitter
    elif name == "EmbeddingModel":
        from .embedding import EmbeddingModel
        return EmbeddingModel
    elif name == "SentenceTransformerEmbedding":
        from .embedding import SentenceTransformerEmbedding
        return SentenceTransformerEmbedding
    elif name == "VectorStore":
        from .vector_store import VectorStore
        return VectorStore
    elif name == "InMemoryVectorStore":
        from .vector_store import InMemoryVectorStore
        return InMemoryVectorStore
    elif name == "FAISSVectorStore":
        from .vector_store import FAISSVectorStore
        return FAISSVectorStore
    elif name == "Retriever":
        from .retriever import Retriever
        return Retriever
    elif name == "QueryRewriter":
        from .query_rewriter import QueryRewriter
        return QueryRewriter
    elif name == "HypotheticalDocumentEmbedder":
        from .query_rewriter import HypotheticalDocumentEmbedder
        return HypotheticalDocumentEmbedder
    elif name == "Generator":
        from .generator import Generator
        return Generator
    elif name == "RAGPipeline":
        from .pipeline import RAGPipeline
        return RAGPipeline
    elif name == "Agent":
        from .agent import Agent
        return Agent
    elif name == "AgentExecutor":
        from .agent import AgentExecutor
        return AgentExecutor
    elif name == "AgentResult":
        from .agent import AgentResult
        return AgentResult
    elif name == "Tool":
        from .tools import Tool
        return Tool
    elif name == "ToolRegistry":
        from .tools import ToolRegistry
        return ToolRegistry
    elif name == "ToolParameter":
        from .tools import ToolParameter
        return ToolParameter
    raise AttributeError(f"module 'rag' has no attribute '{name}'")
