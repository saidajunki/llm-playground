"""
RAGシステムのテスト
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


class TestDocument:
    """Documentクラスのテスト"""
    
    def test_document_creation(self):
        from src.rag.document import Document
        
        doc = Document(content="テスト内容", metadata={"source": "test.txt"})
        assert doc.content == "テスト内容"
        assert doc.metadata["source"] == "test.txt"
        assert doc.id is not None
    
    def test_document_to_dict(self):
        from src.rag.document import Document
        
        doc = Document(content="テスト", metadata={"key": "value"})
        d = doc.to_dict()
        assert d["content"] == "テスト"
        assert d["metadata"]["key"] == "value"
    
    def test_document_from_dict(self):
        from src.rag.document import Document
        
        data = {"content": "テスト", "metadata": {"key": "value"}, "id": "test-id"}
        doc = Document.from_dict(data)
        assert doc.content == "テスト"
        assert doc.id == "test-id"


class TestDocumentLoader:
    """DocumentLoaderのテスト"""
    
    def test_load_file(self, tmp_path):
        from src.rag.loader import DocumentLoader
        
        # テストファイル作成
        test_file = tmp_path / "test.txt"
        test_file.write_text("これはテストファイルです。", encoding="utf-8")
        
        loader = DocumentLoader()
        doc = loader.load_file(str(test_file))
        
        assert "これはテストファイルです" in doc.content
        assert doc.metadata["type"] == "text"  # loaderは"text"を設定
    
    def test_load_directory(self, tmp_path):
        from src.rag.loader import DocumentLoader
        
        # テストファイル作成
        (tmp_path / "file1.txt").write_text("ファイル1", encoding="utf-8")
        (tmp_path / "file2.txt").write_text("ファイル2", encoding="utf-8")
        
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        
        assert len(docs) == 2


class TestTextSplitter:
    """TextSplitterのテスト"""
    
    def test_split_text(self):
        from src.rag.splitter import TextSplitter
        
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "これは長いテキストです。" * 10
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 60  # 少し余裕を持たせる
    
    def test_split_documents(self):
        from src.rag.document import Document
        from src.rag.splitter import TextSplitter
        
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        docs = [Document(content="テスト" * 20, metadata={"source": "test"})]
        chunks = splitter.split_documents(docs)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert "source" in chunk.metadata


class TestEmbedding:
    """Embeddingのテスト"""
    
    def test_embedding_dimension(self):
        from src.rag.embedding import SentenceTransformerEmbedding
        
        embedding = SentenceTransformerEmbedding()
        assert embedding.dimension == 384  # multilingual-e5-small
    
    def test_embed_single(self):
        from src.rag.embedding import SentenceTransformerEmbedding
        
        embedding = SentenceTransformerEmbedding()
        vec = embedding.embed("テストテキスト")
        
        assert len(vec) == 384
    
    def test_embed_batch(self):
        from src.rag.embedding import SentenceTransformerEmbedding
        
        embedding = SentenceTransformerEmbedding()
        texts = ["テキスト1", "テキスト2", "テキスト3"]
        vecs = embedding.embed_batch(texts)
        
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)


class TestRetriever:
    """Retrieverのテスト"""
    
    def test_retrieve(self):
        from src.rag.document import Document
        from src.rag.embedding import SentenceTransformerEmbedding
        from src.rag.retriever import Retriever
        from src.rag.vector_store import InMemoryVectorStore
        
        # セットアップ
        embedding = SentenceTransformerEmbedding()
        store = InMemoryVectorStore()
        
        docs = [
            Document(content="Pythonはプログラミング言語です", metadata={"id": "1"}),
            Document(content="機械学習は人工知能の一分野です", metadata={"id": "2"}),
            Document(content="東京は日本の首都です", metadata={"id": "3"}),
        ]
        
        embeddings = embedding.embed_batch([d.content for d in docs])
        store.add_documents(docs, embeddings)
        
        retriever = Retriever(store, embedding)
        results = retriever.retrieve("プログラミング言語について", k=2)
        
        assert len(results) == 2
        # Pythonに関するドキュメントが上位に来るはず
        assert "Python" in results[0][0].content


    def test_retrieve_with_rerank(self):
        from src.rag.document import Document
        from src.rag.embedding import SentenceTransformerEmbedding
        from src.rag.retriever import Retriever
        from src.rag.vector_store import InMemoryVectorStore
        
        embedding = SentenceTransformerEmbedding()
        store = InMemoryVectorStore()
        
        docs = [
            Document(content="Pythonプログラミング入門", metadata={"id": "1"}),
            Document(content="機械学習とPython", metadata={"id": "2"}),
            Document(content="JavaScriptの基礎", metadata={"id": "3"}),
        ]
        
        embeddings = embedding.embed_batch([d.content for d in docs])
        store.add_documents(docs, embeddings)
        
        retriever = Retriever(store, embedding)
        results = retriever.retrieve_with_rerank("Python", k=2)
        
        assert len(results) == 2
        # Pythonを含むドキュメントが上位に来るはず
        assert all("Python" in r[0].content for r in results)


class TestGenerator:
    """Generatorのテスト"""
    
    def test_generate_without_model(self):
        from src.rag.document import Document
        from src.rag.generator import Generator
        
        generator = Generator()
        docs = [(Document(content="テスト内容", metadata={"source": "test.txt"}), 0.9)]
        
        response = generator.generate("質問", docs)
        
        # モデルなしの場合はコンテキストモードで返す
        assert "[Context-only mode]" in response
        assert "テスト内容" in response
    
    def test_generate_with_citations(self):
        from src.rag.document import Document
        from src.rag.generator import Generator
        
        generator = Generator()
        docs = [
            (Document(content="内容1", metadata={"source": "file1.txt"}), 0.9),
            (Document(content="内容2", metadata={"source": "file2.txt"}), 0.8),
        ]
        
        response, sources = generator.generate_with_citations("質問", docs)
        
        assert len(sources) == 2
        assert "file1.txt" in sources
        assert "file2.txt" in sources
    
    def test_build_context(self):
        from src.rag.document import Document
        from src.rag.generator import Generator
        
        generator = Generator(max_context_length=100)
        docs = [
            (Document(content="短いテキスト", metadata={"source": "a.txt"}), 0.9),
        ]
        
        context = generator._build_context(docs)
        assert "短いテキスト" in context
        assert "a.txt" in context


class TestDatabaseLoader:
    """DatabaseLoaderのテスト"""
    
    def test_sqlite_loader(self, tmp_path):
        from src.rag.database import SQLiteLoader
        
        # テスト用SQLiteデータベース作成
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO articles (title, content) VALUES (?, ?)",
            ("テスト記事", "これはテスト記事の内容です。")
        )
        conn.commit()
        conn.close()
        
        # ローダーでテスト
        loader = SQLiteLoader(str(db_path))
        docs = loader.load_table(
            table_name="articles",
            content_column="content",
            metadata_columns=["title"]
        )
        
        assert len(docs) == 1
        assert "テスト記事の内容" in docs[0].content
        assert docs[0].metadata["title"] == "テスト記事"
        
        loader.close()
    
    def test_list_tables(self, tmp_path):
        from src.rag.database import SQLiteLoader
        
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE table1 (id INTEGER)")
        cursor.execute("CREATE TABLE table2 (id INTEGER)")
        conn.commit()
        conn.close()
        
        loader = SQLiteLoader(str(db_path))
        tables = loader.list_tables()
        
        assert "table1" in tables
        assert "table2" in tables
        
        loader.close()
    
    def test_get_table_schema(self, tmp_path):
        from src.rag.database import SQLiteLoader
        
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        conn.commit()
        conn.close()
        
        loader = SQLiteLoader(str(db_path))
        schema = loader.get_table_schema("test_table")
        
        assert "id" in schema
        assert "name" in schema
        assert "value" in schema
        
        loader.close()


class TestRAGPipeline:
    """RAGPipelineの統合テスト"""
    
    def test_pipeline_index_and_query(self, tmp_path):
        from src.rag.pipeline import RAGPipeline
        
        # テストファイル作成
        (tmp_path / "python.txt").write_text(
            "Pythonは汎用プログラミング言語です。機械学習やWeb開発に広く使われています。",
            encoding="utf-8"
        )
        (tmp_path / "javascript.txt").write_text(
            "JavaScriptはWebブラウザで動作するスクリプト言語です。",
            encoding="utf-8"
        )
        
        # パイプライン作成
        pipeline = RAGPipeline(use_faiss=False)  # テスト用にInMemory使用
        
        # インデックス
        num_chunks = pipeline.index_directory(str(tmp_path))
        assert num_chunks > 0
        
        # クエリ
        response, sources = pipeline.query("Pythonとは何ですか？", with_citations=True)
        
        assert response is not None
        assert len(sources) > 0
    
    def test_pipeline_save_load(self, tmp_path):
        from src.rag.pipeline import RAGPipeline
        
        # テストファイル作成
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.txt").write_text("テストデータ", encoding="utf-8")
        
        # インデックス作成
        pipeline = RAGPipeline(use_faiss=False)
        pipeline.index_directory(str(data_dir))
        
        # 保存
        index_dir = tmp_path / "index"
        pipeline.save(str(index_dir))
        
        # 読み込み
        pipeline2 = RAGPipeline.from_index(str(index_dir), use_faiss=False)
        
        # クエリが動作することを確認
        response, _ = pipeline2.query("テスト", with_citations=True)
        assert response is not None


class TestVectorStore:
    """VectorStoreのテスト"""
    
    def test_inmemory_store(self):
        from src.rag.document import Document
        from src.rag.vector_store import InMemoryVectorStore
        import numpy as np
        
        store = InMemoryVectorStore()
        
        docs = [Document(content=f"Doc {i}", metadata={"id": str(i)}) for i in range(3)]
        embeddings = [np.random.rand(384).tolist() for _ in range(3)]
        
        store.add_documents(docs, embeddings)
        
        # 検索
        query_vec = embeddings[0]  # 最初のドキュメントと同じベクトル
        results = store.search(query_vec, k=2)
        
        assert len(results) == 2
        assert results[0][0].content == "Doc 0"  # 最も類似
    
    def test_faiss_store(self):
        from src.rag.document import Document
        from src.rag.vector_store import FAISSVectorStore
        import numpy as np
        
        store = FAISSVectorStore(dimension=384)
        
        docs = [Document(content=f"Doc {i}", metadata={"id": str(i)}) for i in range(3)]
        embeddings = [np.random.rand(384).tolist() for _ in range(3)]
        
        store.add_documents(docs, embeddings)
        
        # 検索
        query_vec = embeddings[0]
        results = store.search(query_vec, k=2)
        
        assert len(results) == 2


class TestQueryRewriter:
    """QueryRewriterのテスト"""
    
    def test_normalize_query(self):
        from src.rag.query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter()
        
        # 口語→文語変換
        result = rewriter._normalize_query("Transformerって何")
        assert "とは" in result
        
        result = rewriter._normalize_query("教えて")
        assert "について" in result
    
    def test_extract_keywords(self):
        from src.rag.query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter()
        
        # 英単語の抽出
        result = rewriter._extract_keywords("Transformerについて")
        assert "Transformer" in result
    
    def test_rewrite_rule_based(self):
        from src.rag.query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter()
        
        # ルールベースの書き換え
        original = "Transformerって何ですか"
        rewritten = rewriter.rewrite(original)
        
        # 何らかの変換が行われる
        assert rewritten is not None
        assert len(rewritten) > 0
    
    def test_history_management(self):
        from src.rag.query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter(max_history=3)
        
        # 履歴追加
        rewriter.add_to_history("質問1")
        rewriter.add_to_history("質問2")
        rewriter.add_to_history("質問3")
        
        history = rewriter.get_history()
        assert len(history) == 3
        assert "質問1" in history
        
        # クリア
        rewriter.clear_history()
        assert len(rewriter.get_history()) == 0
    
    def test_pronoun_resolution(self):
        from src.rag.query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter()
        rewriter.add_to_history("Transformerのアーキテクチャについて")
        
        # 代名詞解決
        result = rewriter._resolve_pronouns("それについて詳しく")
        # Transformerが含まれるはず
        assert "Transformer" in result or "それ" not in result


class TestTools:
    """Toolsのテスト"""
    
    def test_tool_creation(self):
        from src.rag.tools import Tool, ToolParameter
        
        tool = Tool(
            name="test_tool",
            description="テストツール",
            parameters=[
                ToolParameter("arg1", "引数1", "string"),
                ToolParameter("arg2", "引数2", "number", required=False, default=10)
            ],
            function=lambda arg1, arg2=10: f"{arg1}: {arg2}"
        )
        
        assert tool.name == "test_tool"
        assert len(tool.parameters) == 2
    
    def test_tool_execution(self):
        from src.rag.tools import Tool, ToolParameter
        
        tool = Tool(
            name="add",
            description="足し算",
            parameters=[
                ToolParameter("a", "数値A", "number"),
                ToolParameter("b", "数値B", "number")
            ],
            function=lambda a, b: a + b
        )
        
        result = tool.execute(a=3, b=5)
        assert result == 8
    
    def test_tool_schema(self):
        from src.rag.tools import Tool, ToolParameter
        
        tool = Tool(
            name="search",
            description="検索",
            parameters=[
                ToolParameter("query", "検索クエリ", "string"),
            ]
        )
        
        schema = tool.to_schema()
        assert schema["name"] == "search"
        assert "query" in schema["parameters"]["properties"]
    
    def test_tool_registry(self):
        from src.rag.tools import Tool, ToolParameter, ToolRegistry
        
        registry = ToolRegistry()
        
        tool = Tool(
            name="test",
            description="テスト",
            parameters=[],
            function=lambda: "ok"
        )
        
        registry.register(tool)
        
        assert "test" in registry.list_tools()
        assert registry.get("test") is not None
    
    def test_calculator_tool(self):
        from src.rag.tools import create_calculator_tool
        
        calc = create_calculator_tool()
        
        result = calc.execute(expression="2 + 3 * 4")
        assert "14" in result
        
        result = calc.execute(expression="sqrt(16)")
        assert "4" in result
    
    def test_datetime_tool(self):
        from src.rag.tools import create_datetime_tool
        
        dt_tool = create_datetime_tool()
        
        result = dt_tool.execute()
        assert len(result) > 0
        
        result = dt_tool.execute(format="%Y-%m-%d")
        assert "-" in result


class TestAgent:
    """Agentのテスト"""
    
    def test_agent_creation(self):
        from src.rag.agent import Agent
        from src.rag.tools import ToolRegistry
        
        registry = ToolRegistry()
        agent = Agent(tool_registry=registry)
        
        assert agent.max_iterations == 5
    
    def test_agent_rule_based(self):
        from src.rag.agent import Agent
        from src.rag.tools import Tool, ToolParameter, ToolRegistry
        
        registry = ToolRegistry()
        registry.register(Tool(
            name="calculate",
            description="計算",
            parameters=[ToolParameter("expression", "数式", "string")],
            function=lambda expression: f"結果: {eval(expression)}"
        ))
        
        agent = Agent(tool_registry=registry)
        result = agent.run("2 + 3を計算して")
        
        assert result.answer is not None
        assert len(result.steps) > 0
    
    def test_agent_executor(self):
        from src.rag.agent import Agent, AgentExecutor
        
        agent = Agent()
        executor = AgentExecutor(agent, include_builtin_tools=True)
        
        # 計算ツールが登録されているはず
        assert "calculate" in agent.tool_registry.list_tools()
        
        # 実行
        answer = executor.run("今日の日付は？")
        assert answer is not None
