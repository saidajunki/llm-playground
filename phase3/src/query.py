"""
クエリ実行CLI

使用方法:
    python -m src.query --index ./index --query "質問"
"""

import argparse
import logging
import sys
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query RAG index")
    
    parser.add_argument(
        "--index", "-i",
        type=str,
        required=True,
        help="Path to index directory"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Query string"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use reranking for better results"
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Embedding model name (default: intfloat/multilingual-e5-small)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    # インデックスパスの確認
    index_path = Path(args.index)
    if not index_path.exists():
        logger.error(f"Index path does not exist: {args.index}")
        sys.exit(1)
    
    # パイプライン読み込み
    print(f"Loading index from: {args.index}")
    
    from src.rag.embedding import SentenceTransformerEmbedding
    
    embedding_model = SentenceTransformerEmbedding(model_name=args.embedding_model)
    pipeline = RAGPipeline.from_index(
        args.index,
        embedding_model=embedding_model
    )
    
    print(f"Index loaded: {pipeline.vector_store.count} documents\n")
    
    # クエリ実行
    print(f"Query: {args.query}")
    print("-" * 50)
    
    response, sources = pipeline.query(
        args.query,
        k=args.top_k,
        with_citations=True,
        rerank=args.rerank
    )
    
    print(f"\n{response}")
    
    if not args.no_sources and sources:
        print("\n📚 Sources:")
        for source in sources:
            print(f"  - {source}")


if __name__ == "__main__":
    main()
