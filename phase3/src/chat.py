"""
対話モードCLI

使用方法:
    python -m src.chat --index ./index
"""

import argparse
import logging
import sys
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat")
    
    parser.add_argument(
        "--index", "-i",
        type=str,
        required=True,
        help="Path to index directory"
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
    
    print(f"Index loaded: {pipeline.vector_store.count} documents")
    print("\n" + "=" * 50)
    print("RAG Chat - Type 'quit' or 'exit' to end")
    print("=" * 50 + "\n")
    
    # 対話ループ
    history = []
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if query.lower() == "history":
                print("\n--- Chat History ---")
                for i, (q, a) in enumerate(history, 1):
                    print(f"{i}. Q: {q}")
                    print(f"   A: {a[:100]}...")
                print("-------------------\n")
                continue
            
            if query.lower() == "clear":
                history.clear()
                print("History cleared.\n")
                continue
            
            # クエリ実行
            response, sources = pipeline.query(
                query,
                k=args.top_k,
                with_citations=True,
                rerank=args.rerank
            )
            
            print(f"\nAssistant: {response}")
            
            if not args.no_sources and sources:
                print("\n📚 Sources:")
                for source in sources[:3]:  # 最大3つ表示
                    source_name = source.split("/")[-1] if "/" in source else source
                    print(f"  - {source_name}")
            
            print()
            
            # 履歴に追加
            history.append((query, response))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
