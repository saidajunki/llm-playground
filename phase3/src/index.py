"""
インデックス作成CLI

使用方法:
    python -m src.index --source ./data --output ./index
"""

import argparse
import logging
import sys
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Create RAG index from documents")
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Source directory or file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./index",
        help="Output directory for index (default: ./index)"
    )
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to include (default: .txt .md .py .json)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for text splitting (default: 50)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Embedding model name (default: intfloat/multilingual-e5-small)"
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Use in-memory vector store instead of FAISS"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    # ソースパスの確認
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source path does not exist: {args.source}")
        sys.exit(1)
    
    # パイプライン作成
    logger.info("Initializing RAG pipeline...")
    
    from src.rag.embedding import SentenceTransformerEmbedding
    from src.rag.splitter import TextSplitter
    
    embedding_model = SentenceTransformerEmbedding(model_name=args.embedding_model)
    text_splitter = TextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        text_splitter=text_splitter,
        use_faiss=not args.no_faiss
    )
    
    # インデックス作成
    logger.info(f"Indexing documents from: {args.source}")
    
    if source_path.is_file():
        num_chunks = pipeline.index_file(str(source_path))
    else:
        num_chunks = pipeline.index_directory(
            str(source_path),
            recursive=True,
            extensions=args.extensions
        )
    
    # 保存
    logger.info(f"Saving index to: {args.output}")
    pipeline.save(args.output)
    
    logger.info(f"Done! Indexed {num_chunks} chunks.")
    print(f"\n✅ Index created successfully!")
    print(f"   - Chunks: {num_chunks}")
    print(f"   - Output: {args.output}")
    print(f"\nTo query the index, run:")
    print(f"   python -m src.query --index {args.output} --query \"your question\"")


if __name__ == "__main__":
    main()
