# Implementation Plan: RAG System for Phase 3

## Overview

RAG（Retrieval-Augmented Generation）システムをスクラッチで実装する。ローカルのテキストファイルやデータベースから関連情報を検索し、LLMの回答生成に活用する。

## Tasks

- [x] 1. プロジェクト構造とテスト環境のセットアップ
  - phase3/ ディレクトリを作成
  - phase3/src/rag/ ディレクトリを作成
  - phase3/tests/ ディレクトリを作成
  - requirements.txt を作成（sentence-transformers, faiss-cpu, pytest, hypothesis）
  - _Requirements: 全体_

- [x] 2. Documentクラスの実装
  - [x] 2.1 Documentデータクラスを実装
    - content, metadata, id フィールド
    - to_dict / from_dict メソッド
    - UUID自動生成
    - _Requirements: 1.4_
  - [ ]* 2.2 Document のプロパティテストを作成
    - **Property 1: Document Loading Preserves Content**
    - **Validates: Requirements 1.2, 1.4**

- [x] 3. DocumentLoaderの実装
  - [x] 3.1 DocumentLoaderクラスを実装
    - load_file メソッド（単一ファイル読み込み）
    - load_directory メソッド（再帰的ディレクトリ読み込み）
    - SUPPORTED_EXTENSIONS 定義
    - メタデータ付与（source, type, timestamp）
    - _Requirements: 1.1, 1.2, 1.4, 1.5_
  - [ ]* 3.2 DocumentLoader のユニットテストを作成
    - 各ファイル形式の読み込みテスト
    - エラーハンドリングテスト
    - _Requirements: 1.1, 1.2, 1.5_

- [x] 4. Checkpoint - DocumentLoader動作確認
  - 単体でDocumentLoaderが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. TextSplitterの実装
  - [x] 5.1 TextSplitterクラスを実装
    - chunk_size, chunk_overlap, separators パラメータ
    - split_text メソッド
    - split_documents メソッド
    - チャンクメタデータ付与
    - _Requirements: 2.1, 2.2, 2.4, 2.5_
  - [ ]* 5.2 TextSplitter のプロパティテストを作成
    - **Property 2: Text Splitter Chunk Size Constraint**
    - **Property 3: Text Splitter Overlap Preservation**
    - **Property 4: Chunk Metadata Preservation**
    - **Validates: Requirements 2.1, 2.2, 2.4, 2.5**

- [x] 6. CodeSplitterの実装
  - [x] 6.1 CodeSplitterクラスを実装
    - Python用の関数/クラス単位分割
    - TextSplitterを継承
    - _Requirements: 2.3_
  - [ ]* 6.2 CodeSplitter のユニットテストを作成
    - Python コードの分割テスト
    - _Requirements: 2.3_

- [x] 7. Checkpoint - TextSplitter動作確認
  - 単体でTextSplitterが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. EmbeddingModelの実装
  - [x] 8.1 EmbeddingModel抽象基底クラスを実装
    - embed, embed_batch, dimension 抽象メソッド
    - _Requirements: 3.1_
  - [x] 8.2 SentenceTransformerEmbeddingを実装
    - sentence-transformers ラッパー
    - キャッシュ機能
    - multilingual-e5-small をデフォルトモデルに
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 8.3 EmbeddingModel のプロパティテストを作成
    - **Property 5: Embedding Dimension Consistency**
    - **Property 6: Embedding Batch Equivalence**
    - **Property 7: Embedding Cache Consistency**
    - **Validates: Requirements 3.1, 3.3, 3.4, 3.5**

- [x] 9. Checkpoint - EmbeddingModel動作確認
  - 単体でEmbeddingModelが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. VectorStoreの実装
  - [x] 10.1 VectorStore抽象基底クラスを実装
    - add_documents, search, save, load 抽象メソッド
    - _Requirements: 4.1, 4.2, 4.4, 4.5_
  - [x] 10.2 InMemoryVectorStoreを実装
    - コサイン類似度計算
    - メタデータフィルタリング
    - JSON永続化
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [x] 10.3 FAISSVectorStoreを実装
    - FAISS IndexFlatIP 使用
    - L2正規化でコサイン類似度
    - faiss.write_index / read_index で永続化
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ]* 10.4 VectorStore のプロパティテストを作成
    - **Property 8: Vector Store Add-Search Round Trip**
    - **Property 9: Vector Store Persistence Round Trip**
    - **Property 10: Vector Store Filter Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

- [x] 11. Checkpoint - VectorStore動作確認
  - 単体でVectorStoreが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Retrieverの実装
  - [x] 12.1 Retrieverクラスを実装
    - retrieve メソッド（基本検索）
    - retrieve_with_rerank メソッド（リランキング付き）
    - top_k パラメータ
    - _Requirements: 5.1, 5.2, 5.3, 5.5_
  - [ ]* 12.2 Retriever のプロパティテストを作成
    - **Property 11: Retriever Top-K Constraint**
    - **Validates: Requirements 5.1, 5.3**

- [x] 13. Generatorの実装
  - [x] 13.1 Generatorクラスを実装
    - DEFAULT_TEMPLATE 定義
    - generate メソッド
    - generate_with_citations メソッド
    - _build_context ヘルパー
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ]* 13.2 Generator のプロパティテストを作成
    - **Property 12: Generator Template Application**
    - **Validates: Requirements 6.3**

- [x] 14. Checkpoint - Retriever/Generator動作確認
  - 単体でRetrieverとGeneratorが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. RAGPipelineの実装
  - [x] 15.1 RAGPipelineクラスを実装
    - index_documents メソッド
    - query メソッド
    - save / load メソッド
    - _Requirements: 全体_
  - [ ]* 15.2 RAGPipeline の統合テストを作成
    - エンドツーエンドのインデックス→クエリテスト
    - _Requirements: 全体_

- [x] 16. データベース連携の実装
  - [x] 16.1 DatabaseLoaderクラスを実装
    - SQLite接続サポート
    - load_from_query メソッド
    - テーブルスキーマ取得
    - _Requirements: 7.1, 7.3, 7.4_
  - [ ]* 16.2 DatabaseLoader のユニットテストを作成
    - SQLiteでのテスト
    - _Requirements: 7.1, 7.3, 7.4_

- [x] 17. Checkpoint - データベース連携動作確認
  - 単体でDatabaseLoaderが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 18. CLIスクリプトの作成
  - [x] 18.1 index.py を作成
    - ディレクトリ/ファイルのインデックス作成
    - --source, --output オプション
    - _Requirements: 8.1_
  - [x] 18.2 query.py を作成
    - 単発クエリ実行
    - --index, --query, --top-k オプション
    - ソース表示
    - _Requirements: 8.2, 8.4_
  - [x] 18.3 chat.py を作成
    - 対話モード
    - 履歴表示
    - _Requirements: 8.3, 8.5_

- [x] 19. エラーハンドリングの実装
  - DocumentLoadError クラス
  - EmbeddingError クラス
  - DatabaseConnectionError クラス
  - ロギング設定
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 20. Final Checkpoint - 全テスト実行
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. ドキュメントとREADME作成
  - phase3/README.md を作成
  - 使用方法、実行例を記載
  - _Requirements: 全体_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Hypothesisライブラリを使用してプロパティベーステストを実装
- sentence-transformers の multilingual-e5-small を使用（日本語対応）
