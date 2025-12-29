# Implementation Plan: Mini Transformer Phase 1

## Overview

LLMの仕組みを理解するため、ミニTransformerをフルスクラッチで実装する。各コンポーネントを順番に実装し、最終的に小規模なテキスト生成モデルを完成させる。

## Tasks

- [x] 1. プロジェクト構造のセットアップ
  - phase1ディレクトリとサブディレクトリを作成
  - requirements.txtを作成（pytorch, hypothesis, pytest）
  - README.mdを作成
  - _Requirements: 全体_

- [x] 2. Tokenizerの実装
  - [x] 2.1 CharTokenizerクラスを実装
    - build_vocab, encode, decode, save, loadメソッド
    - 特殊トークン（PAD, UNK, BOS, EOS）の処理
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ]* 2.2 Tokenizerのプロパティテストを実装
    - **Property 1: Tokenizer Round Trip**
    - **Validates: Requirements 1.2, 1.4**

- [x] 3. Embeddingレイヤーの実装
  - [x] 3.1 TokenEmbeddingクラスを実装
    - nn.Embeddingをラップ
    - _Requirements: 2.1, 2.2_
  - [x] 3.2 PositionalEncodingクラスを実装
    - Sinusoidal Positional Encoding
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ]* 3.3 Embeddingのプロパティテストを実装
    - **Property 2: Embedding Output Shape**
    - **Property 3: Positional Encoding Uniqueness**
    - **Property 4: Positional Encoding Shape Preservation**
    - **Validates: Requirements 2.1, 2.3, 3.1, 3.3**

- [x] 4. Checkpoint - Embedding完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Self-Attentionの実装
  - [x] 5.1 SelfAttentionクラスを実装
    - Query, Key, Value計算
    - Scaled Dot-Product Attention
    - Causal Mask対応
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 5.2 MultiHeadAttentionクラスを実装
    - 複数ヘッドの並列実行
    - 出力の結合と線形変換
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ]* 5.3 Attentionのプロパティテストを実装
    - **Property 5: Attention Weights Sum to One**
    - **Property 6: Causal Mask Prevents Future Attention**
    - **Property 7: Multi-Head Attention Output Shape**
    - **Validates: Requirements 4.2, 4.3, 5.1, 5.2**

- [x] 6. Feed Forward Networkの実装
  - [x] 6.1 FeedForwardクラスを実装
    - 2層の全結合層
    - GELU活性化関数
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ]* 6.2 FeedForwardのプロパティテストを実装
    - **Property 8: Feed Forward Shape Preservation**
    - **Validates: Requirements 6.1, 6.3**

- [x] 7. Transformer Blockの実装
  - [x] 7.1 TransformerBlockクラスを実装
    - Multi-Head Attention + Layer Norm + FFN
    - 残差接続
    - Pre-LN構成
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ]* 7.2 TransformerBlockのプロパティテストを実装
    - **Property 9: Transformer Block Residual Connection**
    - **Validates: Requirements 7.2**

- [x] 8. Checkpoint - Transformer Block完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. 完全なモデルの組み立て
  - [x] 9.1 ModelConfigデータクラスを実装
    - ハイパーパラメータの設定
    - _Requirements: 8.2_
  - [x] 9.2 MiniGPTクラスを実装
    - 全コンポーネントの統合
    - Weight tying
    - _Requirements: 8.1, 8.3_
  - [ ]* 9.3 モデルのプロパティテストを実装
    - **Property 10: Model Output is Probability Distribution**
    - **Validates: Requirements 8.3**

- [x] 10. データセットの準備
  - [x] 10.1 TextDatasetクラスを実装
    - テキストのトークン化と分割
    - 固定長シーケンス
    - _Requirements: 11.2_
  - [x] 10.2 DataLoaderのセットアップ
    - バッチ処理対応
    - _Requirements: 11.3_
  - [x] 10.3 サンプルデータの準備
    - 小規模な日本語/英語テキスト
    - _Requirements: 11.1_
  - [ ]* 10.4 データセットのプロパティテストを実装
    - **Property 16: Dataset Fixed Length**
    - **Property 17: DataLoader Batch Shape**
    - **Validates: Requirements 11.2, 11.3**

- [x] 11. 学習ループの実装
  - [x] 11.1 TrainConfigデータクラスを実装
    - 学習ハイパーパラメータ
    - _Requirements: 9.1, 9.2_
  - [x] 11.2 学習率スケジューラを実装
    - Warmup + Cosine Decay
    - _Requirements: 9.4_
  - [x] 11.3 学習ループを実装
    - Cross-Entropy損失
    - AdamWオプティマイザ
    - Gradient Clipping
    - ログ出力
    - _Requirements: 9.1, 9.2, 9.3_
  - [x] 11.4 チェックポイント保存/読み込みを実装
    - モデル状態の保存
    - _Requirements: 9.5_
  - [ ]* 11.5 学習関連のプロパティテストを実装
    - **Property 11: Learning Rate Schedule**
    - **Property 12: Checkpoint Round Trip**
    - **Validates: Requirements 9.4, 9.5**

- [x] 12. Checkpoint - 学習ループ完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. テキスト生成の実装
  - [x] 13.1 自己回帰生成を実装
    - 次トークン予測の繰り返し
    - _Requirements: 10.1_
  - [x] 13.2 サンプリング戦略を実装
    - Temperature調整
    - Top-kサンプリング
    - Greedy decoding
    - _Requirements: 10.2, 10.3_
  - [x] 13.3 生成パラメータの設定
    - 最大生成長
    - _Requirements: 10.4_
  - [ ]* 13.4 生成のプロパティテストを実装
    - **Property 13: Autoregressive Generation Length**
    - **Property 14: Temperature Sampling Determinism**
    - **Property 15: Top-k Sampling Constraint**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [x] 14. 統合と動作確認
  - [x] 14.1 全コンポーネントの統合テスト
    - エンドツーエンドの動作確認
    - _Requirements: 全体_
  - [x] 14.2 小規模データでの学習実行
    - 損失の減少を確認
    - _Requirements: 9.3_
  - [x] 14.3 テキスト生成のデモ
    - 学習済みモデルでの生成
    - _Requirements: 10.1_

- [x] 15. Final Checkpoint - 全テスト通過確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. ドキュメントと可視化
  - [x] 16.1 README.mdの完成
    - 使い方、アーキテクチャ説明
    - _Requirements: 全体_
  - [x] 16.2 Attention可視化ノートブックの作成
    - Attention重みの可視化
    - 学習曲線のプロット
    - _Requirements: 学習目的_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- 各タスクは前のタスクに依存するため、順番に実行
- Checkpointでは必ずテストを実行して動作確認
- Property testsはhypothesisライブラリを使用
- M3 Proで動作する小規模モデル（約2000万パラメータ）を想定
