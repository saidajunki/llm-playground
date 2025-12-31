# Implementation Plan: LoRA Fine-tuning for Qwen2-0.5B

## Overview

Qwen2-0.5BモデルにLoRAを適用し、パラメータ効率的なファインチューニングを実装する。PEFTライブラリを使わず、LoRAの仕組みを理解できるスクラッチ実装を行う。

## Tasks

- [x] 1. プロジェクト構造とテスト環境のセットアップ
  - phase2/src/lora/ ディレクトリを作成
  - phase2/tests/ ディレクトリを作成
  - pytest と hypothesis を requirements.txt に追加
  - _Requirements: 全体_

- [x] 2. LoRAConfigの実装
  - [x] 2.1 LoRAConfigクラスを実装
    - rank, alpha, dropout, target_modules を保持するdataclass
    - scalingプロパティ（alpha/rank）
    - to_dict/from_dict メソッド
    - save/load メソッド（JSON形式）
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 2.2 LoRAConfig のプロパティテストを作成
    - **Property 3: Scaling Factor Computation**
    - **Property 4: LoRA Config Round-Trip**
    - **Validates: Requirements 1.3, 2.3, 2.4**

- [x] 3. LoRALinearの実装
  - [x] 3.1 LoRALinearクラスを実装
    - lora_A, lora_B パラメータの定義
    - Kaiming uniform初期化（A）、ゼロ初期化（B）
    - forward メソッド（元の出力 + LoRA出力）
    - merge_weights メソッド
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]* 3.2 LoRALinear のプロパティテストを作成
    - **Property 1: LoRA Forward Computation**
    - **Property 2: LoRA Initialization**
    - **Validates: Requirements 1.1, 1.2, 1.4**

- [x] 4. Checkpoint - LoRALinear動作確認
  - 単体でLoRALinearが正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. LoRAModelの実装
  - [x] 5.1 LoRAModelクラスを実装
    - _apply_lora メソッド（対象モジュールの置換）
    - _replace_module メソッド（モジュール置換ヘルパー）
    - get_trainable_parameters メソッド
    - count_parameters メソッド
    - save_lora_weights / load_lora_weights メソッド
    - merge_and_unload メソッド
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 7.1, 7.3_
  - [ ]* 5.2 LoRAModel のプロパティテストを作成
    - **Property 5: Base Model Freezing**
    - **Property 6: Target Module Replacement**
    - **Property 7: Trainable Parameter Count**
    - **Validates: Requirements 3.1, 3.2, 3.4, 3.5**

- [x] 6. Checkpoint - LoRAModel動作確認
  - 小さなモデルでLoRA適用が正しく動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. InstructionDatasetの実装
  - [x] 7.1 InstructionDatasetクラスを実装
    - _load_data メソッド（タブ区切りファイル読み込み）
    - _format_example メソッド（Qwen2チャットテンプレート適用）
    - __getitem__ メソッド（トークナイズ）
    - split メソッド（訓練/検証分割）
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 7.2 InstructionDataset のプロパティテストを作成
    - **Property 11: Dataset Tokenization Length**
    - **Property 12: Dataset Split Integrity**
    - **Validates: Requirements 5.4, 5.5**

- [x] 8. サンプル学習データの作成
  - phase2/data/instruction_sample.txt を作成
  - 日本語の指示-応答ペアを10-20件程度
  - _Requirements: 5.1_

- [x] 9. LoRATrainerの実装
  - [x] 9.1 LoRATrainerクラスを実装
    - _setup メソッド（DataLoader, Optimizer, Scheduler）
    - train メソッド（トレーニングループ）
    - evaluate メソッド（検証）
    - save_checkpoint / load_checkpoint メソッド
    - gradient accumulation サポート
    - mixed precision (fp16) サポート
    - gradient checkpointing サポート
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.2, 9.3_
  - [ ]* 9.2 LoRATrainer のプロパティテストを作成
    - **Property 8: Training Preserves Base Weights**
    - **Property 9: LoRA Weights Save/Load Round-Trip**
    - **Property 10: Checkpoint Size**
    - **Validates: Requirements 4.1, 4.4, 4.5, 6.4**

- [x] 10. Checkpoint - トレーニング動作確認
  - 小さなデータセットで学習が動作することを確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. LoRAInferenceの実装
  - [x] 11.1 LoRAInferenceクラスを実装
    - from_lora_checkpoint クラスメソッド
    - generate メソッド
    - compare_outputs 静的メソッド
    - _Requirements: 8.1, 8.2, 8.3_
  - [ ]* 11.2 マージ出力等価性のプロパティテストを作成
    - **Property 13: Merge Computation**
    - **Property 14: Merge Output Equivalence**
    - **Property 15: Merged Model Structure**
    - **Validates: Requirements 7.1, 7.2, 7.3, 8.3**

- [x] 12. CLIスクリプトの作成
  - [x] 12.1 finetune_lora.py を作成
    - コマンドライン引数でハイパーパラメータ指定
    - 学習の実行と進捗表示
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 12.2 inference_lora.py を作成
    - LoRAチェックポイントからの推論
    - マージオプション
    - 対話モード
    - _Requirements: 8.1, 8.2, 8.4_

- [x] 13. エラーハンドリングの実装
  - MemoryError クラス（メモリ不足時の提案付き）
  - LoRAConfigError クラス
  - CheckpointError クラス
  - validate_config / validate_checkpoint 関数
  - _Requirements: 9.4_

- [x] 14. 統合テストの作成
  - [x] 14.1 エンドツーエンドの学習テスト
    - 小さなモデルとデータで学習→保存→読み込み→推論
    - _Requirements: 全体_
  - [ ]* 14.2 マージ前後の出力比較テスト
    - LoRAモデルとマージモデルの出力一致確認
    - _Requirements: 7.2, 8.3_

- [x] 15. Final Checkpoint - 全テスト実行
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. ドキュメントとREADME更新
  - phase2/README.md にLoRAファインチューニングの使い方を追加
  - 実行例とコマンドを記載
  - _Requirements: 全体_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Hypothesisライブラリを使用してプロパティベーステストを実装
