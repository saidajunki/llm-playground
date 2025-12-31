# Requirements Document

## Introduction

Phase 2でQwen2-0.5Bモデルに対してLoRA（Low-Rank Adaptation）によるパラメータ効率的なファインチューニングを実装する。LoRAは元のモデルの重みを凍結し、低ランク行列のみを学習することで、少ないメモリと計算リソースで効果的なファインチューニングを実現する手法である。

## Glossary

- **LoRA_Module**: 低ランク行列A, Bを保持し、元の重みに加算する差分を計算するモジュール
- **LoRA_Config**: LoRAのハイパーパラメータ（ランク、アルファ、対象レイヤー等）を管理する設定クラス
- **LoRA_Trainer**: LoRAパラメータのみを学習するトレーニングループを実行するクラス
- **Base_Model**: ファインチューニング対象の事前学習済みモデル（Qwen2-0.5B）
- **Adapter_Weights**: LoRAで学習された低ランク行列の重み
- **Merged_Model**: LoRA重みを元のモデルに統合した推論用モデル
- **Instruction_Dataset**: 指示-応答ペアで構成されるファインチューニング用データセット

## Requirements

### Requirement 1: LoRAモジュールの実装

**User Story:** 開発者として、線形層に適用可能なLoRAモジュールを実装したい。これにより、元のモデルの重みを変更せずに効率的な学習ができる。

#### Acceptance Criteria

1. THE LoRA_Module SHALL maintain two low-rank matrices A and B where output = W*x + (B*A)*x * scaling
2. WHEN initializing LoRA_Module, THE system SHALL initialize matrix A with Kaiming uniform and matrix B with zeros
3. THE LoRA_Module SHALL compute scaling factor as alpha / rank
4. WHEN forward pass is called, THE LoRA_Module SHALL add the low-rank adaptation to the original linear layer output
5. THE LoRA_Module SHALL support configurable rank (r) and alpha parameters

### Requirement 2: LoRA設定管理

**User Story:** 開発者として、LoRAのハイパーパラメータを一元管理したい。これにより、実験の再現性と設定の変更が容易になる。

#### Acceptance Criteria

1. THE LoRA_Config SHALL store rank, alpha, dropout rate, and target module names
2. THE LoRA_Config SHALL provide default values suitable for Qwen2-0.5B model
3. WHEN serializing LoRA_Config, THE system SHALL save all parameters to JSON format
4. WHEN deserializing LoRA_Config, THE system SHALL restore all parameters from JSON format

### Requirement 3: モデルへのLoRA適用

**User Story:** 開発者として、既存のTransformerモデルにLoRAを適用したい。これにより、指定したレイヤーのみを効率的に学習できる。

#### Acceptance Criteria

1. WHEN applying LoRA to Base_Model, THE system SHALL freeze all original model parameters
2. WHEN applying LoRA to Base_Model, THE system SHALL replace target linear layers with LoRA-enhanced versions
3. THE system SHALL support targeting query, key, value, and output projection layers
4. WHEN counting trainable parameters, THE system SHALL report only LoRA parameters as trainable
5. THE system SHALL achieve at least 99% parameter reduction compared to full fine-tuning

### Requirement 4: LoRAトレーニング

**User Story:** 開発者として、LoRAパラメータのみを効率的に学習したい。これにより、限られたGPUメモリでもファインチューニングが可能になる。

#### Acceptance Criteria

1. WHEN training with LoRA, THE LoRA_Trainer SHALL update only LoRA parameters while keeping base model frozen
2. THE LoRA_Trainer SHALL support gradient accumulation for effective batch size scaling
3. THE LoRA_Trainer SHALL log training loss at configurable intervals
4. THE LoRA_Trainer SHALL save checkpoints containing only LoRA weights
5. WHEN loading checkpoints, THE LoRA_Trainer SHALL restore LoRA weights without requiring base model weights
6. IF training is interrupted, THEN THE LoRA_Trainer SHALL support resuming from the last checkpoint

### Requirement 5: 学習データの準備

**User Story:** 開発者として、指示-応答形式のデータセットを準備したい。これにより、モデルを特定のタスクに適応させることができる。

#### Acceptance Criteria

1. THE Instruction_Dataset SHALL load data from text files with instruction-response pairs
2. THE Instruction_Dataset SHALL tokenize inputs using the base model's tokenizer
3. THE Instruction_Dataset SHALL apply chat template formatting consistent with Qwen2 format
4. WHEN a sequence exceeds max length, THE Instruction_Dataset SHALL truncate appropriately
5. THE Instruction_Dataset SHALL support train/validation split

### Requirement 6: LoRA重みの保存と読み込み

**User Story:** 開発者として、学習したLoRA重みを保存・読み込みしたい。これにより、学習結果を再利用し、異なるベースモデルにも適用できる。

#### Acceptance Criteria

1. WHEN saving Adapter_Weights, THE system SHALL save only LoRA parameters (not base model)
2. WHEN saving Adapter_Weights, THE system SHALL include LoRA_Config in the saved files
3. WHEN loading Adapter_Weights, THE system SHALL apply them to a compatible base model
4. THE saved Adapter_Weights SHALL be significantly smaller than the full model (< 1% of base model size)

### Requirement 7: LoRA重みのマージ

**User Story:** 開発者として、LoRA重みを元のモデルにマージしたい。これにより、推論時のオーバーヘッドなしで学習効果を得られる。

#### Acceptance Criteria

1. WHEN merging LoRA weights, THE system SHALL compute W_merged = W_original + B*A*scaling
2. WHEN merging is complete, THE Merged_Model SHALL produce identical outputs to the LoRA-applied model
3. THE Merged_Model SHALL not require LoRA modules for inference
4. WHEN saving Merged_Model, THE system SHALL save as a standard model format

### Requirement 8: 推論と評価

**User Story:** 開発者として、ファインチューニング後のモデルで推論を行いたい。これにより、学習効果を確認できる。

#### Acceptance Criteria

1. THE system SHALL support inference with LoRA-applied model (without merging)
2. THE system SHALL support inference with Merged_Model
3. WHEN comparing outputs, THE system SHALL verify LoRA-applied and merged models produce equivalent results
4. THE system SHALL provide a script for interactive testing of the fine-tuned model

### Requirement 9: メモリ効率

**User Story:** 開発者として、限られたGPUメモリでファインチューニングを実行したい。これにより、コンシューマーGPUでも学習が可能になる。

#### Acceptance Criteria

1. THE system SHALL support training on GPUs with 8GB or more VRAM
2. THE system SHALL support gradient checkpointing to reduce memory usage
3. THE system SHALL support mixed precision training (fp16/bf16)
4. WHEN memory is insufficient, THE system SHALL provide clear error messages with suggestions
