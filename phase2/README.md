# Phase 2: 軽量LLMのファインチューニング

既存の軽量オープンソースモデル（Qwen2-0.5B）を使って、実用的な会話モデルを構築するプロジェクト。

## 目的

- 既存モデルのファインチューニング手法を学ぶ
- LoRA/QLoRAによる効率的な学習を体験
- RLHF/DPOによるアライメントを実装
- Phase 1で学んだ知識を実践に活かす

## 使用モデル

- **Qwen2-0.5B**: Alibaba製の軽量モデル（5億パラメータ）
- 日本語対応、ローカルで学習可能

## ディレクトリ構成

```
phase2/
├── src/
│   ├── download_model.py   # モデルダウンロード
│   ├── inference.py        # 推論・生成
│   ├── finetune_sft.py     # SFT（教師ありファインチューニング）
│   ├── finetune_lora.py    # LoRAファインチューニング
│   └── dpo.py              # DPO（Direct Preference Optimization）
├── data/                   # 学習データ
├── checkpoints/            # モデルチェックポイント
└── notebooks/              # 実験用ノートブック
```

## セットアップ

```bash
cd phase2
pip install -r requirements.txt
```

## LoRAファインチューニング

LoRA（Low-Rank Adaptation）を使って、Qwen2-0.5Bを効率的にファインチューニングできます。

### LoRAとは

LoRAは、元のモデルの重みを凍結し、低ランク行列のみを学習する手法です：
- **パラメータ効率**: 学習パラメータは全体の1%未満
- **メモリ効率**: 8GB GPUでも学習可能
- **高速**: フル学習より大幅に高速

### 使い方

#### 1. モデルのダウンロード

```bash
python -m src.download_model
```

#### 2. LoRAファインチューニング

```bash
python -m src.finetune_lora --data data/instruction_sample.txt
```

主なオプション：
- `--rank`: LoRAランク（デフォルト: 8）
- `--alpha`: スケーリング係数（デフォルト: 16.0）
- `--epochs`: エポック数（デフォルト: 3）
- `--batch-size`: バッチサイズ（デフォルト: 4）
- `--lr`: 学習率（デフォルト: 1e-4）

#### 3. 推論

```bash
# 単発の質問
python -m src.inference_lora --lora checkpoints/lora/final --prompt "こんにちは"

# 対話モード
python -m src.inference_lora --lora checkpoints/lora/final

# マージして推論（オーバーヘッドなし）
python -m src.inference_lora --lora checkpoints/lora/final --merge
```

### データ形式

学習データはタブ区切りのテキストファイル：

```
質問1	回答1
質問2	回答2
```

### LoRAモジュール構成

```
src/lora/
├── config.py      # LoRA設定
├── linear.py      # LoRA線形層
├── model.py       # LoRAモデルラッパー
├── dataset.py     # データセット
├── trainer.py     # トレーナー
├── inference.py   # 推論
└── errors.py      # エラー処理
```

## Phase 1 との違い

| 観点 | Phase 1 | Phase 2 |
|------|---------|---------|
| モデル | フルスクラッチ（15M） | 既存モデル（500M） |
| 目的 | 仕組みの理解 | 実用的な活用 |
| 学習 | 事前学習から | ファインチューニング |
| 能力 | パターン模倣 | 実際の会話 |
