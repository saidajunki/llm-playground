# Phase 1: Mini Transformer from Scratch

LLMの仕組みを理解するため、ミニTransformerをフルスクラッチで実装するプロジェクト。

## 目的

- Transformerアーキテクチャの各コンポーネントを自分の手で実装
- Attention機構、Positional Encoding、学習ループなどの動作原理を体得
- 小規模なテキスト生成モデルを完成させる

## モデルスペック

| パラメータ | 値 |
|-----------|-----|
| パラメータ数 | 約2000万〜3000万 |
| 層数 | 6 |
| 埋め込み次元 | 384 |
| ヘッド数 | 6 |
| コンテキスト長 | 256トークン |

## ディレクトリ構成

```
phase1/
├── src/
│   ├── tokenizer.py      # 文字レベルトークナイザー
│   ├── embedding.py      # Embedding + Positional Encoding
│   ├── attention.py      # Self-Attention, Multi-Head Attention
│   ├── feedforward.py    # Feed Forward Network
│   ├── transformer.py    # Transformer Block
│   ├── model.py          # 完全なMiniGPTモデル
│   ├── train.py          # 学習ループ
│   ├── generate.py       # テキスト生成
│   └── dataset.py        # データセット処理
├── tests/                # テストコード
├── data/                 # 学習データ
├── checkpoints/          # モデルチェックポイント
└── notebooks/            # 実験・可視化用ノートブック
```

## セットアップ

```bash
cd phase1
pip install -r requirements.txt
```

## 使い方

### 学習

```bash
python -m src.train --data data/sample.txt --epochs 10
```

オプション:
- `--batch-size`: バッチサイズ（デフォルト: 32）
- `--lr`: 学習率（デフォルト: 3e-4）
- `--seq-len`: シーケンス長（デフォルト: 128）

### テキスト生成

```bash
python -m src.generate --checkpoint checkpoints/checkpoint_final.pt --prompt "今日は"
```

オプション:
- `--max-tokens`: 最大生成トークン数（デフォルト: 100）
- `--temperature`: サンプリング温度（デフォルト: 0.8）
- `--top-k`: Top-kサンプリング（デフォルト: 50）

### テスト実行

```bash
python -m pytest tests/ -v
```

## アーキテクチャ

```
Input Text → Tokenizer → Token IDs
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────┐
│     Transformer Block × 6       │
│  ┌───────────────────────────┐  │
│  │ Layer Norm                │  │
│  │ Multi-Head Attention      │  │
│  │ + Residual Connection     │  │
│  │ Layer Norm                │  │
│  │ Feed Forward Network      │  │
│  │ + Residual Connection     │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final Layer Norm → Linear → Softmax
    ↓
Next Token Probabilities
```

## 主要コンポーネント解説

### 1. Tokenizer (tokenizer.py)
文字レベルのトークナイザー。テキストを文字単位でトークンIDに変換。

### 2. Embedding (embedding.py)
- **TokenEmbedding**: トークンIDを密なベクトルに変換
- **PositionalEncoding**: sin/cosを使った位置情報の付与

### 3. Attention (attention.py)
- **SelfAttention**: Scaled Dot-Product Attention
- **MultiHeadAttention**: 複数のAttentionヘッドを並列実行

### 4. Transformer Block (transformer.py)
Pre-LN構成のTransformerブロック。Attention + FFN + 残差接続。

### 5. MiniGPT (model.py)
全コンポーネントを統合したDecoder-only Transformer。

## 学習目的での注意点

このモデルは「会話ができる」レベルではありません。目的は：
- Transformerの仕組みを理解すること
- 学習データのパターンを模倣した短いテキスト生成ができること
- Phase 2で軽量モデルを使う際の基礎知識を得ること

## 次のステップ（Phase 2）

Phase 2では、TinyLlamaやPhi-3-miniなどの軽量オープンソースモデルを使って、実際に会話ができるレベルのモデルを構築します。
