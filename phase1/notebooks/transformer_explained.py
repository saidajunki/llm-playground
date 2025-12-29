"""
Transformerの仕組みを理解するためのインタラクティブな解説

このスクリプトでは、Transformerの各コンポーネントを
実際のコードと出力を見ながら理解していきます。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

# =============================================================================
# 1. トークナイザー: テキスト → 数値
# =============================================================================
print("=" * 60)
print("1. トークナイザー: テキストを数値に変換")
print("=" * 60)

from src.tokenizer import CharTokenizer

tokenizer = CharTokenizer()
tokenizer.build_vocab(["こんにちは世界", "Hello World"])

text = "こんにちは"
token_ids = tokenizer.encode(text)

print(f"\n入力テキスト: '{text}'")
print(f"トークンID: {token_ids}")
print(f"\n各文字とIDの対応:")
for char, id in zip(text, token_ids):
    print(f"  '{char}' → {id}")

# デコード（逆変換）
decoded = tokenizer.decode(token_ids)
print(f"\nデコード結果: '{decoded}'")
print(f"語彙サイズ: {tokenizer.vocab_size}")

# =============================================================================
# 2. Embedding: 数値 → ベクトル
# =============================================================================
print("\n" + "=" * 60)
print("2. Embedding: トークンIDをベクトルに変換")
print("=" * 60)

from src.embedding import TokenEmbedding

embed_dim = 8  # 説明用に小さい次元
vocab_size = tokenizer.vocab_size

embedding = TokenEmbedding(vocab_size, embed_dim)

# トークンIDをテンソルに変換
input_ids = torch.tensor([token_ids])  # (1, 5) = (batch, seq_len)
print(f"\n入力形状: {input_ids.shape} (batch_size=1, seq_len={len(token_ids)})")

# Embedding適用
embedded = embedding(input_ids)
print(f"Embedding後の形状: {embedded.shape} (batch, seq_len, embed_dim)")

print(f"\n各トークンのベクトル表現:")
for i, (char, vec) in enumerate(zip(text, embedded[0])):
    print(f"  '{char}': [{vec[0].item():.3f}, {vec[1].item():.3f}, ..., {vec[-1].item():.3f}]")

print("""
【ポイント】
- 各トークン（文字）が embed_dim 次元のベクトルになる
- このベクトルは学習によって更新される
- 似た意味の単語は似たベクトルになる（学習後）
""")

# =============================================================================
# 3. Positional Encoding: 位置情報の付与
# =============================================================================
print("=" * 60)
print("3. Positional Encoding: 位置情報を追加")
print("=" * 60)

from src.embedding import PositionalEncoding

pos_encoding = PositionalEncoding(embed_dim, max_len=100, dropout=0.0)

# 位置エンコーディングを取得
pe = pos_encoding.get_encoding(5)  # 5位置分
print(f"\n位置エンコーディングの形状: {pe.shape}")

print("\n各位置のエンコーディング（最初の4次元）:")
for i in range(5):
    print(f"  位置{i}: [{pe[i, 0].item():.3f}, {pe[i, 1].item():.3f}, {pe[i, 2].item():.3f}, {pe[i, 3].item():.3f}]")

# Embeddingに位置情報を加算
with_position = pos_encoding(embedded)
print(f"\n位置情報追加後の形状: {with_position.shape}")

print("""
【ポイント】
- Transformerは順序を考慮しないので、位置情報を明示的に追加
- sin/cos関数を使って各位置に固有のパターンを生成
- 位置0と位置1は異なるベクトルが加算される
""")

# =============================================================================
# 4. Self-Attention: 最も重要な部分！
# =============================================================================
print("=" * 60)
print("4. Self-Attention: トークン間の関係性を計算")
print("=" * 60)

from src.attention import MultiHeadAttention, create_causal_mask

# 簡単な例で説明
print("\n--- 簡単な例 ---")
simple_text = "猫は"
simple_ids = tokenizer.encode(simple_text)
simple_input = torch.tensor([simple_ids])
simple_embedded = embedding(simple_input)

# 1ヘッドのAttention
attention = MultiHeadAttention(embed_dim, num_heads=1, dropout=0.0)

# Causal Mask（未来を見ない）
mask = create_causal_mask(len(simple_text))
print(f"\nCausal Mask (True=参照可能):")
print(mask.int())

output, attn_weights = attention(simple_embedded, mask)
print(f"\nAttention重み (どのトークンに注目しているか):")
print(f"形状: {attn_weights.shape} (batch, heads, seq, seq)")

# Attention重みを可視化
weights = attn_weights[0, 0].detach()  # (seq, seq)
print(f"\n各トークンが他のトークンにどれだけ注目しているか:")
for i, char in enumerate(simple_text):
    print(f"  '{char}' の注目先:")
    for j, target_char in enumerate(simple_text):
        if j <= i:  # Causal maskにより未来は見えない
            print(f"    → '{target_char}': {weights[i, j].item():.3f}")

print("""
【ポイント】
- 各トークンが「どの他のトークンに注目すべきか」を学習
- Causal Maskにより、未来のトークンは参照できない
- 「猫」を生成するとき、「は」はまだ見えない
- Attention重みの各行は合計1.0（確率分布）
""")

# =============================================================================
# 5. Query, Key, Value の仕組み
# =============================================================================
print("=" * 60)
print("5. Query, Key, Value の仕組み")
print("=" * 60)

print("""
Attentionは「検索」のようなもの:

1. Query (Q): 「何を探しているか」
   - 各トークンが「自分に関連する情報は何か？」を表現

2. Key (K): 「何を持っているか」
   - 各トークンが「自分はこういう情報を持っている」を表現

3. Value (V): 「実際の情報」
   - 各トークンが持つ実際の情報

計算の流れ:
  1. Q と K の類似度を計算 → Attention Score
  2. Softmaxで正規化 → Attention Weight
  3. Weight で V を重み付け平均 → 出力
""")

# 実際の計算を見てみる
import math

# 入力
x = simple_embedded  # (1, 3, 8)
batch_size, seq_len, _ = x.shape

# Q, K, V を計算（線形変換）
Q = attention.q_proj(x)  # (1, 3, 8)
K = attention.k_proj(x)
V = attention.v_proj(x)

print(f"Q, K, V の形状: {Q.shape}")

# Attention Score = Q @ K^T / sqrt(d)
scale = math.sqrt(embed_dim)
scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
print(f"\nAttention Score (正規化前):")
print(scores[0].detach())

# Causal Maskを適用
scores_masked = scores.masked_fill(~mask, float('-inf'))
print(f"\nMask適用後 (-inf = 参照不可):")
print(scores_masked[0].detach())

# Softmaxで確率に変換
weights_manual = F.softmax(scores_masked, dim=-1)
print(f"\nSoftmax後 (Attention Weight):")
print(weights_manual[0].detach())

# =============================================================================
# 6. Multi-Head Attention
# =============================================================================
print("\n" + "=" * 60)
print("6. Multi-Head Attention: 複数の視点で見る")
print("=" * 60)

multi_head = MultiHeadAttention(embed_dim, num_heads=2, dropout=0.0)
output, attn_weights = multi_head(simple_embedded, mask)

print(f"\nAttention重みの形状: {attn_weights.shape}")
print("(batch=1, heads=2, seq=3, seq=3)")

print("\nHead 0 の注目パターン:")
print(attn_weights[0, 0].detach())

print("\nHead 1 の注目パターン:")
print(attn_weights[0, 1].detach())

print("""
【ポイント】
- 複数のヘッドが異なる「視点」で情報を見る
- Head 0: 文法的な関係を見るかも
- Head 1: 意味的な関係を見るかも
- 最終的に全ヘッドの出力を結合
""")

# =============================================================================
# 7. Feed Forward Network
# =============================================================================
print("=" * 60)
print("7. Feed Forward Network: 非線形変換")
print("=" * 60)

from src.feedforward import FeedForward

ff = FeedForward(embed_dim, ff_dim=32, dropout=0.0)
ff_output = ff(output)

print(f"\n入力形状: {output.shape}")
print(f"出力形状: {ff_output.shape}")

print("""
【ポイント】
- Attentionの後に適用される2層のニューラルネット
- 中間層は通常4倍の次元（8 → 32 → 8）
- GELU活性化関数で非線形性を追加
- 各位置に独立して適用（position-wise）
""")

# =============================================================================
# 8. Transformer Block 全体
# =============================================================================
print("=" * 60)
print("8. Transformer Block: 全部組み合わせる")
print("=" * 60)

from src.transformer import TransformerBlock

block = TransformerBlock(embed_dim, num_heads=2, ff_dim=32, dropout=0.0)
block_output, block_attn = block(simple_embedded, mask)

print(f"\n入力形状: {simple_embedded.shape}")
print(f"出力形状: {block_output.shape}")

print("""
Transformer Block の構造 (Pre-LN):

    入力
      ↓
    Layer Norm
      ↓
    Multi-Head Attention
      ↓
    + ←── 残差接続（入力を足す）
      ↓
    Layer Norm
      ↓
    Feed Forward
      ↓
    + ←── 残差接続
      ↓
    出力

【残差接続のポイント】
- 入力をそのまま出力に足す
- 勾配が消失しにくくなる
- 深いネットワークでも学習できる
""")

# =============================================================================
# 9. 完全なモデル
# =============================================================================
print("=" * 60)
print("9. 完全なMiniGPTモデル")
print("=" * 60)

from src.model import MiniGPT, ModelConfig

config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    ff_dim=256,
    max_len=32
)

model = MiniGPT(config)
print(f"\nモデル構成:")
print(f"  語彙サイズ: {config.vocab_size}")
print(f"  埋め込み次元: {config.embed_dim}")
print(f"  ヘッド数: {config.num_heads}")
print(f"  層数: {config.num_layers}")
print(f"  パラメータ数: {model.count_parameters():,}")

# Forward pass
test_input = torch.tensor([[4, 5, 6, 7, 8]])  # 適当なトークンID
logits = model(test_input)

print(f"\n入力形状: {test_input.shape} (batch=1, seq=5)")
print(f"出力形状: {logits.shape} (batch=1, seq=5, vocab={config.vocab_size})")

# 次のトークンの予測
probs = F.softmax(logits[0, -1], dim=-1)  # 最後の位置の確率分布
top5 = torch.topk(probs, 5)
print(f"\n最後の位置での次トークン予測 (Top 5):")
for prob, idx in zip(top5.values, top5.indices):
    char = tokenizer.decode([idx.item()])
    print(f"  '{char}' (ID={idx.item()}): {prob.item():.4f}")

# =============================================================================
# 10. 学習の仕組み
# =============================================================================
print("\n" + "=" * 60)
print("10. 学習の仕組み: 次のトークンを予測")
print("=" * 60)

print("""
【学習の目標】
入力: "こんにち"
正解: "んにちは"  ← 1つずらしたもの

各位置で「次のトークン」を予測:
  位置0: "こ" → "ん" を予測
  位置1: "ん" → "に" を予測
  位置2: "に" → "ち" を予測
  位置3: "ち" → "は" を予測

【損失関数: Cross-Entropy】
- 予測確率と正解の差を計算
- 正解トークンの確率が高いほど損失が小さい
""")

# 実際の損失計算
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# 例: "こんにち" → "んにちは"
input_text = "こんにち"
target_text = "んにちは"

input_ids = torch.tensor([tokenizer.encode(input_text)])
target_ids = torch.tensor([tokenizer.encode(target_text)])

print(f"\n入力: '{input_text}' → {input_ids.tolist()}")
print(f"正解: '{target_text}' → {target_ids.tolist()}")

# Forward
logits = model(input_ids)

# 損失計算
loss = criterion(
    logits.view(-1, config.vocab_size),  # (batch*seq, vocab)
    target_ids.view(-1)                   # (batch*seq,)
)
print(f"\n損失: {loss.item():.4f}")

print("""
【学習ループ】
1. 入力をモデルに通す → 予測を得る
2. 予測と正解の損失を計算
3. 損失を逆伝播 → 勾配を計算
4. 勾配でパラメータを更新
5. 繰り返し → 損失が下がる → 予測精度が上がる
""")

# =============================================================================
# 11. テキスト生成の仕組み
# =============================================================================
print("=" * 60)
print("11. テキスト生成: 自己回帰生成")
print("=" * 60)

print("""
【自己回帰生成】
1. プロンプト "今日は" を入力
2. モデルが次のトークンの確率を出力
3. 確率に基づいてトークンをサンプリング
4. サンプリングしたトークンを入力に追加
5. 2-4を繰り返す

例:
  "今日は" → "天" を予測
  "今日は天" → "気" を予測
  "今日は天気" → "が" を予測
  ...
""")

# 簡単な生成デモ
print("\n--- 生成デモ ---")
prompt = "今"
prompt_ids = tokenizer.encode(prompt)
generated = prompt_ids.copy()

print(f"プロンプト: '{prompt}'")
print(f"生成過程:")

for step in range(5):
    input_tensor = torch.tensor([generated])
    logits = model(input_tensor)
    
    # 最後の位置の確率
    probs = F.softmax(logits[0, -1] / 0.8, dim=-1)  # temperature=0.8
    
    # サンプリング
    next_token = torch.multinomial(probs, 1).item()
    generated.append(next_token)
    
    current_text = tokenizer.decode(generated)
    next_char = tokenizer.decode([next_token])
    print(f"  Step {step+1}: '{next_char}' を追加 → '{current_text}'")

print(f"\n最終結果: '{tokenizer.decode(generated)}'")

print("""
【Temperature】
- 低い (0.1): 確率の高いトークンを選びやすい → 決定的
- 高い (1.5): 確率の低いトークンも選ばれやすい → 多様性

【Top-k サンプリング】
- 上位k個のトークンからのみサンプリング
- 変な出力を防ぐ
""")

print("\n" + "=" * 60)
print("まとめ")
print("=" * 60)
print("""
Transformerの核心:
1. Embedding: テキストをベクトルに変換
2. Positional Encoding: 位置情報を追加
3. Self-Attention: トークン間の関係性を学習
4. Feed Forward: 非線形変換
5. 残差接続: 深いネットワークでも学習可能に

学習:
- 「次のトークンを予測する」タスクで学習
- 大量のテキストで学習 → パターンを記憶

生成:
- 自己回帰的に1トークンずつ生成
- Temperature/Top-kで多様性を制御
""")
