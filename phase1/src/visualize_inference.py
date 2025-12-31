"""
推論時の内部状態を可視化するツール

各レイヤーでの:
1. Attention重み（どのトークンに注目しているか）
2. 隠れ状態の変化
3. 最終的な予測確率

使用方法:
    python -m src.visualize_inference \
        --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "日本の首都は"
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなしで動作
import numpy as np
import seaborn as sns

from .model import MiniGPT, ModelConfig
from .tokenizer import CharTokenizer


def load_model(checkpoint_path: str, device: str) -> Tuple[MiniGPT, CharTokenizer]:
    """モデルとトークナイザーを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_or_dict = checkpoint.get('config', {})
    
    # ModelConfigオブジェクトの場合はそのまま使用
    if isinstance(config_or_dict, ModelConfig):
        config = config_or_dict
    else:
        # 辞書の場合は新しいModelConfigを作成
        config = ModelConfig(
            vocab_size=config_or_dict.get('vocab_size', 5000),
            embed_dim=config_or_dict.get('embed_dim', 384),
            num_heads=config_or_dict.get('num_heads', 6),
            num_layers=config_or_dict.get('num_layers', 6),
            ff_dim=config_or_dict.get('ff_dim', 1536),
            max_len=config_or_dict.get('max_len', 128),
            dropout=config_or_dict.get('dropout', 0.1),
        )
    
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer_path = Path(checkpoint_path).parent / "tokenizer.json"
    tokenizer = CharTokenizer()
    tokenizer.load(str(tokenizer_path))
    
    return model, tokenizer


def get_attention_weights(
    model: MiniGPT, 
    input_ids: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Attention重みを取得"""
    with torch.no_grad():
        logits, attention_weights = model(input_ids, return_attention=True)
    return logits, attention_weights


def get_hidden_states(
    model: MiniGPT,
    input_ids: torch.Tensor
) -> List[torch.Tensor]:
    """各層の隠れ状態を取得"""
    hidden_states = []
    
    with torch.no_grad():
        # Embedding
        x = model.token_embedding(input_ids)
        x = model.pos_encoding(x)
        hidden_states.append(x.clone())
        
        # 各Transformer Block
        from .attention import create_causal_mask
        mask = create_causal_mask(input_ids.size(1), input_ids.device)
        
        for block in model.blocks:
            x, _ = block(x, mask)
            hidden_states.append(x.clone())
        
        # Final LayerNorm
        x = model.ln_final(x)
        hidden_states.append(x.clone())
    
    return hidden_states


def plot_attention_heatmap(
    attention_weights: List[torch.Tensor],
    tokens: List[str],
    output_path: str = None,
    layer_idx: int = None
):
    """Attention重みをヒートマップで表示"""
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)
    
    if layer_idx is not None:
        # 特定の層のみ
        layers_to_plot = [layer_idx]
    else:
        # 全層
        layers_to_plot = range(num_layers)
    
    for layer in layers_to_plot:
        attn = attention_weights[layer][0]  # (num_heads, seq_len, seq_len)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Layer {layer} Attention Weights', fontsize=14)
        
        for head in range(min(6, num_heads)):
            ax = axes[head // 3, head % 3]
            head_attn = attn[head].cpu().numpy()
            
            sns.heatmap(
                head_attn,
                ax=ax,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                vmin=0,
                vmax=1,
            )
            ax.set_title(f'Head {head}')
            ax.set_xlabel('Key (attending to)')
            ax.set_ylabel('Query (from)')
        
        plt.tight_layout()
        
        if output_path:
            save_path = output_path.replace('.png', f'_layer{layer}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存: {save_path}")
        plt.close()


def plot_attention_flow(
    attention_weights: List[torch.Tensor],
    tokens: List[str],
    output_path: str = None
):
    """全層のAttentionフローを可視化（最後のトークンがどこに注目しているか）"""
    num_layers = len(attention_weights)
    
    # 各層で最後のトークンがどこに注目しているか（全ヘッドの平均）
    attention_to_last = []
    for layer_attn in attention_weights:
        # (1, num_heads, seq_len, seq_len) -> 最後の行を取得
        last_token_attn = layer_attn[0, :, -1, :].mean(dim=0)  # 全ヘッド平均
        attention_to_last.append(last_token_attn.cpu().numpy())
    
    attention_matrix = np.array(attention_to_last)  # (num_layers, seq_len)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        attention_matrix,
        ax=ax,
        xticklabels=tokens,
        yticklabels=[f'Layer {i}' for i in range(num_layers)],
        cmap='Reds',
        vmin=0,
    )
    ax.set_title('Attention Flow: Where the last token looks at each layer')
    ax.set_xlabel('Input Tokens')
    ax.set_ylabel('Layer')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    plt.close()


def plot_hidden_state_norms(
    hidden_states: List[torch.Tensor],
    tokens: List[str],
    output_path: str = None
):
    """各層での隠れ状態のノルムを可視化"""
    norms = []
    for h in hidden_states:
        # (1, seq_len, embed_dim) -> (seq_len,)
        norm = h[0].norm(dim=-1).cpu().numpy()
        norms.append(norm)
    
    norms_matrix = np.array(norms)  # (num_layers+2, seq_len)
    
    labels = ['Embedding'] + [f'Layer {i}' for i in range(len(hidden_states)-2)] + ['Final LN']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        norms_matrix,
        ax=ax,
        xticklabels=tokens,
        yticklabels=labels,
        cmap='viridis',
    )
    ax.set_title('Hidden State Norms Through Layers')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Layer')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    plt.close()


def plot_top_predictions(
    logits: torch.Tensor,
    tokenizer: CharTokenizer,
    tokens: List[str],
    output_path: str = None,
    top_k: int = 10
):
    """各位置でのTop-K予測を表示"""
    probs = F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
    
    fig, axes = plt.subplots(len(tokens), 1, figsize=(10, 3 * len(tokens)))
    if len(tokens) == 1:
        axes = [axes]
    
    for pos, (ax, token) in enumerate(zip(axes, tokens)):
        pos_probs = probs[pos].cpu().numpy()
        top_indices = np.argsort(pos_probs)[-top_k:][::-1]
        top_probs = pos_probs[top_indices]
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
        colors = ['green' if i == 0 else 'steelblue' for i in range(top_k)]
        ax.barh(range(top_k), top_probs, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel('Probability')
        ax.set_title(f'After "{token}" → Top {top_k} predictions')
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    plt.close()


def print_inference_summary(
    tokens: List[str],
    logits: torch.Tensor,
    attention_weights: List[torch.Tensor],
    tokenizer: CharTokenizer
):
    """推論のサマリーをテキストで表示"""
    print("\n" + "="*60)
    print("推論サマリー")
    print("="*60)
    
    print(f"\n入力トークン: {tokens}")
    print(f"トークン数: {len(tokens)}")
    
    # 最後の位置での予測
    probs = F.softmax(logits[0, -1], dim=-1)
    top_probs, top_indices = probs.topk(5)
    
    print(f"\n次のトークン予測 (Top 5):")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        print(f"  '{token}': {prob.item():.4f} ({prob.item()*100:.1f}%)")
    
    # 各層のAttention統計
    print(f"\n各層のAttention統計 (最後のトークン):")
    for layer_idx, attn in enumerate(attention_weights):
        # 最後のトークンが最も注目している位置
        last_attn = attn[0, :, -1, :].mean(dim=0)  # 全ヘッド平均
        max_pos = last_attn.argmax().item()
        max_val = last_attn[max_pos].item()
        print(f"  Layer {layer_idx}: 最も注目 → '{tokens[max_pos]}' ({max_val:.3f})")


def main():
    parser = argparse.ArgumentParser(description="推論の可視化")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="visualizations")
    parser.add_argument("--layer", type=int, default=None, help="特定の層のみ可視化")
    args = parser.parse_args()
    
    device = "cpu"  # 可視化はCPUで十分
    
    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, device)
    
    # トークナイズ
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], device=device)
    tokens = [tokenizer.decode([t]) for t in input_ids]
    
    print(f"Input: {args.prompt}")
    print(f"Tokens: {tokens}")
    
    # 推論
    logits, attention_weights = get_attention_weights(model, input_tensor)
    hidden_states = get_hidden_states(model, input_tensor)
    
    # サマリー表示
    print_inference_summary(tokens, logits, attention_weights, tokenizer)
    
    # 可視化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Attention ヒートマップ
    plot_attention_heatmap(
        attention_weights, tokens,
        str(output_dir / "attention_heatmap.png"),
        layer_idx=args.layer
    )
    
    # 2. Attention フロー
    plot_attention_flow(
        attention_weights, tokens,
        str(output_dir / "attention_flow.png")
    )
    
    # 3. 隠れ状態のノルム
    plot_hidden_state_norms(
        hidden_states, tokens,
        str(output_dir / "hidden_norms.png")
    )
    
    # 4. Top予測
    plot_top_predictions(
        logits, tokenizer, tokens,
        str(output_dir / "top_predictions.png")
    )
    
    print(f"\n可視化を {output_dir} に保存しました")


if __name__ == "__main__":
    main()
