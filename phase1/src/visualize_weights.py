"""
学習中のパラメータ変化を可視化するツール

使用方法:
    python -m src.visualize_weights --checkpoints checkpoints/
"""

import argparse
from pathlib import Path
from typing import Dict, List
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_checkpoint_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """チェックポイントから重みを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['model_state_dict']


def get_weight_stats(weights: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """各層の重みの統計情報を計算"""
    stats = {}
    for name, tensor in weights.items():
        if tensor.numel() > 0:
            stats[name] = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
            }
    return stats


def compare_checkpoints(checkpoint_paths: List[str]) -> Dict[str, List[Dict]]:
    """複数のチェックポイントを比較"""
    all_stats = {}
    
    for path in checkpoint_paths:
        weights = load_checkpoint_weights(path)
        stats = get_weight_stats(weights)
        
        for name, stat in stats.items():
            if name not in all_stats:
                all_stats[name] = []
            all_stats[name].append(stat)
    
    return all_stats


def plot_weight_evolution(checkpoint_dir: str, output_path: str = None):
    """重みの変化をプロット"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # チェックポイントを時系列順にソート
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    
    if not checkpoints:
        print("チェックポイントが見つかりません")
        return
    
    print(f"見つかったチェックポイント: {len(checkpoints)}個")
    
    # 各チェックポイントの統計を収集
    steps = []
    layer_stats = {}
    
    for cp_path in checkpoints:
        # ステップ数を抽出
        step = int(cp_path.stem.split('step')[-1])
        steps.append(step)
        
        weights = load_checkpoint_weights(str(cp_path))
        stats = get_weight_stats(weights)
        
        for name, stat in stats.items():
            if name not in layer_stats:
                layer_stats[name] = {'mean': [], 'std': []}
            layer_stats[name]['mean'].append(stat['mean'])
            layer_stats[name]['std'].append(stat['std'])
    
    # 主要な層のみプロット
    key_layers = [
        'token_embedding.embedding.weight',
        'blocks.0.attention.q_proj.weight',
        'blocks.0.attention.out_proj.weight',
        'blocks.0.ff.net.0.weight',
        'blocks.5.attention.q_proj.weight',
        'blocks.5.ff.net.0.weight',
        'output.weight',
    ]
    
    # 存在する層のみフィルタ
    key_layers = [l for l in key_layers if l in layer_stats]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 平均値の変化
    ax1 = axes[0]
    for layer in key_layers:
        short_name = layer.split('.')[-2] + '.' + layer.split('.')[-1]
        ax1.plot(steps, layer_stats[layer]['mean'], label=short_name, marker='o', markersize=3)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Weight')
    ax1.set_title('Weight Mean Evolution')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 標準偏差の変化
    ax2 = axes[1]
    for layer in key_layers:
        short_name = layer.split('.')[-2] + '.' + layer.split('.')[-1]
        ax2.plot(steps, layer_stats[layer]['std'], label=short_name, marker='o', markersize=3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Std Weight')
    ax2.set_title('Weight Std Evolution')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    else:
        plt.show()


def plot_weight_histogram(checkpoint_path: str, output_path: str = None):
    """特定チェックポイントの重み分布をヒストグラムで表示"""
    weights = load_checkpoint_weights(checkpoint_path)
    
    # 主要な層を選択
    key_layers = [
        ('token_embedding.embedding.weight', 'Token Embedding'),
        ('blocks.0.attention.q_proj.weight', 'Attention Q (Layer 0)'),
        ('blocks.0.ff.net.0.weight', 'FFN (Layer 0)'),
        ('blocks.5.attention.q_proj.weight', 'Attention Q (Layer 5)'),
        ('output.weight', 'Output'),
    ]
    
    key_layers = [(k, n) for k, n in key_layers if k in weights]
    
    fig, axes = plt.subplots(len(key_layers), 1, figsize=(10, 3 * len(key_layers)))
    if len(key_layers) == 1:
        axes = [axes]
    
    for ax, (layer_name, display_name) in zip(axes, key_layers):
        data = weights[layer_name].flatten().numpy()
        ax.hist(data, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{display_name}\nmean={data.mean():.4f}, std={data.std():.4f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    else:
        plt.show()


def plot_attention_weights_heatmap(checkpoint_path: str, output_path: str = None):
    """Attention重み行列をヒートマップで表示"""
    weights = load_checkpoint_weights(checkpoint_path)
    
    # 各層のQ, K, V, Out重みを取得
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for layer_idx in range(min(6, len([k for k in weights if 'blocks' in k and 'q_proj' in k]))):
        ax = axes[layer_idx // 3, layer_idx % 3]
        
        q_weight = weights[f'blocks.{layer_idx}.attention.q_proj.weight'].numpy()
        
        # 大きすぎる場合はサブサンプル
        if q_weight.shape[0] > 100:
            q_weight = q_weight[::4, ::4]
        
        im = ax.imshow(q_weight, cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
        ax.set_title(f'Layer {layer_idx} Q-proj')
        ax.set_xlabel('Input dim')
        ax.set_ylabel('Output dim')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存: {output_path}")
    else:
        plt.show()


def print_weight_summary(checkpoint_path: str):
    """重みのサマリーを表示"""
    weights = load_checkpoint_weights(checkpoint_path)
    stats = get_weight_stats(weights)
    
    print(f"\n{'='*80}")
    print(f"チェックポイント: {checkpoint_path}")
    print(f"{'='*80}")
    print(f"{'Layer':<50} {'Shape':<20} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*80}")
    
    total_params = 0
    for name, stat in sorted(stats.items()):
        shape_str = str(stat['shape'])
        print(f"{name:<50} {shape_str:<20} {stat['mean']:>10.4f} {stat['std']:>10.4f}")
        total_params += stat['numel']
    
    print(f"{'-'*80}")
    print(f"総パラメータ数: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(description="重みの可視化")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="チェックポイントディレクトリ")
    parser.add_argument("--checkpoint", type=str, default=None, help="特定のチェックポイント")
    parser.add_argument("--mode", type=str, default="evolution", 
                        choices=["evolution", "histogram", "heatmap", "summary"],
                        help="可視化モード")
    parser.add_argument("--output", type=str, default=None, help="出力ファイルパス")
    args = parser.parse_args()
    
    if args.mode == "evolution":
        plot_weight_evolution(args.checkpoints, args.output)
    elif args.mode == "histogram":
        cp = args.checkpoint or str(sorted(Path(args.checkpoints).glob("checkpoint_step*.pt"))[-1])
        plot_weight_histogram(cp, args.output)
    elif args.mode == "heatmap":
        cp = args.checkpoint or str(sorted(Path(args.checkpoints).glob("checkpoint_step*.pt"))[-1])
        plot_attention_weights_heatmap(cp, args.output)
    elif args.mode == "summary":
        cp = args.checkpoint or str(sorted(Path(args.checkpoints).glob("checkpoint_step*.pt"))[-1])
        print_weight_summary(cp)


if __name__ == "__main__":
    main()
