"""
Text generation for Mini Transformer.

自己回帰生成、Temperature調整、Top-kサンプリングを実装。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import MiniGPT, ModelConfig
from .tokenizer import CharTokenizer
from .train import load_checkpoint


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    テキストを生成する。
    
    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_new_tokens: 生成する最大トークン数
        temperature: サンプリング温度（0に近いほど決定的）
        top_k: Top-kサンプリングのk値（Noneで無効）
        device: デバイス
        
    Returns:
        生成されたテキスト
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # プロンプトをトークン化
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 自己回帰生成
    for _ in range(max_new_tokens):
        # コンテキスト長を超えないように切り詰め
        if input_ids.size(1) > model.config.max_len:
            input_ids = input_ids[:, -model.config.max_len:]
        
        # Forward pass
        logits = model(input_ids)
        
        # 最後の位置のlogitsを取得
        logits = logits[:, -1, :]  # (batch, vocab)
        
        # Temperature調整
        if temperature > 0:
            logits = logits / temperature
        
        # Top-kサンプリング
        if top_k is not None and top_k > 0:
            logits = top_k_filtering(logits, top_k)
        
        # 確率分布に変換
        probs = F.softmax(logits, dim=-1)
        
        # サンプリング
        if temperature == 0:
            # Greedy decoding
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # 確率的サンプリング
            next_token = torch.multinomial(probs, num_samples=1)
        
        # 入力に追加
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # EOSトークンで終了
        if next_token.item() == tokenizer.eos_id:
            break
    
    # デコード
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def top_k_filtering(
    logits: torch.Tensor,
    top_k: int
) -> torch.Tensor:
    """
    Top-kサンプリング用にlogitsをフィルタリング。
    
    上位k個以外のlogitsを-infに設定。
    
    Args:
        logits: (batch, vocab) logits
        top_k: 保持する上位k個
        
    Returns:
        フィルタリングされたlogits
    """
    if top_k <= 0:
        return logits
    
    # 上位k個の値を取得
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    
    # k番目の値を閾値として使用
    threshold = values[:, -1].unsqueeze(-1)
    
    # 閾値未満を-infに
    logits = torch.where(
        logits < threshold,
        torch.full_like(logits, float('-inf')),
        logits
    )
    
    return logits


@torch.no_grad()
def generate_with_attention(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: Optional[torch.device] = None
) -> Tuple[str, List[torch.Tensor]]:
    """
    テキストを生成し、Attention重みも返す。
    
    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_new_tokens: 生成する最大トークン数
        temperature: サンプリング温度
        device: デバイス
        
    Returns:
        generated_text: 生成されたテキスト
        attention_weights: 各ステップのAttention重み
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    all_attention_weights = []
    
    for _ in range(max_new_tokens):
        if input_ids.size(1) > model.config.max_len:
            input_ids = input_ids[:, -model.config.max_len:]
        
        logits, attention_weights = model(input_ids, return_attention=True)
        all_attention_weights.append(attention_weights)
        
        logits = logits[:, -1, :]
        
        if temperature > 0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        
        if temperature == 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_id:
            break
    
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text, all_attention_weights


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer file")
    parser.add_argument("--prompt", type=str, default="今日は", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 to disable)")
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # チェックポイント読み込み
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config_or_dict = checkpoint['config']
    
    # 設定を復元（ModelConfigオブジェクトまたは辞書に対応）
    if isinstance(config_or_dict, ModelConfig):
        config = config_or_dict
    else:
        config = ModelConfig(
            vocab_size=config_or_dict.get('vocab_size', 5000),
            embed_dim=config_or_dict.get('embed_dim', 384),
            num_heads=config_or_dict.get('num_heads', 6),
            num_layers=config_or_dict.get('num_layers', 6),
            ff_dim=config_or_dict.get('ff_dim', 1536),
            max_len=config_or_dict.get('max_len', 128),
            dropout=config_or_dict.get('dropout', 0.1),
        )
    
    # モデル作成と重み読み込み
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # トークナイザー読み込み
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = Path(args.checkpoint).parent / "tokenizer.json"
    
    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    # 生成
    generated = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        device=device
    )
    
    print(f"Generated:\n{generated}")


if __name__ == "__main__":
    main()
