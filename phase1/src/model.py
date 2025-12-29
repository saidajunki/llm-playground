"""
Complete Mini GPT Model.

全コンポーネントを統合したDecoder-only Transformerモデル。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .embedding import TokenEmbedding, PositionalEncoding
from .transformer import TransformerBlock
from .attention import create_causal_mask


@dataclass
class ModelConfig:
    """モデルのハイパーパラメータ設定"""
    vocab_size: int = 5000
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    ff_dim: int = 1536  # 4 * embed_dim
    max_len: int = 256
    dropout: float = 0.1
    
    def __post_init__(self):
        # ff_dimが指定されていない場合は4 * embed_dim
        if self.ff_dim is None:
            self.ff_dim = 4 * self.embed_dim


class MiniGPT(nn.Module):
    """
    Mini GPT Model (Decoder-only Transformer).
    
    構成:
    - Token Embedding
    - Positional Encoding
    - N x Transformer Block
    - Final Layer Norm
    - Output Linear (weight tied with embedding)
    
    Args:
        config: ModelConfig
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed_dim)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(
            config.embed_dim, 
            config.max_len, 
            config.dropout
        )
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ff_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final Layer Norm
        self.ln_final = nn.LayerNorm(config.embed_dim)
        
        # Output Linear (vocab_size次元に変換)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Weight Tying: 出力層とEmbeddingの重みを共有
        # これによりパラメータ数を削減し、学習を安定させる
        self.output.weight = self.token_embedding.embedding.weight
        
        # パラメータの初期化
        self._init_weights()
    
    def _init_weights(self):
        """パラメータを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len) トークンID
            return_attention: Attention重みも返すか
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) 各位置での次トークン予測
            attention_weights: (optional) 各層のAttention重みのリスト
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Causal maskを作成
        mask = create_causal_mask(seq_len, device)
        
        # Token Embedding + Positional Encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer Blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_weights.append(attn)
        
        # Final Layer Norm + Output
        x = self.ln_final(x)
        logits = self.output(x)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def count_parameters(self) -> int:
        """学習可能なパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"MiniGPT(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  embed_dim={self.config.embed_dim},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
