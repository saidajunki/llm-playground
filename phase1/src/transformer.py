"""
Transformer Block for Mini Transformer.

Multi-Head Attention + Feed Forward Network + Layer Normalization + Residual Connection
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block (Pre-LN構成).
    
    Pre-LN: Layer Normを先に適用する構成。学習が安定しやすい。
    
    x -> LN -> Attention -> + -> LN -> FFN -> + -> output
    |________________________|   |______________|
           (residual)              (residual)
    
    Args:
        embed_dim: 埋め込み次元
        num_heads: Attentionヘッド数
        ff_dim: Feed Forward中間層の次元
        dropout: ドロップアウト率
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed Forward Network
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        # Dropout for residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transformer Blockを適用。
        
        Args:
            x: (batch_size, seq_len, embed_dim) 入力
            mask: (seq_len, seq_len) Causal mask
            
        Returns:
            output: (batch_size, seq_len, embed_dim) 出力
            attention_weights: (batch_size, num_heads, seq_len, seq_len) Attention重み
        """
        # Pre-LN + Attention + Residual
        normed = self.ln1(x)
        attn_out, attn_weights = self.attention(normed, mask)
        x = x + self.dropout(attn_out)
        
        # Pre-LN + FFN + Residual
        normed = self.ln2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x, attn_weights
