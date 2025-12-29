"""
Feed Forward Network for Mini Transformer.

Attention層の後に適用される2層の全結合ネットワーク。
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    Args:
        embed_dim: 入出力の次元
        ff_dim: 中間層の次元（通常は4 * embed_dim）
        dropout: ドロップアウト率
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed Forward Networkを適用。
        
        Args:
            x: (batch_size, seq_len, embed_dim) 入力
            
        Returns:
            (batch_size, seq_len, embed_dim) 出力
        """
        return self.net(x)
