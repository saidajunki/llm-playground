"""
Attention mechanisms for Mini Transformer.

Self-AttentionとMulti-Head Attentionを実装。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Args:
        embed_dim: 埋め込み次元
        dropout: ドロップアウト率
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-Attentionを計算。
        
        Args:
            x: (batch_size, seq_len, embed_dim) 入力
            mask: (seq_len, seq_len) Causal mask (Trueで参照可能)
            
        Returns:
            output: (batch_size, seq_len, embed_dim) 出力
            attention_weights: (batch_size, seq_len, seq_len) Attention重み
        """
        # Q, K, V を計算
        Q = self.query(x)  # (batch, seq, embed)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, seq, seq)
        
        # Causal maskを適用
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmaxで正規化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 重み付き和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    
    複数のAttentionヘッドを並列に実行し、結果を結合する。
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Args:
        embed_dim: 埋め込み次元
        num_heads: ヘッド数
        dropout: ドロップアウト率
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V の線形変換
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 出力の線形変換
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-Head Attentionを計算。
        
        Args:
            x: (batch_size, seq_len, embed_dim) 入力
            mask: (seq_len, seq_len) Causal mask (Trueで参照可能)
            
        Returns:
            output: (batch_size, seq_len, embed_dim) 出力
            attention_weights: (batch_size, num_heads, seq_len, seq_len) Attention重み
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算して複数ヘッドに分割
        # (batch, seq, embed) -> (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (batch, num_heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal maskを適用
        if mask is not None:
            # maskを(1, 1, seq, seq)に拡張してブロードキャスト
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmaxで正規化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 重み付き和: (batch, num_heads, seq, head_dim)
        context = torch.matmul(attention_weights, V)
        
        # ヘッドを結合: (batch, seq, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 出力の線形変換
        output = self.out_proj(context)
        
        return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Causal mask（下三角行列）を作成。
    
    位置iは位置j <= iのみ参照可能。
    
    Args:
        seq_len: シーケンス長
        device: デバイス
        
    Returns:
        (seq_len, seq_len) マスク (Trueで参照可能)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask
