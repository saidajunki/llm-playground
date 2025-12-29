"""
Embedding layers for Mini Transformer.

Token EmbeddingとPositional Encodingを実装。
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    トークンIDを埋め込みベクトルに変換するレイヤー。
    
    Args:
        vocab_size: 語彙サイズ
        embed_dim: 埋め込み次元
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        トークンIDを埋め込みベクトルに変換。
        
        Args:
            x: (batch_size, seq_len) トークンID
            
        Returns:
            (batch_size, seq_len, embed_dim) 埋め込みベクトル
        """
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    位置情報をsin/cosで表現し、埋め込みベクトルに加算する。
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        embed_dim: 埋め込み次元
        max_len: 最大シーケンス長
        dropout: ドロップアウト率
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        max_len: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional Encodingを事前計算
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        # 偶数インデックスにsin、奇数インデックスにcos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (1, max_len, embed_dim)の形状でバッファとして登録
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        埋め込みベクトルに位置情報を加算。
        
        Args:
            x: (batch_size, seq_len, embed_dim) 埋め込みベクトル
            
        Returns:
            (batch_size, seq_len, embed_dim) 位置情報が加算されたベクトル
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        指定した長さのPositional Encodingを取得。
        
        Args:
            seq_len: シーケンス長
            
        Returns:
            (seq_len, embed_dim) Positional Encoding
        """
        return self.pe[0, :seq_len, :]
