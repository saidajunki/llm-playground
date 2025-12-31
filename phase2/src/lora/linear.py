"""
LoRA線形層

元の線形層にLoRA（低ランク適応）を適用するモジュール

数学的原理:
    元の線形層: y = Wx + b
    LoRA適用後: y = Wx + (B @ A)x * (α/r) + b
    
    - W: 元の重み行列 (d_out × d_in)、凍結
    - A: 低ランク行列 (r × d_in)、学習対象
    - B: 低ランク行列 (d_out × r)、学習対象
    - r: ランク
    - α: スケーリング係数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRAを適用した線形層"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0
    ):
        """
        Args:
            original_layer: 元の線形層
            rank: 低ランク行列のランク
            alpha: スケーリング係数
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA行列
        # A: (rank, in_features) - 入力を低次元に射影
        # B: (out_features, rank) - 低次元から出力次元に射影
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank), requires_grad=True)
        
        # ドロップアウト
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初期化
        # A: Kaiming uniform初期化（学習開始時に適切な勾配を得るため）
        # B: ゼロ初期化（初期状態でLoRAの寄与をゼロにするため）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 元の重みを凍結
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル (..., in_features)
            
        Returns:
            出力テンソル (..., out_features)
        """
        # 元の線形層の出力
        original_output = self.original_layer(x)
        
        # LoRA出力: (B @ A) @ x * scaling
        # 効率的な計算: x @ A^T @ B^T * scaling
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(lora_input, self.lora_A)  # x @ A^T -> (..., rank)
        lora_output = F.linear(lora_output, self.lora_B)  # (..., rank) @ B^T -> (..., out_features)
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """
        LoRA重みを元の重みにマージして新しい線形層を返す
        
        マージ後は推論時のオーバーヘッドなしで学習効果を得られる
        
        Returns:
            マージされた重みを持つ新しい線形層
        """
        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None,
            device=self.original_layer.weight.device,
            dtype=self.original_layer.weight.dtype
        )
        
        # W_merged = W + B @ A * scaling
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.original_layer.weight.data + delta_w
        
        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data.clone()
        
        return merged
    
    def get_lora_parameters(self):
        """LoRAパラメータのみを返す"""
        return [self.lora_A, self.lora_B]
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.original_layer.in_features}, "
            f"out_features={self.original_layer.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )
