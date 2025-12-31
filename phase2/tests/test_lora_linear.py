"""
LoRALinear 単体テスト
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# phase2/src をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora.linear import LoRALinear
from lora.config import LoRAConfig


class TestLoRALinear:
    """LoRALinearの単体テスト"""
    
    def test_initialization_b_zeros(self):
        """B行列がゼロで初期化されることを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))
    
    def test_initialization_a_not_zeros(self):
        """A行列がゼロでないことを確認（Kaiming初期化）"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))
    
    def test_original_weights_frozen(self):
        """元の重みが凍結されていることを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        assert not lora.original_layer.weight.requires_grad
        assert not lora.original_layer.bias.requires_grad
    
    def test_lora_weights_trainable(self):
        """LoRA重みが学習可能であることを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad
    
    def test_scaling_factor(self):
        """スケーリング係数が正しく計算されることを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        assert lora.scaling == 16.0 / 8
    
    def test_initial_output_equals_original(self):
        """初期状態でLoRAの出力が元の出力と等しいことを確認"""
        original = nn.Linear(64, 32)
        original_weight = original.weight.data.clone()
        original_bias = original.bias.data.clone()
        
        lora = LoRALinear(original, rank=8, alpha=16.0, dropout=0.0)
        
        x = torch.randn(4, 64)
        
        # 元の出力を計算
        expected = x @ original_weight.T + original_bias
        
        # LoRAの出力
        actual = lora(x)
        
        # B=0なので、LoRAの寄与はゼロ
        assert torch.allclose(actual, expected, atol=1e-6)
    
    def test_forward_shape(self):
        """出力形状が正しいことを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        x = torch.randn(4, 16, 64)  # (batch, seq, features)
        output = lora(x)
        
        assert output.shape == (4, 16, 32)
    
    def test_merge_weights(self):
        """重みのマージが正しく動作することを確認"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0, dropout=0.0)
        
        # LoRA重みを変更
        lora.lora_A.data = torch.randn_like(lora.lora_A)
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        # マージ
        merged = lora.merge_weights()
        
        # 同じ入力で出力を比較
        x = torch.randn(4, 64)
        lora_output = lora(x)
        merged_output = merged(x)
        
        assert torch.allclose(lora_output, merged_output, atol=1e-5)
    
    def test_merge_weights_formula(self):
        """マージの計算式が正しいことを確認: W_merged = W + B @ A * scaling"""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0)
        
        # LoRA重みを設定
        lora.lora_A.data = torch.randn_like(lora.lora_A)
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        # 期待されるマージ重み
        expected_weight = original.weight.data + (lora.lora_B @ lora.lora_A) * lora.scaling
        
        # マージ
        merged = lora.merge_weights()
        
        assert torch.allclose(merged.weight.data, expected_weight, atol=1e-6)


class TestLoRAConfig:
    """LoRAConfigの単体テスト"""
    
    def test_default_values(self):
        """デフォルト値が正しいことを確認"""
        config = LoRAConfig()
        
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules
    
    def test_scaling_property(self):
        """scalingプロパティが正しく計算されることを確認"""
        config = LoRAConfig(rank=4, alpha=8.0)
        
        assert config.scaling == 2.0
    
    def test_to_dict_from_dict(self):
        """辞書への変換と復元が正しく動作することを確認"""
        config = LoRAConfig(rank=16, alpha=32.0, dropout=0.1)
        
        d = config.to_dict()
        restored = LoRAConfig.from_dict(d)
        
        assert restored.rank == config.rank
        assert restored.alpha == config.alpha
        assert restored.dropout == config.dropout
        assert restored.target_modules == config.target_modules
    
    def test_save_load(self, tmp_path):
        """ファイルへの保存と読み込みが正しく動作することを確認"""
        config = LoRAConfig(rank=16, alpha=32.0, dropout=0.1)
        
        save_path = tmp_path / "lora_config.json"
        config.save(str(save_path))
        
        loaded = LoRAConfig.load(str(save_path))
        
        assert loaded.rank == config.rank
        assert loaded.alpha == config.alpha
        assert loaded.dropout == config.dropout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
