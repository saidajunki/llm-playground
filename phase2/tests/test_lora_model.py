"""
LoRAModel 単体テスト
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora.config import LoRAConfig
from lora.linear import LoRALinear
from lora.model import LoRAModel


class SimpleTransformer(nn.Module):
    """テスト用の簡易Transformerモデル"""
    
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        
        # Attention層（LoRAの対象）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # FFN層
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.output = nn.Linear(d_model, 1000)
    
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 簡易attention
        attn = torch.softmax(q @ k.transpose(-2, -1) / 8, dim=-1)
        x = attn @ v
        x = self.o_proj(x)
        
        x = x + self.ffn(x)
        
        return self.output(x)


class TestLoRAModel:
    """LoRAModelの単体テスト"""
    
    def test_apply_lora_to_target_modules(self):
        """対象モジュールにLoRAが適用されることを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        lora_model = LoRAModel(base_model, config)
        
        # 4つのモジュールにLoRAが適用されている
        assert len(lora_model.lora_layers) == 4
        assert "q_proj" in lora_model.lora_layers
        assert "k_proj" in lora_model.lora_layers
        assert "v_proj" in lora_model.lora_layers
        assert "o_proj" in lora_model.lora_layers
    
    def test_base_model_frozen(self):
        """ベースモデルのパラメータが凍結されていることを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        # LoRA以外のパラメータは凍結
        for name, param in base_model.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"{name} should be frozen"
    
    def test_lora_parameters_trainable(self):
        """LoRAパラメータが学習可能であることを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        trainable_params = list(lora_model.get_trainable_parameters())
        
        # 4モジュール × 2パラメータ(A, B) = 8
        assert len(trainable_params) == 8
        
        for param in trainable_params:
            assert param.requires_grad
    
    def test_parameter_count(self):
        """パラメータ数のカウントが正しいことを確認"""
        base_model = SimpleTransformer(d_model=64)
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        trainable, total = lora_model.count_parameters()
        
        # LoRAパラメータ: 4モジュール × (4×64 + 64×4) = 4 × 512 = 2048
        expected_trainable = 4 * (4 * 64 + 64 * 4)
        assert trainable == expected_trainable
        
        # 学習パラメータは全体の5%未満（テストモデルは小さいので緩和）
        # 実際のQwen2-0.5Bでは1%未満になる
        assert trainable / total < 0.05
    
    def test_forward_pass(self):
        """順伝播が正しく動作することを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        input_ids = torch.randint(0, 1000, (2, 16))
        output = lora_model(input_ids=input_ids)
        
        assert output.shape == (2, 16, 1000)
    
    def test_save_load_lora_weights(self, tmp_path):
        """LoRA重みの保存と読み込みが正しく動作することを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        # LoRA重みを変更
        for layer in lora_model.lora_layers.values():
            layer.lora_A.data = torch.randn_like(layer.lora_A)
            layer.lora_B.data = torch.randn_like(layer.lora_B)
        
        # 保存
        save_path = tmp_path / "lora_checkpoint"
        lora_model.save_lora_weights(str(save_path))
        
        # 新しいモデルに読み込み
        base_model2 = SimpleTransformer()
        lora_model2 = LoRAModel(base_model2, config)
        lora_model2.load_lora_weights(str(save_path))
        
        # 重みが一致することを確認
        for name in lora_model.lora_layers:
            assert torch.allclose(
                lora_model.lora_layers[name].lora_A,
                lora_model2.lora_layers[name].lora_A
            )
            assert torch.allclose(
                lora_model.lora_layers[name].lora_B,
                lora_model2.lora_layers[name].lora_B
            )
    
    def test_merge_and_unload(self):
        """マージが正しく動作することを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
        
        lora_model = LoRAModel(base_model, config)
        
        # LoRA重みを設定
        for layer in lora_model.lora_layers.values():
            layer.lora_A.data = torch.randn_like(layer.lora_A)
            layer.lora_B.data = torch.randn_like(layer.lora_B)
        
        # マージ前の出力
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output_before = lora_model(input_ids=input_ids).clone()
        
        # マージ
        merged_model = lora_model.merge_and_unload()
        
        # マージ後の出力
        with torch.no_grad():
            output_after = merged_model(input_ids)
        
        # 出力が一致（浮動小数点の誤差を考慮）
        assert torch.allclose(output_before, output_after, atol=1e-2, rtol=1e-3)
    
    def test_merged_model_has_no_lora_layers(self):
        """マージ後にLoRAレイヤーがないことを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        merged_model = lora_model.merge_and_unload()
        
        # LoRAレイヤーがクリアされている
        assert len(lora_model.lora_layers) == 0
        
        # モデル内にLoRALinearがない
        for module in merged_model.modules():
            assert not isinstance(module, LoRALinear)
    
    def test_checkpoint_size(self, tmp_path):
        """チェックポイントサイズが小さいことを確認"""
        base_model = SimpleTransformer()
        config = LoRAConfig(rank=4, alpha=8.0)
        
        lora_model = LoRAModel(base_model, config)
        
        # 保存
        save_path = tmp_path / "lora_checkpoint"
        lora_model.save_lora_weights(str(save_path))
        
        # ファイルサイズを確認
        weights_size = (save_path / "lora_weights.pt").stat().st_size
        config_size = (save_path / "lora_config.json").stat().st_size
        
        # ベースモデルのサイズを概算
        base_size = sum(p.numel() * 4 for p in base_model.parameters())  # float32
        
        # LoRA重みはベースモデルの5%未満（テストモデルは小さいので緩和）
        # 実際のQwen2-0.5Bでは1%未満になる
        assert weights_size < base_size * 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
