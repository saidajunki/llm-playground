"""
LoRAモデルラッパー

既存のTransformerモデルにLoRAを適用するラッパークラス
"""

from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional, Any

import torch
import torch.nn as nn

from .config import LoRAConfig
from .linear import LoRALinear


class LoRAModel(nn.Module):
    """LoRAを適用したモデルラッパー"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig
    ):
        """
        Args:
            base_model: LoRAを適用するベースモデル
            config: LoRA設定
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.lora_layers: Dict[str, LoRALinear] = {}
        
        # LoRAを適用
        self._apply_lora()
    
    def _apply_lora(self) -> None:
        """対象モジュールにLoRAを適用"""
        # まず全パラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 対象モジュールにLoRAを適用
        for name, module in self.base_model.named_modules():
            if self._should_apply_lora(name, module):
                lora_layer = LoRALinear(
                    module,
                    self.config.rank,
                    self.config.alpha,
                    self.config.dropout
                )
                self._replace_module(name, lora_layer)
                self.lora_layers[name] = lora_layer
    
    def _should_apply_lora(self, name: str, module: nn.Module) -> bool:
        """このモジュールにLoRAを適用すべきか判定"""
        if not isinstance(module, nn.Linear):
            return False
        
        for target in self.config.target_modules:
            if target in name:
                return True
        
        return False
    
    def _replace_module(self, name: str, new_module: nn.Module) -> None:
        """モジュールを置換"""
        parts = name.split(".")
        parent = self.base_model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
    
    def forward(self, **kwargs) -> Any:
        """順伝播（ベースモデルに委譲）"""
        return self.base_model(**kwargs)
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """学習対象パラメータ（LoRAパラメータのみ）を返す"""
        for layer in self.lora_layers.values():
            yield layer.lora_A
            yield layer.lora_B
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        パラメータ数をカウント
        
        Returns:
            (学習パラメータ数, 全パラメータ数)
        """
        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        total = sum(p.numel() for p in self.base_model.parameters())
        return trainable, total
    
    def print_trainable_parameters(self) -> None:
        """学習パラメータの情報を表示"""
        trainable, total = self.count_parameters()
        percentage = 100 * trainable / total if total > 0 else 0
        
        print(f"Trainable parameters: {trainable:,} / {total:,} ({percentage:.4f}%)")
        print(f"LoRA layers: {len(self.lora_layers)}")
        for name in self.lora_layers:
            print(f"  - {name}")
    
    def save_lora_weights(self, path: str) -> None:
        """LoRA重みのみを保存"""
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = layer.lora_A.data.cpu()
            state_dict[f"{name}.lora_B"] = layer.lora_B.data.cpu()
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(state_dict, save_path / "lora_weights.pt")
        self.config.save(str(save_path / "lora_config.json"))
    
    def load_lora_weights(self, path: str) -> None:
        """LoRA重みを読み込み"""
        weights_path = Path(path) / "lora_weights.pt"
        state_dict = torch.load(weights_path, map_location="cpu")
        
        for name, layer in self.lora_layers.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key in state_dict:
                layer.lora_A.data = state_dict[a_key].to(layer.lora_A.device)
            if b_key in state_dict:
                layer.lora_B.data = state_dict[b_key].to(layer.lora_B.device)
    
    def merge_and_unload(self) -> nn.Module:
        """
        LoRA重みをベースモデルにマージして返す
        
        マージ後はLoRAモジュールが不要になり、
        通常のモデルとして推論できる
        
        Returns:
            マージされたベースモデル
        """
        for name, lora_layer in self.lora_layers.items():
            merged_layer = lora_layer.merge_weights()
            self._replace_module(name, merged_layer)
        
        # LoRAレイヤーの参照をクリア
        self.lora_layers.clear()
        
        return self.base_model
    
    @property
    def device(self) -> torch.device:
        """モデルのデバイスを返す"""
        return next(self.base_model.parameters()).device
    
    def generate(self, *args, **kwargs):
        """テキスト生成（ベースモデルに委譲）"""
        return self.base_model.generate(*args, **kwargs)
