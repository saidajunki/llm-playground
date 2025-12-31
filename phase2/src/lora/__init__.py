"""
LoRA (Low-Rank Adaptation) モジュール

Qwen2-0.5B モデルのパラメータ効率的なファインチューニングを実装
"""

from .config import LoRAConfig
from .linear import LoRALinear

__all__ = [
    "LoRAConfig",
    "LoRALinear",
]

# 遅延インポート用（モジュールが実装されたら追加）
def __getattr__(name):
    if name == "LoRAModel":
        from .model import LoRAModel
        return LoRAModel
    elif name == "InstructionDataset":
        from .dataset import InstructionDataset
        return InstructionDataset
    elif name == "LoRATrainer":
        from .trainer import LoRATrainer
        return LoRATrainer
    elif name == "LoRAInference":
        from .inference import LoRAInference
        return LoRAInference
    raise AttributeError(f"module 'lora' has no attribute '{name}'")
