"""
LoRAエラークラス

LoRAファインチューニングで発生するエラーを定義
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class LoRAError(Exception):
    """LoRA関連エラーの基底クラス"""
    pass


class LoRAConfigError(LoRAError):
    """LoRA設定エラー"""
    pass


class CheckpointError(LoRAError):
    """チェックポイント関連エラー"""
    pass


class MemoryError(LoRAError):
    """GPUメモリ不足エラー"""
    
    def __init__(
        self,
        required_gb: Optional[float] = None,
        available_gb: Optional[float] = None,
        message: Optional[str] = None
    ):
        self.required_gb = required_gb
        self.available_gb = available_gb
        
        if message:
            suggestions = [message]
        else:
            suggestions = ["GPU memory insufficient for training."]
        
        if required_gb and available_gb:
            suggestions.append(
                f"Required: {required_gb:.1f}GB, Available: {available_gb:.1f}GB"
            )
        
        suggestions.extend([
            "",
            "Suggestions to reduce memory usage:",
            "  1. Reduce batch_size (e.g., --batch-size 2)",
            "  2. Enable gradient checkpointing (--gradient-checkpointing)",
            "  3. Reduce max_length (e.g., --max-length 256)",
            "  4. Use mixed precision training (--fp16)",
            "  5. Reduce LoRA rank (e.g., --rank 4)",
        ])
        
        super().__init__("\n".join(suggestions))


def validate_config(config, model: nn.Module) -> None:
    """
    LoRA設定の妥当性を検証
    
    Args:
        config: LoRAConfig
        model: 対象モデル
        
    Raises:
        LoRAConfigError: 設定が無効な場合
    """
    # ランクの検証
    if config.rank <= 0:
        raise LoRAConfigError(f"rank must be positive, got {config.rank}")
    
    if config.rank > 256:
        raise LoRAConfigError(
            f"rank {config.rank} is unusually large. "
            "Typical values are 4-64. Are you sure?"
        )
    
    # アルファの検証
    if config.alpha <= 0:
        raise LoRAConfigError(f"alpha must be positive, got {config.alpha}")
    
    # ドロップアウトの検証
    if not 0 <= config.dropout < 1:
        raise LoRAConfigError(
            f"dropout must be in [0, 1), got {config.dropout}"
        )
    
    # 対象モジュールの存在確認
    module_names = [name for name, _ in model.named_modules()]
    
    for target in config.target_modules:
        found = any(target in name for name in module_names)
        if not found:
            raise LoRAConfigError(
                f"Target module '{target}' not found in model. "
                f"Available modules contain: {module_names[:10]}..."
            )


def validate_checkpoint(path: str, config=None) -> None:
    """
    チェックポイントの妥当性を検証
    
    Args:
        path: チェックポイントパス
        config: 期待するLoRAConfig（オプション）
        
    Raises:
        CheckpointError: チェックポイントが無効な場合
    """
    from .config import LoRAConfig
    
    checkpoint_path = Path(path)
    
    # ディレクトリの存在確認
    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint directory not found: {path}")
    
    # 重みファイルの存在確認
    weights_path = checkpoint_path / "lora_weights.pt"
    if not weights_path.exists():
        raise CheckpointError(f"LoRA weights not found at {weights_path}")
    
    # 設定ファイルの存在確認
    config_path = checkpoint_path / "lora_config.json"
    if not config_path.exists():
        raise CheckpointError(f"LoRA config not found at {config_path}")
    
    # 設定の互換性確認
    if config is not None:
        saved_config = LoRAConfig.load(str(config_path))
        
        if saved_config.rank != config.rank:
            raise CheckpointError(
                f"Rank mismatch: checkpoint has rank={saved_config.rank}, "
                f"but model expects rank={config.rank}"
            )
        
        if saved_config.target_modules != config.target_modules:
            raise CheckpointError(
                f"Target modules mismatch:\n"
                f"  Checkpoint: {saved_config.target_modules}\n"
                f"  Model: {config.target_modules}"
            )


def check_gpu_memory(required_gb: float = 8.0) -> None:
    """
    GPUメモリが十分かチェック
    
    Args:
        required_gb: 必要なメモリ量（GB）
        
    Raises:
        MemoryError: メモリが不足している場合
    """
    if not torch.cuda.is_available():
        return  # CPUの場合はスキップ
    
    # 利用可能なメモリを取得
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    available_memory = total_memory - allocated_memory
    
    available_gb = available_memory / (1024 ** 3)
    
    if available_gb < required_gb:
        raise MemoryError(
            required_gb=required_gb,
            available_gb=available_gb
        )
