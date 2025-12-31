"""
LoRA設定クラス

LoRAのハイパーパラメータを一元管理する
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRAのハイパーパラメータ設定"""
    
    rank: int = 8
    """低ランク行列のランク（通常 4〜64）"""
    
    alpha: float = 16.0
    """スケーリング係数（通常 rank と同じか2倍）"""
    
    dropout: float = 0.05
    """ドロップアウト率"""
    
    target_modules: List[str] = field(default_factory=list)
    """LoRAを適用する対象モジュール名のリスト"""
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if not self.target_modules:
            # Qwen2のattention層をデフォルトで対象
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]
    
    @property
    def scaling(self) -> float:
        """スケーリング係数を計算: alpha / rank"""
        return self.alpha / self.rank
    
    def to_dict(self) -> dict:
        """設定を辞書に変換"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        """辞書から設定を復元"""
        return cls(
            rank=d.get("rank", 8),
            alpha=d.get("alpha", 16.0),
            dropout=d.get("dropout", 0.05),
            target_modules=d.get("target_modules", [])
        )
    
    def save(self, path: str) -> None:
        """設定をJSONファイルに保存"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "LoRAConfig":
        """JSONファイルから設定を読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def __repr__(self) -> str:
        return (
            f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}, target_modules={self.target_modules})"
        )
