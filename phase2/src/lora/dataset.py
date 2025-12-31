"""
指示-応答形式のデータセット

LoRAファインチューニング用のデータセットクラス
"""

import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """指示-応答形式のデータセット"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            data_path: データファイルのパス（タブ区切り形式）
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[Dict[str, str]]:
        """
        データファイルを読み込み
        
        フォーマット: instruction<TAB>response
        """
        examples = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # タブ区切りで分割
                parts = line.split("\t")
                if len(parts) >= 2:
                    examples.append({
                        "instruction": parts[0].strip(),
                        "response": parts[1].strip()
                    })
                else:
                    print(f"Warning: Line {line_num} has invalid format, skipping")
        
        return examples
    
    def _format_example(self, example: Dict[str, str]) -> str:
        """
        Qwen2のチャットテンプレートでフォーマット
        
        Args:
            example: {"instruction": ..., "response": ...}
            
        Returns:
            フォーマットされたテキスト
        """
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
        
        # チャットテンプレートを適用
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # フォールバック: シンプルなフォーマット
            return f"User: {example['instruction']}\nAssistant: {example['response']}"
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データセットからアイテムを取得
        
        Returns:
            {
                "input_ids": トークンID,
                "attention_mask": アテンションマスク,
                "labels": ラベル（input_idsと同じ、言語モデル学習用）
            }
        """
        example = self.examples[idx]
        text = self._format_example(example)
        
        # トークナイズ
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # バッチ次元を削除
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # ラベルはinput_idsと同じ（言語モデルの学習）
        labels = input_ids.clone()
        
        # パディングトークンのラベルを-100に設定（損失計算から除外）
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def split(
        self,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple["InstructionDataset", "InstructionDataset"]:
        """
        訓練/検証に分割
        
        Args:
            val_ratio: 検証データの割合
            seed: ランダムシード
            
        Returns:
            (訓練データセット, 検証データセット)
        """
        n_val = int(len(self.examples) * val_ratio)
        n_val = max(1, n_val)  # 最低1件は検証データに
        
        # シャッフル
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self.examples), generator=generator).tolist()
        
        # 分割
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # 新しいデータセットを作成
        train_dataset = copy.copy(self)
        val_dataset = copy.copy(self)
        
        train_dataset.examples = [self.examples[i] for i in train_indices]
        val_dataset.examples = [self.examples[i] for i in val_indices]
        
        return train_dataset, val_dataset
    
    @classmethod
    def from_list(
        cls,
        examples: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512
    ) -> "InstructionDataset":
        """
        リストからデータセットを作成
        
        Args:
            examples: [{"instruction": ..., "response": ...}, ...]
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        dataset = cls.__new__(cls)
        dataset.tokenizer = tokenizer
        dataset.max_length = max_length
        dataset.examples = examples
        return dataset
