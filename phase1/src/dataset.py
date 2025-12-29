"""
Dataset utilities for Mini Transformer.

テキストデータの読み込み、トークン化、バッチ処理を行う。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    テキストデータセット。
    
    テキストをトークン化し、固定長のシーケンスに分割する。
    各サンプルは(input, target)のペアで、targetはinputを1トークンずらしたもの。
    
    Args:
        texts: テキストのリスト
        tokenizer: トークナイザー
        seq_len: シーケンス長
    """
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: CharTokenizer, 
        seq_len: int
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = self._prepare_data(texts)
    
    def _prepare_data(self, texts: List[str]) -> torch.Tensor:
        """全テキストをトークン化して連結"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)
    
    def __len__(self) -> int:
        # seq_len + 1 のウィンドウが必要（input + target）
        return max(0, (len(self.data) - 1) // self.seq_len)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        インデックスに対応するサンプルを返す。
        
        Returns:
            x: (seq_len,) 入力トークン
            y: (seq_len,) ターゲットトークン（xを1つずらしたもの）
        """
        start = idx * self.seq_len
        end = start + self.seq_len
        
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        
        return x, y


def load_text_file(path: Union[str, Path]) -> str:
    """テキストファイルを読み込む"""
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_dataloader(
    texts: List[str],
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    DataLoaderを作成する。
    
    Args:
        texts: テキストのリスト
        tokenizer: トークナイザー
        seq_len: シーケンス長
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        num_workers: ワーカー数
        
    Returns:
        DataLoader
    """
    dataset = TextDataset(texts, tokenizer, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
