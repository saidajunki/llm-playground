"""
Character-level Tokenizer for Mini Transformer.

文字レベルのトークナイザー。学習目的のためシンプルな実装。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union


class CharTokenizer:
    """文字レベルトークナイザー"""
    
    # 特殊トークン
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self._init_special_tokens()
    
    def _init_special_tokens(self) -> None:
        """特殊トークンを初期化"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        self.vocab_size = len(special_tokens)
    
    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]
    
    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]
    
    @property
    def bos_id(self) -> int:
        return self.char_to_id[self.BOS_TOKEN]
    
    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.EOS_TOKEN]
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        テキストから語彙を構築する。
        
        Args:
            texts: 語彙構築に使用するテキストのリスト
        """
        # 全テキストからユニークな文字を収集
        chars = set()
        for text in texts:
            chars.update(text)
        
        # ソートして一貫性を保つ
        sorted_chars = sorted(chars)
        
        # 特殊トークンの後に追加
        for char in sorted_chars:
            if char not in self.char_to_id:
                idx = self.vocab_size
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                self.vocab_size += 1
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        テキストをトークンIDのリストに変換する。
        
        Args:
            text: エンコードするテキスト
            add_special_tokens: BOSとEOSを追加するか
            
        Returns:
            トークンIDのリスト
        """
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_id)
        
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.unk_id)
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        トークンIDのリストをテキストに復元する。
        
        Args:
            ids: デコードするトークンIDのリスト
            skip_special_tokens: 特殊トークンをスキップするか
            
        Returns:
            復元されたテキスト
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        chars = []
        
        for idx in ids:
            if idx in self.id_to_char:
                if skip_special_tokens and idx in special_ids:
                    continue
                chars.append(self.id_to_char[idx])
            # 無効なIDは無視
        
        return ''.join(chars)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        語彙をJSONファイルに保存する。
        
        Args:
            path: 保存先のパス
        """
        path = Path(path)
        data = {
            'char_to_id': self.char_to_id,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        語彙をJSONファイルから読み込む。
        
        Args:
            path: 読み込み元のパス
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.char_to_id = data['char_to_id']
        self.vocab_size = data['vocab_size']
        
        # id_to_charを再構築
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
