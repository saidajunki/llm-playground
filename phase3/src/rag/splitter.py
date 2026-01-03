"""
TextSplitter - テキスト分割クラス

ドキュメントをチャンクに分割する
"""

import re
from typing import List, Optional

from .document import Document


class TextSplitter:
    """テキスト分割クラス"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            chunk_size: チャンクの最大サイズ
            chunk_overlap: チャンク間のオーバーラップ
            separators: 分割に使用するセパレータのリスト（優先度順）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        テキストをチャンクに分割
        
        Args:
            text: 分割するテキスト
            
        Returns:
            List[str]: チャンクのリスト
        """
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """再帰的にテキストを分割"""
        if not separators:
            # セパレータがない場合は強制的に分割
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # 空文字の場合は文字単位で分割
            return self._split_by_size(text)
        
        # セパレータで分割
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # セパレータを戻す（最後以外）
            piece = split + separator if split != splits[-1] else split
            
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    # 現在のチャンクが大きすぎる場合は再帰的に分割
                    if len(current_chunk) > self.chunk_size:
                        chunks.extend(self._split_recursive(current_chunk, remaining_separators))
                    else:
                        chunks.append(current_chunk.strip())
                
                # 新しいピースが大きすぎる場合は再帰的に分割
                if len(piece) > self.chunk_size:
                    chunks.extend(self._split_recursive(piece, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = piece
        
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                chunks.extend(self._split_recursive(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk.strip())
        
        # オーバーラップを適用
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        return [c for c in chunks if c]  # 空のチャンクを除去
    
    def _split_by_size(self, text: str) -> List[str]:
        """サイズで強制的に分割"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """チャンク間にオーバーラップを適用"""
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 前のチャンクの末尾をオーバーラップとして追加
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) >= self.chunk_overlap else prev_chunk
            
            # オーバーラップが既に含まれていない場合のみ追加
            if not current_chunk.startswith(overlap_text):
                result.append(overlap_text + current_chunk)
            else:
                result.append(current_chunk)
        
        return result
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントリストを分割
        
        Args:
            documents: 分割するドキュメントのリスト
            
        Returns:
            List[Document]: 分割されたドキュメントのリスト
        """
        result = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                # 元のメタデータをコピーしてチャンク情報を追加
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                metadata["original_doc_id"] = doc.id
                
                result.append(Document(content=chunk, metadata=metadata))
        
        return result


class CodeSplitter(TextSplitter):
    """コード用テキスト分割クラス"""
    
    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Args:
            language: プログラミング言語
            chunk_size: チャンクの最大サイズ
            chunk_overlap: チャンク間のオーバーラップ
        """
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
    
    def split_text(self, text: str) -> List[str]:
        """
        コード構造を考慮してテキストを分割
        
        Args:
            text: 分割するコード
            
        Returns:
            List[str]: チャンクのリスト
        """
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        if self.language == "python":
            return self._split_python(text)
        else:
            # 他の言語は通常の分割
            return super().split_text(text)
    
    def _split_python(self, text: str) -> List[str]:
        """Pythonコードを関数/クラス単位で分割"""
        # 関数とクラスの定義を検出
        pattern = r'^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)'
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # 新しい関数/クラスの開始を検出
            if re.match(pattern, line.strip()):
                # 現在のチャンクを保存
                if current_chunk:
                    chunk_text = ''.join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(line_with_newline)
            current_size += len(line_with_newline)
            
            # チャンクサイズを超えた場合
            if current_size > self.chunk_size:
                chunk_text = ''.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
        
        # 残りを追加
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
