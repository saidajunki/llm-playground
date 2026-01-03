"""
Generator - 回答生成クラス

検索結果を基にLLMで回答を生成する
"""

import logging
from typing import Any, List, Optional, Tuple, Union

from .document import Document

logger = logging.getLogger(__name__)


class Generator:
    """回答生成クラス"""
    
    DEFAULT_TEMPLATE = """以下のコンテキストを参考に、質問に回答してください。
コンテキストに関連する情報がない場合は、その旨を伝えてください。

コンテキスト:
{context}

質問: {query}

回答:"""
    
    DEFAULT_TEMPLATE_EN = """Answer the question based on the following context.
If the context doesn't contain relevant information, please indicate that.

Context:
{context}

Question: {query}

Answer:"""
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        template: Optional[str] = None,
        max_context_length: int = 2000,
        language: str = "ja"
    ):
        """
        Args:
            model: LLMモデル（Noneの場合はコンテキストのみ返す）
            tokenizer: トークナイザー
            template: プロンプトテンプレート
            max_context_length: コンテキストの最大長
            language: 言語（ja/en）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.language = language
        
        if template:
            self.template = template
        elif language == "ja":
            self.template = self.DEFAULT_TEMPLATE
        else:
            self.template = self.DEFAULT_TEMPLATE_EN
    
    def generate(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        検索結果を基に回答を生成
        
        Args:
            query: 質問
            retrieved_docs: 検索結果（ドキュメント, スコア）のリスト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            
        Returns:
            str: 生成された回答
        """
        if not retrieved_docs:
            if self.language == "ja":
                return "関連する情報が見つかりませんでした。"
            else:
                return "No relevant information found."
        
        # コンテキスト構築
        context = self._build_context(retrieved_docs)
        
        # プロンプト作成
        prompt = self.template.format(context=context, query=query)
        
        # モデルがない場合はプロンプトを返す
        if self.model is None:
            return f"[Context-only mode]\n\n{prompt}"
        
        # LLMで生成
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # デバイスに移動
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 入力部分を除いてデコード
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"生成に失敗しました: {e}"
    
    def generate_with_citations(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        引用付きで回答を生成
        
        Args:
            query: 質問
            retrieved_docs: 検索結果
            **kwargs: generateに渡す追加引数
            
        Returns:
            Tuple[str, List[str]]: (回答, ソースのリスト)
        """
        response = self.generate(query, retrieved_docs, **kwargs)
        
        # ソースを抽出
        sources = []
        for doc, score in retrieved_docs:
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            if source not in sources:
                sources.append(source)
        
        return response, sources
    
    def _build_context(self, docs: List[Tuple[Document, float]]) -> str:
        """
        検索結果からコンテキストを構築
        
        Args:
            docs: (ドキュメント, スコア)のリスト
            
        Returns:
            str: コンテキスト文字列
        """
        context_parts = []
        total_length = 0
        
        for doc, score in docs:
            content = doc.content.strip()
            
            # 最大長チェック
            if total_length + len(content) > self.max_context_length:
                # 残りの長さだけ追加
                remaining = self.max_context_length - total_length
                if remaining > 100:  # 最低100文字は追加
                    content = content[:remaining] + "..."
                else:
                    break
            
            # ソース情報を追加
            source = doc.metadata.get("source", doc.metadata.get("filename", ""))
            if source:
                source_name = source.split("/")[-1] if "/" in source else source
                context_parts.append(f"[Source: {source_name}]\n{content}")
            else:
                context_parts.append(content)
            
            total_length += len(content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_response_with_sources(
        self,
        response: str,
        sources: List[str]
    ) -> str:
        """
        回答とソースをフォーマット
        
        Args:
            response: 回答
            sources: ソースのリスト
            
        Returns:
            str: フォーマットされた回答
        """
        if not sources:
            return response
        
        if self.language == "ja":
            sources_text = "\n\n📚 参照元:\n" + "\n".join(f"  - {s}" for s in sources)
        else:
            sources_text = "\n\n📚 Sources:\n" + "\n".join(f"  - {s}" for s in sources)
        
        return response + sources_text
