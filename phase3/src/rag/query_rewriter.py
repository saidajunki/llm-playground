"""
QueryRewriter - クエリ書き換えクラス

ユーザーのクエリを検索に適した形に変換する
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """クエリを検索用に書き換えるクラス"""
    
    DEFAULT_TEMPLATE = """以下の質問を、ドキュメント検索に適した形に書き換えてください。

ルール:
- 代名詞（それ、これ、あれ等）を具体的な名詞に置き換える
- 口語的な表現を文語的な表現に変換
- 会話履歴がある場合は文脈を補完する
- 検索キーワードとして有効な形にする
- 書き換えたクエリのみを出力（説明不要）

会話履歴:
{history}

質問: {query}

検索クエリ:"""
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        template: Optional[str] = None,
        max_history: int = 3
    ):
        """
        Args:
            model: LLMモデル（Noneの場合はルールベースで処理）
            tokenizer: トークナイザー
            template: プロンプトテンプレート
            max_history: 保持する会話履歴の最大数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or self.DEFAULT_TEMPLATE
        self.max_history = max_history
        self.history: List[str] = []

    def rewrite(
        self,
        query: str,
        use_history: bool = True
    ) -> str:
        """
        クエリを検索用に書き換え
        
        Args:
            query: 元のクエリ
            use_history: 会話履歴を使用するかどうか
            
        Returns:
            str: 書き換えられたクエリ
        """
        # LLMがある場合はLLMで書き換え
        if self.model is not None:
            return self._rewrite_with_llm(query, use_history)
        
        # LLMがない場合はルールベースで書き換え
        return self._rewrite_rule_based(query, use_history)
    
    def _rewrite_with_llm(self, query: str, use_history: bool) -> str:
        """LLMを使ってクエリを書き換え"""
        try:
            import torch
            
            # 履歴を整形
            history_text = self._format_history() if use_history else "なし"
            
            # プロンプト作成
            prompt = self.template.format(history=history_text, query=query)
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,  # 低めで安定した出力
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            rewritten = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 改行以降を削除（最初の行のみ使用）
            rewritten = rewritten.strip().split('\n')[0].strip()
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten if rewritten else query
            
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}, using original query")
            return query

    def _rewrite_rule_based(self, query: str, use_history: bool) -> str:
        """ルールベースでクエリを書き換え"""
        rewritten = query
        
        # 代名詞の解決（履歴がある場合）
        if use_history and self.history:
            rewritten = self._resolve_pronouns(rewritten)
        
        # 口語→文語変換
        rewritten = self._normalize_query(rewritten)
        
        # キーワード抽出・強調
        rewritten = self._extract_keywords(rewritten)
        
        if rewritten != query:
            logger.info(f"Query rewritten (rule-based): '{query}' -> '{rewritten}'")
        
        return rewritten
    
    def _resolve_pronouns(self, query: str) -> str:
        """代名詞を解決"""
        pronouns = {
            "それ": None,
            "これ": None,
            "あれ": None,
            "その": None,
            "この": None,
            "あの": None,
        }
        
        # 履歴から名詞を抽出
        if self.history:
            last_query = self.history[-1]
            # 簡易的に最後のクエリから主要な名詞を抽出
            nouns = self._extract_nouns(last_query)
            if nouns:
                main_noun = nouns[0]
                for pronoun in pronouns:
                    if pronoun in query:
                        query = query.replace(pronoun, main_noun)
                        break
        
        return query
    
    def _normalize_query(self, query: str) -> str:
        """口語を文語に正規化"""
        import re
        
        # 順序が重要：長いパターンから先にマッチ
        replacements = [
            ("ってなに", "とは何か"),
            ("って何", "とは"),
            ("なんですか", "とは"),
            ("教えて", "について"),
            ("知りたい", "について"),
            ("がわからない", "について"),
            ("わからない", "について"),
        ]
        
        for old, new in replacements:
            if old in query:
                query = query.replace(old, new)
        
        # 「どうやって〜するの」→「〜の方法」
        query = re.sub(r'どうやって(.+?)するの', r'\1 方法', query)
        query = re.sub(r'どうすれば(.+?)できる', r'\1 方法', query)
        
        # 残った「って」をスペースに
        query = query.replace("って", " ")
        
        # 重複する「について」を除去
        while "についてについて" in query:
            query = query.replace("についてについて", "について")
        
        # 余分なスペースを除去
        query = " ".join(query.split())
        
        return query

    def _extract_keywords(self, query: str) -> str:
        """キーワードを抽出・強調"""
        import re
        
        # 疑問詞を除去
        question_words = ["何", "どう", "なぜ", "いつ", "どこ", "誰", "どの", "どれ"]
        
        # 助詞を除去してキーワードを抽出
        particles = ["は", "が", "を", "に", "で", "と", "の", "へ", "から", "まで", "より"]
        
        # 簡易的なキーワード抽出
        # 実際にはMeCabなどの形態素解析を使うとより精度が上がる
        keywords = []
        
        # 英単語を抽出
        english_words = re.findall(r'[A-Za-z]+', query)
        keywords.extend(english_words)
        
        # 日本語の名詞らしきものを抽出（2文字以上の連続）
        japanese_words = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}', query)
        for word in japanese_words:
            # 助詞で終わる場合は除去
            for particle in particles:
                if word.endswith(particle):
                    word = word[:-len(particle)]
                    break
            if len(word) >= 2:
                keywords.append(word)
        
        # キーワードがあれば元のクエリに追加
        if keywords:
            # 重複を除去しつつ順序を保持
            seen = set()
            unique_keywords = []
            for kw in keywords:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    unique_keywords.append(kw)
            
            # 元のクエリにキーワードが含まれていない場合のみ追加
            additional = [kw for kw in unique_keywords if kw not in query]
            if additional:
                return query + " " + " ".join(additional[:3])
        
        return query
    
    def _extract_nouns(self, text: str) -> List[str]:
        """テキストから名詞を抽出（簡易版）"""
        import re
        
        nouns = []
        
        # 英単語
        english = re.findall(r'[A-Z][a-z]+|[A-Z]+', text)
        nouns.extend(english)
        
        # 日本語（カタカナ語、漢字語）
        katakana = re.findall(r'[\u30a0-\u30ff]{2,}', text)
        nouns.extend(katakana)
        
        kanji = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        nouns.extend(kanji)
        
        return nouns

    def _format_history(self) -> str:
        """会話履歴を整形"""
        if not self.history:
            return "なし"
        
        recent = self.history[-self.max_history:]
        return "\n".join(f"- {q}" for q in recent)
    
    def add_to_history(self, query: str) -> None:
        """クエリを履歴に追加"""
        self.history.append(query)
        
        # 最大数を超えたら古いものを削除
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history:]
    
    def clear_history(self) -> None:
        """履歴をクリア"""
        self.history = []
    
    def get_history(self) -> List[str]:
        """履歴を取得"""
        return self.history.copy()


class HypotheticalDocumentEmbedder:
    """
    HyDE (Hypothetical Document Embeddings)
    
    クエリから仮想的な回答ドキュメントを生成し、
    そのドキュメントのエンベディングで検索する手法
    """
    
    DEFAULT_TEMPLATE = """以下の質問に対する回答を、ドキュメントの一部として書いてください。
実際の回答ではなく、この質問に答えるドキュメントがあったとしたら、
どのような内容が書かれているかを想像して書いてください。

質問: {query}

ドキュメント:"""
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        template: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or self.DEFAULT_TEMPLATE
    
    def generate_hypothetical_document(self, query: str) -> str:
        """仮想ドキュメントを生成"""
        if self.model is None:
            # モデルがない場合はクエリをそのまま返す
            return query
        
        try:
            import torch
            
            prompt = self.template.format(query=query)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            hypothetical_doc = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"Generated hypothetical document for: {query[:50]}...")
            return hypothetical_doc.strip()
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query
