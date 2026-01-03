"""
Agent - ツール選択・実行エージェント

Bedrock Agents に相当する機能
ReActパターンでツールを選択・実行し、回答を生成する
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """エージェントの実行ステップ"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    """エージェントの実行結果"""
    answer: str
    steps: List[AgentStep]
    tool_calls: List[Tuple[str, Dict[str, Any], str]]  # (tool_name, args, result)


class Agent:
    """
    ReActパターンのエージェント
    
    Thought → Action → Observation のループで問題を解決
    """
    
    SYSTEM_PROMPT = """あなたは質問に答えるアシスタントです。
以下のツールを使って情報を収集し、回答を生成してください。

{tools_description}

回答形式:
1. まず「Thought:」で何をすべきか考えます
2. ツールを使う場合は「Action:」でツール名、「Action Input:」でJSON形式の引数を指定します
3. ツールの結果は「Observation:」として返されます
4. 最終回答は「Final Answer:」で始めてください

例:
Thought: ユーザーはTransformerについて知りたいようです。ドキュメントを検索します。
Action: search_documents
Action Input: {{"query": "Transformer アーキテクチャ"}}
Observation: [検索結果...]
Thought: 検索結果を基に回答できます。
Final Answer: Transformerは...

質問: {query}
"""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        Args:
            model: LLMモデル
            tokenizer: トークナイザー
            tool_registry: ツールレジストリ
            max_iterations: 最大イテレーション数
            verbose: 詳細ログを出力するか
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def register_tool(self, tool: Tool) -> None:
        """ツールを登録"""
        self.tool_registry.register(tool)
    
    def run(self, query: str) -> AgentResult:
        """
        エージェントを実行
        
        Args:
            query: ユーザーの質問
            
        Returns:
            AgentResult: 実行結果
        """
        steps = []
        tool_calls = []
        
        # システムプロンプト生成
        prompt = self.SYSTEM_PROMPT.format(
            tools_description=self.tool_registry.get_tools_description(),
            query=query
        )
        
        # LLMがない場合はルールベースで処理
        if self.model is None:
            return self._run_rule_based(query)
        
        # ReActループ
        for i in range(self.max_iterations):
            if self.verbose:
                logger.info(f"Iteration {i + 1}/{self.max_iterations}")
            
            # LLMで次のステップを生成
            response = self._generate(prompt)
            
            # レスポンスをパース
            step = self._parse_response(response)
            steps.append(step)
            
            # Final Answerがあれば終了
            if "Final Answer:" in response:
                answer = self._extract_final_answer(response)
                return AgentResult(answer=answer, steps=steps, tool_calls=tool_calls)
            
            # Actionがあればツールを実行
            if step.action and step.action_input is not None:
                tool = self.tool_registry.get(step.action)
                if tool:
                    try:
                        result = tool.execute(**step.action_input)
                        step.observation = str(result)
                        tool_calls.append((step.action, step.action_input, str(result)))
                    except Exception as e:
                        step.observation = f"エラー: {e}"
                else:
                    step.observation = f"ツール '{step.action}' が見つかりません"
                
                # プロンプトに結果を追加
                prompt += f"\n{response}\nObservation: {step.observation}\n"
        
        # 最大イテレーションに達した場合
        return AgentResult(
            answer="申し訳ありません。回答を生成できませんでした。",
            steps=steps,
            tool_calls=tool_calls
        )

    def _generate(self, prompt: str) -> str:
        """LLMでテキスト生成"""
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stop_strings=["Observation:"],
                    tokenizer=self.tokenizer
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Final Answer: 生成に失敗しました: {e}"
    
    def _parse_response(self, response: str) -> AgentStep:
        """レスポンスをパース"""
        thought = ""
        action = None
        action_input = None
        
        # Thoughtを抽出
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Actionを抽出
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action = action_match.group(1).strip()
        
        # Action Inputを抽出
        input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse action input: {input_match.group(1)}")
        
        return AgentStep(thought=thought, action=action, action_input=action_input)
    
    def _extract_final_answer(self, response: str) -> str:
        """Final Answerを抽出"""
        match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response

    def _run_rule_based(self, query: str) -> AgentResult:
        """
        ルールベースでツールを選択・実行（LLMなしの場合）
        """
        steps = []
        tool_calls = []
        
        # キーワードベースでツール選択
        selected_tool = self._select_tool_by_keywords(query)
        
        if selected_tool:
            tool = self.tool_registry.get(selected_tool)
            if tool:
                # ツール実行
                step = AgentStep(
                    thought=f"キーワードから '{selected_tool}' ツールを選択",
                    action=selected_tool,
                    action_input=self._extract_args_from_query(query, tool)
                )
                
                try:
                    result = tool.execute(**step.action_input)
                    step.observation = str(result)
                    tool_calls.append((selected_tool, step.action_input, str(result)))
                except Exception as e:
                    step.observation = f"エラー: {e}"
                
                steps.append(step)
                
                # 結果を回答として返す
                return AgentResult(
                    answer=f"[{selected_tool}の結果]\n{step.observation}",
                    steps=steps,
                    tool_calls=tool_calls
                )
        
        # ツールが選択されなかった場合
        return AgentResult(
            answer="適切なツールが見つかりませんでした。質問を具体的にしてください。",
            steps=[AgentStep(thought="適切なツールが見つからない")],
            tool_calls=[]
        )
    
    def _select_tool_by_keywords(self, query: str) -> Optional[str]:
        """キーワードからツールを選択"""
        query_lower = query.lower()
        
        # キーワードマッピング（優先度順）
        keyword_map = [
            ("read_file", ["ファイル", "読み込", "読んで", "開い", ".md", ".txt", ".py", ".json"]),
            ("query_database", ["データベース", "DB", "SQL", "テーブル", "レコード"]),
            ("calculate", ["計算", "足し", "引き", "掛け", "割り", "sqrt", "sin", "cos", "pow", "+", "-", "*", "/"]),
            ("get_datetime", ["日時", "今日", "明日", "昨日", "時間", "日付", "何日", "日後"]),
            ("web_search", ["Web", "ウェブ", "インターネット", "最新"]),
            ("search_documents", ["検索", "調べ", "探し", "ドキュメント", "文書", "について", "教えて", "とは"]),
        ]
        
        for tool_name, keywords in keyword_map:
            if tool_name in self.tool_registry.tools:
                for keyword in keywords:
                    if keyword in query_lower or keyword.lower() in query_lower:
                        return tool_name
        
        # デフォルトは検索
        if "search_documents" in self.tool_registry.tools:
            return "search_documents"
        
        return None
    
    def _extract_args_from_query(self, query: str, tool: Tool) -> Dict[str, Any]:
        """クエリからツール引数を抽出"""
        import re
        
        args = {}
        
        for param in tool.parameters:
            if param.name == "query":
                args["query"] = query
            elif param.name == "expression":
                # 数式を抽出（関数呼び出しも含む）
                # sqrt(144), pow(2, 10) などをサポート
                # ASCII文字のみを抽出
                expr = ''.join(c for c in query if ord(c) < 128 or c in '+-*/(). ,')
                expr = ' '.join(expr.split())  # 余分なスペースを除去
                args["expression"] = expr.strip() if expr.strip() else "0"
            elif param.name == "sql":
                args["sql"] = query
            elif param.name == "path":
                # パスを抽出
                match = re.search(r'[\.\/\w\-]+\.\w+', query)
                if match:
                    args["path"] = match.group()
            elif param.name == "offset_days":
                # 日数オフセットを抽出
                match = re.search(r'(\d+)日(後|前)', query)
                if match:
                    days = int(match.group(1))
                    if "前" in match.group(2):
                        days = -days
                    args["offset_days"] = days
                elif "明日" in query:
                    args["offset_days"] = 1
                elif "昨日" in query:
                    args["offset_days"] = -1
                else:
                    args["offset_days"] = 0
            elif param.default is not None:
                args[param.name] = param.default
        
        return args


class AgentExecutor:
    """
    エージェント実行のラッパー
    
    RAGパイプラインと統合して使用
    """
    
    def __init__(
        self,
        agent: Agent,
        rag_pipeline = None,
        db_loader = None,
        include_builtin_tools: bool = True
    ):
        """
        Args:
            agent: エージェント
            rag_pipeline: RAGパイプライン（検索ツール用）
            db_loader: データベースローダー（DBツール用）
            include_builtin_tools: 組み込みツールを含めるか
        """
        self.agent = agent
        self.rag_pipeline = rag_pipeline
        self.db_loader = db_loader
        
        if include_builtin_tools:
            self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """組み込みツールを登録"""
        from .tools import (
            create_search_tool,
            create_database_tool,
            create_calculator_tool,
            create_datetime_tool,
            create_file_reader_tool,
        )
        
        # 検索ツール
        if self.rag_pipeline:
            self.agent.register_tool(create_search_tool(self.rag_pipeline))
        
        # データベースツール
        if self.db_loader:
            self.agent.register_tool(create_database_tool(self.db_loader))
        
        # その他の組み込みツール
        self.agent.register_tool(create_calculator_tool())
        self.agent.register_tool(create_datetime_tool())
        self.agent.register_tool(create_file_reader_tool())
    
    def run(self, query: str) -> str:
        """
        クエリを実行して回答を返す
        
        Args:
            query: ユーザーの質問
            
        Returns:
            str: 回答
        """
        result = self.agent.run(query)
        return result.answer
    
    def run_with_details(self, query: str) -> AgentResult:
        """
        クエリを実行して詳細な結果を返す
        
        Args:
            query: ユーザーの質問
            
        Returns:
            AgentResult: 詳細な実行結果
        """
        return self.agent.run(query)
    
    def chat(self, query: str, show_steps: bool = False) -> str:
        """
        チャット形式で実行
        
        Args:
            query: ユーザーの質問
            show_steps: ステップを表示するか
            
        Returns:
            str: フォーマットされた回答
        """
        result = self.agent.run(query)
        
        output = []
        
        if show_steps and result.steps:
            output.append("🔍 実行ステップ:")
            for i, step in enumerate(result.steps, 1):
                output.append(f"  {i}. {step.thought[:50]}...")
                if step.action:
                    output.append(f"     → {step.action}")
            output.append("")
        
        output.append(result.answer)
        
        if result.tool_calls:
            output.append("\n📚 使用したツール:")
            for tool_name, args, _ in result.tool_calls:
                output.append(f"  - {tool_name}")
        
        return "\n".join(output)
