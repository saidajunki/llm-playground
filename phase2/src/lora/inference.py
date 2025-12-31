"""
LoRA推論クラス

LoRAモデルでの推論とマージ機能を提供
"""

from pathlib import Path
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn

from .config import LoRAConfig
from .model import LoRAModel
from .linear import LoRALinear


class LoRAInference:
    """LoRAモデルでの推論"""
    
    def __init__(
        self,
        model: Union[LoRAModel, nn.Module],
        tokenizer: Any
    ):
        """
        Args:
            model: LoRAモデルまたはマージ済みモデル
            tokenizer: トークナイザー
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    @classmethod
    def from_lora_checkpoint(
        cls,
        base_model_path: str,
        lora_path: str,
        merge: bool = False,
        device: str = "auto"
    ) -> "LoRAInference":
        """
        LoRAチェックポイントから読み込み
        
        Args:
            base_model_path: ベースモデルのパス
            lora_path: LoRAチェックポイントのパス
            merge: マージするかどうか
            device: デバイス（"auto", "cuda", "cpu"）
            
        Returns:
            LoRAInferenceインスタンス
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # ベースモデル読み込み
        if device == "auto":
            device_map = "auto"
        else:
            device_map = None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )
        
        if device != "auto" and device_map is None:
            base_model = base_model.to(device)
        
        # LoRA設定読み込み
        config = LoRAConfig.load(f"{lora_path}/lora_config.json")
        
        # LoRAモデル構築
        lora_model = LoRAModel(base_model, config)
        lora_model.load_lora_weights(lora_path)
        
        if merge:
            # マージして通常のモデルとして使用
            merged_model = lora_model.merge_and_unload()
            return cls(merged_model, tokenizer)
        else:
            return cls(lora_model, tokenizer)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            do_sample: サンプリングを使用するか
            
        Returns:
            生成されたテキスト
        """
        # チャットテンプレートを適用
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"User: {prompt}\nAssistant:"
        
        # トークナイズ
        inputs = self.tokenizer([text], return_tensors="pt")
        
        # デバイスに移動
        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 入力部分を除いてデコード
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def _get_device(self) -> torch.device:
        """モデルのデバイスを取得"""
        if hasattr(self.model, 'device'):
            return self.model.device
        return next(self.model.parameters()).device
    
    def interactive_chat(self) -> None:
        """対話モード"""
        print("\n=== LoRA Fine-tuned Model Chat ===")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                response = self.generate(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    @staticmethod
    def compare_outputs(
        lora_inference: "LoRAInference",
        merged_inference: "LoRAInference",
        prompts: List[str]
    ) -> bool:
        """
        LoRAモデルとマージモデルの出力を比較
        
        Args:
            lora_inference: LoRAモデルの推論インスタンス
            merged_inference: マージモデルの推論インスタンス
            prompts: テストプロンプトのリスト
            
        Returns:
            全ての出力が一致すればTrue
        """
        all_match = True
        
        for prompt in prompts:
            # 同じシードで生成（決定的生成）
            torch.manual_seed(42)
            lora_output = lora_inference.generate(
                prompt, 
                temperature=0.0,
                do_sample=False
            )
            
            torch.manual_seed(42)
            merged_output = merged_inference.generate(
                prompt,
                temperature=0.0,
                do_sample=False
            )
            
            if lora_output != merged_output:
                print(f"Mismatch for prompt: {prompt}")
                print(f"  LoRA: {lora_output[:100]}...")
                print(f"  Merged: {merged_output[:100]}...")
                all_match = False
        
        return all_match
    
    @staticmethod
    def has_lora_layers(model: nn.Module) -> bool:
        """モデルにLoRAレイヤーが含まれているか確認"""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                return True
        return False
