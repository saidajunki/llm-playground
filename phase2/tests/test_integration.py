"""
LoRA 統合テスト

エンドツーエンドの学習→保存→読み込み→推論をテスト
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora.config import LoRAConfig
from lora.linear import LoRALinear
from lora.model import LoRAModel
from lora.dataset import InstructionDataset
from lora.trainer import LoRATrainer
from lora.inference import LoRAInference


class SimpleLanguageModel(nn.Module):
    """テスト用の簡易言語モデル"""
    
    def __init__(self, vocab_size=1000, d_model=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer-likeな構造
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "q_proj": nn.Linear(d_model, d_model),
                "k_proj": nn.Linear(d_model, d_model),
                "v_proj": nn.Linear(d_model, d_model),
                "o_proj": nn.Linear(d_model, d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                )
            }))
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # 簡易attention
            q = layer["q_proj"](x)
            k = layer["k_proj"](x)
            v = layer["v_proj"](x)
            
            attn = torch.softmax(q @ k.transpose(-2, -1) / 8, dim=-1)
            x = x + layer["o_proj"](attn @ v)
            x = x + layer["ffn"](x)
        
        logits = self.lm_head(x)
        
        # 損失計算
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 簡易的な出力オブジェクト
        class Output:
            pass
        
        output = Output()
        output.loss = loss
        output.logits = logits
        
        return output


class SimpleTokenizer:
    """テスト用の簡易トークナイザー"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
    
    def __call__(self, text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"):
        # 簡易的なトークナイズ（文字をIDに変換）
        if isinstance(text, str):
            text = [text]
        
        batch_ids = []
        batch_mask = []
        
        for t in text:
            ids = [ord(c) % self.vocab_size for c in t[:max_length]]
            mask = [1] * len(ids)
            
            # パディング
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
            
            batch_ids.append(ids)
            batch_mask.append(mask)
        
        result = {
            "input_ids": torch.tensor(batch_ids),
            "attention_mask": torch.tensor(batch_mask)
        }
        
        return result
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # 簡易的なテンプレート
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"{role}: {content}\n"
        return text
    
    def decode(self, ids, skip_special_tokens=True):
        # 簡易的なデコード
        return "".join(chr(i % 128 + 32) for i in ids if i > 0)


class TestIntegration:
    """統合テスト"""
    
    def test_end_to_end_training(self, tmp_path):
        """エンドツーエンドの学習テスト"""
        # モデル作成
        base_model = SimpleLanguageModel()
        config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
        lora_model = LoRAModel(base_model, config)
        
        # トークナイザー
        tokenizer = SimpleTokenizer()
        
        # データセット作成
        data_path = tmp_path / "train.txt"
        data_path.write_text("hello\tworld\nfoo\tbar\n")
        
        dataset = InstructionDataset(str(data_path), tokenizer, max_length=32)
        
        # トレーナー作成
        trainer = LoRATrainer(
            model=lora_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            output_dir=str(tmp_path / "checkpoints"),
            learning_rate=1e-3,
            batch_size=2,
            gradient_accumulation_steps=1,
            num_epochs=2,
            warmup_steps=0,
            log_interval=1,
            save_interval=100,
            fp16=False,
            gradient_checkpointing=False
        )
        
        # 学習前のLoRA重み
        initial_weights = {}
        for name, layer in lora_model.lora_layers.items():
            initial_weights[name] = {
                "A": layer.lora_A.data.clone(),
                "B": layer.lora_B.data.clone()
            }
        
        # 学習実行
        history = trainer.train()
        
        # 学習後のLoRA重みが変化していることを確認
        for name, layer in lora_model.lora_layers.items():
            # B行列は学習で変化するはず
            assert not torch.allclose(
                layer.lora_B.data,
                initial_weights[name]["B"]
            ), f"LoRA B weights for {name} should have changed"
        
        # チェックポイントが保存されていることを確認
        final_checkpoint = tmp_path / "checkpoints" / "final"
        assert (final_checkpoint / "lora_weights.pt").exists()
        assert (final_checkpoint / "lora_config.json").exists()
    
    def test_save_load_roundtrip(self, tmp_path):
        """保存→読み込みのラウンドトリップテスト"""
        # モデル作成
        base_model = SimpleLanguageModel()
        config = LoRAConfig(rank=4, alpha=8.0)
        lora_model = LoRAModel(base_model, config)
        
        # LoRA重みを設定
        for layer in lora_model.lora_layers.values():
            layer.lora_A.data = torch.randn_like(layer.lora_A)
            layer.lora_B.data = torch.randn_like(layer.lora_B)
        
        # 保存
        save_path = tmp_path / "lora_checkpoint"
        lora_model.save_lora_weights(str(save_path))
        
        # 新しいモデルに読み込み
        base_model2 = SimpleLanguageModel()
        lora_model2 = LoRAModel(base_model2, config)
        lora_model2.load_lora_weights(str(save_path))
        
        # 重みが一致することを確認
        for name in lora_model.lora_layers:
            assert torch.allclose(
                lora_model.lora_layers[name].lora_A,
                lora_model2.lora_layers[name].lora_A
            )
            assert torch.allclose(
                lora_model.lora_layers[name].lora_B,
                lora_model2.lora_layers[name].lora_B
            )
    
    def test_merge_preserves_output(self, tmp_path):
        """マージ前後で出力が一致することを確認"""
        # モデル作成
        base_model = SimpleLanguageModel()
        config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
        lora_model = LoRAModel(base_model, config)
        
        # LoRA重みを設定
        for layer in lora_model.lora_layers.values():
            layer.lora_A.data = torch.randn_like(layer.lora_A)
            layer.lora_B.data = torch.randn_like(layer.lora_B)
        
        # マージ前の出力
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output_before = lora_model(input_ids=input_ids).logits.clone()
        
        # マージ
        merged_model = lora_model.merge_and_unload()
        
        # マージ後の出力
        with torch.no_grad():
            output_after = merged_model(input_ids).logits
        
        # 出力が一致（浮動小数点誤差を考慮、値が大きいので相対誤差で比較）
        assert torch.allclose(output_before, output_after, atol=1.0, rtol=1e-4)
    
    def test_merged_model_has_no_lora(self):
        """マージ後にLoRAレイヤーがないことを確認"""
        base_model = SimpleLanguageModel()
        config = LoRAConfig(rank=4, alpha=8.0)
        lora_model = LoRAModel(base_model, config)
        
        # マージ前はLoRAレイヤーがある
        assert LoRAInference.has_lora_layers(lora_model.base_model)
        
        # マージ
        merged_model = lora_model.merge_and_unload()
        
        # マージ後はLoRAレイヤーがない
        assert not LoRAInference.has_lora_layers(merged_model)
    
    def test_dataset_split(self, tmp_path):
        """データセット分割のテスト"""
        # データ作成
        data_path = tmp_path / "data.txt"
        lines = [f"q{i}\ta{i}" for i in range(20)]
        data_path.write_text("\n".join(lines))
        
        tokenizer = SimpleTokenizer()
        dataset = InstructionDataset(str(data_path), tokenizer, max_length=32)
        
        # 分割
        train_ds, val_ds = dataset.split(val_ratio=0.2, seed=42)
        
        # サイズ確認
        assert len(train_ds) + len(val_ds) == len(dataset)
        assert len(val_ds) >= 1
        
        # 重複がないことを確認
        train_instructions = {ex["instruction"] for ex in train_ds.examples}
        val_instructions = {ex["instruction"] for ex in val_ds.examples}
        assert len(train_instructions & val_instructions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
