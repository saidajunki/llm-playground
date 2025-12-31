# Design Document: LoRA Fine-tuning for Qwen2-0.5B

## Overview

本設計では、Qwen2-0.5BモデルにLoRA（Low-Rank Adaptation）を適用し、パラメータ効率的なファインチューニングを実現する。LoRAは元のモデルの重み行列Wに対して、低ランク分解 ΔW = BA を加算することで、学習パラメータ数を大幅に削減しながら効果的な適応を可能にする。

### 設計方針

1. **教育的実装**: PEFTライブラリを使わず、LoRAの仕組みを理解できるスクラッチ実装
2. **実用性**: 実際にQwen2-0.5Bをファインチューニングできる完全な実装
3. **効率性**: 8GB GPUでも動作する省メモリ設計

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LoRA Fine-tuning System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  LoRAConfig  │───▶│  LoRALinear  │───▶│ LoRAModel    │  │
│  │              │    │              │    │              │  │
│  │ - rank       │    │ - lora_A     │    │ - base_model │  │
│  │ - alpha      │    │ - lora_B     │    │ - lora_layers│  │
│  │ - target_mods│    │ - scaling    │    │ - config     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                    │         │
│                              ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │LoRATrainer   │◀───│InstructData  │    │LoRAInference │  │
│  │              │    │              │    │              │  │
│  │ - train()    │    │ - tokenize() │    │ - generate() │  │
│  │ - save()     │    │ - format()   │    │ - merge()    │  │
│  │ - load()     │    │ - split()    │    │ - compare()  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### LoRAの数学的原理

元の線形層: y = Wx + b
LoRA適用後: y = Wx + BAx * (α/r) + b

- W: 元の重み行列 (d_out × d_in)、凍結
- A: 低ランク行列 (r × d_in)、学習対象
- B: 低ランク行列 (d_out × r)、学習対象
- r: ランク（通常 4〜64）
- α: スケーリング係数（通常 r と同じか2倍）

## Components and Interfaces

### 1. LoRAConfig

```python
@dataclass
class LoRAConfig:
    """LoRAのハイパーパラメータ設定"""
    rank: int = 8                    # 低ランク行列のランク
    alpha: float = 16.0              # スケーリング係数
    dropout: float = 0.05            # ドロップアウト率
    target_modules: List[str] = None # 対象モジュール名
    
    # デフォルトでQwen2のattention層を対象
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]
    
    @property
    def scaling(self) -> float:
        return self.alpha / self.rank
    
    def to_dict(self) -> dict: ...
    
    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig": ...
    
    def save(self, path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> "LoRAConfig": ...
```

### 2. LoRALinear

```python
class LoRALinear(nn.Module):
    """LoRAを適用した線形層"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA行列
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # 初期化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 元の重みを凍結
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 元の出力
        original_output = self.original_layer(x)
        
        # LoRA出力: (B @ A) @ x * scaling
        lora_output = self.lora_dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # x @ A^T
        lora_output = F.linear(lora_output, self.lora_B)  # (x @ A^T) @ B^T
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """LoRA重みを元の重みにマージ"""
        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None
        )
        
        # W_merged = W + B @ A * scaling
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.original_layer.weight.data + delta_w
        
        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data.clone()
        
        return merged
```

### 3. LoRAModel

```python
class LoRAModel(nn.Module):
    """LoRAを適用したモデルラッパー"""
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: LoRAConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.lora_layers: Dict[str, LoRALinear] = {}
        
        self._apply_lora()
    
    def _apply_lora(self) -> None:
        """対象モジュールにLoRAを適用"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(
                        module,
                        self.config.rank,
                        self.config.alpha,
                        self.config.dropout
                    )
                    self._replace_module(name, lora_layer)
                    self.lora_layers[name] = lora_layer
    
    def _replace_module(self, name: str, new_module: nn.Module) -> None:
        """モジュールを置換"""
        parts = name.split(".")
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def forward(self, **kwargs) -> CausalLMOutput:
        return self.base_model(**kwargs)
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """学習対象パラメータのみを返す"""
        for layer in self.lora_layers.values():
            yield layer.lora_A
            yield layer.lora_B
    
    def count_parameters(self) -> Tuple[int, int]:
        """(学習パラメータ数, 全パラメータ数)を返す"""
        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        total = sum(p.numel() for p in self.base_model.parameters())
        return trainable, total
    
    def save_lora_weights(self, path: str) -> None:
        """LoRA重みのみを保存"""
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = layer.lora_A.data
            state_dict[f"{name}.lora_B"] = layer.lora_B.data
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, save_path / "lora_weights.pt")
        self.config.save(str(save_path / "lora_config.json"))
    
    def load_lora_weights(self, path: str) -> None:
        """LoRA重みを読み込み"""
        state_dict = torch.load(Path(path) / "lora_weights.pt")
        for name, layer in self.lora_layers.items():
            layer.lora_A.data = state_dict[f"{name}.lora_A"]
            layer.lora_B.data = state_dict[f"{name}.lora_B"]
    
    def merge_and_unload(self) -> PreTrainedModel:
        """LoRA重みをマージして元のモデル形式で返す"""
        for name, lora_layer in self.lora_layers.items():
            merged_layer = lora_layer.merge_weights()
            self._replace_module(name, merged_layer)
        
        return self.base_model
```

### 4. InstructionDataset

```python
class InstructionDataset(Dataset):
    """指示-応答形式のデータセット"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[Dict[str, str]]:
        """データファイルを読み込み"""
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # フォーマット: instruction\tresponse
                parts = line.split("\t")
                if len(parts) >= 2:
                    examples.append({
                        "instruction": parts[0],
                        "response": parts[1]
                    })
        return examples
    
    def _format_example(self, example: Dict[str, str]) -> str:
        """Qwen2のチャットテンプレートでフォーマット"""
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = self._format_example(example)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }
    
    def split(self, val_ratio: float = 0.1) -> Tuple["InstructionDataset", "InstructionDataset"]:
        """訓練/検証に分割"""
        n_val = int(len(self.examples) * val_ratio)
        indices = torch.randperm(len(self.examples)).tolist()
        
        train_dataset = copy.copy(self)
        val_dataset = copy.copy(self)
        
        train_dataset.examples = [self.examples[i] for i in indices[n_val:]]
        val_dataset.examples = [self.examples[i] for i in indices[:n_val]]
        
        return train_dataset, val_dataset
```

### 5. LoRATrainer

```python
class LoRATrainer:
    """LoRAファインチューニング用トレーナー"""
    
    def __init__(
        self,
        model: LoRAModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: InstructionDataset,
        val_dataset: Optional[InstructionDataset] = None,
        output_dir: str = "checkpoints/lora",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 10,
        save_interval: int = 500,
        fp16: bool = True,
        gradient_checkpointing: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        
        # ハイパーパラメータ
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        
        # 初期化
        self._setup()
    
    def _setup(self) -> None:
        """トレーニング環境のセットアップ"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # Optimizer（LoRAパラメータのみ）
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # Gradient checkpointing
        if self.gradient_checkpointing:
            self.model.base_model.gradient_checkpointing_enable()
    
    def train(self) -> Dict[str, List[float]]:
        """トレーニングループ"""
        self.model.train()
        device = next(self.model.parameters()).device
        
        history = {"train_loss": [], "val_loss": []}
        global_step = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                # バッチをデバイスに移動
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # ログ出力
                    if global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / (step + 1)
                        print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                        history["train_loss"].append(avg_loss)
                    
                    # チェックポイント保存
                    if global_step % self.save_interval == 0:
                        self.save_checkpoint(f"step_{global_step}")
            
            # エポック終了時の検証
            if self.val_dataset:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            # エポック終了時のチェックポイント
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        return history
    
    def evaluate(self) -> float:
        """検証データでの評価"""
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, name: str) -> None:
        """チェックポイントを保存"""
        checkpoint_path = self.output_dir / name
        self.model.save_lora_weights(str(checkpoint_path))
        
        # トレーニング状態も保存
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None
        }, checkpoint_path / "trainer_state.pt")
    
    def load_checkpoint(self, name: str) -> None:
        """チェックポイントから復元"""
        checkpoint_path = self.output_dir / name
        self.model.load_lora_weights(str(checkpoint_path))
        
        state = torch.load(checkpoint_path / "trainer_state.pt")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler and state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
```

### 6. LoRAInference

```python
class LoRAInference:
    """LoRAモデルでの推論"""
    
    def __init__(
        self,
        model: Union[LoRAModel, PreTrainedModel],
        tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    @classmethod
    def from_lora_checkpoint(
        cls,
        base_model_path: str,
        lora_path: str,
        merge: bool = False
    ) -> "LoRAInference":
        """LoRAチェックポイントから読み込み"""
        # ベースモデル読み込み
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
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
        top_p: float = 0.9
    ) -> str:
        """テキスト生成"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    @staticmethod
    def compare_outputs(
        lora_inference: "LoRAInference",
        merged_inference: "LoRAInference",
        prompts: List[str],
        tolerance: float = 1e-5
    ) -> bool:
        """LoRAモデルとマージモデルの出力を比較"""
        for prompt in prompts:
            # 同じシードで生成
            torch.manual_seed(42)
            lora_output = lora_inference.generate(prompt, temperature=0.0)
            
            torch.manual_seed(42)
            merged_output = merged_inference.generate(prompt, temperature=0.0)
            
            if lora_output != merged_output:
                print(f"Mismatch for prompt: {prompt}")
                print(f"LoRA: {lora_output}")
                print(f"Merged: {merged_output}")
                return False
        
        return True
```

## Data Models

### LoRAConfig JSON Format

```json
{
    "rank": 8,
    "alpha": 16.0,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

### Instruction Dataset Format

タブ区切りテキストファイル:
```
質問1\t回答1
質問2\t回答2
```

### LoRA Checkpoint Structure

```
checkpoints/lora/
├── lora_weights.pt      # LoRA行列のみ（数MB）
├── lora_config.json     # 設定ファイル
└── trainer_state.pt     # オプティマイザ状態（再開用）
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: LoRA Forward Computation

*For any* input tensor x and LoRALinear layer with original weight W, rank r, and alpha α, the output should equal W*x + (B @ A)*x * (α/r), where A and B are the low-rank matrices.

**Validates: Requirements 1.1, 1.4**

### Property 2: LoRA Initialization

*For any* newly created LoRALinear layer, matrix B should be initialized to all zeros, ensuring the initial LoRA contribution is zero.

**Validates: Requirements 1.2**

### Property 3: Scaling Factor Computation

*For any* LoRA configuration with rank r and alpha α, the scaling factor should equal α/r.

**Validates: Requirements 1.3**

### Property 4: LoRA Config Round-Trip

*For any* valid LoRAConfig, serializing to JSON and deserializing should produce an equivalent configuration with all parameters preserved.

**Validates: Requirements 2.3, 2.4**

### Property 5: Base Model Freezing

*For any* model with LoRA applied, all original model parameters should have requires_grad=False.

**Validates: Requirements 3.1**

### Property 6: Target Module Replacement

*For any* model with LoRA applied, all modules matching target_modules patterns should be replaced with LoRALinear instances.

**Validates: Requirements 3.2**

### Property 7: Trainable Parameter Count

*For any* LoRA model, the count of trainable parameters should equal exactly the sum of LoRA A and B matrix elements across all LoRA layers, and this should be less than 1% of total model parameters.

**Validates: Requirements 3.4, 3.5**

### Property 8: Training Preserves Base Weights

*For any* training step on a LoRA model, the base model weights should remain unchanged (frozen).

**Validates: Requirements 4.1**

### Property 9: LoRA Weights Save/Load Round-Trip

*For any* LoRA model, saving weights and loading them into a fresh LoRA model should produce identical LoRA parameters.

**Validates: Requirements 4.4, 4.5, 6.1, 6.3**

### Property 10: Checkpoint Size

*For any* saved LoRA checkpoint, the file size should be less than 1% of the full model size.

**Validates: Requirements 6.4**

### Property 11: Dataset Tokenization Length

*For any* instruction-response pair in the dataset, the tokenized output length should not exceed max_length.

**Validates: Requirements 5.4**

### Property 12: Dataset Split Integrity

*For any* dataset split with ratio r, the train and validation sets should be non-overlapping and their combined size should equal the original dataset size.

**Validates: Requirements 5.5**

### Property 13: Merge Computation

*For any* LoRALinear layer, the merged weight should equal W_original + (B @ A) * scaling.

**Validates: Requirements 7.1**

### Property 14: Merge Output Equivalence

*For any* input, the LoRA-applied model and the merged model should produce identical outputs (within floating-point tolerance).

**Validates: Requirements 7.2, 8.3**

### Property 15: Merged Model Structure

*For any* merged model, there should be no LoRALinear modules remaining—all should be replaced with standard Linear layers.

**Validates: Requirements 7.3**

## Error Handling

### Memory Errors

```python
class MemoryError(Exception):
    """GPU メモリ不足エラー"""
    
    def __init__(self, required_gb: float, available_gb: float):
        self.required_gb = required_gb
        self.available_gb = available_gb
        suggestions = [
            f"Required: {required_gb:.1f}GB, Available: {available_gb:.1f}GB",
            "Suggestions:",
            "  1. Reduce batch_size",
            "  2. Enable gradient_checkpointing=True",
            "  3. Reduce max_length",
            "  4. Use fp16=True for mixed precision"
        ]
        super().__init__("\n".join(suggestions))
```

### Configuration Errors

```python
class LoRAConfigError(Exception):
    """LoRA設定エラー"""
    pass

def validate_config(config: LoRAConfig, model: nn.Module) -> None:
    """設定の妥当性を検証"""
    # ランクの検証
    if config.rank <= 0:
        raise LoRAConfigError(f"rank must be positive, got {config.rank}")
    
    # 対象モジュールの存在確認
    module_names = [name for name, _ in model.named_modules()]
    for target in config.target_modules:
        if not any(target in name for name in module_names):
            raise LoRAConfigError(f"Target module '{target}' not found in model")
```

### Checkpoint Errors

```python
class CheckpointError(Exception):
    """チェックポイント関連エラー"""
    pass

def validate_checkpoint(path: str, config: LoRAConfig) -> None:
    """チェックポイントの妥当性を検証"""
    checkpoint_path = Path(path)
    
    if not (checkpoint_path / "lora_weights.pt").exists():
        raise CheckpointError(f"LoRA weights not found at {path}")
    
    if not (checkpoint_path / "lora_config.json").exists():
        raise CheckpointError(f"LoRA config not found at {path}")
    
    # 設定の互換性確認
    saved_config = LoRAConfig.load(str(checkpoint_path / "lora_config.json"))
    if saved_config.rank != config.rank:
        raise CheckpointError(
            f"Rank mismatch: checkpoint has {saved_config.rank}, "
            f"model expects {config.rank}"
        )
```

## Testing Strategy

### Unit Tests

単体テストは具体的な例とエッジケースを検証する：

1. **LoRALinear Tests**
   - 初期化時のB行列がゼロであることを確認
   - 様々なランク/アルファ値での動作確認
   - 空入力やバッチサイズ1での動作

2. **LoRAConfig Tests**
   - デフォルト値の確認
   - JSON保存/読み込みの動作確認
   - 無効な設定値でのエラー処理

3. **LoRAModel Tests**
   - 対象モジュールの置換確認
   - パラメータ凍結の確認
   - 重みの保存/読み込み

4. **InstructionDataset Tests**
   - ファイル読み込みの動作確認
   - トークナイゼーションの動作確認
   - 分割の動作確認

### Property-Based Tests

プロパティベーステストは普遍的な性質を検証する。Hypothesisライブラリを使用：

```python
from hypothesis import given, strategies as st

# Property 1: LoRA Forward Computation
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=32),
    in_features=st.integers(min_value=16, max_value=64),
    out_features=st.integers(min_value=16, max_value=64),
    rank=st.integers(min_value=1, max_value=8),
    alpha=st.floats(min_value=1.0, max_value=32.0)
)
def test_lora_forward_computation(batch_size, seq_len, in_features, out_features, rank, alpha):
    """Feature: lora-finetuning, Property 1: LoRA Forward Computation"""
    # ... test implementation
```

### Test Configuration

- Property-based tests: minimum 100 iterations per property
- Use Hypothesis for Python property-based testing
- Each property test references its design document property number
- Tag format: `Feature: lora-finetuning, Property {number}: {property_text}`

### Test File Structure

```
phase2/tests/
├── __init__.py
├── test_lora_linear.py      # LoRALinear unit tests
├── test_lora_config.py      # LoRAConfig unit tests
├── test_lora_model.py       # LoRAModel unit tests
├── test_dataset.py          # InstructionDataset tests
├── test_trainer.py          # LoRATrainer tests
├── test_inference.py        # Inference tests
└── test_properties.py       # Property-based tests
```
