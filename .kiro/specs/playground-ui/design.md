# Design Document: Playground UI

## Overview

Mini TransformerモデルをブラウザでテストできるChatGPT風Playground UIを実装する。バックエンドはPython (aiohttp + WebSocket)、フロントエンドはVanilla JS + CSSで構築し、既存のvisualizerディレクトリに追加する。

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              playground.html                          │   │
│  │  ┌──────────┐  ┌──────────────────────────────┐    │   │
│  │  │ Settings │  │      Chat Messages Area       │    │   │
│  │  │ Panel    │  │  ┌─────────────────────────┐ │    │   │
│  │  │          │  │  │ User: こんにちは        │ │    │   │
│  │  │ Checkpoint│  │  └─────────────────────────┘ │    │   │
│  │  │ Selector │  │  ┌─────────────────────────┐ │    │   │
│  │  │          │  │  │ AI: こんにちは、今日は...│ │    │   │
│  │  │ Params   │  │  └─────────────────────────┘ │    │   │
│  │  └──────────┘  └──────────────────────────────┘    │   │
│  │              ┌──────────────────────────────┐       │   │
│  │              │  Input Field        [Send]   │       │   │
│  │              └──────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket / HTTP
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend (playground_server.py)               │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ HTTP Endpoints  │  │      WebSocket Handler          │  │
│  │ /checkpoints    │  │  - Stream token generation      │  │
│  │ /model-info     │  │  - Real-time response           │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
│                              │                               │
│  ┌───────────────────────────┴───────────────────────────┐  │
│  │                  Model Manager                         │  │
│  │  - Load/switch checkpoints                            │  │
│  │  - Generate text with streaming                       │  │
│  │  - Cache loaded models                                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### Backend Components

#### 1. PlaygroundServer (playground_server.py)

```python
class PlaygroundServer:
    """Playground APIサーバー"""
    
    def __init__(self, checkpoint_dir: str, host: str, port: int):
        self.checkpoint_dir = checkpoint_dir
        self.model_manager = ModelManager(checkpoint_dir)
        self.app = web.Application()
        
    async def get_checkpoints(self, request) -> web.Response:
        """GET /api/checkpoints - チェックポイント一覧を返す"""
        
    async def get_model_info(self, request) -> web.Response:
        """GET /api/model-info - 現在のモデル情報を返す"""
        
    async def load_checkpoint(self, request) -> web.Response:
        """POST /api/load-checkpoint - チェックポイントを読み込む"""
        
    async def websocket_handler(self, request) -> web.WebSocketResponse:
        """WebSocket /ws/generate - ストリーミング生成"""
```

#### 2. ModelManager

```python
class ModelManager:
    """モデルの読み込みと生成を管理"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.current_model: Optional[MiniGPT] = None
        self.current_checkpoint: Optional[str] = None
        self.tokenizer: Optional[CharTokenizer] = None
        
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """利用可能なチェックポイント一覧"""
        
    def load_checkpoint(self, checkpoint_name: str) -> ModelInfo:
        """チェックポイントを読み込み"""
        
    async def generate_stream(
        self,
        prompt: str,
        temperature: float,
        top_k: int,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
```

### Frontend Components

#### 1. HTML Structure (playground.html)

```html
<div class="playground-container">
    <!-- サイドバー: 設定パネル -->
    <aside class="settings-panel">
        <div class="checkpoint-selector">...</div>
        <div class="generation-params">...</div>
        <div class="model-info">...</div>
    </aside>
    
    <!-- メインエリア: チャット -->
    <main class="chat-area">
        <div class="messages-container">...</div>
        <div class="input-area">...</div>
    </main>
</div>
```

#### 2. JavaScript Modules

```javascript
// PlaygroundApp - メインアプリケーション
class PlaygroundApp {
    constructor()
    async init()
    async loadCheckpoints()
    async selectCheckpoint(name)
    async sendMessage(prompt)
    clearChat()
}

// ChatUI - チャットUI管理
class ChatUI {
    addUserMessage(text)
    addAssistantMessage()
    appendToLastMessage(token)
    finishLastMessage()
    showTypingIndicator()
    hideTypingIndicator()
    clearMessages()
}

// WebSocketClient - WebSocket通信
class WebSocketClient {
    connect()
    send(data)
    onMessage(callback)
    onError(callback)
    close()
}
```

## Data Models

### CheckpointInfo

```python
@dataclass
class CheckpointInfo:
    name: str           # "checkpoint_step4000.pt"
    step: int           # 4000
    loss: float         # 4.09
    timestamp: str      # "2024-12-30 12:00:00"
    file_size: int      # bytes
```

### ModelInfo

```python
@dataclass
class ModelInfo:
    checkpoint_name: str
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    max_len: int
    total_params: int
    training_loss: float
```

### GenerateRequest (WebSocket)

```json
{
    "type": "generate",
    "prompt": "今日は",
    "temperature": 0.8,
    "top_k": 50,
    "max_tokens": 100
}
```

### GenerateResponse (WebSocket Stream)

```json
{"type": "token", "token": "天"}
{"type": "token", "token": "気"}
{"type": "token", "token": "が"}
{"type": "done", "full_text": "今日は天気が..."}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system.*

### Property 1: Checkpoint list completeness
*For any* checkpoint directory containing valid checkpoint files, the `/api/checkpoints` endpoint SHALL return all checkpoints with valid step numbers, loss values, and timestamps.
**Validates: Requirements 1.1**

### Property 2: Checkpoint loading consistency
*For any* valid checkpoint name, loading that checkpoint SHALL result in a model that can generate text, and the model info SHALL match the checkpoint's stored configuration.
**Validates: Requirements 1.2, 5.1, 5.2, 5.3**

### Property 3: Conversation history persistence
*For any* sequence of user messages sent within a session, all messages SHALL remain visible in the chat history until explicitly cleared.
**Validates: Requirements 2.5**

### Property 4: Parameter bounds validation
*For any* generation parameter input, temperature SHALL be clamped to [0.0, 2.0], top_k SHALL be clamped to [0, 100], and max_tokens SHALL be clamped to [1, 500].
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 5: Parameter application
*For any* generation request, the parameters used SHALL match the current UI settings at the time of request.
**Validates: Requirements 3.4**

### Property 6: Streaming token delivery
*For any* generation request, tokens SHALL be delivered incrementally via WebSocket, and the final concatenated result SHALL equal the complete generated text.
**Validates: Requirements 4.1**

## Error Handling

| Error Condition | Response | User Feedback |
|----------------|----------|---------------|
| Server unavailable | WebSocket close | "サーバーに接続できません" |
| Checkpoint not found | 404 JSON error | "チェックポイントが見つかりません" |
| Checkpoint load failure | 500 JSON error | "モデルの読み込みに失敗しました" |
| Generation error | WebSocket error message | "生成中にエラーが発生しました" |
| Invalid parameters | 400 JSON error | "パラメータが無効です" |

## Testing Strategy

### Unit Tests
- Checkpoint discovery and parsing
- Parameter validation and clamping
- WebSocket message serialization

### Integration Tests
- Full generation flow with streaming
- Checkpoint switching
- Error handling scenarios

### Property-Based Tests
- Use `hypothesis` for Python backend tests
- Test checkpoint list completeness with various directory states
- Test parameter bounds with random inputs

