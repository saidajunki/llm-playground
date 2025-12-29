# Requirements Document

## Introduction

Mini Transformerモデルをブラウザ上でインタラクティブにテストできるPlayground UIを実装する。ChatGPT風のチャットインターフェースで、チェックポイントを選択してテキスト生成を試すことができる。

## Glossary

- **Playground**: モデルをインタラクティブにテストするためのWebベースUI
- **Checkpoint**: 学習途中または完了時に保存されたモデルの重み
- **Generation_Parameters**: テキスト生成時の設定（temperature, top_k, max_tokens等）
- **Chat_Interface**: ユーザーがプロンプトを入力し、モデルの応答を表示するUI
- **Backend_Server**: チェックポイント管理とテキスト生成APIを提供するサーバー

## Requirements

### Requirement 1: チェックポイント選択

**User Story:** As a developer, I want to select different checkpoints, so that I can compare model outputs at different training stages.

#### Acceptance Criteria

1. WHEN the Playground loads, THE Backend_Server SHALL return a list of available checkpoints with step numbers and timestamps
2. WHEN a user selects a checkpoint from the dropdown, THE Backend_Server SHALL load that checkpoint into memory
3. WHILE a checkpoint is loading, THE Chat_Interface SHALL display a loading indicator
4. WHEN a checkpoint is successfully loaded, THE Chat_Interface SHALL display the checkpoint info (step number, loss, parameters)

### Requirement 2: チャットインターフェース

**User Story:** As a user, I want a ChatGPT-like interface, so that I can easily interact with the model.

#### Acceptance Criteria

1. THE Chat_Interface SHALL display a message input field at the bottom of the screen
2. WHEN a user types a prompt and presses Enter or clicks send, THE Chat_Interface SHALL display the user's message in a chat bubble
3. WHEN a generation request is sent, THE Chat_Interface SHALL display a typing indicator while waiting for response
4. WHEN the model generates text, THE Chat_Interface SHALL display the response in a distinct chat bubble
5. THE Chat_Interface SHALL maintain conversation history within the session
6. WHEN a user clicks the clear button, THE Chat_Interface SHALL clear all messages from the conversation

### Requirement 3: 生成パラメータ設定

**User Story:** As a developer, I want to adjust generation parameters, so that I can experiment with different sampling strategies.

#### Acceptance Criteria

1. THE Chat_Interface SHALL provide controls for temperature (0.0 to 2.0, default 0.8)
2. THE Chat_Interface SHALL provide controls for top_k (0 to 100, default 50)
3. THE Chat_Interface SHALL provide controls for max_tokens (1 to 500, default 100)
4. WHEN generation parameters are changed, THE Chat_Interface SHALL apply them to subsequent generations
5. THE Chat_Interface SHALL display current parameter values in a collapsible settings panel

### Requirement 4: リアルタイム生成表示

**User Story:** As a user, I want to see text being generated in real-time, so that I can observe the model's behavior.

#### Acceptance Criteria

1. WHEN the model generates text, THE Chat_Interface SHALL stream tokens as they are generated (WebSocket)
2. WHILE streaming, THE Chat_Interface SHALL display a cursor or animation at the end of the text
3. IF generation is interrupted, THEN THE Chat_Interface SHALL display the partial response with an indication

### Requirement 5: モデル情報表示

**User Story:** As a developer, I want to see model information, so that I can understand the current model configuration.

#### Acceptance Criteria

1. THE Chat_Interface SHALL display model configuration (vocab_size, embed_dim, num_heads, num_layers)
2. THE Chat_Interface SHALL display total parameter count
3. THE Chat_Interface SHALL display current checkpoint's training loss
4. WHEN hovering over model info, THE Chat_Interface SHALL show detailed configuration in a tooltip

### Requirement 6: エラーハンドリング

**User Story:** As a user, I want clear error messages, so that I can understand and resolve issues.

#### Acceptance Criteria

1. IF the Backend_Server is unavailable, THEN THE Chat_Interface SHALL display a connection error message
2. IF checkpoint loading fails, THEN THE Chat_Interface SHALL display an error with the reason
3. IF generation fails, THEN THE Chat_Interface SHALL display an error message in the chat
4. WHEN an error occurs, THE Chat_Interface SHALL provide a retry option where applicable

### Requirement 7: レスポンシブデザイン

**User Story:** As a user, I want the interface to work on different screen sizes, so that I can use it on various devices.

#### Acceptance Criteria

1. THE Chat_Interface SHALL adapt layout for desktop (>1024px), tablet (768-1024px), and mobile (<768px)
2. THE Chat_Interface SHALL maintain usability on all supported screen sizes
3. THE Chat_Interface SHALL use a dark theme consistent with modern chat applications
