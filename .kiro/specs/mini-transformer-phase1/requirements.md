# Requirements Document

## Introduction

Phase 1では、LLMの仕組みを深く理解するために、ミニTransformerモデルをフルスクラッチで実装する。目的は「会話ができるモデルを作る」ことではなく、「Transformerアーキテクチャの各コンポーネントを自分の手で実装し、動作原理を体得する」ことである。

## Glossary

- **Transformer**: Attention機構をベースにした深層学習アーキテクチャ
- **Tokenizer**: テキストをトークン（数値ID）に変換するコンポーネント
- **Embedding**: トークンIDを密なベクトル表現に変換するレイヤー
- **Positional_Encoding**: トークンの位置情報をベクトルに付与する仕組み
- **Self_Attention**: 入力シーケンス内の各トークンが他のトークンとの関係性を計算する機構
- **Multi_Head_Attention**: 複数のAttentionヘッドを並列に実行し、異なる観点からの関係性を捉える機構
- **Feed_Forward_Network**: Attention層の出力を非線形変換する全結合層
- **Layer_Normalization**: 学習を安定させるための正規化手法
- **Causal_Mask**: 未来のトークンを参照できないようにするマスク（自己回帰生成用）
- **Training_Loop**: データを使ってモデルのパラメータを更新する反復処理
- **Inference**: 学習済みモデルを使ってテキストを生成する処理

## Requirements

### Requirement 1: トークナイザーの実装

**User Story:** As a 学習者, I want to テキストをトークンに変換する仕組みを実装する, so that LLMがテキストをどのように数値として扱うかを理解できる

#### Acceptance Criteria

1. WHEN テキストが入力された時, THE Tokenizer SHALL テキストを文字単位またはサブワード単位でトークンIDのリストに変換する
2. WHEN トークンIDのリストが入力された時, THE Tokenizer SHALL 元のテキストに復元（デコード）できる
3. THE Tokenizer SHALL 語彙辞書（vocabulary）を構築し、未知語を特殊トークンで処理する
4. FOR ALL 有効なテキスト, エンコードしてデコードした結果 SHALL 元のテキストと等価である（ラウンドトリップ）

### Requirement 2: Embeddingレイヤーの実装

**User Story:** As a 学習者, I want to トークンIDをベクトルに変換する仕組みを実装する, so that 離散的なトークンが連続的なベクトル空間でどう表現されるかを理解できる

#### Acceptance Criteria

1. WHEN トークンIDが入力された時, THE Embedding SHALL 対応する埋め込みベクトルを返す
2. THE Embedding SHALL 学習可能なパラメータとして埋め込み行列を保持する
3. WHEN バッチ処理が行われる時, THE Embedding SHALL 複数のシーケンスを同時に処理できる

### Requirement 3: Positional Encodingの実装

**User Story:** As a 学習者, I want to 位置情報をベクトルに付与する仕組みを実装する, so that Transformerが順序情報をどう扱うかを理解できる

#### Acceptance Criteria

1. WHEN 埋め込みベクトルが入力された時, THE Positional_Encoding SHALL 位置情報を加算したベクトルを返す
2. THE Positional_Encoding SHALL sinとcosを使った固定的なエンコーディングを実装する
3. WHEN 異なる位置のトークンが処理される時, THE Positional_Encoding SHALL それぞれ異なる位置ベクトルを付与する

### Requirement 4: Self-Attention機構の実装

**User Story:** As a 学習者, I want to Attention機構を実装する, so that Transformerの核心部分がどう動作するかを理解できる

#### Acceptance Criteria

1. WHEN 入力ベクトルが与えられた時, THE Self_Attention SHALL Query, Key, Valueを計算する
2. WHEN Query, Key, Valueが計算された時, THE Self_Attention SHALL Attention重みをsoftmaxで計算する
3. WHEN 自己回帰生成モードの時, THE Self_Attention SHALL Causal_Maskを適用して未来のトークンを参照しない
4. THE Self_Attention SHALL スケーリング（√d_k で割る）を適用してAttentionスコアを安定させる

### Requirement 5: Multi-Head Attentionの実装

**User Story:** As a 学習者, I want to 複数のAttentionヘッドを並列実行する仕組みを実装する, so that 異なる観点からの関係性をどう捉えるかを理解できる

#### Acceptance Criteria

1. WHEN 入力が与えられた時, THE Multi_Head_Attention SHALL 複数のAttentionヘッドを並列に実行する
2. WHEN 各ヘッドの出力が得られた時, THE Multi_Head_Attention SHALL それらを結合して線形変換する
3. THE Multi_Head_Attention SHALL ヘッド数を設定可能なパラメータとして持つ

### Requirement 6: Feed Forward Networkの実装

**User Story:** As a 学習者, I want to Attention後の非線形変換を実装する, so that Transformerブロックの構造を理解できる

#### Acceptance Criteria

1. WHEN Attention出力が入力された時, THE Feed_Forward_Network SHALL 2層の全結合層で変換する
2. THE Feed_Forward_Network SHALL 中間層でGELUまたはReLU活性化関数を適用する
3. THE Feed_Forward_Network SHALL 入力と出力の次元を同じに保つ

### Requirement 7: Transformer Blockの実装

**User Story:** As a 学習者, I want to Transformerの1ブロックを組み立てる, so that 各コンポーネントがどう連携するかを理解できる

#### Acceptance Criteria

1. THE Transformer_Block SHALL Multi_Head_Attention、Layer_Normalization、Feed_Forward_Networkを含む
2. THE Transformer_Block SHALL 残差接続（Residual Connection）を実装する
3. WHEN 入力が処理される時, THE Transformer_Block SHALL Pre-LN（Layer Normを先に適用）またはPost-LN構成を選択可能にする

### Requirement 8: 完全なTransformerモデルの組み立て

**User Story:** As a 学習者, I want to 全コンポーネントを組み合わせて完全なモデルを作る, so that GPTライクなアーキテクチャの全体像を理解できる

#### Acceptance Criteria

1. THE Transformer_Model SHALL Embedding、Positional_Encoding、複数のTransformer_Block、出力層を含む
2. THE Transformer_Model SHALL 設定可能なハイパーパラメータ（層数、ヘッド数、埋め込み次元など）を持つ
3. WHEN 推論時, THE Transformer_Model SHALL 次のトークンの確率分布を出力する

### Requirement 9: 学習ループの実装

**User Story:** As a 学習者, I want to モデルを学習させる仕組みを実装する, so that ニューラルネットワークの学習プロセスを理解できる

#### Acceptance Criteria

1. THE Training_Loop SHALL Cross-Entropy損失関数を使用する
2. THE Training_Loop SHALL AdamまたはAdamWオプティマイザを使用する
3. WHEN 学習が進行する時, THE Training_Loop SHALL 損失値をログ出力する
4. THE Training_Loop SHALL 学習率スケジューリング（warmup + decay）を実装する
5. THE Training_Loop SHALL チェックポイント保存機能を持つ

### Requirement 10: テキスト生成（推論）の実装

**User Story:** As a 学習者, I want to 学習済みモデルでテキストを生成する, so that 自己回帰生成の仕組みを理解できる

#### Acceptance Criteria

1. WHEN プロンプトが与えられた時, THE Inference SHALL 次のトークンを予測して追加する自己回帰生成を行う
2. THE Inference SHALL Temperature調整によるサンプリングを実装する
3. THE Inference SHALL Top-kサンプリングを実装する
4. THE Inference SHALL 最大生成長を設定可能にする

### Requirement 11: 学習用データセットの準備

**User Story:** As a 学習者, I want to 小規模なデータセットを準備する, so that モデルを実際に学習させて動作確認できる

#### Acceptance Criteria

1. THE Dataset SHALL 小規模な日本語または英語テキストコーパスを使用する
2. THE Dataset SHALL テキストを固定長のシーケンスに分割する
3. THE Dataset SHALL バッチ処理に対応したDataLoaderを提供する
