# Implementation Plan: Playground UI

## Overview

Mini TransformerモデルをブラウザでテストできるChatGPT風Playground UIを実装する。バックエンドサーバーとフロントエンドUIを段階的に構築する。

## Tasks

- [x] 1. バックエンドサーバーの基盤構築
  - [x] 1.1 ModelManagerクラスの実装
    - チェックポイント一覧取得機能
    - チェックポイント読み込み機能
    - モデル情報取得機能
    - _Requirements: 1.1, 1.2, 5.1, 5.2, 5.3_
  - [x] 1.2 PlaygroundServerの基本構造
    - aiohttpアプリケーション設定
    - 静的ファイル配信設定
    - CORSヘッダー設定
    - _Requirements: 6.1_

- [x] 2. REST APIエンドポイントの実装
  - [x] 2.1 GET /api/checkpoints エンドポイント
    - チェックポイント一覧をJSON形式で返す
    - _Requirements: 1.1_
  - [x] 2.2 POST /api/load-checkpoint エンドポイント
    - 指定チェックポイントを読み込み
    - _Requirements: 1.2_
  - [x] 2.3 GET /api/model-info エンドポイント
    - 現在のモデル情報を返す
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 3. ストリーミング生成の実装
  - [x] 3.1 generate_stream関数の実装
    - トークン単位でyieldする非同期ジェネレータ
    - temperature, top_k, max_tokensパラメータ対応
    - _Requirements: 4.1, 3.4_
  - [x] 3.2 WebSocket /ws/generate ハンドラ
    - 生成リクエスト受信
    - トークンストリーミング送信
    - エラーハンドリング
    - _Requirements: 4.1, 4.3, 6.3_

- [x] 4. フロントエンドHTML/CSS構築
  - [x] 4.1 playground.htmlの基本構造
    - サイドバー（設定パネル）
    - メインエリア（チャット）
    - 入力エリア
    - _Requirements: 2.1, 7.3_
  - [x] 4.2 CSSスタイリング
    - ダークテーマ
    - チャットバブルスタイル
    - レスポンシブレイアウト
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 5. フロントエンドJavaScript実装
  - [x] 5.1 WebSocketClientクラス
    - 接続管理
    - メッセージ送受信
    - 再接続ロジック
    - _Requirements: 4.1, 6.1_
  - [x] 5.2 ChatUIクラス
    - メッセージ追加/削除
    - ストリーミング表示
    - タイピングインジケータ
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_
  - [x] 5.3 PlaygroundAppクラス
    - 初期化とイベントバインディング
    - チェックポイント選択処理
    - パラメータ管理
    - _Requirements: 1.4, 3.1, 3.2, 3.3, 3.5_

- [x] 6. 統合とエラーハンドリング
  - [x] 6.1 エラー表示の実装
    - 接続エラー表示
    - 生成エラー表示
    - リトライボタン
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [x] 6.2 ローディング状態の実装
    - チェックポイント読み込み中表示
    - 生成中インジケータ
    - _Requirements: 1.3, 2.3, 4.2_

- [x] 7. チェックポイント - 動作確認
  - サーバー起動とUI表示確認
  - チェックポイント切り替え確認
  - テキスト生成確認
  - Ensure all features work, ask the user if questions arise.

## Notes

- 既存の`phase1/visualizer/`ディレクトリに追加
- `python -m visualizer.playground_server`で起動
- http://localhost:8081 でアクセス（既存visualizerと別ポート）
