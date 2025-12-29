"""
Playground Server for Mini Transformer.

ChatGPT風のPlayground UIを提供するWebサーバー。
チェックポイント管理、ストリーミング生成をサポート。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import torch
import torch.nn.functional as F
from aiohttp import web, WSMsgType

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import MiniGPT, ModelConfig
from src.tokenizer import CharTokenizer


@dataclass
class CheckpointInfo:
    """チェックポイント情報"""
    name: str
    step: int
    loss: float
    timestamp: str
    file_size: int


@dataclass
class ModelInfo:
    """モデル情報"""
    checkpoint_name: str
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    max_len: int
    total_params: int
    training_loss: float


class ModelManager:
    """モデルの読み込みと生成を管理"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.current_model: Optional[MiniGPT] = None
        self.current_checkpoint: Optional[str] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.model_info: Optional[ModelInfo] = None
        self.device = torch.device("cpu")  # 生成はCPUで実行
        
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """利用可能なチェックポイント一覧を取得"""
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint_step*.pt"):
            # ステップ番号を抽出
            match = re.search(r'checkpoint_step(\d+)\.pt', path.name)
            if not match:
                continue
                
            step = int(match.group(1))
            
            # ファイル情報を取得
            stat = path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # 損失値を取得（チェックポイントを読み込まずにメタデータから）
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                loss = checkpoint.get('loss', 0.0)
            except Exception:
                loss = 0.0
            
            checkpoints.append(CheckpointInfo(
                name=path.name,
                step=step,
                loss=round(loss, 4),
                timestamp=timestamp,
                file_size=stat.st_size
            ))
        
        # ステップ番号でソート
        checkpoints.sort(key=lambda x: x.step)
        return checkpoints
    
    def load_checkpoint(self, checkpoint_name: str) -> ModelInfo:
        """チェックポイントを読み込み"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        
        # チェックポイント読み込み
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config: ModelConfig = checkpoint['config']
        
        # モデル作成と重み読み込み
        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # トークナイザー読み込み
        tokenizer_path = self.checkpoint_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError("Tokenizer not found")
        
        tokenizer = CharTokenizer()
        tokenizer.load(tokenizer_path)
        
        # 状態を更新
        self.current_model = model
        self.current_checkpoint = checkpoint_name
        self.tokenizer = tokenizer
        
        # モデル情報を作成
        self.model_info = ModelInfo(
            checkpoint_name=checkpoint_name,
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_len=config.max_len,
            total_params=model.count_parameters(),
            training_loss=round(checkpoint.get('loss', 0.0), 4)
        )
        
        return self.model_info
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """現在のモデル情報を取得"""
        return self.model_info
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
        max_tokens: int = 100
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
        if self.current_model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")
        
        model = self.current_model
        tokenizer = self.tokenizer
        
        # プロンプトをトークン化
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 自己回帰生成
        with torch.no_grad():
            for _ in range(max_tokens):
                # コンテキスト長を超えないように切り詰め
                if input_ids.size(1) > model.config.max_len:
                    input_ids = input_ids[:, -model.config.max_len:]
                
                # Forward pass
                logits = model(input_ids)
                logits = logits[:, -1, :]  # 最後の位置
                
                # Temperature調整
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-kサンプリング
                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k_val, dim=-1)
                    threshold = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < threshold,
                        torch.full_like(logits, float('-inf')),
                        logits
                    )
                
                # サンプリング
                probs = F.softmax(logits, dim=-1)
                if temperature == 0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # トークンをデコード
                token_id = next_token.item()
                token_str = tokenizer.decode([token_id])
                
                # トークンをyield
                yield token_str
                
                # 入力に追加
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # EOSで終了
                if token_id == tokenizer.eos_id:
                    break
                
                # 少し待機（ストリーミング効果）
                await asyncio.sleep(0.02)



class PlaygroundServer:
    """Playground APIサーバー"""
    
    def __init__(self, checkpoint_dir: str, host: str = "localhost", port: int = 8081):
        self.checkpoint_dir = checkpoint_dir
        self.host = host
        self.port = port
        self.model_manager = ModelManager(checkpoint_dir)
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """ルートを設定"""
        self.app.router.add_get("/", self.index_handler)
        self.app.router.add_get("/api/checkpoints", self.get_checkpoints)
        self.app.router.add_post("/api/load-checkpoint", self.load_checkpoint)
        self.app.router.add_get("/api/model-info", self.get_model_info)
        self.app.router.add_get("/ws/generate", self.websocket_handler)
        
        # 静的ファイル配信
        static_dir = Path(__file__).parent
        self.app.router.add_static("/static/", static_dir)
    
    async def index_handler(self, request: web.Request) -> web.Response:
        """インデックスページを返す"""
        html_path = Path(__file__).parent / "playground.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="Playground HTML not found", status=404)
    
    async def get_checkpoints(self, request: web.Request) -> web.Response:
        """GET /api/checkpoints - チェックポイント一覧"""
        try:
            checkpoints = self.model_manager.list_checkpoints()
            data = [asdict(cp) for cp in checkpoints]
            return web.json_response({"checkpoints": data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def load_checkpoint(self, request: web.Request) -> web.Response:
        """POST /api/load-checkpoint - チェックポイント読み込み"""
        try:
            body = await request.json()
            checkpoint_name = body.get("checkpoint")
            
            if not checkpoint_name:
                return web.json_response(
                    {"error": "checkpoint name required"},
                    status=400
                )
            
            model_info = self.model_manager.load_checkpoint(checkpoint_name)
            return web.json_response({"model_info": asdict(model_info)})
            
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_model_info(self, request: web.Request) -> web.Response:
        """GET /api/model-info - モデル情報"""
        model_info = self.model_manager.get_model_info()
        
        if model_info is None:
            return web.json_response(
                {"error": "No model loaded"},
                status=400
            )
        
        return web.json_response({"model_info": asdict(model_info)})
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket /ws/generate - ストリーミング生成"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        if data.get("type") == "generate":
                            prompt = data.get("prompt", "")
                            temperature = float(data.get("temperature", 0.8))
                            top_k = int(data.get("top_k", 50))
                            max_tokens = int(data.get("max_tokens", 100))
                            
                            # パラメータをクランプ
                            temperature = max(0.0, min(2.0, temperature))
                            top_k = max(0, min(100, top_k))
                            max_tokens = max(1, min(500, max_tokens))
                            
                            # ストリーミング生成
                            full_text = prompt
                            async for token in self.model_manager.generate_stream(
                                prompt, temperature, top_k, max_tokens
                            ):
                                full_text += token
                                await ws.send_json({
                                    "type": "token",
                                    "token": token
                                })
                            
                            # 完了通知
                            await ws.send_json({
                                "type": "done",
                                "full_text": full_text
                            })
                            
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "error": str(e)
                        })
                        
                elif msg.type == WSMsgType.ERROR:
                    print(f"WebSocket error: {ws.exception()}")
                    
        except Exception as e:
            print(f"WebSocket handler error: {e}")
        
        return ws
    
    def run(self):
        """サーバーを起動"""
        print(f"Starting Playground server at http://{self.host}:{self.port}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        web.run_app(self.app, host=self.host, port=self.port)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Playground Server")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    
    server = PlaygroundServer(
        checkpoint_dir=args.checkpoint_dir,
        host=args.host,
        port=args.port
    )
    server.run()


if __name__ == "__main__":
    main()
