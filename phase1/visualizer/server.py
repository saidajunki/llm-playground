"""
学習状況をリアルタイムでWebSocketで配信するサーバー

使用方法:
    python -m visualizer.server
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import torch
from aiohttp import web
import aiohttp

# グローバル変数
connected_clients: List[web.WebSocketResponse] = []
training_state = {
    'step': 0,
    'loss': 0.0,
    'lr': 0.0,
    'current_text': '',
    'weights': {},
    'is_training': False,
}


async def websocket_handler(request):
    """WebSocket接続ハンドラ"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    connected_clients.append(ws)
    print(f"クライアント接続: {len(connected_clients)}人")
    
    try:
        # 初期状態を送信
        await ws.send_json({
            'type': 'init',
            'data': training_state
        })
        
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get('type') == 'ping':
                    await ws.send_json({'type': 'pong'})
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f'WebSocketエラー: {ws.exception()}')
    finally:
        connected_clients.remove(ws)
        print(f"クライアント切断: {len(connected_clients)}人")
    
    return ws


async def broadcast(message: dict):
    """全クライアントにメッセージを送信"""
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"送信エラー: {e}")


async def index_handler(request):
    """HTMLページを返す"""
    html_path = Path(__file__).parent / 'index.html'
    return web.FileResponse(html_path)


async def static_handler(request):
    """静的ファイルを返す"""
    filename = request.match_info['filename']
    file_path = Path(__file__).parent / filename
    if file_path.exists():
        return web.FileResponse(file_path)
    return web.Response(status=404)


def extract_weight_samples(state_dict: dict, sample_size: int = 100) -> dict:
    """重みからサンプルを抽出"""
    samples = {}
    
    key_layers = [
        'token_embedding.embedding.weight',
        'blocks.0.attention.q_proj.weight',
        'blocks.0.ff.net.0.weight',
        'blocks.2.attention.q_proj.weight',
        'blocks.5.attention.q_proj.weight',
        'output.weight',
    ]
    
    for layer_name in key_layers:
        if layer_name in state_dict:
            tensor = state_dict[layer_name]
            flat = tensor.flatten()
            
            # ランダムサンプリング
            indices = torch.randperm(len(flat))[:sample_size]
            sample = flat[indices].tolist()
            
            samples[layer_name] = {
                'values': sample,
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
            }
    
    return samples


async def load_checkpoint_and_broadcast(checkpoint_path: str):
    """チェックポイントを読み込んでブロードキャスト"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        training_state['step'] = checkpoint.get('step', 0)
        training_state['loss'] = checkpoint.get('loss', 0.0)
        training_state['weights'] = extract_weight_samples(checkpoint['model_state_dict'])
        
        await broadcast({
            'type': 'checkpoint',
            'data': training_state
        })
    except Exception as e:
        print(f"チェックポイント読み込みエラー: {e}")


async def watch_training_log(log_path: str):
    """学習ログを監視してブロードキャスト"""
    import re
    
    log_path = Path(log_path)
    last_size = 0
    
    while True:
        try:
            if log_path.exists():
                current_size = log_path.stat().st_size
                
                if current_size > last_size:
                    with open(log_path, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                    
                    # ログからステップ情報を抽出
                    # Training:   0%|          | 100/297021 [01:28<69:38:12,  1.18it/s, loss=6.0986, lr=3.00e-04]
                    matches = re.findall(
                        r'(\d+)/\d+.*loss=([0-9.]+).*lr=([0-9.e-]+)',
                        new_content
                    )
                    
                    if matches:
                        step, loss, lr = matches[-1]
                        training_state['step'] = int(step)
                        training_state['loss'] = float(loss)
                        training_state['lr'] = float(lr)
                        training_state['is_training'] = True
                        
                        await broadcast({
                            'type': 'training_update',
                            'data': {
                                'step': training_state['step'],
                                'loss': training_state['loss'],
                                'lr': training_state['lr'],
                            }
                        })
                    
                    last_size = current_size
        except Exception as e:
            print(f"ログ監視エラー: {e}")
        
        await asyncio.sleep(1)


async def watch_checkpoints(checkpoint_dir: str):
    """チェックポイントディレクトリを監視"""
    checkpoint_dir = Path(checkpoint_dir)
    last_checkpoint = None
    
    while True:
        try:
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
            
            if checkpoints:
                latest = checkpoints[-1]
                
                if latest != last_checkpoint:
                    print(f"新しいチェックポイント検出: {latest}")
                    await load_checkpoint_and_broadcast(str(latest))
                    last_checkpoint = latest
        except Exception as e:
            print(f"チェックポイント監視エラー: {e}")
        
        await asyncio.sleep(5)


async def watch_training_data(data_path: str):
    """学習データの現在位置を監視（デモ用）"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 定期的にランダムな位置のテキストを送信
    while True:
        try:
            if training_state['is_training']:
                # ランダムな位置から128文字を取得
                start = random.randint(0, max(0, len(text) - 200))
                sample = text[start:start + 128]
                
                training_state['current_text'] = sample
                
                await broadcast({
                    'type': 'training_text',
                    'data': {
                        'text': sample,
                        'position': start,
                        'total_length': len(text),
                    }
                })
        except Exception as e:
            print(f"テキスト監視エラー: {e}")
        
        await asyncio.sleep(2)


async def start_background_tasks(app):
    """バックグラウンドタスクを開始"""
    app['log_watcher'] = asyncio.create_task(
        watch_training_log('training.log')
    )
    app['checkpoint_watcher'] = asyncio.create_task(
        watch_checkpoints('checkpoints')
    )
    app['data_watcher'] = asyncio.create_task(
        watch_training_data('data/wikipedia_ja_1gb.txt')
    )


async def cleanup_background_tasks(app):
    """バックグラウンドタスクをクリーンアップ"""
    for task_name in ['log_watcher', 'checkpoint_watcher', 'data_watcher']:
        if task_name in app:
            app[task_name].cancel()
            try:
                await app[task_name]
            except asyncio.CancelledError:
                pass


def create_app():
    """アプリケーションを作成"""
    app = web.Application()
    
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/{filename}', static_handler)
    
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    return app


def main():
    app = create_app()
    print("サーバー起動: http://localhost:8080")
    web.run_app(app, host='localhost', port=8080)


if __name__ == '__main__':
    main()
