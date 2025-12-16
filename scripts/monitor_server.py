#!/usr/bin/env python3
"""简单的训练监控网页服务器"""
import os
import json
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import argparse

LOG_FILE = "data/checkpoints/moe_dyt/train_stable.log"
JSON_LOG = "data/checkpoints/moe_dyt/train_log.json"

def get_html_template():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>MoE-DyT Training Monitor</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="10">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ color: #aaa; font-size: 14px; }}
        pre {{ background: #0a0a15; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 13px; max-height: 400px; overflow-y: auto; }}
        .gpu {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .gpu-card {{ flex: 1; min-width: 200px; }}
        .progress {{ background: #0f3460; border-radius: 5px; height: 20px; overflow: hidden; }}
        .progress-bar {{ background: linear-gradient(90deg, #00d4ff, #00ff88); height: 100%%; transition: width 0.3s; }}
        .status-running {{ color: #00ff88; }}
        .status-stopped {{ color: #ff4444; }}
        .refresh-info {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MoE-DyT Training Monitor</h1>
        <p class="refresh-info">Auto-refresh every 10 seconds | Last update: {timestamp}</p>

        <div class="card">
            <h2>📊 Training Status: <span class="{status_class}">{status}</span></h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{epoch}</div>
                    <div class="stat-label">Epoch</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{step}</div>
                    <div class="stat-label">Global Step</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{loss}</div>
                    <div class="stat-label">Train Loss</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{accuracy}</div>
                    <div class="stat-label">Train Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{val_loss}</div>
                    <div class="stat-label">Val Loss</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{lr}</div>
                    <div class="stat-label">Learning Rate</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎮 GPU Status</h2>
            <div class="gpu">{gpu_info}</div>
        </div>

        <div class="card">
            <h2>📜 Recent Logs (last 50 lines)</h2>
            <pre>{log_tail}</pre>
        </div>
    </div>
</body>
</html>'''

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    idx, name, mem_used, mem_total, util = parts[:5]
                    mem_pct = int(mem_used) / int(mem_total) * 100 if int(mem_total) > 0 else 0
                    gpus.append(f'''<div class="gpu-card stat-box">
                        <div class="stat-value">GPU {idx}</div>
                        <div class="stat-label">{name}</div>
                        <div style="margin:10px 0">Memory: {mem_used}/{mem_total} MiB</div>
                        <div class="progress"><div class="progress-bar" style="width:{mem_pct:.1f}%"></div></div>
                        <div style="margin-top:5px">Utilization: {util}%</div>
                    </div>''')
        return ''.join(gpus) if gpus else '<div class="stat-box">No GPU info available</div>'
    except:
        return '<div class="stat-box">Error getting GPU info</div>'

def get_log_tail(n=50):
    try:
        if os.path.exists(LOG_FILE):
            result = subprocess.run(['tail', '-n', str(n), LOG_FILE], capture_output=True, text=True, timeout=5)
            return result.stdout.replace('<', '&lt;').replace('>', '&gt;')
    except: pass
    return "Log file not found"

def get_training_stats():
    stats = {'epoch': '-', 'step': '-', 'loss': '-', 'accuracy': '-', 'val_loss': '-', 'lr': '-'}
    try:
        if os.path.exists(JSON_LOG):
            with open(JSON_LOG) as f:
                data = json.load(f)
                if data:
                    last = data[-1]
                    stats['epoch'] = f"{last.get('epoch', '-')}"
                    stats['step'] = f"{last.get('train', {}).get('global_step', '-'):,}"
                    stats['loss'] = f"{last.get('train', {}).get('loss', 0):.4f}"
                    stats['accuracy'] = f"{last.get('train', {}).get('accuracy', 0)*100:.2f}%"
                    stats['val_loss'] = f"{last.get('val', {}).get('loss', 0):.4f}"
                    stats['lr'] = f"{last.get('lr', 0):.2e}"
    except: pass
    return stats

def is_training_running():
    try:
        pid_file = "data/checkpoints/moe_dyt/train_stable.pid"
        if os.path.exists(pid_file):
            with open(pid_file) as f:
                pid = f.read().strip()
            result = subprocess.run(['ps', '-p', pid], capture_output=True, timeout=5)
            return result.returncode == 0
    except: pass
    return False

class MonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            stats = get_training_stats()
            running = is_training_running()
            html = get_html_template().format(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                status='RUNNING' if running else 'STOPPED',
                status_class='status-running' if running else 'status-stopped',
                gpu_info=get_gpu_info(),
                log_tail=get_log_tail(),
                **stats
            )
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    
    server = HTTPServer(('0.0.0.0', args.port), MonitorHandler)
    print(f"Monitor server started at http://0.0.0.0:{args.port}")
    print(f"Press Ctrl+C to stop")
    server.serve_forever()

