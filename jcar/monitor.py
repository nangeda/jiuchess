#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet 训练监控器

基于 Python 内置 http.server 的 Web 监控系统（无需 Flask），提供：
- 实时训练指标可视化（损失、准确率、学习率）
- 系统资源监控（GPU、CPU、内存）
- 日志查看（tail -f 风格）
- 检查点管理

Author: JiuqiNet Team
"""

import argparse
import json
import subprocess
import urllib.parse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>JiuqiNet Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: #16213e; padding: 20px; text-align: center; }
        .header h1 { color: #00d9ff; }
        .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; }
        .card h2 { color: #00d9ff; margin-bottom: 15px; font-size: 1.2em; }
        .chart-container { height: 250px; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .stat { background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 1.8em; color: #00d9ff; font-weight: bold; }
        .stat-label { font-size: 0.9em; color: #888; }
        .log-container { background: #0a0a15; border-radius: 8px; padding: 15px; height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        .log-line { padding: 2px 0; border-bottom: 1px solid #222; }
        .log-line.error { color: #ff6b6b; }
        .log-line.warning { color: #ffd93d; }
        .checkpoint-list { max-height: 200px; overflow-y: auto; }
        .checkpoint-item { display: flex; justify-content: space-between; padding: 8px; background: #0f3460; margin: 5px 0; border-radius: 5px; }
        .refresh-info { text-align: center; color: #666; padding: 10px; }
    </style>
</head>
<body>
    <div class="header"><h1>JiuqiNet Training Monitor</h1><p id="update-time">--</p></div>
    <div class="container">
        <div class="card"><h2>Progress</h2><div class="stats">
            <div class="stat"><div class="stat-value" id="epoch">--</div><div class="stat-label">Epoch</div></div>
            <div class="stat"><div class="stat-value" id="loss">--</div><div class="stat-label">Loss</div></div>
            <div class="stat"><div class="stat-value" id="acc">--</div><div class="stat-label">Accuracy</div></div>
        </div></div>
        <div class="card"><h2>Loss</h2><div class="chart-container"><canvas id="lossChart"></canvas></div></div>
        <div class="card"><h2>Accuracy</h2><div class="chart-container"><canvas id="accChart"></canvas></div></div>
        <div class="card"><h2>LR</h2><div class="chart-container"><canvas id="lrChart"></canvas></div></div>
        <div class="card"><h2>System</h2><div class="stats">
            <div class="stat"><div class="stat-value" id="gpu-mem">--</div><div class="stat-label">GPU Mem</div></div>
            <div class="stat"><div class="stat-value" id="gpu-util">--</div><div class="stat-label">GPU Util</div></div>
            <div class="stat"><div class="stat-value" id="cpu">--</div><div class="stat-label">CPU</div></div>
        </div></div>
        <div class="card"><h2>Log</h2><div class="log-container" id="log-container"></div></div>
        <div class="card"><h2>Checkpoints</h2><div class="checkpoint-list" id="checkpoint-list"></div></div>
        <div class="card"><h2>Config</h2><pre id="config" style="font-size:11px;overflow:auto;max-height:200px;"></pre></div>
    </div>
    <div class="refresh-info">Auto-refresh every {{ refresh_interval }}s</div>
    <script>
        const R={{ refresh_interval }}*1000; let lC,aC,lrC;
        function init(){const c=(l,cl)=>({type:'line',data:{labels:[],datasets:[{label:l,data:[],borderColor:cl,tension:0.1,fill:false}]},options:{responsive:true,maintainAspectRatio:false}});
        lC=new Chart(document.getElementById('lossChart'),c('Loss','#ff6b6b'));aC=new Chart(document.getElementById('accChart'),c('Acc','#4ecdc4'));lrC=new Chart(document.getElementById('lrChart'),c('LR','#ffd93d'));}
        async function upd(){try{const[m,lg,ck,cf,sy]=await Promise.all([fetch('/api/metrics').then(r=>r.json()),fetch('/api/logs').then(r=>r.json()),fetch('/api/checkpoints').then(r=>r.json()),fetch('/api/config').then(r=>r.json()),fetch('/api/system').then(r=>r.json())]);
        if(m.train_loss&&m.train_loss.length){document.getElementById('epoch').textContent=m.train_loss.length;document.getElementById('loss').textContent=m.train_loss.slice(-1)[0].toFixed(4);document.getElementById('acc').textContent=(m.train_acc.slice(-1)[0]*100).toFixed(1)+'%';
        const lb=Array.from({length:m.train_loss.length},(_,i)=>i+1);lC.data.labels=lb;lC.data.datasets[0].data=m.train_loss;lC.update('none');aC.data.labels=lb;aC.data.datasets[0].data=m.train_acc.map(x=>x*100);aC.update('none');lrC.data.labels=lb;lrC.data.datasets[0].data=m.lr;lrC.update('none');}
        const lc=document.getElementById('log-container');lc.innerHTML=lg.lines.map(l=>'<div class="log-line '+(l.includes('ERROR')?'error':l.includes('WARNING')?'warning':'')+'">'+l+'</div>').join('');lc.scrollTop=lc.scrollHeight;
        document.getElementById('checkpoint-list').innerHTML=ck.files.map(f=>'<div class="checkpoint-item"><span>'+f.name+'</span><span>'+f.size+'</span></div>').join('');
        document.getElementById('config').textContent=JSON.stringify(cf,null,2);document.getElementById('gpu-mem').textContent=sy.gpu_memory||'--';document.getElementById('gpu-util').textContent=sy.gpu_util||'--';document.getElementById('cpu').textContent=sy.cpu||'--';
        document.getElementById('update-time').textContent='Updated: '+new Date().toLocaleTimeString();}catch(e){console.error(e);}}
        init();upd();setInterval(upd,R);
    </script>
</body></html>'''


class MonitorHandler(SimpleHTTPRequestHandler):
    """HTTP 请求处理器"""

    def __init__(self, *args, log_dir: Path, refresh_interval: int, max_log_lines: int, **kwargs):
        self.log_dir = log_dir
        self.refresh_interval = refresh_interval
        self.max_log_lines = max_log_lines
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """静默日志"""
        pass

    def send_json(self, data: dict):
        """发送 JSON 响应"""
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def send_html(self, html: str):
        """发送 HTML 响应"""
        content = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        """处理 GET 请求"""
        path = urllib.parse.urlparse(self.path).path

        if path == '/':
            html = HTML_TEMPLATE.replace('{{ refresh_interval }}', str(self.refresh_interval))
            self.send_html(html)
        elif path == '/api/metrics':
            self.handle_metrics()
        elif path == '/api/logs':
            self.handle_logs()
        elif path == '/api/checkpoints':
            self.handle_checkpoints()
        elif path == '/api/config':
            self.handle_config()
        elif path == '/api/system':
            self.handle_system()
        elif path.startswith('/download/'):
            self.handle_download(path[10:])
        else:
            self.send_error(404)

    def handle_metrics(self):
        history_path = self.log_dir / 'history.json'
        if history_path.exists():
            with open(history_path) as f:
                self.send_json(json.load(f))
        else:
            self.send_json({})

    def handle_logs(self):
        log_path = self.log_dir / 'train.log'
        lines = []
        if log_path.exists():
            with open(log_path) as f:
                lines = [l.strip() for l in f.readlines()[-self.max_log_lines:]]
        self.send_json({'lines': lines})

    def handle_checkpoints(self):
        files = []
        for f in sorted(self.log_dir.glob('*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            size = f.stat().st_size
            size_str = f"{size/1e6:.1f}MB" if size > 1e6 else f"{size/1e3:.1f}KB"
            files.append({'name': f.name, 'size': size_str})
        self.send_json({'files': files})

    def handle_config(self):
        config_path = self.log_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                self.send_json(json.load(f))
        else:
            self.send_json({})

    def handle_system(self):
        result = {'gpu_memory': '--', 'gpu_util': '--', 'cpu': '--'}
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'], timeout=5
            ).decode().strip().split('\n')[0].split(',')
            if len(output) >= 3:
                mem_used, mem_total, util = [x.strip() for x in output]
                result['gpu_memory'] = f"{mem_used}/{mem_total}MB"
                result['gpu_util'] = f"{util}%"
        except Exception:
            pass
        try:
            with open('/proc/loadavg') as f:
                result['cpu'] = f"{float(f.read().split()[0]):.1f}"
        except Exception:
            pass
        self.send_json(result)

    def handle_download(self, filename: str):
        path = self.log_dir / filename
        if path.exists() and path.suffix == '.pt':
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.send_header('Content-Length', path.stat().st_size)
            self.end_headers()
            with open(path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)


class TrainingMonitor:
    """训练监控器（无需 Flask，使用 Python 内置 http.server）"""

    def __init__(self, log_dir: str, host: str = '0.0.0.0', port: int = 8889,
                 refresh_interval: int = 5, max_log_lines: int = 1000):
        self.log_dir = Path(log_dir)
        self.host = host
        self.port = port
        self.refresh_interval = refresh_interval
        self.max_log_lines = max_log_lines

    def run(self):
        """启动监控服务器"""
        print("=" * 60)
        print("JiuqiNet Training Monitor (No Flask Required)")
        print("=" * 60)
        print(f"Log directory: {self.log_dir}")
        print(f"Server: http://{self.host}:{self.port}")
        print(f"Refresh interval: {self.refresh_interval}s")
        print("=" * 60)

        handler = partial(
            MonitorHandler,
            log_dir=self.log_dir,
            refresh_interval=self.refresh_interval,
            max_log_lines=self.max_log_lines
        )
        server = HTTPServer((self.host, self.port), handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='JiuqiNet Training Monitor')
    parser.add_argument('--log-dir', type=str, default='exp/jcar', help='Training log directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8889, help='Server port')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh interval in seconds')
    parser.add_argument('--max-lines', type=int, default=1000, help='Maximum log lines to display')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    monitor = TrainingMonitor(
        log_dir=args.log_dir, host=args.host, port=args.port,
        refresh_interval=args.refresh, max_log_lines=args.max_lines
    )
    monitor.run()


if __name__ == '__main__':
    main()
