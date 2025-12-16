#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习训练脚本

使用 Gumbel MCTS 自我对弈 + PPO 风格的策略优化。
训练流程：
1. 自我对弈生成训练数据
2. 使用生成的数据更新神经网络
3. 重复上述过程

Author: JCAR Team
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from .model import JiuqiNet, create_model
from .config import JiuqiNetConfig
from .gumbel_mcts import MCTSConfig
from .self_play import SelfPlayManager, ReplayBuffer


class RLTrainer:
    """强化学习训练器"""

    def __init__(self, model: JiuqiNet, config: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('lr_restart_period', 10),
            T_mult=2
        )

        # 混合精度
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # MCTS 配置 (默认 64 次模拟)
        mcts_config = MCTSConfig(
            num_simulations=config.get('mcts_simulations', 64),
            num_sampled_actions=config.get('mcts_sampled_actions', 16),
            c_puct=config.get('mcts_c_puct', 1.5),
            root_dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
            root_noise_frac=config.get('noise_frac', 0.25)
        )

        # 自我对弈管理器
        self.self_play_manager = SelfPlayManager(
            model, mcts_config, device,
            buffer_capacity=config.get('buffer_capacity', 100000)
        )

        # 最大步数
        self.max_steps = config.get('max_steps_per_game', 1000)

        # 训练统计
        self.iteration = 0
        self.total_games = 0
        self.total_steps = 0

        # 保存目录
        self.out_dir = Path(config.get('out_dir', 'exp/jcar_rl'))
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, num_iterations: int = 100):
        """主训练循环"""
        # 检测是否在终端中运行（而不是重定向到文件）
        import sys
        is_tty = sys.stdout.isatty()

        # 只在终端中使用 tqdm
        has_tqdm = False
        if is_tty:
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                pass

        print("=" * 60, flush=True)
        print("🚀 JCAR 强化学习训练 (Gumbel MCTS)", flush=True)
        print("=" * 60, flush=True)
        print(f"  迭代次数: {num_iterations}", flush=True)
        print(f"  每次自我对弈: {self.config.get('games_per_iteration', 10)} 局", flush=True)
        print(f"  MCTS 模拟次数: {self.config.get('mcts_simulations', 64)}", flush=True)
        print(f"  批大小: {self.config.get('batch_size', 256)}", flush=True)
        print(f"  输出目录: {self.out_dir}", flush=True)
        print("=" * 60, flush=True)

        total_start = time.time()

        # 主迭代循环 - 从当前迭代继续
        start_iter = self.iteration  # 恢复训练时从上次迭代继续
        remaining_iters = num_iterations - start_iter

        if start_iter > 0:
            print(f"📌 从迭代 {start_iter + 1} 继续训练...", flush=True)

        iter_range = range(remaining_iters)
        if has_tqdm:
            iter_pbar = tqdm(iter_range, desc="RL Training", unit="iter",
                            position=0, leave=True, ncols=100)
        else:
            iter_pbar = iter_range

        for i in iter_pbar:
            self.iteration = start_iter + i + 1
            iter_start = time.time()

            if has_tqdm:
                iter_pbar.set_description(f"Iter {self.iteration}/{num_iterations}")
            else:
                print(f"\n{'='*60}", flush=True)
                print(f"📍 迭代 {self.iteration}/{num_iterations}", flush=True)
                print(f"{'='*60}", flush=True)

            # 1. 自我对弈
            self.model.eval()
            games_per_iter = self.config.get('games_per_iteration', 10)
            num_workers = self.config.get('num_workers', 4)
            use_parallel = self.config.get('use_parallel', True)

            if use_parallel and num_workers > 1:
                print(f"\n[1/3] 🎮 自我对弈 {games_per_iter} 局 (并行 {num_workers} workers)...", flush=True)
            else:
                print(f"\n[1/3] 🎮 自我对弈 {games_per_iter} 局...", flush=True)

            sp_start = time.time()

            # 选择串行或并行模式
            if use_parallel and num_workers > 1:
                records = self.self_play_manager.generate_games_parallel(
                    games_per_iter,
                    max_steps=self.max_steps,
                    num_workers=num_workers,
                    verbose=True
                )
            else:
                records = self.self_play_manager.generate_games(
                    games_per_iter,
                    max_steps=self.max_steps,
                    verbose=True,
                    use_tqdm=has_tqdm
                )
            sp_time = time.time() - sp_start

            self.total_games += len(records)

            # 统计
            wins = {'white': 0, 'black': 0, 'draw': 0}
            total_steps = 0
            for r in records:
                total_steps += r.total_steps
                if r.winner is None:
                    wins['draw'] += 1
                elif r.winner.name == 'white':
                    wins['white'] += 1
                else:
                    wins['black'] += 1

            avg_steps = total_steps / max(1, len(records))
            stats = self.self_play_manager.get_stats()

            print(f"   ✅ 完成: W={wins['white']} B={wins['black']} D={wins['draw']}", flush=True)
            print(f"   📊 平均步数: {avg_steps:.0f}, Buffer: {stats['buffer_size']} 样本", flush=True)
            print(f"   ⏱️ 耗时: {sp_time:.1f}s", flush=True)

            # 2. 训练
            self.model.train()
            epochs = self.config.get('train_epochs', 3)
            batch_size = self.config.get('batch_size', 256)

            print(f"\n[2/3] 🧠 训练 {epochs} epochs...", flush=True)

            train_start = time.time()
            if len(self.self_play_manager.buffer) >= batch_size:
                train_losses = self._train_epoch(epochs, batch_size, use_tqdm=has_tqdm)
                train_time = time.time() - train_start

                print(f"   ✅ Loss: policy={train_losses['policy']:.4f}, "
                      f"value={train_losses['value']:.4f}, "
                      f"total={train_losses['total']:.4f}", flush=True)
                print(f"   ⏱️ 耗时: {train_time:.1f}s", flush=True)
            else:
                train_losses = {'policy': 0, 'value': 0, 'total': 0}
                print("   ⚠️ Buffer 不足，跳过训练", flush=True)

            # 3. 保存检查点
            print(f"\n[3/3] 💾 保存检查点...", flush=True)
            self._save_checkpoint()

            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            completed_iters = self.iteration - start_iter
            if completed_iters > 0:
                eta = (total_time / completed_iters) * (num_iterations - self.iteration)
            else:
                eta = 0

            if has_tqdm:
                iter_pbar.set_postfix({
                    'W/B/D': f"{wins['white']}/{wins['black']}/{wins['draw']}",
                    'loss': f"{train_losses['total']:.3f}",
                    'buf': stats['buffer_size'],
                    'ETA': f"{eta/60:.0f}m"
                })

            print(f"\n   ✅ 迭代完成: {iter_time:.1f}s", flush=True)
            print(f"   🕐 总耗时: {total_time/60:.1f}m, ETA: {eta/60:.1f}m", flush=True)

            # 学习率调度
            self.scheduler.step()

        if has_tqdm:
            iter_pbar.close()

        print(f"\n{'='*60}", flush=True)
        print(f"🎉 训练完成！总耗时: {(time.time()-total_start)/60:.1f} 分钟", flush=True)
        print(f"   总对弈: {self.total_games} 局", flush=True)
        print(f"   总样本: {stats['total_samples']}", flush=True)
        print(f"{'='*60}", flush=True)
    
    def _train_epoch(self, epochs: int, batch_size: int, use_tqdm: bool = False) -> dict:
        """训练一个 epoch"""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        steps_per_epoch = self.config.get('steps_per_epoch', 100)
        total_steps = epochs * steps_per_epoch

        # 尝试使用 tqdm
        if use_tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_steps, desc="Training", unit="step",
                           position=1, leave=False, ncols=100)
            except ImportError:
                pbar = None
        else:
            pbar = None

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                batch = self.self_play_manager.get_batch(batch_size)
                losses = self._train_step(batch)

                total_policy_loss += losses['policy']
                total_value_loss += losses['value']
                total_loss += losses['total']
                num_batches += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{losses['total']:.4f}",
                        'p_loss': f"{losses['policy']:.4f}",
                        'v_loss': f"{losses['value']:.4f}"
                    })

        if pbar:
            pbar.close()

        return {
            'policy': total_policy_loss / max(num_batches, 1),
            'value': total_value_loss / max(num_batches, 1),
            'total': total_loss / max(num_batches, 1)
        }

    def _train_step(self, batch: dict) -> dict:
        """单步训练"""
        obs = batch['obs'].to(self.device)
        phase_ids = batch['phase_ids'].to(self.device)
        values_target = batch['values'].to(self.device)
        cand_feats_list = [c.to(self.device) for c in batch['cand_feats_list']]
        policies_target = batch['policies']

        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss, policy_loss, value_loss = self._compute_loss(
                    obs, phase_ids, cand_feats_list, policies_target, values_target
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, policy_loss, value_loss = self._compute_loss(
                obs, phase_ids, cand_feats_list, policies_target, values_target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'total': loss.item()
        }

    def _compute_loss(self, obs, phase_ids, cand_feats_list, policies_target, values_target):
        """计算损失"""
        logits_list, values_pred = self.model.score_candidates(obs, phase_ids, cand_feats_list)

        # Policy loss: KL 散度
        policy_loss = 0.0
        for logits, target in zip(logits_list, policies_target):
            if logits.numel() == 0:
                continue
            target = target.to(self.device)
            log_probs = F.log_softmax(logits, dim=0)
            # 使用交叉熵而非 KL（更稳定）
            policy_loss += -(target * log_probs).sum()
        policy_loss = policy_loss / len(logits_list)

        # Value loss: MSE
        values_pred = values_pred.squeeze(-1)
        value_loss = F.mse_loss(values_pred, values_target)

        # 总损失
        value_weight = self.config.get('value_loss_weight', 0.5)
        total_loss = policy_loss + value_weight * value_loss

        return total_loss, policy_loss, value_loss

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_games': self.total_games,
            'config': self.config
        }

        # 保存最新
        path = self.out_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)

        # 每 10 次迭代保存一个
        if self.iteration % 10 == 0:
            path = self.out_dir / f'checkpoint_iter_{self.iteration:04d}.pt'
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']

        print(f"Loaded checkpoint from {path} (iteration {self.iteration})")


def get_default_config() -> dict:
    """默认 RL 训练配置"""
    return {
        # 训练参数
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'value_loss_weight': 0.5,
        'batch_size': 256,
        'train_epochs': 3,
        'steps_per_epoch': 100,
        'lr_restart_period': 10,
        'use_amp': True,

        # MCTS 参数 (Gumbel MCTS)
        'mcts_simulations': 64,         # 增加到 64 次，提高策略质量
        'mcts_sampled_actions': 16,
        'mcts_c_puct': 1.5,
        'dirichlet_alpha': 0.3,
        'noise_frac': 0.25,

        # 自我对弈
        'games_per_iteration': 10,
        'buffer_capacity': 100000,
        'max_steps_per_game': 1000,     # 每局最大步数

        # 并行化
        'use_parallel': True,           # 启用并行自我对弈
        'num_workers': 4,               # 并行进程数

        # 输出
        'out_dir': 'exp/jcar_rl'
    }


def main():
    parser = argparse.ArgumentParser(description='JCAR 强化学习训练 (Gumbel MCTS)')
    parser.add_argument('--checkpoint', type=str, default=None, help='SFT 检查点路径')
    parser.add_argument('--resume', type=str, default=None, help='RL 检查点路径（恢复训练）')
    parser.add_argument('--iterations', type=int, default=100, help='训练迭代次数')
    parser.add_argument('--games', type=int, default=10, help='每次迭代自我对弈局数')
    parser.add_argument('--mcts-sims', type=int, default=64, help='MCTS 模拟次数 (默认64)')
    parser.add_argument('--max-steps', type=int, default=1000, help='每局最大步数')
    parser.add_argument('--num-workers', type=int, default=4, help='并行进程数')
    parser.add_argument('--no-parallel', action='store_true', help='禁用并行模式')
    parser.add_argument('--out-dir', type=str, default='exp/jcar_rl', help='输出目录')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 配置
    config = get_default_config()
    config['games_per_iteration'] = args.games
    config['mcts_simulations'] = args.mcts_sims
    config['max_steps_per_game'] = args.max_steps
    config['out_dir'] = args.out_dir
    config['num_workers'] = args.num_workers
    config['use_parallel'] = not args.no_parallel

    # 创建模型
    model = create_model()

    # 加载 SFT 检查点
    if args.checkpoint:
        print(f"Loading SFT checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # 创建训练器
    trainer = RLTrainer(model, config, device)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train(num_iterations=args.iterations)


if __name__ == '__main__':
    main()
