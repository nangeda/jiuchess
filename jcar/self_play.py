#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我对弈模块

使用 Gumbel MCTS 进行自我对弈，生成强化学习训练数据。
支持多进程并行对弈以加速数据收集。

Author: JCAR Team
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random
import time
import pickle
import queue

from jiu.jiuboard_fast import GameState, Move, Board
from jiu.jiutypes import Player, Point, board_size, board_gild
from jiu.scoring import compute_game_result

from .gumbel_mcts import GumbelMCTS, MCTSConfig, create_mcts

# 设置多进程启动方式
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


@dataclass
class GameSample:
    """单个训练样本"""
    obs: np.ndarray              # (6, 14, 14) 棋盘状态
    phase_id: int                # 阶段 ID
    cand_feats: np.ndarray       # (K, 14) 候选特征
    policy: np.ndarray           # (K,) MCTS 改进后的策略
    value: float                 # 游戏结果 (-1, 0, 1)
    action_idx: int              # 选择的动作索引


@dataclass
class GameRecord:
    """完整对局记录"""
    samples: List[GameSample] = field(default_factory=list)
    winner: Optional[Player] = None
    total_steps: int = 0
    white_stones: int = 0
    black_stones: int = 0


class SelfPlayWorker:
    """自我对弈工作器"""
    
    def __init__(self, model, mcts_config: MCTSConfig = None, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.mcts = GumbelMCTS(model, mcts_config or MCTSConfig(), device)
    
    def play_game(self, temperature_schedule: Dict[int, float] = None,
                  max_steps: int = 1000, verbose: bool = False) -> GameRecord:
        """
        进行一局自我对弈
        
        Args:
            temperature_schedule: 步数 -> 温度的映射，默认前30步温度=1，之后=0
            max_steps: 最大步数
            verbose: 是否打印详情
            
        Returns:
            完整对局记录
        """
        if temperature_schedule is None:
            temperature_schedule = {0: 1.0, 30: 0.1, 100: 0.0}
        
        state = GameState.new_game(board_size)
        record = GameRecord()
        samples_white = []  # 白方样本
        samples_black = []  # 黑方样本
        
        step = 0
        while step < max_steps:
            legal_moves = state.legal_moves()
            if not legal_moves:
                break
            
            # 获取当前温度
            temp = self._get_temperature(step, temperature_schedule)
            
            # MCTS 搜索
            policy, best_decision = self.mcts.search(state)
            
            if best_decision is None:
                break
            
            # 根据温度采样动作
            if temp > 0 and len(policy) > 1:
                policy_temp = np.power(policy + 1e-8, 1.0 / temp)
                policy_temp = policy_temp / policy_temp.sum()
                # 确保精确归一化
                policy_temp = np.clip(policy_temp, 0, 1)
                policy_temp = policy_temp / policy_temp.sum()
                action_idx = np.random.choice(len(legal_moves), p=policy_temp)
            else:
                action_idx = np.argmax(policy)
            
            selected_decision = legal_moves[action_idx]
            
            # 记录样本
            sample = self._create_sample(state, legal_moves, policy, action_idx)
            if state.next_player == Player.white:
                samples_white.append(sample)
            else:
                samples_black.append(sample)
            
            # 执行动作
            move = self._decision_to_move(selected_decision)
            state = state.apply_move(move)
            step += 1
            
            if verbose and step % 50 == 0:
                result = compute_game_result(state)
                print(f"  Step {step}: White={result.w}, Black={result.b}")
        
        # 计算最终结果
        result = compute_game_result(state)
        record.total_steps = step
        record.white_stones = result.w
        record.black_stones = result.b
        
        # 确定胜者和价值
        if result.w > result.b:
            record.winner = Player.white
            white_value, black_value = 1.0, -1.0
        elif result.b > result.w:
            record.winner = Player.black
            white_value, black_value = -1.0, 1.0
        else:
            white_value, black_value = 0.0, 0.0
        
        # 设置样本价值
        for s in samples_white:
            s.value = white_value
        for s in samples_black:
            s.value = black_value
        
        record.samples = samples_white + samples_black
        random.shuffle(record.samples)  # 打乱顺序
        
        return record
    
    def _get_temperature(self, step: int, schedule: Dict[int, float]) -> float:
        """根据步数获取温度"""
        temp = 1.0
        for threshold, t in sorted(schedule.items()):
            if step >= threshold:
                temp = t
        return temp

    def _create_sample(self, state: GameState, legal_moves, policy: np.ndarray,
                       action_idx: int) -> GameSample:
        """创建训练样本"""
        # 编码状态
        obs = self._encode_state(state)
        phase_id = self._get_phase_id(state)
        cand_feats = self.mcts._build_candidate_features(legal_moves, phase_id, state)

        return GameSample(
            obs=obs,
            phase_id=phase_id,
            cand_feats=cand_feats,
            policy=policy,
            value=0.0,  # 稍后填充
            action_idx=action_idx
        )

    def _encode_state(self, state: GameState) -> np.ndarray:
        """编码棋盘状态"""
        return self.mcts._encode_state(state)

    def _get_phase_id(self, state: GameState) -> int:
        """获取阶段 ID"""
        return self.mcts._get_phase_id(state)

    def _decision_to_move(self, dec) -> Move:
        """Decision 转 Move"""
        return self.mcts._decision_to_move(dec)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, record: GameRecord):
        """添加一局游戏的所有样本"""
        for sample in record.samples:
            self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[GameSample]:
        """随机采样"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        """保存到文件"""
        data = []
        for s in self.buffer:
            data.append({
                'obs': s.obs,
                'phase_id': s.phase_id,
                'cand_feats': s.cand_feats,
                'policy': s.policy,
                'value': s.value,
                'action_idx': s.action_idx
            })
        torch.save(data, path)
        print(f"Saved {len(data)} samples to {path}")

    def load(self, path: str):
        """从文件加载"""
        data = torch.load(path)
        for d in data:
            sample = GameSample(
                obs=d['obs'],
                phase_id=d['phase_id'],
                cand_feats=d['cand_feats'],
                policy=d['policy'],
                value=d['value'],
                action_idx=d['action_idx']
            )
            self.buffer.append(sample)
        print(f"Loaded {len(data)} samples from {path}")


def _play_single_game_worker(args):
    """单局对弈工作函数（用于多进程）

    注意：每个子进程独立加载模型到 CPU，避免 GPU 共享问题。
    虽然 CPU 推理较慢，但多进程并行可以弥补。
    """
    model_state_dict, mcts_config_dict, device_hint, max_steps, game_id, temperature_schedule = args

    # 在子进程中重建模型 - 使用 CPU 避免多进程 GPU 共享问题
    # 也可以根据 game_id 分配不同 GPU
    import os

    # 禁用 CUDA 以避免多进程共享问题
    worker_device = 'cpu'  # 子进程使用 CPU

    from .model import create_model
    model = create_model()
    model.load_state_dict(model_state_dict)
    model = model.to(worker_device)
    model.eval()

    # 重建 MCTS 配置
    mcts_config = MCTSConfig(**mcts_config_dict)

    # 创建 worker 并对弈
    worker = SelfPlayWorker(model, mcts_config, worker_device)

    start_time = time.time()
    record = worker.play_game(
        temperature_schedule=temperature_schedule,
        max_steps=max_steps,
        verbose=False
    )
    elapsed = time.time() - start_time

    # 返回结果
    winner_str = "White" if record.winner == Player.white else \
                "Black" if record.winner == Player.black else "Draw"

    return {
        'game_id': game_id,
        'record': record,
        'elapsed': elapsed,
        'winner_str': winner_str,
        'steps': record.total_steps,
        'samples': len(record.samples)
    }


class SelfPlayManager:
    """自我对弈管理器（支持并行）"""

    def __init__(self, model, mcts_config: MCTSConfig = None, device: str = 'cuda',
                 buffer_capacity: int = 100000, num_workers: int = 1):
        self.model = model
        self.device = device
        self.mcts_config = mcts_config or MCTSConfig()
        self.num_workers = num_workers
        self.worker = SelfPlayWorker(model, self.mcts_config, device)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.games_played = 0
        self.total_samples = 0

    def generate_games(self, num_games: int, max_steps: int = 1000,
                       verbose: bool = True, use_tqdm: bool = False) -> List[GameRecord]:
        """生成多局自我对弈（串行模式，更稳定）"""
        records = []

        # 检测是否在终端中运行
        import sys
        is_tty = sys.stdout.isatty()

        # 只在终端中使用 tqdm
        if use_tqdm and is_tty:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
        else:
            use_tqdm = False

        iterator = range(num_games)
        if use_tqdm and verbose:
            iterator = tqdm(iterator, desc="Self-Play", unit="game", ncols=100)

        for i in iterator:
            start_time = time.time()
            record = self.worker.play_game(max_steps=max_steps, verbose=False)
            elapsed = time.time() - start_time

            self.buffer.add_game(record)
            records.append(record)
            self.games_played += 1
            self.total_samples += len(record.samples)

            winner_str = "White" if record.winner == Player.white else \
                        "Black" if record.winner == Player.black else "Draw"

            if use_tqdm and verbose:
                iterator.set_postfix({
                    'winner': winner_str,
                    'steps': record.total_steps,
                    'time': f'{elapsed:.1f}s'
                })
            elif verbose:
                print(f"  Game {i+1}/{num_games}: {winner_str}, "
                      f"{record.total_steps} steps, {len(record.samples)} samples, "
                      f"{elapsed:.1f}s", flush=True)

        return records

    def generate_games_parallel(self, num_games: int, max_steps: int = 1000,
                                num_workers: int = 4, verbose: bool = True) -> List[GameRecord]:
        """并行生成多局自我对弈（使用多进程）"""
        records = []

        # 准备参数
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        mcts_config_dict = {
            'num_simulations': self.mcts_config.num_simulations,
            'c_puct': self.mcts_config.c_puct,
            'root_dirichlet_alpha': self.mcts_config.root_dirichlet_alpha,
            'root_noise_frac': self.mcts_config.root_noise_frac,
            'num_sampled_actions': self.mcts_config.num_sampled_actions,
            'temperature': self.mcts_config.temperature,
            'value_scale': self.mcts_config.value_scale,
        }
        temperature_schedule = {0: 1.0, 30: 0.1, 100: 0.0}

        args_list = [
            (model_state_dict, mcts_config_dict, self.device, max_steps, i, temperature_schedule)
            for i in range(num_games)
        ]

        # 使用进程池
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        completed = 0
        if use_tqdm and verbose:
            pbar = tqdm(total=num_games, desc="Self-Play (Parallel)", unit="game", ncols=100)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_play_single_game_worker, args): args[4]
                      for args in args_list}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    record = result['record']

                    self.buffer.add_game(record)
                    records.append(record)
                    self.games_played += 1
                    self.total_samples += len(record.samples)
                    completed += 1

                    if use_tqdm and verbose:
                        pbar.update(1)
                        pbar.set_postfix({
                            'winner': result['winner_str'],
                            'steps': result['steps'],
                            'time': f"{result['elapsed']:.1f}s"
                        })
                    elif verbose:
                        print(f"  Game {completed}/{num_games}: {result['winner_str']}, "
                              f"{result['steps']} steps, {result['samples']} samples, "
                              f"{result['elapsed']:.1f}s")
                except Exception as e:
                    print(f"  Game failed: {e}")

        if use_tqdm and verbose:
            pbar.close()

        return records

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取训练批次"""
        samples = self.buffer.sample(batch_size)

        obs = torch.stack([torch.from_numpy(s.obs) for s in samples])
        phase_ids = torch.tensor([s.phase_id for s in samples], dtype=torch.long)
        values = torch.tensor([s.value for s in samples], dtype=torch.float32)

        cand_feats_list = [torch.from_numpy(s.cand_feats) for s in samples]
        policies = [torch.from_numpy(s.policy) for s in samples]

        return {
            'obs': obs,
            'phase_ids': phase_ids,
            'values': values,
            'cand_feats_list': cand_feats_list,
            'policies': policies
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'games_played': self.games_played,
            'total_samples': self.total_samples,
            'buffer_size': len(self.buffer),
            'avg_samples_per_game': self.total_samples / max(1, self.games_played)
        }

