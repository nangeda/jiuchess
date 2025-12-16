#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gumbel MCTS (Gumbel AlphaZero)

基于 DeepMind 2022 论文 "Policy Improvement by Planning with Gumbel" 实现。
相比原版 MCTS，Gumbel MCTS 更加 sample-efficient，适合强化学习训练。

核心思想：
1. 使用 Gumbel-Top-k 采样替代 UCB 选择
2. 用更少的模拟次数达到更好的策略改进
3. 理论上保证策略单调改进

Author: JCAR Team
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from jiu.jiuboard_fast import GameState, Move
from jiu.jiutypes import Decision, Point, Go, Skip_eat, board_size, board_gild, Player


@dataclass
class MCTSConfig:
    """MCTS 配置"""
    num_simulations: int = 32          # 模拟次数（Gumbel MCTS 需要更少）
    c_puct: float = 1.5                # 探索常数
    root_dirichlet_alpha: float = 0.3  # 根节点 Dirichlet 噪声
    root_noise_frac: float = 0.25      # 根节点噪声比例
    num_sampled_actions: int = 16      # Gumbel Top-k 采样的 k 值
    temperature: float = 1.0           # 温度参数
    value_scale: float = 1.0           # 价值缩放


class MCTSNode:
    """MCTS 树节点"""
    __slots__ = ['state', 'player', 'parent', 'action_idx', 'decision',
                 'children', 'visit_count', 'value_sum', 'prior',
                 'is_expanded', 'gumbel']
    
    def __init__(self, state: GameState = None, parent: 'MCTSNode' = None,
                 action_idx: int = -1, decision: Decision = None, prior: float = 0.0):
        self.state = state
        self.player = state.next_player if state is not None else None
        self.parent = parent
        self.action_idx = action_idx
        self.decision = decision
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.gumbel = 0.0  # Gumbel 噪声
    
    @property
    def q_value(self) -> float:
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """UCB 分数"""
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


class GumbelMCTS:
    """
    Gumbel MCTS 搜索
    
    特点：
    1. 使用 Gumbel-Top-k 进行动作采样
    2. Sequential Halving 进行高效搜索
    3. 更少的模拟次数，更好的策略改进
    """
    
    def __init__(self, model, config: MCTSConfig = None, device: str = 'cuda'):
        self.model = model
        self.config = config or MCTSConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def search(self, state: GameState) -> Tuple[np.ndarray, Optional[Decision]]:
        """
        执行 Gumbel MCTS 搜索
        
        Args:
            state: 当前游戏状态
            
        Returns:
            policy: 改进后的策略分布
            best_decision: 最佳动作
        """
        legal_moves = state.legal_moves()
        if not legal_moves:
            return np.array([]), None
        
        num_actions = len(legal_moves)
        if num_actions == 1:
            return np.array([1.0]), legal_moves[0]
        
        # 创建根节点并评估
        root = MCTSNode(state)
        self._expand_node(root, legal_moves)
        
        # 添加 Dirichlet 噪声到根节点
        self._add_dirichlet_noise(root, num_actions)
        
        # Gumbel-Top-k 采样
        priors = np.array([root.children[i].prior for i in range(num_actions)])
        gumbels = self._sample_gumbel(num_actions)
        
        # 计算初始分数: log(prior) + gumbel
        log_priors = np.log(priors + 1e-8)
        scores = log_priors + gumbels
        
        # 选择 top-k 动作进行搜索
        k = min(self.config.num_sampled_actions, num_actions)
        top_k_indices = np.argsort(scores)[-k:]
        
        # Sequential Halving
        candidates = list(top_k_indices)
        sims_per_action = max(1, self.config.num_simulations // max(1, k * int(np.ceil(np.log2(max(2, k))))))

        # 执行模拟
        rounds = 0
        max_rounds = int(np.ceil(np.log2(max(2, k)))) + 1

        while len(candidates) > 1 and rounds < max_rounds:
            rounds += 1
            for action_idx in candidates:
                for _ in range(sims_per_action):
                    self._simulate(root, action_idx)

            # 基于 Q + Gumbel 分数减半
            q_scores = []
            for idx in candidates:
                child = root.children[idx]
                q_score = self._compute_gumbel_q(child, gumbels[idx], root.visit_count)
                q_scores.append((idx, q_score))

            q_scores.sort(key=lambda x: x[1], reverse=True)
            # 确保至少减少一个，避免无限循环
            new_size = max(1, len(candidates) // 2)
            candidates = [x[0] for x in q_scores[:new_size]]
        
        # 计算最终策略
        visit_counts = np.array([root.children[i].visit_count for i in range(num_actions)])
        policy = visit_counts / (visit_counts.sum() + 1e-8)
        
        # 选择最佳动作
        best_idx = candidates[0] if candidates else np.argmax(visit_counts)
        best_decision = legal_moves[best_idx]

        return policy, best_decision

    def _sample_gumbel(self, n: int) -> np.ndarray:
        """采样 Gumbel(0, 1) 噪声"""
        u = np.random.uniform(0, 1, size=n)
        return -np.log(-np.log(u + 1e-10) + 1e-10)

    def _add_dirichlet_noise(self, node: MCTSNode, num_actions: int):
        """添加 Dirichlet 噪声到根节点"""
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * num_actions)
        frac = self.config.root_noise_frac
        for i in range(num_actions):
            if i in node.children:
                node.children[i].prior = (1 - frac) * node.children[i].prior + frac * noise[i]

    def _compute_gumbel_q(self, child: MCTSNode, gumbel: float, parent_visits: int) -> float:
        """计算 Gumbel Q 分数"""
        if child.visit_count == 0:
            return gumbel + np.log(child.prior + 1e-8)

        # 使用完成的 Q 值估计
        sigma_q = self._sigma(child.q_value)
        return gumbel + np.log(child.prior + 1e-8) + sigma_q

    def _sigma(self, q: float) -> float:
        """将 Q 值转换到合适的尺度"""
        return q * self.config.value_scale

    def _expand_node(self, node: MCTSNode, legal_moves: List[Decision]):
        """扩展节点"""
        if node.is_expanded:
            return

        # 获取神经网络预测
        obs = self._encode_state(node.state)
        phase_id = self._get_phase_id(node.state)
        cand_feats = self._build_candidate_features(legal_moves, phase_id, node.state)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        cand_tensor = torch.from_numpy(cand_feats).to(self.device)

        logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        logits = logits_list[0]

        # 计算策略
        probs = F.softmax(logits, dim=0).cpu().numpy()
        value = value.item()

        # 创建子节点
        for i, dec in enumerate(legal_moves):
            child = MCTSNode(
                state=None,  # 延迟创建
                parent=node,
                action_idx=i,
                decision=dec,
                prior=probs[i]
            )
            node.children[i] = child

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1

    def _simulate(self, root: MCTSNode, action_idx: int):
        """执行单次模拟"""
        child = root.children[action_idx]

        # 如果子节点状态未创建，先创建
        if child.state is None:
            move = self._decision_to_move(child.decision)
            child.state = root.state.apply_move(move)

        # 获取子节点的价值估计
        if not child.is_expanded:
            legal_moves = child.state.legal_moves()
            if legal_moves:
                self._expand_node(child, legal_moves)
                value = -child.value_sum  # 对手视角
            else:
                # 终局
                value = self._get_terminal_value(child.state, root.player)
        else:
            value = -child.q_value

        # 回传
        child.visit_count += 1
        child.value_sum += value
        root.visit_count += 1

    def _encode_state(self, state: GameState) -> np.ndarray:
        """编码棋盘状态为 (6, 14, 14)"""
        obs = np.zeros((6, board_size, board_size), dtype=np.float32)
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                pt = Point(r, c)
                pl = state.board.get(pt)
                if pl == Player.white:
                    obs[0, r-1, c-1] = 1.0
                elif pl == Player.black:
                    obs[1, r-1, c-1] = 1.0
                else:
                    obs[2, r-1, c-1] = 1.0
        return obs

    def _get_phase_id(self, state: GameState) -> int:
        """获取阶段 ID: 0=布局, 1=对战, 2=飞子"""
        if state.step < board_gild:
            return 0
        elif state.board.get_player_total(state.next_player) <= 14:
            return 2
        return 1

    def _build_candidate_features(self, decisions: List[Decision], phase_id: int,
                                   state: GameState) -> np.ndarray:
        """构建候选动作特征"""
        from dyt.candidate_features import build_features_for_candidates

        cand_dicts = [self._decision_to_dict(d) for d in decisions]
        flying = state.board.get_player_total(state.next_player) <= 14
        feats = build_features_for_candidates(cand_dicts, phase_id, flying)
        return feats[:, :14]  # 只取前14维

    def _decision_to_dict(self, dec: Decision) -> dict:
        """将 Decision 转换为字典格式"""
        if dec.act == 'put_piece':
            return {'act': 'put_piece', 'point': {'r': dec.points.row, 'c': dec.points.col}}
        elif dec.act == 'is_go':
            go = dec.points
            return {'act': 'is_go',
                    'go': {'r': go.go.row, 'c': go.go.col},
                    'to': {'r': go.to.row, 'c': go.to.col}}
        elif dec.act == 'fly':
            go = dec.points
            return {'act': 'fly',
                    'go': {'r': go.go.row, 'c': go.go.col},
                    'to': {'r': go.to.row, 'c': go.to.col}}
        elif dec.act == 'skip_move':
            se = dec.points
            return {'act': 'skip_move',
                    'go': {'r': se.go.row, 'c': se.go.col},
                    'to': {'r': se.to.row, 'c': se.to.col},
                    'eat': {'r': se.eat.row, 'c': se.eat.col}}
        elif dec.act == 'skip_eat_seq':
            seq = []
            for se in dec.points:
                seq.append({'go': {'r': se.go.row, 'c': se.go.col},
                           'to': {'r': se.to.row, 'c': se.to.col},
                           'eat': {'r': se.eat.row, 'c': se.eat.col}})
            return {'act': 'skip_eat_seq', 'seq': seq}
        return {'act': 'unknown'}

    def _decision_to_move(self, dec: Decision) -> Move:
        """将 Decision 转换为 Move"""
        if dec.act == 'put_piece':
            return Move.put_piece(dec.points)
        elif dec.act == 'is_go':
            return Move.go_piece(dec.points)
        elif dec.act == 'fly':
            return Move.fly_piece(dec.points)
        elif dec.act == 'skip_move':
            return Move.move_skip(dec.points)
        elif dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(dec.points)
        raise ValueError(f"Unknown decision type: {dec.act}")

    def _get_terminal_value(self, state: GameState, player: Player) -> float:
        """获取终局价值"""
        from jiu.scoring import compute_game_result
        result = compute_game_result(state)

        if player == Player.white:
            if result.w > result.b:
                return 1.0
            elif result.w < result.b:
                return -1.0
        else:
            if result.b > result.w:
                return 1.0
            elif result.b < result.w:
                return -1.0
        return 0.0


class SimpleMCTS:
    """
    简化版 MCTS，用于快速推理

    不使用 Gumbel，但保持高效。用于正式对弈。
    """

    def __init__(self, model, num_simulations: int = 100, c_puct: float = 1.5,
                 device: str = 'cuda'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._gumbel_mcts = GumbelMCTS(model, MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct
        ), device)

    @torch.no_grad()
    def search(self, state: GameState, temperature: float = 0.0) -> Decision:
        """
        执行 MCTS 搜索并返回最佳动作

        Args:
            state: 当前游戏状态
            temperature: 采样温度（0 = 贪婪）

        Returns:
            最佳动作
        """
        policy, best_decision = self._gumbel_mcts.search(state)

        if temperature == 0 or best_decision is None:
            return best_decision

        # 温度采样
        legal_moves = state.legal_moves()
        if len(policy) != len(legal_moves):
            return best_decision

        # 应用温度
        policy = np.power(policy, 1.0 / temperature)
        policy = policy / (policy.sum() + 1e-8)

        # 采样
        idx = np.random.choice(len(legal_moves), p=policy)
        return legal_moves[idx]


def create_mcts(model, mode: str = 'train', device: str = 'cuda', **kwargs) -> GumbelMCTS:
    """
    创建 MCTS 实例

    Args:
        model: 神经网络模型
        mode: 'train' (Gumbel MCTS) 或 'eval' (SimpleMCTS)
        device: 设备
        **kwargs: 额外配置

    Returns:
        MCTS 实例
    """
    if mode == 'train':
        config = MCTSConfig(
            num_simulations=kwargs.get('num_simulations', 32),
            num_sampled_actions=kwargs.get('num_sampled_actions', 16),
            c_puct=kwargs.get('c_puct', 1.5),
            root_dirichlet_alpha=kwargs.get('dirichlet_alpha', 0.3),
            root_noise_frac=kwargs.get('noise_frac', 0.25),
        )
        return GumbelMCTS(model, config, device)
    else:
        return SimpleMCTS(
            model,
            num_simulations=kwargs.get('num_simulations', 100),
            c_puct=kwargs.get('c_puct', 1.5),
            device=device
        )

