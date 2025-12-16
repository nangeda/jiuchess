#!/usr/bin/env python3
"""
MCTS自对弈数据生成脚本

使用传统MCTS进行自对弈，生成高质量训练数据
用于DyT+Gumbel MCTS的监督学习或强化学习
"""
import os
import sys
import json
import time
import pickle
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import deque
from copy import deepcopy

import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiu.jiuboard_fast import GameState, Move, Board, count_squares
from jiu.jiutypes import Player, Point, Decision, board_size, board_gild
import math
import random


def encode_board_state(state: GameState, history: deque = None) -> np.ndarray:
    """将GameState编码为 (6, H, W) 的观察张量"""
    H, W = board_size, board_size
    obs = np.zeros((6, H, W), dtype=np.float32)
    
    # 通道0-2: 当前局面
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            pt = Point(r, c)
            pl = state.board.get(pt)
            if pl == Player.white:
                obs[0, r - 1, c - 1] = 1.0
            elif pl == Player.black:
                obs[1, r - 1, c - 1] = 1.0
            else:
                obs[2, r - 1, c - 1] = 1.0
    
    # 通道3-5: 历史帧
    if history is not None and len(history) > 0:
        me = state.next_player
        hist_list = list(history)[-3:]
        for i, past_board in enumerate(hist_list):
            for r in range(1, H + 1):
                for c in range(1, W + 1):
                    pt = Point(r, c)
                    if past_board.get(pt) == me:
                        obs[3 + i, r - 1, c - 1] = 1.0
    return obs


def get_phase_id(state: GameState) -> int:
    """获取阶段ID: 0=布局, 1=对战, 2=飞子"""
    if state.step < board_gild:
        return 0
    elif state.board.get_player_total(state.next_player) <= 14:
        return 2
    else:
        return 1


def decision_to_dict(dec: Decision, state: GameState) -> dict:
    """将Decision转换为字典"""
    from jiu.jiutypes import Go, Skip_eat
    
    if dec.act == 'put_piece':
        p = dec.points
        return {'act': 'put_piece', 'point': {'r': p.row, 'c': p.col}}
    elif dec.act == 'is_go':
        go_obj = dec.points
        return {'act': 'is_go', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col},
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    elif dec.act == 'fly':
        go_obj = dec.points
        return {'act': 'fly', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col},
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    elif dec.act == 'skip_move':
        se = dec.points
        return {'act': 'skip_move', 'go': {'r': se.go.row, 'c': se.go.col},
                'to': {'r': se.to.row, 'c': se.to.col},
                'eat': {'r': se.eat.row, 'c': se.eat.col}}
    elif dec.act == 'skip_eat_seq':
        seq = dec.points
        seq_data = []
        for se in seq:
            seq_data.append({
                'go': {'r': se.go.row, 'c': se.go.col},
                'to': {'r': se.to.row, 'c': se.to.col},
                'eat': {'r': se.eat.row, 'c': se.eat.col}
            })
        return {'act': 'skip_eat_seq', 'seq': seq_data}
    return {'act': dec.act}


# ============== 快速MCTS实现 ==============

class FastMCTSNode:
    """快速MCTS节点"""
    __slots__ = ['state', 'parent', 'move', 'children', 'untried_moves', 'wins', 'visits']

    def __init__(self, state: GameState, parent=None, move: Decision = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = state.legal_moves()
        self.wins = 0.0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_over()

    def best_child(self, c=1.414):
        best_score = -float('inf')
        best_child = None
        log_parent = math.log(self.visits) if self.visits > 0 else 0

        for child in self.children:
            if child.visits == 0:
                return child  # 优先访问未访问节点
            score = child.wins / child.visits + c * math.sqrt(log_parent / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        dec = self.untried_moves.pop()
        move_obj = decision_to_move(dec)
        next_state = self.state.apply_move(move_obj)
        child = FastMCTSNode(next_state, parent=self, move=dec)
        self.children.append(child)
        return child


def decision_to_move(dec: Decision) -> Move:
    """将Decision转换为Move"""
    if dec.act == 'put_piece':
        return Move.put_piece(dec.points)
    if dec.act == 'is_go':
        return Move.go_piece(dec.points)
    if dec.act == 'skip_move':
        return Move.move_skip(dec.points)
    if dec.act == 'skip_eat_seq':
        return Move.move_skip_seq(dec.points)
    if dec.act == 'fly':
        return Move.fly_piece(dec.points)
    if dec.act == 'eat_point':
        return Move.eat(dec.points)
    return None


class FastMCTS:
    """快速MCTS - 使用浅层rollout"""

    def __init__(self, simulation_time=1.5, max_simulations=100, rollout_depth=10, c=1.4):
        self.simulation_time = simulation_time
        self.max_simulations = max_simulations
        self.rollout_depth = rollout_depth  # 浅层rollout
        self.c = c

    def search(self, state: GameState) -> Tuple[Decision, np.ndarray]:
        """执行MCTS搜索，返回最佳动作和访问分布"""
        candidates = state.legal_moves()
        if not candidates:
            return None, np.array([])
        if len(candidates) == 1:
            return candidates[0], np.array([1.0])

        root = FastMCTSNode(state)
        start_time = time.time()
        simulations = 0

        while (time.time() - start_time < self.simulation_time and
               simulations < self.max_simulations):
            node = self._select(root)

            if not node.is_terminal():
                if not node.is_fully_expanded():
                    node = node.expand()
                reward = self._rollout(node.state, state.next_player)
            else:
                reward = self._terminal_reward(node.state, state.next_player)

            self._backpropagate(node, reward)
            simulations += 1

        # 构建访问分布
        visit_counts = np.zeros(len(candidates))
        for child in root.children:
            # 找到对应的候选索引
            for i, cand in enumerate(candidates):
                if self._decisions_equal(child.move, cand):
                    visit_counts[i] = child.visits
                    break

        # 归一化
        total = visit_counts.sum()
        if total > 0:
            visit_dist = visit_counts / total
        else:
            visit_dist = np.ones(len(candidates)) / len(candidates)

        # 选择访问次数最多的
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move, visit_dist
        else:
            return random.choice(candidates), visit_dist

    def _select(self, node: FastMCTSNode) -> FastMCTSNode:
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c)
        return node

    def _rollout(self, state: GameState, root_player: Player) -> float:
        """浅层rollout + 启发式评估 + 多样性策略"""
        current = state
        depth = 0

        while not current.is_over() and depth < self.rollout_depth:
            decisions = current.legal_moves()
            if not decisions:
                break

            if current.step >= board_gild:
                # 对战/飞子阶段：混合策略增加多样性
                eat_moves = [d for d in decisions if d.eats > 0]

                if eat_moves:
                    # 80%概率选最大吃子，20%概率随机吃子（增加多样性）
                    if random.random() < 0.8:
                        dec = max(eat_moves, key=lambda d: d.eats)
                    else:
                        dec = random.choice(eat_moves)
                else:
                    # 无吃子时：70%随机，30%选择"进攻性"走法
                    if random.random() < 0.3:
                        # 选择靠近对方棋子的走法（更激进）
                        dec = self._select_aggressive_move(current, decisions)
                    else:
                        dec = random.choice(decisions)
            else:
                # 布局阶段：偏好中心区域
                if random.random() < 0.3:
                    dec = self._select_center_move(decisions)
                else:
                    dec = random.choice(decisions)

            move = decision_to_move(dec)
            if move is None:
                break
            current = current.apply_move(move)
            depth += 1

        return self._evaluate(current, root_player)

    def _select_aggressive_move(self, state: GameState, decisions: list) -> Decision:
        """选择进攻性走法：靠近对方棋子"""
        if not decisions:
            return None

        opponent = state.next_player.other
        opp_positions = []

        # 获取对方棋子位置
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                pt = Point(r, c)
                if state.board.get(pt) == opponent:
                    opp_positions.append((r, c))

        if not opp_positions:
            return random.choice(decisions)

        # 计算每个走法的"进攻分数"（靠近对方棋子）
        def aggression_score(dec):
            # 获取目标位置
            if dec.act == 'is_go' and hasattr(dec.points, 'to'):
                target = dec.points.to
            elif dec.act == 'fly' and hasattr(dec.points, 'to'):
                target = dec.points.to
            else:
                return 0

            # 计算到最近对方棋子的距离
            min_dist = float('inf')
            for opp_r, opp_c in opp_positions:
                dist = abs(target.row - opp_r) + abs(target.col - opp_c)
                min_dist = min(min_dist, dist)

            return -min_dist  # 距离越近分数越高

        # 选择进攻分数最高的走法
        best = max(decisions, key=aggression_score)
        return best

    def _select_center_move(self, decisions: list) -> Decision:
        """布局时偏好中心区域"""
        if not decisions:
            return None

        center = board_size // 2 + 1

        def center_score(dec):
            if dec.act == 'put_piece' and hasattr(dec.points, 'row'):
                r, c = dec.points.row, dec.points.col
                return -abs(r - center) - abs(c - center)
            return 0

        # 70%选最中心，30%随机（保持多样性）
        if random.random() < 0.7:
            return max(decisions, key=center_score)
        return random.choice(decisions)

    def _evaluate(self, state: GameState, player: Player) -> float:
        """启发式评估"""
        winner = state.winner()
        if winner == player:
            return 1.0
        elif winner == player.other:
            return 0.0

        # 基于棋子数评估
        my_pieces = state.board.get_player_total(player)
        opp_pieces = state.board.get_player_total(player.other)
        total = my_pieces + opp_pieces
        if total == 0:
            return 0.5

        # 加入成方评估
        my_squares = count_squares(state.board, player)
        opp_squares = count_squares(state.board, player.other)

        piece_score = my_pieces / total
        square_bonus = (my_squares - opp_squares) * 0.05

        return max(0.0, min(1.0, piece_score + square_bonus))

    def _terminal_reward(self, state: GameState, player: Player) -> float:
        winner = state.winner()
        if winner == player:
            return 1.0
        elif winner == player.other:
            return 0.0
        return 0.5

    def _backpropagate(self, node: FastMCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            # 从当前节点玩家角度更新
            if node.parent is not None:
                if node.state.next_player != node.parent.state.next_player:
                    node.wins += reward
                else:
                    node.wins += (1.0 - reward)
            node = node.parent

    def _decisions_equal(self, d1: Decision, d2: Decision) -> bool:
        if d1.act != d2.act:
            return False
        if d1.act == 'put_piece':
            return d1.points == d2.points
        return str(d1.points) == str(d2.points)


# ============== 自对弈工作器 ==============

class MCTSSelfPlayWorker:
    """MCTS自对弈工作器"""

    def __init__(
        self,
        simulation_time: float = 1.5,
        max_simulations: int = 100,
        rollout_depth: int = 10,
        exploration_weight: float = 1.4,
        max_game_steps: int = 1000,
        verbose: bool = True
    ):
        self.simulation_time = simulation_time
        self.max_simulations = max_simulations
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.max_game_steps = max_game_steps
        self.verbose = verbose

        # 创建MCTS搜索器
        self.mcts = FastMCTS(
            simulation_time=simulation_time,
            max_simulations=max_simulations,
            rollout_depth=rollout_depth,
            c=exploration_weight
        )
    
    def play_one_game(self) -> Dict:
        """进行一局完整对弈，返回对局数据"""
        state = GameState.new_game(board_size)
        history = deque(maxlen=6)
        trajectory = []

        start_time = time.time()
        step_times = []

        # 僵局检测：连续N步棋子数不变则判定僵局
        stalemate_counter = 0
        last_piece_count = 0
        STALEMATE_THRESHOLD = 50  # 连续50步无变化则结束

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始新对局 | 棋盘大小: {board_size}x{board_size}")
            print(f"MCTS配置: 模拟{self.max_simulations}次, 时限{self.simulation_time}秒")
            print(f"{'='*60}")

        while not state.is_over() and state.step < self.max_game_steps:
            step_start = time.time()
            
            # 获取合法动作
            candidates = state.legal_moves()
            if not candidates:
                if self.verbose:
                    print(f"Step {state.step}: 无合法动作，游戏结束")
                break

            # 记录当前状态
            obs = encode_board_state(state, history)
            phase_id = get_phase_id(state)

            # MCTS搜索选择动作
            chosen_dec, visit_dist = self.mcts.search(state)

            if chosen_dec is None:
                if self.verbose:
                    print(f"Step {state.step}: MCTS返回None，尝试随机选择")
                chosen_dec = random.choice(candidates)
                visit_dist = np.ones(len(candidates)) / len(candidates)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # 转换为Move
            move = decision_to_move(chosen_dec)

            # 找到选中动作的索引
            chosen_idx = 0
            for i, c in enumerate(candidates):
                if self.mcts._decisions_equal(c, chosen_dec):
                    chosen_idx = i
                    break

            # 记录轨迹
            trajectory.append({
                'obs': obs,
                'phase_id': phase_id,
                'candidates': [decision_to_dict(c, state) for c in candidates],
                'chosen_idx': chosen_idx,
                'visit_dist': visit_dist,
                'player': 'white' if state.next_player == Player.white else 'black',
                'step': state.step
            })

            # 更新历史
            history.append(deepcopy(state.board))

            # 执行动作
            state = state.apply_move(move)

            # 僵局检测（对战期和飞子期）
            if state.step >= board_gild:
                current_pieces = state.board.get_player_total(Player.white) + state.board.get_player_total(Player.black)
                if current_pieces == last_piece_count:
                    stalemate_counter += 1
                else:
                    stalemate_counter = 0
                    last_piece_count = current_pieces

                if stalemate_counter >= STALEMATE_THRESHOLD:
                    if self.verbose:
                        print(f"Step {state.step}: 检测到僵局（连续{STALEMATE_THRESHOLD}步无变化），结束对局")
                    break

            # 打印进度
            if self.verbose and state.step % 20 == 0:
                w_count = state.board.get_player_total(Player.white)
                b_count = state.board.get_player_total(Player.black)
                phase_name = ['布局', '对战', '飞子'][get_phase_id(state)]
                print(f"Step {state.step:3d} | {phase_name} | 白:{w_count:2d} 黑:{b_count:2d} | "
                      f"本步{step_time:.2f}s")

        # 获取胜者
        winner = state.winner()
        if winner is None:
            winner = state.winner_by_timeout()

        total_time = time.time() - start_time

        # 计算value标签
        for step_data in trajectory:
            player = Player.white if step_data['player'] == 'white' else Player.black
            if winner is None:
                step_data['value'] = 0.0
            elif winner == player:
                step_data['value'] = 1.0
            else:
                step_data['value'] = -1.0

        result = {
            'trajectory': trajectory,
            'winner': 'white' if winner == Player.white else ('black' if winner == Player.black else 'draw'),
            'total_steps': state.step,
            'total_time': total_time,
            'avg_step_time': np.mean(step_times) if step_times else 0,
            'config': {
                'simulation_time': self.simulation_time,
                'max_simulations': self.max_simulations
            }
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"对局结束!")
            print(f"  总步数: {state.step}")
            print(f"  总耗时: {total_time:.1f}秒")
            print(f"  平均每步: {result['avg_step_time']:.2f}秒")
            print(f"  胜者: {result['winner']}")
            print(f"  白子: {state.board.get_player_total(Player.white)}")
            print(f"  黑子: {state.board.get_player_total(Player.black)}")
            print(f"{'='*60}")

        return result


def save_game_data(result: Dict, output_dir: str, game_id: int):
    """保存对局数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存为pickle（保留numpy数组）
    pkl_path = os.path.join(output_dir, f"game_{game_id:04d}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)

    # 保存摘要为JSON
    summary = {
        'game_id': game_id,
        'winner': result['winner'],
        'total_steps': result['total_steps'],
        'total_time': result['total_time'],
        'avg_step_time': result['avg_step_time'],
        'num_samples': len(result['trajectory']),
        'config': result['config']
    }
    json_path = os.path.join(output_dir, f"game_{game_id:04d}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return pkl_path


def main():
    parser = argparse.ArgumentParser(description='MCTS自对弈数据生成')
    parser.add_argument('--num_games', type=int, default=1, help='生成对局数量')
    parser.add_argument('--simulations', type=int, default=100, help='每步MCTS模拟次数')
    parser.add_argument('--time_limit', type=float, default=1.0, help='每步时间限制(秒)')
    parser.add_argument('--rollout_depth', type=int, default=10, help='Rollout深度')
    parser.add_argument('--output_dir', type=str, default='data/mcts_selfplay', help='输出目录')
    parser.add_argument('--max_steps', type=int, default=1000, help='最大对局步数')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# MCTS 自对弈数据生成")
    print(f"{'#'*60}")
    print(f"配置:")
    print(f"  - 对局数量: {args.num_games}")
    print(f"  - MCTS模拟次数: {args.simulations}")
    print(f"  - 每步时限: {args.time_limit}秒")
    print(f"  - Rollout深度: {args.rollout_depth}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 最大步数: {args.max_steps}")
    print(f"{'#'*60}\n")

    # 创建worker
    worker = MCTSSelfPlayWorker(
        simulation_time=args.time_limit,
        max_simulations=args.simulations,
        rollout_depth=args.rollout_depth,
        max_game_steps=args.max_steps,
        verbose=not args.quiet
    )

    all_results = []
    total_samples = 0
    total_time = 0

    for game_id in range(args.num_games):
        print(f"\n>>> 对局 {game_id + 1}/{args.num_games}")

        result = worker.play_one_game()
        all_results.append(result)

        # 保存数据
        save_path = save_game_data(result, args.output_dir, game_id)

        total_samples += len(result['trajectory'])
        total_time += result['total_time']

        print(f"已保存: {save_path}")
        print(f"累计样本: {total_samples}, 累计时间: {total_time:.1f}秒")

    # 打印总结
    print(f"\n{'='*60}")
    print(f"生成完成!")
    print(f"  - 总对局数: {args.num_games}")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 总耗时: {total_time:.1f}秒")
    print(f"  - 平均每局: {total_time/args.num_games:.1f}秒")
    print(f"  - 数据目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

