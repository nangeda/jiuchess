#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Value标签生成策略

改进点：
1. 修复CSV解析（使用已验证的解析逻辑）
2. 从最终状态推断胜负
3. 改进启发式评估（增加位置、威胁等特征）
4. 支持多种Value计算策略
"""
import argparse
import copy
import os
import sys
from typing import List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import Point
from agent.dyt_agent import encode_board_state, get_phase_id
from dyt.candidate_features import build_features_for_candidates

# 导入已验证的解析函数
from scripts.build_dataset_no_label import (
    parse_stage1_tokens,
    parse_stage2_segments,
    build_move_from_segment,
    move_to_dict
)


class ImprovedValueLabeler:
    """改进的价值标签生成器"""
    
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.stats = {
            'from_final_state': 0,
            'from_piece_count': 0,
            'from_advantage': 0,
            'from_heuristic': 0,
            'total': 0
        }
    
    def label_game(self, line: str) -> List[dict]:
        """为一局对局生成所有样本（带改进的value标签）"""
        try:
            # 解析对局
            game_states, moves = self._parse_game(line)
            if len(game_states) < 2:
                return []
            
            # 判断对局结果（从最终状态推断）
            final_state = game_states[-1]
            winner, method = self._infer_winner(final_state)
            
            # 生成样本
            samples = []
            for t, (state, actual_move) in enumerate(zip(game_states[:-1], moves)):
                try:
                    sample = self._create_sample(state, actual_move, t, len(game_states), winner, method)
                    if sample is not None:
                        samples.append(sample)
                except Exception as e:
                    continue
            
            return samples
            
        except Exception as e:
            return []
    
    def _parse_game(self, line: str) -> Tuple[List[GameState], List[Move]]:
        """解析对局，返回所有状态和动作"""
        try:
            # 移除引号和回车
            line = line.strip().strip('"').strip()
            
            # 分割Stage[1]和Stage[2]
            if 'Stage[1]:' not in line:
                return [], []
            
            head, rest = line.split('Stage[1]:', 1)
            
            if ';Stage[2]:' in rest:
                stage1_str, rest2 = rest.split(';Stage[2]:', 1)
                stage2_str = rest2
            else:
                stage1_str, stage2_str = rest, ''
        except ValueError:
            return [], []
        
        states = []
        moves = []
        state = GameState.new_game(14)
        states.append(copy.deepcopy(state))
        
        # Stage 1: 布局
        try:
            for player, pt in parse_stage1_tokens(stage1_str):
                move = Move.put_piece(pt)
                moves.append(move)
                state = state.apply_move(move)
                states.append(copy.deepcopy(state))
        except Exception:
            pass
        
        # Stage 2: 对战
        try:
            segments = parse_stage2_segments(stage2_str)
            for seg in segments:
                try:
                    move = build_move_from_segment(state, seg)
                    if move is None:
                        continue
                    moves.append(move)
                    state = state.apply_move(move)
                    states.append(copy.deepcopy(state))
                except Exception:
                    break
        except Exception:
            pass
        
        return states, moves
    
    def _infer_winner(self, final_state: GameState) -> Tuple[Optional[Player], str]:
        """从最终状态推断获胜方"""
        # 方法1：游戏已结束
        if final_state.is_over():
            winner = final_state.winner()
            return winner, 'game_over'
        
        # 方法2：检查棋子数
        black_pieces = final_state.board.get_player_total(Player.black)
        white_pieces = final_state.board.get_player_total(Player.white)
        
        # 少于3个子失败
        if black_pieces < 3 and white_pieces >= 3:
            return Player.white, 'piece_count'
        if white_pieces < 3 and black_pieces >= 3:
            return Player.black, 'piece_count'
        
        # 方法3：显著优势（>6子差距）
        if black_pieces - white_pieces > 6:
            return Player.black, 'large_advantage'
        if white_pieces - black_pieces > 6:
            return Player.white, 'large_advantage'
        
        # 方法4：中等优势（3-6子差距）
        if black_pieces - white_pieces >= 3:
            return Player.black, 'medium_advantage'
        if white_pieces - black_pieces >= 3:
            return Player.white, 'medium_advantage'
        
        # 无法判断
        return None, 'unknown'
    
    def _create_sample(
        self, 
        state: GameState, 
        actual_move: Move, 
        step_num: int,
        total_steps: int,
        winner: Optional[Player],
        method: str
    ) -> Optional[dict]:
        """创建一个带value标签的样本"""
        
        # 编码棋盘
        obs = encode_board_state(state, history=[])
        phase_id = get_phase_id(state)
        
        # 获取候选动作
        legal_decisions = state.legal_moves()
        if not legal_decisions:
            return None
        
        # 构建候选特征
        legal_moves = [self._dec_to_move(d) for d in legal_decisions]
        cand_dicts = [move_to_dict(m) for m in legal_moves]
        flying = state._is_flying_stage() if hasattr(state, '_is_flying_stage') else False
        cand_feats = build_features_for_candidates(cand_dicts, phase_id, flying)
        
        # 匹配标签索引
        label_idx = self._match_move(actual_move, legal_moves)
        
        if label_idx < 0:
            return None
        
        # ⭐ 生成改进的价值标签
        value = self._compute_value(state, step_num, total_steps, winner, method)
        
        self.stats['total'] += 1
        
        return {
            'obs': obs,
            'phase_id': phase_id,
            'cand_feats': cand_feats,
            'label_idx': label_idx,
            'value': value
        }
    
    def _compute_value(
        self,
        state: GameState,
        step_num: int,
        total_steps: int,
        winner: Optional[Player],
        method: str
    ) -> float:
        """计算改进的价值标签"""
        
        current_player = state.next_player
        steps_to_end = total_steps - step_num
        
        # 策略1：使用推断的胜负结果
        if winner is not None and method in ['game_over', 'piece_count', 'large_advantage']:
            # 高置信度的胜负
            discount = self.gamma ** steps_to_end
            
            if winner == current_player:
                value = discount
            elif winner == current_player.other:
                value = -discount
            else:
                value = 0.0
            
            self.stats['from_final_state'] += 1
            return float(np.clip(value, -1.0, 1.0))
        
        # 策略2：中等优势（部分折扣）
        if winner is not None and method == 'medium_advantage':
            discount = self.gamma ** steps_to_end
            
            if winner == current_player:
                value = 0.5 * discount  # 降低置信度
            elif winner == current_player.other:
                value = -0.5 * discount
            else:
                value = 0.0
            
            self.stats['from_advantage'] += 1
            return float(np.clip(value, -1.0, 1.0))
        
        # 策略3：改进的启发式评估
        value = self._improved_heuristic_evaluate(state)
        self.stats['from_heuristic'] += 1
        return value
    
    def _improved_heuristic_evaluate(self, state: GameState) -> float:
        """改进的启发式评估（增加更多特征）"""
        player = state.next_player
        opponent = player.other
        
        # 特征1：棋子数差异（基础）
        my_pieces = state.board.get_player_total(player)
        opp_pieces = state.board.get_player_total(opponent)
        piece_diff = my_pieces - opp_pieces
        piece_score = piece_diff / 14.0  # 归一化
        
        # 特征2：褡裢数差异（重要）
        try:
            from jiu.jiuboard_fast import count_independent_dalians
            my_dalians = count_independent_dalians(state.board, player)
            opp_dalians = count_independent_dalians(state.board, opponent)
            dalian_diff = my_dalians - opp_dalians
            dalian_score = dalian_diff / 2.0
        except:
            dalian_score = 0.0
        
        # 特征3：中心控制（棋子在中心区域的数量）
        center_score = self._evaluate_center_control(state, player, opponent)
        
        # 特征4：灵活性（可移动的位置数）
        mobility_score = self._evaluate_mobility(state, player, opponent)
        
        # 特征5：连接性（棋子的连通性）
        connectivity_score = self._evaluate_connectivity(state, player, opponent)
        
        # 综合评分（调整权重）
        score = (
            0.25 * piece_score +      # 棋子数
            0.35 * dalian_score +      # 褡裢（最重要）
            0.15 * center_score +      # 中心控制
            0.15 * mobility_score +    # 灵活性
            0.10 * connectivity_score  # 连接性
        )
        
        return float(np.clip(np.tanh(score), -1.0, 1.0))
    
    def _evaluate_center_control(self, state: GameState, player: Player, opponent: Player) -> float:
        """评估中心控制"""
        center_points = [
            Point(7, 7), Point(7, 8), Point(8, 7), Point(8, 8),
            Point(6, 7), Point(6, 8), Point(9, 7), Point(9, 8),
            Point(7, 6), Point(7, 9), Point(8, 6), Point(8, 9)
        ]
        
        my_center = sum(1 for pt in center_points if state.board.get(pt) == player)
        opp_center = sum(1 for pt in center_points if state.board.get(pt) == opponent)
        
        return (my_center - opp_center) / len(center_points)
    
    def _evaluate_mobility(self, state: GameState, player: Player, opponent: Player) -> float:
        """评估移动灵活性（简化版）"""
        # 统计每个玩家的棋子周围空位数
        my_mobility = 0
        opp_mobility = 0
        
        for r in range(1, 15):
            for c in range(1, 15):
                pt = Point(r, c)
                owner = state.board.get(pt)
                
                if owner is None:
                    continue
                
                # 统计相邻空位
                adjacent_empty = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 1 <= nr <= 14 and 1 <= nc <= 14:
                        if state.board.get(Point(nr, nc)) is None:
                            adjacent_empty += 1
                
                if owner == player:
                    my_mobility += adjacent_empty
                else:
                    opp_mobility += adjacent_empty
        
        total = my_mobility + opp_mobility
        if total == 0:
            return 0.0
        
        return (my_mobility - opp_mobility) / total
    
    def _evaluate_connectivity(self, state: GameState, player: Player, opponent: Player) -> float:
        """评估棋子连接性（相邻己方棋子数）"""
        my_connections = 0
        opp_connections = 0
        
        for r in range(1, 15):
            for c in range(1, 15):
                pt = Point(r, c)
                owner = state.board.get(pt)
                
                if owner is None:
                    continue
                
                # 统计相邻同色棋子
                adjacent_same = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 1 <= nr <= 14 and 1 <= nc <= 14:
                        if state.board.get(Point(nr, nc)) == owner:
                            adjacent_same += 1
                
                if owner == player:
                    my_connections += adjacent_same
                else:
                    opp_connections += adjacent_same
        
        total = my_connections + opp_connections
        if total == 0:
            return 0.0
        
        return (my_connections - opp_connections) / total
    
    def _dec_to_move(self, dec) -> Move:
        """Decision转Move"""
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
        return None
    
    def _match_move(self, actual_move: Move, candidates: List[Move]) -> int:
        """匹配动作索引"""
        actual_dict = move_to_dict(actual_move)
        for i, cand_mv in enumerate(candidates):
            if move_to_dict(cand_mv) == actual_dict:
                return i
        return -1


def main():
    parser = argparse.ArgumentParser(description='生成改进的Value标签')
    parser.add_argument('--csv', default='data/2024.5.13(clean1).csv', help='输入CSV文件')
    parser.add_argument('--out_dir', default='exp/datasets_improved_value', help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 创建标签器
    labeler = ImprovedValueLabeler(gamma=args.gamma)
    
    # 处理数据
    print(f"\n处理对局数据: {args.csv}")
    all_samples = []
    
    with open(args.csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总对局数: {len(lines)}\n")
    
    for line in tqdm(lines, desc='生成改进Value标签'):
        line = line.strip()
        if not line:
            continue
        
        samples = labeler.label_game(line)
        all_samples.extend(samples)
    
    print(f'\n✅ 收集到 {len(all_samples)} 个样本（改进Value标签）')
    
    # 统计
    print(f"\n标签来源统计:")
    print(f"  最终状态推断: {labeler.stats['from_final_state']} ({labeler.stats['from_final_state']/max(labeler.stats['total'],1)*100:.1f}%)")
    print(f"  中等优势: {labeler.stats['from_advantage']} ({labeler.stats['from_advantage']/max(labeler.stats['total'],1)*100:.1f}%)")
    print(f"  改进启发式: {labeler.stats['from_heuristic']} ({labeler.stats['from_heuristic']/max(labeler.stats['total'],1)*100:.1f}%)")
    
    # 拆分训练/验证集
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = int(len(all_samples) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    
    # 保存
    train_path = os.path.join(args.out_dir, 'train_improved_value.pt')
    val_path = os.path.join(args.out_dir, 'val_improved_value.pt')
    
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    
    print(f'\n✅ 保存完成:')
    print(f'  训练集: {train_path} ({len(train_samples)} 样本)')
    print(f'  验证集: {val_path} ({len(val_samples)} 样本)')
    
    # 保存统计信息
    stats_path = os.path.join(args.out_dir, 'improved_value_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"改进Value标签生成统计\n")
        f.write(f"==================\n\n")
        f.write(f"总样本数: {len(all_samples)}\n")
        f.write(f"训练集: {len(train_samples)}\n")
        f.write(f"验证集: {len(val_samples)}\n\n")
        
        f.write(f"标签来源:\n")
        f.write(f"  最终状态推断: {labeler.stats['from_final_state']}\n")
        f.write(f"  中等优势: {labeler.stats['from_advantage']}\n")
        f.write(f"  改进启发式: {labeler.stats['from_heuristic']}\n\n")
        
        # 价值分布统计
        values = [s['value'] for s in all_samples]
        f.write(f"Value分布:\n")
        f.write(f"  均值: {np.mean(values):.3f}\n")
        f.write(f"  标准差: {np.std(values):.3f}\n")
        f.write(f"  最小值: {np.min(values):.3f}\n")
        f.write(f"  最大值: {np.max(values):.3f}\n")
        f.write(f"  中位数: {np.median(values):.3f}\n")
        
        # Value分布直方图
        f.write(f"\nValue分布直方图:\n")
        hist, bins = np.histogram(values, bins=10, range=(-1, 1))
        for i in range(len(hist)):
            bar = '█' * int(hist[i] / max(hist) * 50)
            f.write(f"  [{bins[i]:5.2f}, {bins[i+1]:5.2f}): {hist[i]:6d} {bar}\n")
    
    print(f'  统计信息: {stats_path}')


if __name__ == '__main__':
    main()

