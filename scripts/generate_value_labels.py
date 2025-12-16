#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为久棋对局数据生成 Value 标签

策略：
1. 优先使用对局真实结果（最准确）
2. 使用DyT模型Rollout（中等可靠）
3. 启发式评估（快速回退）
"""
import argparse
import copy
import os
import sys
from typing import List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
import re

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiu.jiuboard_fast import GameState, Move, board_gild, count_independent_dalians
from jiu.jiutypes import Player, Point, dataset_coord_to_point, Go, Skip_eat
from agent.dyt_agent import DyTAgent, encode_board_state, get_phase_id
from dyt.candidate_features import build_features_for_candidates


# ========== CSV 解析函数 ==========
COORD_RE = re.compile(r"\(([A-N]),\s*(\d{1,2})\)")


def parse_stage1_tokens(stage1: str) -> List[Tuple[Player, Point]]:
    """解析布局阶段"""
    tokens = []
    for m in re.finditer(r"([BW])\(([A-N]),\s*(\d{1,2})\)", stage1):
        color, letter, row = m.group(1), m.group(2), m.group(3)
        player = Player.black if color == 'B' else Player.white
        try:
            pt = dataset_coord_to_point((letter, row))
            tokens.append((player, pt))
        except ValueError:
            continue
    return tokens


def extract_coords(s: str) -> List[Point]:
    """提取坐标列表"""
    pts = []
    for letter, row in COORD_RE.findall(s):
        try:
            pts.append(dataset_coord_to_point((letter, row)))
        except ValueError:
            continue
    return pts


def parse_stage2_segments(stage2: str) -> List[dict]:
    """解析对战阶段"""
    raw_segments = [seg for seg in stage2.split(';') if seg.strip()]
    moves = []
    for seg in raw_segments:
        if seg.startswith('Stage[2]:'):
            seg = seg[len('Stage[2]:'):]
        seg = seg.strip()
        if not seg:
            continue
        color = seg[0]
        player = Player.black if color == 'B' else Player.white
        start_match = COORD_RE.search(seg)
        if not start_match:
            continue
        start = dataset_coord_to_point((start_match.group(1), start_match.group(2)))
        to_coords: List[Point] = []
        for m in re.finditer(r"-O\(([A-N]),\s*(\d{1,2})\)", seg):
            to_coords.append(dataset_coord_to_point((m.group(1), m.group(2))))
        tc_pts: List[Point] = []
        fc_pts: List[Point] = []
        tc_m = re.search(r"TC:([^;]+)", seg)
        if tc_m:
            tc_pts = extract_coords(tc_m.group(1))
        fc_m = re.search(r"FC:([^;]+)", seg)
        if fc_m:
            fc_pts = extract_coords(fc_m.group(1))
        moves.append({
            'player': player,
            'start': start,
            'tos': to_coords,
            'tc': tc_pts,
            'fc': fc_pts,
        })
    return moves


def midpoint(a: Point, b: Point) -> Point:
    """计算中点"""
    return Point((a.row + b.row) // 2, (a.col + b.col) // 2)


def build_move_from_segment(state: GameState, seg: dict) -> Move:
    """从segment构建Move对象"""
    me: Player = seg['player']
    start: Point = seg['start']
    tos: List[Point] = seg['tos']
    tc: List[Point] = seg['tc']
    
    # 跳吃
    if tc and tos:
        steps: List[Skip_eat] = []
        cur = start
        for to in tos:
            eat = midpoint(cur, to)
            steps.append(Skip_eat(cur, eat, to))
            cur = to
        if len(steps) == 1:
            return Move.move_skip(steps[0])
        return Move.move_skip_seq(steps)
    
    # 普通移动
    final_to = tos[-1] if tos else start
    can_fly = (state.board.get_player_total(me) <= state.board.num_rows)
    if can_fly and final_to != start:
        return Move.fly_piece(Go(start, final_to))
    if final_to != start:
        return Move.go_piece(Go(start, final_to))
    return Move.go_piece(Go(start, start))


def move_to_dict(mv: Move) -> dict:
    """Move转字典"""
    if mv.is_put:
        return {'act': 'put_piece', 'point': {'r': mv.point.row, 'c': mv.point.col}}
    elif mv.is_go:
        return {'act': 'is_go', 'go': {'r': mv.go_to.go.row, 'c': mv.go_to.go.col},
                'to': {'r': mv.go_to.to.row, 'c': mv.go_to.to.col}}
    elif mv.is_fly:
        return {'act': 'fly', 'go': {'r': mv.go_to.go.row, 'c': mv.go_to.go.col},
                'to': {'r': mv.go_to.to.row, 'c': mv.go_to.to.col}}
    elif mv.is_skip_eat:
        se = mv.skip_eat_points
        return {'act': 'skip_move', 'go': {'r': se.go.row, 'c': se.go.col},
                'to': {'r': se.to.row, 'c': se.to.col}}
    elif mv.is_skip_eat_seq:
        seq = mv.skip_eat_points
        return {'act': 'skip_eat_seq', 'seq': [
            {'go': {'r': s.go.row, 'c': s.go.col}, 'to': {'r': s.to.row, 'c': s.to.col}}
            for s in seq
        ]}
    return {}


# ========== 价值标签生成器 ==========
class ValueLabeler:
    """价值标签生成器"""
    
    def __init__(self, dyt_agent: Optional[DyTAgent] = None, gamma: float = 0.95):
        self.dyt_agent = dyt_agent
        self.gamma = gamma
        self.stats = {
            'from_result': 0,
            'from_rollout': 0,
            'from_heuristic': 0,
            'total': 0
        }
    
    def label_game(self, line: str) -> List[dict]:
        """为一局对局生成所有样本（带value标签）"""
        try:
            game_states, moves = self._parse_game(line)
            if len(game_states) < 2:
                return []
            
            # 判断对局结果
            final_state = game_states[-1]
            winner = final_state.winner()
            
            # 生成样本
            samples = []
            for t, (state, actual_move) in enumerate(zip(game_states[:-1], moves)):
                try:
                    sample = self._create_sample(state, actual_move, t, len(game_states), winner)
                    if sample is not None:
                        samples.append(sample)
                except Exception:
                    continue
            
            return samples
            
        except Exception as e:
            return []
    
    def _parse_game(self, line: str) -> Tuple[List[GameState], List[Move]]:
        """解析对局，返回所有状态和动作"""
        try:
            head, rest = line.split('Stage[1]:', 1)
        except ValueError:
            return [], []
        
        if ';Stage[2]:' in rest:
            stage1_str, rest2 = rest.split(';Stage[2]:', 1)
            stage2_str = rest2
        else:
            stage1_str, stage2_str = rest, ''
        
        states = []
        moves = []
        state = GameState.new_game(14)
        states.append(copy.deepcopy(state))
        
        # Stage 1: 布局
        for player, pt in parse_stage1_tokens(stage1_str):
            try:
                move = Move.put_piece(pt)
                moves.append(move)
                state = state.apply_move(move)
                states.append(copy.deepcopy(state))
            except Exception:
                break
        
        # Stage 2: 对战
        segments = parse_stage2_segments(stage2_str)
        for seg in segments:
            try:
                move = build_move_from_segment(state, seg)
                moves.append(move)
                state = state.apply_move(move)
                states.append(copy.deepcopy(state))
            except Exception:
                break
        
        return states, moves
    
    def _create_sample(
        self, 
        state: GameState, 
        actual_move: Move, 
        step_num: int,
        total_steps: int,
        winner: Optional[Player]
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
        legal_moves_objs = [self._dec_to_move(d) for d in legal_decisions]
        cand_dicts = [move_to_dict(m) for m in legal_moves_objs if m is not None]
        flying = state._is_flying_stage() if hasattr(state, '_is_flying_stage') else False
        cand_feats = build_features_for_candidates(cand_dicts, phase_id, flying)
        
        # 匹配标签索引
        label_idx = self._match_move(actual_move, legal_moves_objs)
        
        if label_idx < 0:
            return None
        
        # ⭐ 生成价值标签
        value = self._compute_value(state, step_num, total_steps, winner)
        
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
        winner: Optional[Player]
    ) -> float:
        """计算价值标签（三级策略）"""
        
        # 级别 1: 使用对局真实结果
        if winner is not None:
            current_player = state.next_player
            steps_to_end = total_steps - step_num
            discount = self.gamma ** steps_to_end
            
            if winner == current_player:
                value = discount
            elif winner == current_player.other:
                value = -discount
            else:
                value = 0.0
            
            self.stats['from_result'] += 1
            return float(np.clip(value, -1.0, 1.0))
        
        # 级别 2: DyT Rollout
        if self.dyt_agent is not None:
            try:
                value = self._rollout_with_dyt(state, max_steps=30)
                self.stats['from_rollout'] += 1
                return value
            except Exception:
                pass
        
        # 级别 3: 启发式评估
        value = self._heuristic_evaluate(state)
        self.stats['from_heuristic'] += 1
        return value
    
    def _rollout_with_dyt(self, state: GameState, max_steps: int = 30) -> float:
        """使用DyT模型进行Rollout"""
        current_state = copy.deepcopy(state)
        original_player = state.next_player
        
        for step in range(max_steps):
            if current_state.is_over():
                winner = current_state.winner()
                return self._compute_reward(winner, original_player)
            
            try:
                move, _ = self.dyt_agent.select_move(current_state)
                if move is None:
                    break
                current_state = current_state.apply_move(move)
            except Exception:
                break
        
        # 如果仍未结束，使用启发式
        return self._heuristic_evaluate(current_state, original_player)
    
    def _heuristic_evaluate(self, state: GameState, player: Optional[Player] = None) -> float:
        """启发式评估局面"""
        if player is None:
            player = state.next_player
        
        # 棋子数差异
        my_pieces = state.board.get_player_total(player)
        opp_pieces = state.board.get_player_total(player.other)
        piece_score = (my_pieces - opp_pieces) / 98.0
        
        # 褡裢数差异
        try:
            my_dalians = count_independent_dalians(state.board, player)
            opp_dalians = count_independent_dalians(state.board, player.other)
            dalian_score = (my_dalians - opp_dalians) / 2.0
        except:
            dalian_score = 0.0
        
        # 综合评分
        score = 0.4 * piece_score + 0.6 * dalian_score
        return float(np.clip(np.tanh(score), -1.0, 1.0))
    
    def _compute_reward(self, winner: Optional[Player], player: Player) -> float:
        """根据胜负计算奖励"""
        if winner == player:
            return 1.0
        elif winner == player.other:
            return -1.0
        else:
            return 0.0
    
    def _dec_to_move(self, dec) -> Optional[Move]:
        """Decision转Move"""
        try:
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
            elif dec.act == 'eat_point':
                return Move.eat(dec.points)
        except:
            pass
        return None
    
    def _match_move(self, actual_move: Move, candidates: List[Move]) -> int:
        """匹配动作索引"""
        actual_dict = move_to_dict(actual_move)
        for i, cand_mv in enumerate(candidates):
            if cand_mv is not None and move_to_dict(cand_mv) == actual_dict:
                return i
        return -1


def main():
    parser = argparse.ArgumentParser(description='生成Value标签')
    parser.add_argument('--csv', default='data/2024.5.13(clean1).csv', help='输入CSV文件')
    parser.add_argument('--model', default='exp/real_data_no_augment/best_sft.pth', help='DyT模型路径')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--out_dir', default='exp/datasets_with_value', help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--skip_rollout', action='store_true', help='跳过DyT rollout，仅使用对局结果')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载DyT模型
    dyt_agent = None
    if not args.skip_rollout:
        print(f"加载DyT模型: {args.model}")
        try:
            dyt_agent = DyTAgent(args.model, device=args.device)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️  模型加载失败: {e}")
            print("⚠️  将仅使用对局结果和启发式评估")
    else:
        print("⚠️  跳过DyT rollout模式")
    
    # 创建标签器
    labeler = ValueLabeler(dyt_agent, gamma=args.gamma)
    
    # 处理数据
    print(f"\n处理对局数据: {args.csv}")
    all_samples = []
    
    with open(args.csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc='生成Value标签'):
        line = line.strip()
        if not line:
            continue
        
        samples = labeler.label_game(line)
        all_samples.extend(samples)
    
    print(f'\n✅ 收集到 {len(all_samples)} 个样本（带Value标签）')
    
    # 统计
    print(f"\n标签来源统计:")
    total = max(labeler.stats['total'], 1)
    print(f"  对局结果: {labeler.stats['from_result']} ({labeler.stats['from_result']/total*100:.1f}%)")
    print(f"  DyT Rollout: {labeler.stats['from_rollout']} ({labeler.stats['from_rollout']/total*100:.1f}%)")
    print(f"  启发式: {labeler.stats['from_heuristic']} ({labeler.stats['from_heuristic']/total*100:.1f}%)")
    
    # 拆分训练/验证集
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = int(len(all_samples) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    
    # 保存
    train_path = os.path.join(args.out_dir, 'train_with_value.pt')
    val_path = os.path.join(args.out_dir, 'val_with_value.pt')
    
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    
    print(f'\n✅ 保存完成:')
    print(f'  训练集: {train_path} ({len(train_samples)} 样本)')
    print(f'  验证集: {val_path} ({len(val_samples)} 样本)')
    
    # 保存统计信息
    stats_path = os.path.join(args.out_dir, 'value_label_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Value标签生成统计\n")
        f.write(f"==================\n\n")
        f.write(f"总样本数: {len(all_samples)}\n")
        f.write(f"训练集: {len(train_samples)}\n")
        f.write(f"验证集: {len(val_samples)}\n\n")
        
        f.write(f"标签来源:\n")
        f.write(f"  对局结果: {labeler.stats['from_result']}\n")
        f.write(f"  DyT Rollout: {labeler.stats['from_rollout']}\n")
        f.write(f"  启发式: {labeler.stats['from_heuristic']}\n\n")
        
        # 价值分布统计
        values = [s['value'] for s in all_samples]
        f.write(f"Value分布:\n")
        f.write(f"  均值: {np.mean(values):.3f}\n")
        f.write(f"  标准差: {np.std(values):.3f}\n")
        f.write(f"  最小值: {np.min(values):.3f}\n")
        f.write(f"  最大值: {np.max(values):.3f}\n")
        
        # 分布直方图
        f.write(f"\nValue分布直方图:\n")
        hist, bins = np.histogram(values, bins=10, range=(-1.0, 1.0))
        for i in range(len(hist)):
            bar = '#' * int(hist[i] / max(hist) * 50)
            f.write(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:6d} {bar}\n")
    
    print(f'  统计信息: {stats_path}')
    print(f'\n🎉 完成！')


if __name__ == '__main__':
    main()

