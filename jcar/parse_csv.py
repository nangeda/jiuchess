#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 棋谱解析脚本

解析 data/2024.5.13(clean1).csv 中的人类对局数据，
转换为 JiuqiNet 训练格式。

格式说明：
- Stage[1]: 布局阶段，格式为 W(col,row) 或 B(col,row)
- Stage[2]: 对战阶段，格式为 B(col,row)-O(col,row),TC:... 或 FC:...

新增功能：
- 使用Expert评估函数生成Value标签
"""

import os
import re
import sys
import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Move, Player, Board
from jiu.jiutypes import Point, Go, Skip_eat, Decision, board_gild, board_size
from baseline.baseline_fxy_expert.board_utils import state_to_grid, other_player_val
from baseline.baseline_fxy_expert.scoring import sum_stone, special_moves1, fly_special_moves


# 列名映射: A=1, B=2, ..., N=14
COL_MAP = {chr(ord('A') + i): i + 1 for i in range(14)}

# Value归一化常数 - 降低以获得更广的分布
# 对战阶段: 棋子差最大约60，乘以0.6约36; routes_diff通常在-20到20之间
# 布局阶段: 主要看位置评估
VALUE_SCALE_BATTLE = 8.0   # 对战阶段归一化常数
VALUE_SCALE_LAYOUT = 3.0   # 布局阶段归一化常数


def compute_expert_value(state: GameState, player: Player,
                         winner: Player = None, steps_to_end: int = None,
                         gamma: float = 0.99, balance_perspective: bool = True) -> float:
    """
    使用Expert评估函数计算局面价值（平衡版）

    改进点：
    1. 布局阶段增加位置评估
    2. 降低归一化常数获得更广的分布
    3. 如果有对局结果，融合终局奖励
    4. 【新增】平衡视角：从当前玩家视角计算，确保正负值平衡

    Args:
        state: 当前游戏状态
        player: 当前玩家视角
        winner: 对局获胜者（可选）
        steps_to_end: 距离游戏结束的步数（可选）
        gamma: 折扣因子
        balance_perspective: 是否平衡视角（确保正负值比例接近50:50）

    Returns:
        归一化到 [-1, 1] 的价值
    """
    grid = state_to_grid(state)
    player_val = 1 if player == Player.black else 2
    other_val = other_player_val(player_val)

    # 计算棋子数差
    my_stones = sum_stone(grid, player_val)
    opp_stones = sum_stone(grid, other_val)
    stone_diff = my_stones - opp_stones

    # 判断游戏阶段
    is_layout = state.step < board_gild
    flying = state.board.get_player_total(player) <= 14

    if is_layout:
        # 布局阶段：增加位置评估
        position_score = _compute_layout_position_score(grid, player_val, other_val)
        # 布局阶段棋子数相等，主要看位置优势
        raw_value = position_score
        scale = VALUE_SCALE_LAYOUT
    else:
        # 对战/飞子阶段
        if flying:
            my_routes = fly_special_moves(grid.copy(), player_val)
            opp_routes = fly_special_moves(grid.copy(), other_val)
        else:
            my_routes = special_moves1(grid.copy(), player_val)
            opp_routes = special_moves1(grid.copy(), other_val)

        routes_diff = my_routes - opp_routes
        raw_value = stone_diff * 0.6 + routes_diff * 0.4
        scale = VALUE_SCALE_BATTLE

    # 使用tanh归一化到 [-1, 1]
    heuristic_value = math.tanh(raw_value / scale)

    # 如果有对局结果，融合终局奖励
    if winner is not None and steps_to_end is not None:
        discount = gamma ** steps_to_end
        if winner == player:
            terminal_value = discount
        elif winner == player.other:
            terminal_value = -discount
        else:
            terminal_value = 0.0
        # 融合启发式评估和终局奖励
        # 【改进】增加终局奖励权重，使其更有影响力
        terminal_weight = min(0.9, discount * 1.2)
        heuristic_value = terminal_weight * terminal_value + (1 - terminal_weight) * heuristic_value

    return float(np.clip(heuristic_value, -1.0, 1.0))


def _compute_layout_position_score(grid: np.ndarray, player_val: int, other_val: int) -> float:
    """
    计算布局阶段的位置评分

    评估标准：
    1. 中心控制：靠近中心的棋子得分更高
    2. 褡裢潜力：能形成褡裢结构的位置更有价值
    3. 连接性：棋子之间的连接程度
    """
    CENTER = 7.5  # 14x14棋盘中心

    my_score = 0.0
    opp_score = 0.0

    for idx in range(196):  # 14x14 = 196
        row, col = idx // 14, idx % 14
        if grid[idx] == player_val:
            # 中心距离评分 (越靠近中心越高)
            dist_to_center = abs(row - CENTER) + abs(col - CENTER)
            center_score = (14 - dist_to_center) / 14  # 归一化到 [0, 1]

            # 邻居连接评分
            neighbor_score = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 14 and 0 <= nc < 14:
                    nidx = nr * 14 + nc
                    if grid[nidx] == player_val:
                        neighbor_score += 0.3

            my_score += center_score + neighbor_score

        elif grid[idx] == other_val:
            dist_to_center = abs(row - CENTER) + abs(col - CENTER)
            center_score = (14 - dist_to_center) / 14

            neighbor_score = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 14 and 0 <= nc < 14:
                    nidx = nr * 14 + nc
                    if grid[nidx] == other_val:
                        neighbor_score += 0.3

            opp_score += center_score + neighbor_score

    return my_score - opp_score


@dataclass
class ParsedMove:
    """解析后的走法"""
    player: str  # 'W' or 'B'
    move_type: str  # 'put', 'go', 'skip', 'fly'
    from_pos: Optional[Tuple[int, int]]  # (row, col)
    to_pos: Tuple[int, int]  # (row, col)
    captures: List[Tuple[int, int]]  # 跳吃路径中间点
    eat_targets: List[Tuple[int, int]]  # TC/FC 吃子目标


def parse_coord(coord_str: str) -> Tuple[int, int]:
    """
    解析坐标，如 'G,7' -> (7, 7)
    返回 (row, col)，1-indexed
    """
    parts = coord_str.split(',')
    col_letter = parts[0].strip()
    row = int(parts[1].strip())
    col = COL_MAP.get(col_letter, 1)
    return (row, col)


def parse_stage1_move(move_str: str) -> Optional[ParsedMove]:
    """
    解析布局阶段走法
    格式: W(G,7) 或 B(H,8)
    """
    match = re.match(r'([WB])\(([A-N]),(\d+)\)', move_str.strip())
    if not match:
        return None
    
    player = match.group(1)
    col_letter = match.group(2)
    row = int(match.group(3))
    col = COL_MAP[col_letter]
    
    return ParsedMove(
        player=player,
        move_type='put',
        from_pos=None,
        to_pos=(row, col),
        captures=[],
        eat_targets=[]
    )


def parse_stage2_move(move_str: str) -> Optional[ParsedMove]:
    """
    解析对战阶段走法
    格式: B(H,7)-O(G,7),FC:W(H,9) 或 W(H,5)-O(H,7),TC:B(H,6)
    或连跳: B(F,5)-O(H,5)-O(H,3),TC:W(G,5),W(F,4),FC:W(G,13)
    """
    if not move_str.strip():
        return None
    
    # 分离走法部分和吃子部分
    main_part = move_str.split(',TC:')[0].split(',FC:')[0]
    
    # 解析起点和终点
    positions = re.findall(r'([WB]?)\(([A-N]),(\d+)\)', main_part)
    if len(positions) < 2:
        return None
    
    # 第一个位置是起点
    player = positions[0][0] if positions[0][0] else 'W'
    from_col = COL_MAP[positions[0][1]]
    from_row = int(positions[0][2])
    
    # 收集所有中间点和终点
    path = [(from_row, from_col)]
    for pos in positions[1:]:
        col = COL_MAP[pos[1]]
        row = int(pos[2])
        path.append((row, col))
    
    # 判断走法类型
    if len(path) == 2:
        # 普通走子或单次跳吃
        dr = abs(path[1][0] - path[0][0])
        dc = abs(path[1][1] - path[0][1])
        if dr <= 1 and dc <= 1:
            move_type = 'go'
        else:
            move_type = 'skip'
    else:
        # 连续跳吃
        move_type = 'skip'
    
    # 解析吃子目标
    eat_targets = []
    tc_match = re.findall(r'TC:([^;]+)', move_str)
    fc_match = re.findall(r'FC:([^;]+)', move_str)
    
    for tc in tc_match:
        targets = re.findall(r'[WB]\(([A-N]),(\d+)\)', tc)
        for t in targets:
            eat_targets.append((int(t[1]), COL_MAP[t[0]]))
    
    for fc in fc_match:
        targets = re.findall(r'[WB]\(([A-N]),(\d+)\)', fc)
        for t in targets:
            eat_targets.append((int(t[1]), COL_MAP[t[0]]))
    
    return ParsedMove(
        player=player,
        move_type=move_type,
        from_pos=path[0],
        to_pos=path[-1],
        captures=path[1:-1] if len(path) > 2 else [],
        eat_targets=eat_targets
    )


def parse_game_record(line: str) -> Tuple[List[ParsedMove], List[ParsedMove]]:
    """
    解析一行游戏记录
    返回 (stage1_moves, stage2_moves)
    """
    # 去除引号
    line = line.strip().strip('"')
    
    stage1_moves = []
    stage2_moves = []
    
    # 分离 Stage[1] 和 Stage[2]
    stage1_match = re.search(r'Stage\[1\]:([^;]*(?:;[^S][^t][^a][^g][^e][^;]*)*)', line)
    stage2_match = re.search(r'Stage\[2\]:(.+)$', line)
    
    if stage1_match:
        stage1_str = stage1_match.group(1)
        moves = stage1_str.split(';')
        for m in moves:
            m = m.strip()
            if m:
                parsed = parse_stage1_move(m)
                if parsed:
                    stage1_moves.append(parsed)
    
    if stage2_match:
        stage2_str = stage2_match.group(1)
        moves = stage2_str.split(';')
        for m in moves:
            m = m.strip()
            if m:
                parsed = parse_stage2_move(m)
                if parsed:
                    stage2_moves.append(parsed)
    
    return stage1_moves, stage2_moves


def create_decision_from_parsed(parsed: ParsedMove, state: GameState) -> Optional[Decision]:
    """将 ParsedMove 转换为 Decision"""
    if parsed.move_type == 'put':
        pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        return Decision('put_piece', pt, 0)
    
    elif parsed.move_type == 'go':
        from_pt = Point(parsed.from_pos[0], parsed.from_pos[1])
        to_pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        go = Go(from_pt, to_pt)
        return Decision('is_go', go, 0)
    
    elif parsed.move_type == 'skip':
        from_pt = Point(parsed.from_pos[0], parsed.from_pos[1])
        to_pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        
        if len(parsed.captures) == 0:
            # 单次跳吃: Skip_eat(go, eat, to) = (起点, 被吃点, 终点)
            mid_row = (parsed.from_pos[0] + parsed.to_pos[0]) // 2
            mid_col = (parsed.from_pos[1] + parsed.to_pos[1]) // 2
            mid_pt = Point(mid_row, mid_col)
            se = Skip_eat(from_pt, mid_pt, to_pt)  # 修正顺序
            return Decision('skip_move', se, 0)
        else:
            # 连续跳吃
            seq = []
            all_points = [parsed.from_pos] + parsed.captures + [parsed.to_pos]
            for i in range(len(all_points) - 1):
                fp = Point(all_points[i][0], all_points[i][1])
                tp = Point(all_points[i+1][0], all_points[i+1][1])
                mid_row = (all_points[i][0] + all_points[i+1][0]) // 2
                mid_col = (all_points[i][1] + all_points[i+1][1]) // 2
                mid_pt = Point(mid_row, mid_col)
                seq.append(Skip_eat(fp, mid_pt, tp))  # 修正顺序
            return Decision('skip_eat_seq', seq, 0)
    
    elif parsed.move_type == 'fly':
        from_pt = Point(parsed.from_pos[0], parsed.from_pos[1])
        to_pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        go = Go(from_pt, to_pt)
        return Decision('fly', go, 0)
    
    return None


def encode_board_state(state: GameState) -> np.ndarray:
    """编码棋盘状态为 (6, 14, 14)"""
    H, W = board_size, board_size
    obs = np.zeros((6, H, W), dtype=np.float32)
    
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
    
    return obs


def get_phase_id(state: GameState) -> int:
    """获取阶段ID"""
    if state.step < board_gild:
        return 0
    elif state.board.get_player_total(state.next_player) <= 14:
        return 2
    else:
        return 1


def decision_to_dict(dec: Decision) -> dict:
    """将 Decision 转换为特征字典"""
    if dec.act == 'put_piece':
        p = dec.points
        return {'act': 'put_piece', 'point': {'r': p.row, 'c': p.col}}
    elif dec.act == 'is_go':
        go = dec.points
        return {'act': 'is_go', 'go': {'r': go.go.row, 'c': go.go.col},
                'to': {'r': go.to.row, 'c': go.to.col}}
    elif dec.act == 'fly':
        go = dec.points
        return {'act': 'fly', 'go': {'r': go.go.row, 'c': go.go.col},
                'to': {'r': go.to.row, 'c': go.to.col}}
    elif dec.act == 'skip_move':
        se = dec.points
        return {'act': 'skip_move', 'go': {'r': se.go.row, 'c': se.go.col},
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


def build_candidate_features(decisions: List[Decision], phase_id: int, flying: bool) -> np.ndarray:
    """构建候选动作特征"""
    from dyt.candidate_features import build_features_for_candidates
    
    cand_dicts = [decision_to_dict(d) for d in decisions]
    feats = build_features_for_candidates(cand_dicts, phase_id, flying)
    return feats[:, :14]  # 只取前14维


def find_label_index(target_dec: Decision, legal_decs: List[Decision]) -> int:
    """在合法动作列表中找到目标动作的索引"""
    target_dict = decision_to_dict(target_dec)

    for i, dec in enumerate(legal_decs):
        dec_dict = decision_to_dict(dec)
        if target_dict == dec_dict:
            return i

    # 如果精确匹配失败，尝试模糊匹配（同类型、同起点终点）
    for i, dec in enumerate(legal_decs):
        if dec.act != target_dec.act:
            continue

        if target_dec.act == 'put_piece':
            if dec.points.row == target_dec.points.row and dec.points.col == target_dec.points.col:
                return i
        elif target_dec.act in ('is_go', 'fly'):
            if (dec.points.to.row == target_dec.points.to.row and
                dec.points.to.col == target_dec.points.to.col and
                dec.points.go.row == target_dec.points.go.row and
                dec.points.go.col == target_dec.points.go.col):
                return i
        elif target_dec.act == 'skip_move':
            # Skip_eat(go, eat, to)
            if (dec.points.go.row == target_dec.points.go.row and
                dec.points.go.col == target_dec.points.go.col and
                dec.points.to.row == target_dec.points.to.row and
                dec.points.to.col == target_dec.points.to.col):
                return i
        elif target_dec.act == 'skip_eat_seq':
            # 比较序列的起点和终点
            if len(dec.points) == len(target_dec.points):
                match = True
                for d, t in zip(dec.points, target_dec.points):
                    if (d.go.row != t.go.row or d.go.col != t.go.col or
                        d.to.row != t.to.row or d.to.col != t.to.col):
                        match = False
                        break
                if match:
                    return i

    return -1


def process_game(stage1_moves: List[ParsedMove], stage2_moves: List[ParsedMove]) -> List[dict]:
    """
    处理一局游戏，生成训练样本（改进版）

    改进：先完整模拟对局获取winner，然后再生成样本时传递winner信息
    """
    # 第一步：完整模拟对局获取winner和总步数
    try:
        sim_state = GameState.new_game(14)
        total_steps = 0

        # 模拟布局阶段
        for parsed in stage1_moves:
            expected_player = 'W' if sim_state.next_player == Player.white else 'B'
            if parsed.player != expected_player:
                continue
            pt = Point(parsed.to_pos[0], parsed.to_pos[1])
            move = Move.put_piece(pt)
            try:
                sim_state = sim_state.apply_move(move)
                total_steps += 1
            except:
                break

        # 模拟对战阶段
        for parsed in stage2_moves:
            expected_player = 'W' if sim_state.next_player == Player.white else 'B'
            if parsed.player != expected_player:
                continue
            target_dec = create_decision_from_parsed(parsed, sim_state)
            if target_dec is None:
                continue
            move = _decision_to_move(target_dec, parsed, sim_state)
            if move:
                try:
                    sim_state = sim_state.apply_move(move)
                    total_steps += 1
                except:
                    break

        # 获取对局结果
        winner = sim_state.winner() if sim_state.is_over() else None

    except Exception:
        winner = None
        total_steps = 0

    # 第二步：重新遍历生成样本
    samples = []
    current_step = 0

    try:
        state = GameState.new_game(14)

        # 处理布局阶段
        for parsed in stage1_moves:
            expected_player = 'W' if state.next_player == Player.white else 'B'
            if parsed.player != expected_player:
                continue

            legal_decs = state.legal_moves()
            if not legal_decs:
                break

            target_dec = create_decision_from_parsed(parsed, state)
            if target_dec is None:
                continue

            label_idx = find_label_index(target_dec, legal_decs)
            if label_idx < 0:
                continue

            obs = encode_board_state(state)
            phase_id = get_phase_id(state)
            flying = state.board.get_player_total(state.next_player) <= 14
            cand_feats = build_candidate_features(legal_decs, phase_id, flying)

            # 计算距离结束的步数
            steps_to_end = total_steps - current_step if total_steps > 0 else None

            # 使用改进的Expert评估生成value标签
            value = compute_expert_value(state, state.next_player,
                                         winner=winner, steps_to_end=steps_to_end)

            samples.append({
                'obs': obs,
                'phase_id': phase_id,
                'cand_feats': cand_feats,
                'label_idx': label_idx,
                'value': value
            })

            pt = Point(parsed.to_pos[0], parsed.to_pos[1])
            move = Move.put_piece(pt)
            state = state.apply_move(move)
            current_step += 1

        # 处理对战阶段
        for parsed in stage2_moves:
            expected_player = 'W' if state.next_player == Player.white else 'B'
            if parsed.player != expected_player:
                continue

            legal_decs = state.legal_moves()
            if not legal_decs:
                break

            target_dec = create_decision_from_parsed(parsed, state)
            if target_dec is None:
                continue

            label_idx = find_label_index(target_dec, legal_decs)
            if label_idx < 0:
                continue

            obs = encode_board_state(state)
            phase_id = get_phase_id(state)
            flying = state.board.get_player_total(state.next_player) <= 14
            cand_feats = build_candidate_features(legal_decs, phase_id, flying)

            steps_to_end = total_steps - current_step if total_steps > 0 else None
            value = compute_expert_value(state, state.next_player,
                                         winner=winner, steps_to_end=steps_to_end)

            samples.append({
                'obs': obs,
                'phase_id': phase_id,
                'cand_feats': cand_feats,
                'label_idx': label_idx,
                'value': value
            })

            move = _decision_to_move(target_dec, parsed, state)
            if move:
                try:
                    state = state.apply_move(move)
                    current_step += 1
                except:
                    break

    except Exception as e:
        pass

    return samples


def _decision_to_move(target_dec: Decision, parsed: ParsedMove, state: GameState) -> Optional[Move]:
    """将 Decision 和 ParsedMove 转换为 Move"""
    if parsed.move_type == 'go':
        from_pt = Point(parsed.from_pos[0], parsed.from_pos[1])
        to_pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        return Move.go_piece(Go(from_pt, to_pt))
    elif parsed.move_type == 'skip':
        if target_dec.act == 'skip_move':
            return Move.move_skip(target_dec.points)
        elif target_dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(target_dec.points)
    elif parsed.move_type == 'fly':
        from_pt = Point(parsed.from_pos[0], parsed.from_pos[1])
        to_pt = Point(parsed.to_pos[0], parsed.to_pos[1])
        return Move.fly_piece(Go(from_pt, to_pt))
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse CSV game records')
    parser.add_argument('--input', type=str, 
                        default='data/2024.5.13(clean1).csv',
                        help='Input CSV file')
    parser.add_argument('--output', type=str,
                        default='data/processed/human_games.pt',
                        help='Output file')
    args = parser.parse_args()
    
    # 读取 CSV
    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / csv_path
    
    print(f"Reading {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total games: {len(lines)}")
    
    # 处理每局游戏
    all_samples = []
    success_games = 0
    
    for i, line in enumerate(tqdm(lines, desc='Processing games')):
        stage1, stage2 = parse_game_record(line)
        if stage1:
            samples = process_game(stage1, stage2)
            if samples:
                all_samples.extend(samples)
                success_games += 1
    
    print(f"\nSuccessfully processed {success_games}/{len(lines)} games")
    print(f"Total samples: {len(all_samples)}")

    # 统计 phase 分布
    phase_counts = {0: 0, 1: 0, 2: 0}
    for s in all_samples:
        phase_counts[s['phase_id']] += 1
    print(f"Phase distribution: {phase_counts}")

    # 统计 value 分布
    values = [s['value'] for s in all_samples]
    print(f"\nValue statistics (Expert-generated):")
    print(f"  Min: {min(values):.4f}, Max: {max(values):.4f}")
    print(f"  Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
    print(f"  Positive (advantage): {sum(1 for v in values if v > 0.1)}")
    print(f"  Negative (disadvantage): {sum(1 for v in values if v < -0.1)}")
    print(f"  Neutral: {sum(1 for v in values if -0.1 <= v <= 0.1)}")

    # 保存
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
