"""
构建久棋训练数据集（无需winner标签版本）
1. 解析 CSV 复盘每局
2. 提取每步：(obs, phase_id, candidates, label_idx, value)
3. value统一设为0（因为没有胜负标签）
4. 保存为 train.pt / val.pt
"""
import argparse
import copy
import os
import re
import sys
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiu.jiutypes import Player, Point, dataset_coord_to_point, board_size
from jiu.jiuboard_fast import GameState, Move, Skip_eat, Go, board_gild
from dyt.candidate_features import build_features_for_candidates

COORD_RE = re.compile(r"\(([A-N]),\s*(\d{1,2})\)")

def parse_stage1_tokens(stage1: str) -> List[Tuple[Player, Point]]:
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
    pts = []
    for letter, row in COORD_RE.findall(s):
        try:
            pts.append(dataset_coord_to_point((letter, row)))
        except ValueError:
            continue
    return pts

def parse_stage2_segments(stage2: str) -> List[dict]:
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
    return Point((a.row + b.row) // 2, (a.col + b.col) // 2)

def build_move_from_segment(state: GameState, seg: dict) -> Move:
    me: Player = seg['player']
    start: Point = seg['start']
    tos: List[Point] = seg['tos']
    tc: List[Point] = seg['tc']
    
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
    
    final_to = tos[-1] if tos else start
    can_fly = (state.board.get_player_total(me) <= state.board.num_rows)
    if can_fly and final_to != start:
        return Move.fly_piece(Go(start, final_to))
    if final_to != start:
        return Move.go_piece(Go(start, final_to))
    return Move.go_piece(Go(start, start))

def encode_board_state(state: GameState, history: deque) -> np.ndarray:
    """返回 (6, H, W) 的棋盘编码"""
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
    
    me = state.next_player
    hist_list = list(history)[-3:]
    for i, past_board in enumerate(hist_list):
        for r in range(1, H + 1):
            for c in range(1, W + 1):
                pt = Point(r, c)
                if past_board.get(pt) == me:
                    obs[3 + i, r - 1, c - 1] = 1.0
    return obs

def move_to_dict(mv: Move) -> dict:
    """将 Move 转为字典"""
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

def enumerate_legal_moves(state: GameState) -> List[Move]:
    """枚举合法走法"""
    decisions = state.legal_moves()
    moves: List[Move] = []
    for d in decisions:
        if d.act == 'put_piece':
            moves.append(Move.put_piece(d.points))
        elif d.act == 'is_go':
            moves.append(Move.go_piece(d.points))
        elif d.act == 'fly':
            moves.append(Move.fly_piece(d.points))
        elif d.act == 'skip_move':
            moves.append(Move.move_skip(d.points))
        elif d.act == 'skip_eat_seq':
            moves.append(Move.move_skip_seq(d.points))
    return moves

def match_move(actual_mv: Move, candidates: List[Move]) -> int:
    """返回 actual_mv 在 candidates 中的索引"""
    actual_dict = move_to_dict(actual_mv)
    for i, cand_mv in enumerate(candidates):
        if move_to_dict(cand_mv) == actual_dict:
            return i
    return -1

def replay_and_collect(line: str) -> List[dict]:
    """复盘一局，收集所有样本"""
    try:
        head, rest = line.split('Stage[1]:', 1)
    except ValueError:
        return []
    if ';Stage[2]:' in rest:
        stage1_str, rest2 = rest.split(';Stage[2]:', 1)
        stage2_str = rest2
    else:
        stage1_str, stage2_str = rest, ''
    
    state = GameState.new_game(board_size)
    history = deque(maxlen=3)
    samples: List[dict] = []
    
    # Stage 1: 布局
    for player, pt in parse_stage1_tokens(stage1_str):
        if state.step >= 2:  # 跳过前2步
            try:
                obs = encode_board_state(state, history)
                phase_id = 0
                flying = False
                legal_decisions = state.legal_moves()
                
                if legal_decisions:
                    cand_dicts = []
                    for dec in legal_decisions:
                        cand_dict = {
                            'act': 'put_piece',
                            'point': {'r': dec.points.row, 'c': dec.points.col}
                        }
                        cand_dicts.append(cand_dict)
                    
                    cand_feats = build_features_for_candidates(cand_dicts, phase_id, flying)
                    
                    label_idx = -1
                    for i, dec in enumerate(legal_decisions):
                        if dec.points == pt:
                            label_idx = i
                            break
                    
                    if label_idx >= 0:
                        samples.append({
                            'obs': obs,
                            'phase_id': phase_id,
                            'cand_feats': cand_feats,
                            'label_idx': label_idx,
                            'value': 0.0,  # 没有标签，设为0
                        })
            except Exception:
                pass
        
        state = state.apply_move(Move.put_piece(pt))
        history.append(copy.deepcopy(state.board))
    
    # Stage 2: 对战阶段
    segments = parse_stage2_segments(stage2_str)
    for seg in segments:
        me: Player = seg['player']
        if state.next_player != me:
            state.next_player = me
        
        actual_move = build_move_from_segment(state, seg)
        
        try:
            legal_moves = enumerate_legal_moves(state)
        except Exception:
            state = state.apply_move(actual_move)
            history.append(copy.deepcopy(state.board))
            continue
        
        if not legal_moves:
            state = state.apply_move(actual_move)
            history.append(copy.deepcopy(state.board))
            continue
        
        obs = encode_board_state(state, history)
        phase_id = 0 if state.step < board_gild else (2 if state._is_flying_stage() else 1)
        flying = state._is_flying_stage()
        
        cand_dicts = [move_to_dict(m) for m in legal_moves]
        cand_feats = build_features_for_candidates(cand_dicts, phase_id, flying)
        
        label_idx = match_move(actual_move, legal_moves)
        
        if label_idx >= 0:
            samples.append({
                'obs': obs,
                'phase_id': phase_id,
                'cand_feats': cand_feats,
                'label_idx': label_idx,
                'value': 0.0,  # 没有标签，设为0
            })
        
        try:
            state = state.apply_move(actual_move)
            history.append(copy.deepcopy(state.board))
        except Exception:
            break
    
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/2024.5.13(clean1).csv')
    parser.add_argument('--out_dir', default='exp/datasets')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("构建久棋数据集（无胜负标签版本）")
    print("=" * 60)
    print(f"\n输入CSV: {args.csv}")
    print(f"输出目录: {args.out_dir}")
    print(f"验证集比例: {args.val_ratio}")
    
    # 读取 CSV 并复盘
    all_samples: List[dict] = []
    error_count = 0
    
    with open(args.csv, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"\n总对局数: {len(lines)}")
    print("开始复盘和提取样本...")
    
    for idx, line in enumerate(tqdm(lines, desc='处理对局'), start=1):
        try:
            samples = replay_and_collect(line)
            all_samples.extend(samples)
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 只打印前5个错误
                print(f"\n警告: 第{idx}局解析失败: {e}")
    
    print(f'\n收集到 {len(all_samples)} 个样本')
    if error_count > 0:
        print(f'警告: {error_count} 局对局解析失败（已跳过）')
    
    if len(all_samples) == 0:
        print("错误: 没有成功提取任何样本！")
        return
    
    # 拆分训练集和验证集
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = int(len(all_samples) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    
    # 保存
    train_path = os.path.join(args.out_dir, 'train.pt')
    val_path = os.path.join(args.out_dir, 'val.pt')
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    
    print(f'\n✅ 保存 {len(train_samples)} 个训练样本到 {train_path}')
    print(f'✅ 保存 {len(val_samples)} 个验证样本到 {val_path}')
    
    # 统计信息
    print(f"\n{'=' * 60}")
    print("数据集统计")
    print(f"{'=' * 60}")
    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"总计: {len(all_samples)} 样本")
    
    # 采样显示
    if len(train_samples) > 0:
        sample = train_samples[0]
        print(f"\n样本示例:")
        print(f"  obs shape: {sample['obs'].shape}")
        print(f"  phase_id: {sample['phase_id']}")
        print(f"  候选数量: {len(sample['cand_feats'])}")
        print(f"  label_idx: {sample['label_idx']}")
        print(f"  value: {sample['value']}")

if __name__ == '__main__':
    main()

