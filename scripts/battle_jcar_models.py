#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCAR 模型对战测试：带Value训练的模型 vs 旧SGF模型
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from jcar.model import JiuqiNet
from jcar.config import JiuqiNetConfig
from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import board_gild, Point, Decision, Go, Skip_eat
from dyt.candidate_features import build_features_for_candidates


def encode_board_state(state: GameState) -> np.ndarray:
    from jiu.jiutypes import board_size
    board = state.board
    H, W = board_size, board_size
    obs = np.zeros((6, H, W), dtype=np.float32)
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            if pl == Player.white:
                obs[0, r - 1, c - 1] = 1.0
            elif pl == Player.black:
                obs[1, r - 1, c - 1] = 1.0
            else:
                obs[2, r - 1, c - 1] = 1.0
    return obs


def get_phase_id(state: GameState) -> int:
    if state.step < board_gild:
        return 0
    elif state.board.get_player_total(state.next_player) <= 14:
        return 2
    else:
        return 1


def decision_to_dict(dec: Decision) -> dict:
    if dec.act == 'put_piece':
        p = dec.points if isinstance(dec.points, Point) else Point(1, 1)
        return {'act': 'put_piece', 'point': {'r': p.row, 'c': p.col}}
    elif dec.act == 'is_go':
        go_obj = dec.points if isinstance(dec.points, Go) else Go(Point(1, 1), Point(1, 1))
        return {'act': 'is_go', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col},
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    elif dec.act == 'fly':
        go_obj = dec.points if isinstance(dec.points, Go) else Go(Point(1, 1), Point(1, 1))
        return {'act': 'fly', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col},
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    elif dec.act == 'skip_move':
        se = dec.points if isinstance(dec.points, Skip_eat) else Skip_eat(Point(1,1), Point(1,1), Point(1,1))
        return {'act': 'skip_move', 'go': {'r': se.go.row, 'c': se.go.col},
                'to': {'r': se.to.row, 'c': se.to.col},
                'eat': {'r': se.eat.row, 'c': se.eat.col}}
    elif dec.act == 'skip_eat_seq':
        seq = []
        if isinstance(dec.points, list):
            for se in dec.points:
                if isinstance(se, Skip_eat):
                    seq.append({'go': {'r': se.go.row, 'c': se.go.col},
                                'to': {'r': se.to.row, 'c': se.to.col},
                                'eat': {'r': se.eat.row, 'c': se.eat.col}})
        return {'act': 'skip_eat_seq', 'seq': seq}
    return {'act': 'unknown'}


def decision_to_move(dec: Decision) -> Optional[Move]:
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


class JCARAgent:
    def __init__(self, checkpoint_path: str, device: str, name: str = "JCAR"):
        self.device = device
        self.name = name
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = checkpoint.get('config', JiuqiNetConfig())
        self.model = JiuqiNet(cfg)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
    
    def select_move(self, state: GameState) -> Tuple[Optional[Move], dict]:
        obs = encode_board_state(state)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        phase_id = get_phase_id(state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        legal_decs = state.legal_moves()
        if not legal_decs:
            return None, {'error': 'no legal moves'}
        
        flying = state.board.get_player_total(state.next_player) <= 14
        cand_dicts = [decision_to_dict(d) for d in legal_decs]
        cand_feats = build_features_for_candidates(cand_dicts, phase_id, flying)
        cand_feats = cand_feats[:, :14]
        cand_tensor = torch.from_numpy(cand_feats).float().to(self.device)
        
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        logits = logits_list[0]
        probs = torch.softmax(logits, dim=-1)
        best_idx = probs.argmax().item()
        
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        return move, {'value': value.item(), 'prob': probs[best_idx].item()}


def play_game(black_agent, white_agent, max_steps=500, verbose=False):
    state = GameState.new_game(14)
    agents = {Player.black: black_agent, Player.white: white_agent}
    steps = 0
    
    while not state.is_over() and steps < max_steps:
        current_agent = agents[state.next_player]
        move, info = current_agent.select_move(state)
        if move is None:
            break
        state = state.apply_move(move)
        steps += 1
        if verbose and steps % 50 == 0:
            print(f"  Step {steps}: {state.next_player.other}")
    
    winner = state.winner()
    reason = "normal" if state.is_over() else "max_steps"
    return winner, steps, reason


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-model', default='exp/jcar_sft_with_value/checkpoint_best.pt')
    parser.add_argument('--old-model', default='exp/jcar_sft_v4/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎮 JCAR 模型对战测试")
    print("=" * 70)
    print(f"新模型(带Value): {args.new_model}")
    print(f"旧模型(无Value): {args.old_model}")
    print(f"对战局数: {args.num_games}")
    print("=" * 70)
    
    new_agent = JCARAgent(args.new_model, args.device, "New+Value")
    old_agent = JCARAgent(args.old_model, args.device, "Old-SGF")
    print("✅ 模型加载完成\n")
    
    new_wins, old_wins, draws = 0, 0, 0
    
    for game_idx in range(args.num_games):
        if game_idx % 2 == 0:
            black, white = new_agent, old_agent
            black_name, white_name = "New+Value", "Old-SGF"
        else:
            black, white = old_agent, new_agent
            black_name, white_name = "Old-SGF", "New+Value"
        
        print(f"第 {game_idx+1:2d} 局: 黑={black_name:10s} vs 白={white_name:10s}", end=" ")
        winner, steps, reason = play_game(black, white, args.max_steps)
        
        if winner == Player.black:
            winner_name = black_name
        elif winner == Player.white:
            winner_name = white_name
        else:
            winner_name = "平局"
        
        if winner_name == "New+Value":
            new_wins += 1; sym = "✅"
        elif winner_name == "Old-SGF":
            old_wins += 1; sym = "❌"
        else:
            draws += 1; sym = "🤝"
        
        print(f"-> {sym} {winner_name:10s} ({reason}, {steps}步)")
    
    print("\n" + "=" * 70)
    print("📊 对战结果")
    print("=" * 70)
    print(f"New+Value 胜: {new_wins:2d} ({new_wins/args.num_games*100:5.1f}%)")
    print(f"Old-SGF 胜:   {old_wins:2d} ({old_wins/args.num_games*100:5.1f}%)")
    print(f"平局:         {draws:2d} ({draws/args.num_games*100:5.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()

