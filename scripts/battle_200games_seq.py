#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串行对战: Basic Enhanced vs Expert (100局) + Dalian Enhanced vs Expert (100局)
带实时输出
"""

import sys
import time
from pathlib import Path

# 禁用输出缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from jiu.jiuboard_fast import GameState, Player, count_independent_dalians
from jiu.jiutypes import board_gild
from battle_test import decision_to_move
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def play_game(agent, expert, is_white, agent_type):
    """单局对战"""
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    dalian_formed = 0
    dalian_used = 0
    prev_dalian = 0
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        
        if (current_player == Player.white and is_white) or \
           (current_player == Player.black and not is_white):
            move, info = agent.select_move(state)
            is_agent = True
        else:
            move, _ = expert.select_move(state)
            info = {}
            is_agent = False
        
        if move is None:
            break
        
        state = state.apply_move(move)
        step += 1
        
        # 褡裢统计
        if agent_type == 'dalian' and state.step > board_gild:
            agent_player = Player.white if is_white else Player.black
            curr_dalian = count_independent_dalians(state.board, agent_player)
            
            if is_agent:
                if curr_dalian > prev_dalian:
                    dalian_formed += 1
                if info.get('uses_dalian', False):
                    dalian_used += 1
            
            prev_dalian = curr_dalian
    
    winner = state.winner()
    if winner is None:
        winner = state.winner_by_timeout()
    
    w_count = state.board.get_player_total(Player.white)
    b_count = state.board.get_player_total(Player.black)
    
    agent_count = w_count if is_white else b_count
    expert_count = b_count if is_white else w_count
    
    if winner == Player.white:
        winner_code = 'agent' if is_white else 'expert'
    elif winner == Player.black:
        winner_code = 'agent' if not is_white else 'expert'
    else:
        winner_code = 'draw'
    
    return winner_code, agent_count, expert_count, step, dalian_formed, dalian_used


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--expert-depth', type=int, default=2)
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--output', default='logs/1215_200games.log')
    args = parser.parse_args()
    
    print("=" * 70, flush=True)
    print("Battle: Basic/Dalian Enhanced vs Expert", flush=True)
    print("=" * 70, flush=True)
    print(f"Games per agent: {args.num_games}", flush=True)
    print(f"Expert depth: {args.expert_depth}", flush=True)
    print("=" * 70, flush=True)
    
    # 加载Expert
    expert = ExpertAgent(alpha_beta_depth=args.expert_depth)
    print("Expert loaded", flush=True)
    
    results = []
    
    # === Basic Enhanced vs Expert ===
    print("\n" + "=" * 70, flush=True)
    print("PART 1: Basic Enhanced vs Expert", flush=True)
    print("=" * 70, flush=True)
    
    from agent.basic_enhanced_agent import BasicEnhancedAgent
    basic_agent = BasicEnhancedAgent(args.model, device='cuda:0')
    
    basic_wins, basic_losses = 0, 0
    start_time = time.time()
    
    for i in range(1, args.num_games + 1):
        is_white = (i % 2 == 1)
        game_start = time.time()
        
        winner, a_pieces, e_pieces, steps, _, _ = play_game(
            basic_agent, expert, is_white, 'basic'
        )
        
        game_time = time.time() - game_start
        
        if winner == 'agent':
            basic_wins += 1
        elif winner == 'expert':
            basic_losses += 1
        
        results.append({
            'type': 'basic', 'game': i, 'white': is_white,
            'winner': winner, 'agent_pieces': a_pieces, 
            'expert_pieces': e_pieces, 'steps': steps
        })
        
        side = 'W' if is_white else 'B'
        print(f"  Basic #{i:3d} ({side}) | {winner:6s} | {a_pieces:2d}-{e_pieces:2d} | "
              f"{steps:3d} steps | {game_time:.1f}s | W:{basic_wins} L:{basic_losses}", flush=True)
    
    basic_time = time.time() - start_time
    basic_win_rate = basic_wins / args.num_games * 100
    print(f"\nBasic Enhanced: {basic_wins}W {basic_losses}L ({basic_win_rate:.1f}%) in {basic_time:.1f}s", flush=True)
    
    # 清理GPU内存
    del basic_agent
    torch.cuda.empty_cache()
    
    # === Dalian Enhanced vs Expert ===
    print("\n" + "=" * 70, flush=True)
    print("PART 2: Dalian Enhanced vs Expert", flush=True)
    print("=" * 70, flush=True)
    
    from agent.dalian_enhanced_agent import DalianEnhancedAgent
    dalian_agent = DalianEnhancedAgent(args.model, device='cuda:0')
    
    dalian_wins, dalian_losses = 0, 0
    total_formed, total_used = 0, 0
    start_time = time.time()
    
    for i in range(1, args.num_games + 1):
        is_white = (i % 2 == 1)
        game_start = time.time()
        
        winner, a_pieces, e_pieces, steps, formed, used = play_game(
            dalian_agent, expert, is_white, 'dalian'
        )
        
        game_time = time.time() - game_start
        
        if winner == 'agent':
            dalian_wins += 1
        elif winner == 'expert':
            dalian_losses += 1
        
        total_formed += formed
        total_used += used
        
        results.append({
            'type': 'dalian', 'game': i, 'white': is_white,
            'winner': winner, 'agent_pieces': a_pieces,
            'expert_pieces': e_pieces, 'steps': steps,
            'dalian_formed': formed, 'dalian_used': used
        })
        
        side = 'W' if is_white else 'B'
        print(f"  Dalian #{i:3d} ({side}) | {winner:6s} | {a_pieces:2d}-{e_pieces:2d} | "
              f"{steps:3d} steps | D:{formed}/{used} | {game_time:.1f}s | W:{dalian_wins} L:{dalian_losses}", flush=True)
    
    dalian_time = time.time() - start_time
    dalian_win_rate = dalian_wins / args.num_games * 100
    print(f"\nDalian Enhanced: {dalian_wins}W {dalian_losses}L ({dalian_win_rate:.1f}%) in {dalian_time:.1f}s", flush=True)
    print(f"Dalian stats: formed {total_formed}, used {total_used}", flush=True)
    
    # === 最终报告 ===
    report = []
    report.append("=" * 70)
    report.append("FINAL RESULTS: Basic/Dalian Enhanced vs Expert (depth=2)")
    report.append("=" * 70)
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("-" * 70)
    report.append(f"Basic Enhanced vs Expert ({args.num_games} games)")
    report.append("-" * 70)
    report.append(f"  Wins:   {basic_wins:3d} ({basic_win_rate:.1f}%)")
    report.append(f"  Losses: {basic_losses:3d} ({basic_losses/args.num_games*100:.1f}%)")
    report.append(f"  Time:   {basic_time:.1f}s")
    report.append("")
    report.append("-" * 70)
    report.append(f"Dalian Enhanced vs Expert ({args.num_games} games)")
    report.append("-" * 70)
    report.append(f"  Wins:   {dalian_wins:3d} ({dalian_win_rate:.1f}%)")
    report.append(f"  Losses: {dalian_losses:3d} ({dalian_losses/args.num_games*100:.1f}%)")
    report.append(f"  Dalian formed: {total_formed} (avg {total_formed/args.num_games:.1f}/game)")
    report.append(f"  Dalian used:   {total_used} (avg {total_used/args.num_games:.1f}/game)")
    report.append(f"  Time:   {dalian_time:.1f}s")
    report.append("")
    report.append("=" * 70)
    report.append("COMPARISON")
    report.append("=" * 70)
    diff = dalian_win_rate - basic_win_rate
    report.append(f"  Basic Enhanced:  {basic_win_rate:.1f}% win rate")
    report.append(f"  Dalian Enhanced: {dalian_win_rate:.1f}% win rate")
    report.append(f"  Difference: {diff:+.1f}%")
    report.append("")
    if diff > 0:
        report.append(f"  >>> Dalian Enhanced is BETTER by {diff:.1f}%")
    elif diff < 0:
        report.append(f"  >>> Basic Enhanced is BETTER by {-diff:.1f}%")
    else:
        report.append(f"  >>> Both perform equally")
    report.append("")
    report.append("=" * 70)
    
    # 详细结果
    report.append("\nDETAILED RESULTS:")
    for r in results:
        side = 'W' if r['white'] else 'B'
        line = f"  {r['type']:6s} #{r['game']:3d} ({side}) | {r['winner']:6s} | {r['agent_pieces']:2d}-{r['expert_pieces']:2d} | {r['steps']:3d} steps"
        if r['type'] == 'dalian':
            line += f" | D:{r.get('dalian_formed',0)}/{r.get('dalian_used',0)}"
        report.append(line)
    
    report_text = "\n".join(report)
    
    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text, flush=True)
    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
