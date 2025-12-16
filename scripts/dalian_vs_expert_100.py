#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalian Enhanced vs Expert 100局对战"""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Player
from jiu.jiutypes import Point, board_size
from agent.dalian_enhanced_agent import DalianEnhancedAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def print_board(state: GameState, white_model: str = "?", black_model: str = "?"):
    """打印棋盘状态"""
    board = state.board
    w_count = board.get_player_total(Player.white)
    b_count = board.get_player_total(Player.black)
    
    print(f"\n  白(x)-{white_model}: {w_count}  黑(●)-{black_model}: {b_count}")
    print("      ", end="")
    for c in range(1, board_size + 1):
        print(f"{c:2d}", end="")
    print()
    print("    +" + "--" * board_size + "+")
    
    for r in range(1, board_size + 1):
        print(f" {r:2d} |", end="")
        for c in range(1, board_size + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            if pl == Player.white:
                char = "x "
            elif pl == Player.black:
                char = "● "
            else:
                char = "· "
            print(char, end="")
        print("|")
    print("    +" + "--" * board_size + "+")


def play_game(agent, expert, is_white, game_num, verbose=True):
    state = GameState.new_game(14)
    step = 0
    
    # 记录每个模型的总时间和步数
    agent_total_time = 0.0
    agent_steps = 0
    expert_total_time = 0.0
    expert_steps = 0
    
    agent_side = "白(x)" if is_white else "黑(●)"
    expert_side = "黑(●)" if is_white else "白(x)"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎮 第 {game_num} 局")
        print(f"  {agent_side}: DalianEnhanced")
        print(f"  {expert_side}: Expert(depth=2)")
        print(f"{'='*60}")
    
    while not state.is_over() and step < 800:
        t0 = time.time()
        if (state.next_player == Player.white and is_white) or \
           (state.next_player == Player.black and not is_white):
            move, _ = agent.select_move(state)
            dt = time.time() - t0
            agent_total_time += dt
            agent_steps += 1
        else:
            move, _ = expert.select_move(state)
            dt = time.time() - t0
            expert_total_time += dt
            expert_steps += 1
        if move is None:
            break
        state = state.apply_move(move)
        step += 1
        
        # 每20步输出棋盘状态
        if verbose and step % 20 == 0:
            w = state.board.get_player_total(Player.white)
            b = state.board.get_player_total(Player.black)
            phase = "布局" if step <= 200 else ("飞子" if w <= 12 or b <= 12 else "对战")
            avg_agent = agent_total_time / agent_steps if agent_steps > 0 else 0
            avg_expert = expert_total_time / expert_steps if expert_steps > 0 else 0
            print(f"\n--- Step {step} [{phase}] | Dalian平均:{avg_agent:.2f}s Expert平均:{avg_expert:.2f}s ---")
            white_m = "Dalian" if is_white else "Expert"
            black_m = "Expert" if is_white else "Dalian"
            print_board(state, white_m, black_m)
    
    winner = state.winner() or state.winner_by_timeout()
    w, b = state.board.get_player_total(Player.white), state.board.get_player_total(Player.black)
    agent_p = w if is_white else b
    expert_p = b if is_white else w
    
    avg_agent = agent_total_time / agent_steps if agent_steps > 0 else 0
    avg_expert = expert_total_time / expert_steps if expert_steps > 0 else 0
    
    if verbose:
        print(f"\n🏁 第 {game_num} 局结束")
        white_m = "Dalian" if is_white else "Expert"
        black_m = "Expert" if is_white else "Dalian"
        print_board(state, white_m, black_m)
        print(f"  Dalian平均下子: {avg_agent:.3f}s ({agent_steps}步)")
        print(f"  Expert平均下子: {avg_expert:.3f}s ({expert_steps}步)")
    
    if winner == Player.white:
        result = 'agent' if is_white else 'expert'
    elif winner == Player.black:
        result = 'agent' if not is_white else 'expert'
    else:
        result = 'draw'
    return result, agent_p, expert_p, step, avg_agent, avg_expert


def main():
    print("=" * 60, flush=True)
    print("Dalian Enhanced vs Expert (100 games)", flush=True)
    print("=" * 60, flush=True)
    
    expert = ExpertAgent(alpha_beta_depth=2)
    agent = DalianEnhancedAgent('exp/jcar_sft_2025_balanced/checkpoint_best.pt', device='cuda:0')
    print("Agents loaded!\n", flush=True)
    
    wins, losses = 0, 0
    start = time.time()
    
    total_agent_time = 0.0
    total_expert_time = 0.0
    game_count = 0
    
    for i in range(1, 101):
        is_white = (i % 2 == 1)
        t0 = time.time()
        result, ap, ep, steps, avg_a, avg_e = play_game(agent, expert, is_white, i, verbose=True)
        dt = time.time() - t0
        
        total_agent_time += avg_a
        total_expert_time += avg_e
        game_count += 1
        
        if result == 'agent': wins += 1
        elif result == 'expert': losses += 1
        
        winner_str = "🏆 Dalian胜" if result == 'agent' else ("❌ Expert胜" if result == 'expert' else "🤝 平局")
        print(f"\n{winner_str} | Dalian:{ap} vs Expert:{ep} | {steps}步 | {dt:.1f}s")
        print(f"【战绩】Dalian {wins}胜 {losses}负 | 胜率 {wins*100/(wins+losses):.1f}%" if wins+losses>0 else "", flush=True)
    
    total = time.time() - start
    print(f"\n{'='*60}", flush=True)
    print(f"🏆 最终结果", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Dalian Enhanced: {wins}胜 {losses}负 (胜率 {wins*100/(wins+losses):.1f}%)")
    print(f"  总用时: {total:.1f}s")
    print(f"  Dalian平均下子: {total_agent_time/game_count:.3f}s")
    print(f"  Expert平均下子: {total_expert_time/game_count:.3f}s")
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
