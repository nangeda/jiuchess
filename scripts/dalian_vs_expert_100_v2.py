#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalian Enhanced vs Expert 100局对战 - 详细输出版"""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Player
from jiu.jiutypes import board_gild
from agent.dalian_enhanced_agent import DalianEnhancedAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def play_game_detailed(agent, expert, agent_is_white, game_num):
    """详细输出的对战"""
    state = GameState.new_game(14)
    step = 0
    
    agent_times = []
    expert_times = []
    last_report = 0
    
    agent_color = "白" if agent_is_white else "黑"
    expert_color = "黑" if agent_is_white else "白"
    
    print(f"\n{'═'*60}", flush=True)
    print(f"📍 第 {game_num} 局 | 褡裢增强({agent_color}) vs Expert({expert_color})", flush=True)
    print(f"{'═'*60}", flush=True)
    
    while not state.is_over() and step < 800:
        current = state.next_player
        t0 = time.time()
        
        if (current == Player.white and agent_is_white) or \
           (current == Player.black and not agent_is_white):
            move, _ = agent.select_move(state)
            dt = time.time() - t0
            agent_times.append(dt)
        else:
            move, _ = expert.select_move(state)
            dt = time.time() - t0
            expert_times.append(dt)
        
        if move is None:
            break
        state = state.apply_move(move)
        step += 1
        
        # 每50步输出
        if step % 50 == 0 and step > last_report:
            last_report = step
            phase = "布局" if step < board_gild else "对战"
            w_count = state.board.get_player_total(Player.white)
            b_count = state.board.get_player_total(Player.black)
            
            avg_agent = sum(agent_times[-50:]) / len(agent_times[-50:]) if agent_times else 0
            avg_expert = sum(expert_times[-50:]) / len(expert_times[-50:]) if expert_times else 0
            
            print(f"  Step {step:4d} [{phase}] | 白:{w_count:2d} 黑:{b_count:2d} | "
                  f"用时 Dalian:{avg_agent:.2f}s Expert:{avg_expert:.2f}s", flush=True)
    
    # 结果
    winner = state.winner() or state.winner_by_timeout()
    w_count = state.board.get_player_total(Player.white)
    b_count = state.board.get_player_total(Player.black)
    
    agent_pieces = w_count if agent_is_white else b_count
    expert_pieces = b_count if agent_is_white else w_count
    
    if winner == Player.white:
        result = 'agent' if agent_is_white else 'expert'
    elif winner == Player.black:
        result = 'agent' if not agent_is_white else 'expert'
    else:
        result = 'draw'
    
    avg_agent_total = sum(agent_times) / len(agent_times) if agent_times else 0
    avg_expert_total = sum(expert_times) / len(expert_times) if expert_times else 0
    
    print(f"{'─'*60}", flush=True)
    winner_name = "褡裢增强" if result == 'agent' else ("Expert" if result == 'expert' else "平局")
    print(f"🏁 第 {game_num} 局结束 | 获胜: {winner_name}", flush=True)
    print(f"   最终子数: 白 {w_count} vs 黑 {b_count}", flush=True)
    print(f"   平均走子时间: Dalian {avg_agent_total:.3f}s, Expert {avg_expert_total:.3f}s", flush=True)
    
    return result, agent_pieces, expert_pieces, step


def main():
    print("═" * 60, flush=True)
    print("🎯 Dalian Enhanced vs Expert 对战", flush=True)
    print("═" * 60, flush=True)
    print("  对战局数: 100", flush=True)
    print("  Expert深度: 2", flush=True)
    print("═" * 60, flush=True)
    
    print("\n📦 加载Agent...", flush=True)
    expert = ExpertAgent(alpha_beta_depth=2)
    agent = DalianEnhancedAgent('exp/jcar_sft_2025_balanced/checkpoint_best.pt', device='cuda:1')
    print("✅ 加载完成\n", flush=True)
    
    wins, losses, draws = 0, 0, 0
    start = time.time()
    
    for i in range(1, 101):
        is_white = (i % 2 == 1)
        result, ap, ep, steps = play_game_detailed(agent, expert, is_white, i)
        
        if result == 'agent': 
            wins += 1
        elif result == 'expert': 
            losses += 1
        else:
            draws += 1
        
        total_games = wins + losses + draws
        win_rate = wins / total_games * 100
        print(f"\n  【当前战绩】褡裢增强 {wins} 胜 | Expert {losses} 胜 | 平局 {draws} | 胜率 {win_rate:.1f}%\n", flush=True)
    
    total_time = time.time() - start
    
    print("\n" + "═" * 60, flush=True)
    print("📊 最终结果", flush=True)
    print("═" * 60, flush=True)
    print(f"  Dalian Enhanced: {wins} 胜 ({wins}%)", flush=True)
    print(f"  Expert:          {losses} 胜 ({losses}%)", flush=True)
    print(f"  平局:            {draws}", flush=True)
    print(f"  总用时:          {total_time:.1f}s", flush=True)
    print("═" * 60, flush=True)


if __name__ == '__main__':
    main()
