#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Agent vs Expert(深度2) 对战测试
移除布局阶段特殊处理后的版本
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import board_gild
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def play_game(enhanced_agent, expert_agent, game_idx, enhanced_is_white):
    """对战一局"""
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    if enhanced_is_white:
        white_name, black_name = "Enhanced(白)", "Expert(黑)"
    else:
        white_name, black_name = "Expert(白)", "Enhanced(黑)"
    
    print(f"\n{'═'*50}")
    print(f"📍 第 {game_idx} 局 | {white_name} vs {black_name}")
    print(f"{'═'*50}")
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        
        if (current_player == Player.white and enhanced_is_white) or \
           (current_player == Player.black and not enhanced_is_white):
            result = enhanced_agent.select_move(state)
        else:
            result = expert_agent.select_move(state)
        
        move = result[0] if isinstance(result, tuple) else result
        
        if move is None:
            break
        
        state = state.apply_move(move)
        step += 1
        
        # 每100步输出
        if step % 100 == 0:
            w = state.board.get_player_total(Player.white)
            b = state.board.get_player_total(Player.black)
            if state.step <= board_gild:
                phase = "布局"
            elif w <= 14 or b <= 14:
                phase = "飞子"
            else:
                phase = "对战"
            print(f"  Step {step:4d} [{phase}] | 白:{w:2d} 黑:{b:2d}")
    
    # 结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    w_final = state.board.get_player_total(Player.white)
    b_final = state.board.get_player_total(Player.black)
    
    # Enhanced的棋子数
    enhanced_final = w_final if enhanced_is_white else b_final
    expert_final = b_final if enhanced_is_white else w_final
    
    if winner == Player.white:
        winner_name = "Enhanced" if enhanced_is_white else "Expert"
    elif winner == Player.black:
        winner_name = "Enhanced" if not enhanced_is_white else "Expert"
    else:
        winner_name = "Draw"
    
    lead = enhanced_final - expert_final
    
    print(f"{'─'*50}")
    print(f"🏁 结果: {winner_name} 获胜")
    print(f"   Enhanced: {enhanced_final} | Expert: {expert_final} | 领先: {lead:+d}")
    
    return winner_name, enhanced_final, expert_final, lead


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--expert-depth', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("═" * 50)
    print("🎯 Enhanced Agent vs Expert 对战")
    print("═" * 50)
    print(f"  对战局数: {args.num_games}")
    print(f"  Expert深度: {args.expert_depth}")
    print("═" * 50)
    
    # 加载Agent
    enhanced_agent = EnhancedJiuqiNetAgent(
        args.model,
        args.device,
        dalian_create_weight=12.0,
        dalian_use_weight=20.0,
        dalian_break_weight=10.0,
        verbose=False
    )
    
    expert_agent = ExpertAgent(alpha_beta_depth=args.expert_depth)
    print(f"✅ Expert Agent (depth={args.expert_depth})")
    
    # 统计
    enhanced_wins, expert_wins, draws = 0, 0, 0
    win_leads = []  # 获胜时的领先棋子数
    
    for i in range(1, args.num_games + 1):
        enhanced_is_white = (i % 2 == 1)
        
        winner, e_count, x_count, lead = play_game(
            enhanced_agent, expert_agent, i, enhanced_is_white
        )
        
        if winner == 'Enhanced':
            enhanced_wins += 1
            win_leads.append(lead)
        elif winner == 'Expert':
            expert_wins += 1
        else:
            draws += 1
        
        print(f"\n  【当前战绩】Enhanced {enhanced_wins} : {expert_wins} Expert | 平局 {draws}")
    
    # 最终结果
    print("\n" + "═" * 50)
    print("🏆 最终结果")
    print("═" * 50)
    
    print(f"""
┌──────────────────────────────────────────────────┐
│                    胜率统计                       │
├──────────────────────────────────────────────────┤
│  Enhanced:  {enhanced_wins:2d} 胜  ({enhanced_wins/args.num_games*100:5.1f}%)              │
│  Expert:    {expert_wins:2d} 胜  ({expert_wins/args.num_games*100:5.1f}%)              │
│  平局:       {draws:2d}     ({draws/args.num_games*100:5.1f}%)              │
├──────────────────────────────────────────────────┤
│                  获胜时领先棋子                   │
├──────────────────────────────────────────────────┤""")
    
    if win_leads:
        print(f"│  平均领先: {np.mean(win_leads):+.1f} 子                           │")
        print(f"│  最大领先: {max(win_leads):+d} 子                             │")
        print(f"│  最小领先: {min(win_leads):+d} 子                             │")
    else:
        print(f"│  无获胜记录                                      │")
    
    print("└──────────────────────────────────────────────────┘")
    print("═" * 50)


if __name__ == '__main__':
    main()

