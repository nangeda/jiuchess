#!/usr/bin/env python3
"""基础增强(26维) vs Expert 100局测试"""
import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Player, board_size
from jiu.jiutypes import board_gild
from scripts.test_dalian_game import BasicJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent
from battle_test import decision_to_move

def run_game(agent_white, agent_black, white_name, black_name, game_num):
    state = GameState.new_game(board_size)
    step = 0
    move_times_white, move_times_black = [], []
    
    print(f"\n第{game_num}局: {white_name}(白) vs {black_name}(黑)")
    
    while not state.is_over() and step < 500:
        step += 1
        agent = agent_white if state.next_player == Player.white else agent_black
        times_list = move_times_white if state.next_player == Player.white else move_times_black
        
        start = time.time()
        result = agent.select_move(state)
        move = result[0] if isinstance(result, tuple) else result
        times_list.append(time.time() - start)
        
        if move is None: break
        try:
            state = state.apply_move(move)
        except:
            break
        
        if step % 50 == 0:
            w = state.board.get_player_total(Player.white)
            b = state.board.get_player_total(Player.black)
            avg_w = np.mean(move_times_white) if move_times_white else 0
            avg_b = np.mean(move_times_black) if move_times_black else 0
            print(f"  {step}步: 白{w}子 vs 黑{b}子 | 用时: {white_name}={avg_w:.2f}s {black_name}={avg_b:.2f}s")
    
    w_final = state.board.get_player_total(Player.white)
    b_final = state.board.get_player_total(Player.black)
    
    if w_final > b_final:
        winner, winner_name = "白", white_name
    elif b_final > w_final:
        winner, winner_name = "黑", black_name
    else:
        winner, winner_name = "平", "平局"
    
    print(f"  结果: {w_final}:{b_final} -> {winner_name}胜")
    return winner_name

def main():
    model_path = str(Path(__file__).parent.parent / 'exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    
    print("="*60)
    print("  基础增强(26维) vs Expert(depth=2) 100局测试")
    print("="*60)
    
    basic = BasicJiuqiNetAgent(model_path, device='cuda')
    expert = ExpertAgent(alpha_beta_depth=2)
    
    basic_wins, expert_wins, draws = 0, 0, 0
    
    for i in range(100):
        if i % 2 == 0:
            winner = run_game(basic, expert, "Basic26", "Expert", i+1)
        else:
            winner = run_game(expert, basic, "Expert", "Basic26", i+1)
        
        if "Basic" in winner:
            basic_wins += 1
        elif "Expert" in winner:
            expert_wins += 1
        else:
            draws += 1
        
        win_rate = basic_wins / (i+1) * 100
        print(f"  当前战绩: Basic26 {basic_wins}-{expert_wins} Expert | 胜率:{win_rate:.1f}%\n")
    
    print("\n" + "="*60)
    print(f"  最终结果: Basic26(基础增强) {basic_wins}胜 - Expert {expert_wins}胜 (平局:{draws})")
    print(f"  Basic26胜率: {basic_wins}%")
    print("="*60)

if __name__ == '__main__':
    main()
