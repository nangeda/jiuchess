#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Agent (带褡裢增强) vs Expert AI 对战
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians
from jiu.jiutypes import board_gild
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def play_game(enhanced_agent, expert_agent, game_idx, enhanced_is_white, log_interval=20):
    """
    一局对战
    """
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    # 时间统计
    enhanced_times = []
    expert_times = []
    
    # 褡裢统计
    enhanced_dalian_formed = 0
    enhanced_dalian_used = 0
    prev_enhanced_dalian = 0
    
    game_start = time.time()
    
    # 确定谁执白谁执黑
    if enhanced_is_white:
        white_name = "Enhanced"
        black_name = "Expert"
    else:
        white_name = "Expert"
        black_name = "Enhanced"
    
    print(f"\n{'─'*70}")
    print(f"📍 第 {game_idx} 局")
    print(f"   白○: {white_name}    黑●: {black_name}")
    print(f"{'─'*70}")
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        move_start = time.time()
        
        # 判断当前玩家
        if (current_player == Player.white and enhanced_is_white) or \
           (current_player == Player.black and not enhanced_is_white):
            # Enhanced走子
            move, info = enhanced_agent.select_move(state)
            is_enhanced = True
        else:
            # Expert走子
            move, _ = expert_agent.select_move(state)
            info = {}
            is_enhanced = False
        
        move_time = time.time() - move_start  # 秒
        
        if move is None:
            break
        
        # 记录时间
        if is_enhanced:
            enhanced_times.append(move_time)
        else:
            expert_times.append(move_time)
        
        state = state.apply_move(move)
        step += 1
        
        # 统计褡裢（对战阶段）
        if state.step > board_gild:
            enhanced_player = Player.white if enhanced_is_white else Player.black
            curr_enhanced_dalian = count_independent_dalians(state.board, enhanced_player)
            
            if is_enhanced:  # 刚走完的是Enhanced
                if curr_enhanced_dalian > prev_enhanced_dalian:
                    enhanced_dalian_formed += 1
                if info.get('uses_dalian', False):
                    enhanced_dalian_used += 1
            
            prev_enhanced_dalian = curr_enhanced_dalian
        
        # 每log_interval步输出
        if step % log_interval == 0:
            w_count = state.board.get_player_total(Player.white)
            b_count = state.board.get_player_total(Player.black)
            
            if state.step <= board_gild:
                phase = "布局"
            elif w_count <= 14 or b_count <= 14:
                phase = "飞子"
            else:
                phase = "对战"
            
            enhanced_player = Player.white if enhanced_is_white else Player.black
            expert_player = Player.black if enhanced_is_white else Player.white
            enh_dalian = count_independent_dalians(state.board, enhanced_player) if state.step > board_gild else 0
            exp_dalian = count_independent_dalians(state.board, expert_player) if state.step > board_gild else 0
            
            # 平均时间
            avg_enh = np.mean(enhanced_times[-10:]) if enhanced_times else 0
            avg_exp = np.mean(expert_times[-10:]) if expert_times else 0
            
            # Enhanced的子数
            enh_count = w_count if enhanced_is_white else b_count
            exp_count = b_count if enhanced_is_white else w_count
            
            print(f"   Step {step:4d} [{phase:2s}] | "
                  f"E:{enh_count:2d} vs X:{exp_count:2d} | "
                  f"褡裢 E:{enh_dalian} X:{exp_dalian} | "
                  f"成褡裢:{enhanced_dalian_formed} 用褡裢:{enhanced_dalian_used} | "
                  f"时间 E:{avg_enh:5.2f}s X:{avg_exp:5.2f}s")
    
    # 结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    total_time = time.time() - game_start
    w_final = state.board.get_player_total(Player.white)
    b_final = state.board.get_player_total(Player.black)
    
    # 确定胜者
    if winner == Player.white:
        winner_code = "Enhanced" if enhanced_is_white else "Expert"
    elif winner == Player.black:
        winner_code = "Enhanced" if not enhanced_is_white else "Expert"
    else:
        winner_code = "Draw"
    
    avg_enhanced_time = np.mean(enhanced_times) if enhanced_times else 0
    avg_expert_time = np.mean(expert_times) if expert_times else 0
    
    print(f"{'─'*70}")
    print(f"🏁 第 {game_idx} 局结束")
    print(f"   获胜者: {winner_code}")
    print(f"   最终: Enhanced {w_final if enhanced_is_white else b_final} vs Expert {b_final if enhanced_is_white else w_final}")
    print(f"   步数: {step}, 用时: {total_time:.1f}秒")
    print(f"   ┌────────────────────────────────────────────────────┐")
    print(f"   │ Enhanced 褡裢: 形成 {enhanced_dalian_formed:2d} 次, 利用 {enhanced_dalian_used:2d} 次          │")
    print(f"   │ Enhanced 平均走子时间: {avg_enhanced_time:6.2f} s                │")
    print(f"   │ Expert   平均走子时间: {avg_expert_time:6.2f} s                │")
    print(f"   └────────────────────────────────────────────────────┘")
    
    return {
        'winner': winner_code,
        'steps': step,
        'time': total_time,
        'enhanced_dalian_formed': enhanced_dalian_formed,
        'enhanced_dalian_used': enhanced_dalian_used,
        'enhanced_avg_time': avg_enhanced_time,
        'expert_avg_time': avg_expert_time,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--expert-depth', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 Enhanced Agent (褡裢增强) vs Expert AI 对战")
    print("=" * 70)
    print(f"   对战局数: {args.num_games}")
    print(f"   Expert搜索深度: {args.expert_depth}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 加载Agent
    print("\n📦 加载Agent...")
    enhanced_agent = EnhancedJiuqiNetAgent(
        args.model,
        args.device,
        dalian_create_weight=12.0,
        dalian_use_weight=20.0,
        dalian_break_weight=10.0,
        pre_dalian_weight=5.0,
        verbose=False
    )
    
    expert_agent = ExpertAgent(alpha_beta_depth=args.expert_depth)
    print("✅ Agent加载完成\n")
    
    # 统计
    enhanced_wins, expert_wins, draws = 0, 0, 0
    total_enhanced_dalian_formed = 0
    total_enhanced_dalian_used = 0
    all_enhanced_times = []
    all_expert_times = []
    
    for i in range(1, args.num_games + 1):
        # 交替执白
        enhanced_is_white = (i % 2 == 1)
        
        stats = play_game(enhanced_agent, expert_agent, i, enhanced_is_white)
        
        if stats['winner'] == 'Enhanced':
            enhanced_wins += 1
        elif stats['winner'] == 'Expert':
            expert_wins += 1
        else:
            draws += 1
        
        total_enhanced_dalian_formed += stats['enhanced_dalian_formed']
        total_enhanced_dalian_used += stats['enhanced_dalian_used']
        all_enhanced_times.append(stats['enhanced_avg_time'])
        all_expert_times.append(stats['expert_avg_time'])
    
    # 最终汇总
    print("\n" + "=" * 70)
    print("🏆 最终结果")
    print("=" * 70)
    
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│                          胜率统计                                 │
├──────────────────────────────────────────────────────────────────┤
│  Enhanced (褡裢增强):  {enhanced_wins:2d} 胜  ({enhanced_wins/args.num_games*100:5.1f}%)                       │
│  Expert (深度={args.expert_depth}):      {expert_wins:2d} 胜  ({expert_wins/args.num_games*100:5.1f}%)                       │
│  平局:                 {draws:2d}     ({draws/args.num_games*100:5.1f}%)                       │
├──────────────────────────────────────────────────────────────────┤
│                          褡裢统计                                 │
├──────────────────────────────────────────────────────────────────┤
│  Enhanced 总褡裢: 形成 {total_enhanced_dalian_formed:3d} 次 (平均 {total_enhanced_dalian_formed/args.num_games:.1f}/局)            │
│                   利用 {total_enhanced_dalian_used:3d} 次 (平均 {total_enhanced_dalian_used/args.num_games:.1f}/局)            │
├──────────────────────────────────────────────────────────────────┤
│                          平均走子时间                             │
├──────────────────────────────────────────────────────────────────┤
│  Enhanced: {np.mean(all_enhanced_times):7.3f} s                                         │
│  Expert:   {np.mean(all_expert_times):7.3f} s                                         │
└──────────────────────────────────────────────────────────────────┘
""")
    
    print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()

