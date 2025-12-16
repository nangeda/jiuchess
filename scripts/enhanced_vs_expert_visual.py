#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Agent vs Expert AI 可视化对战
实时显示棋盘和走子，褡裢特别标注
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians, find_all_dalians
from jiu.jiutypes import board_gild, Point, board_size
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def get_dalian_points(state: GameState, player: Player):
    """获取所有褡裢相关的点位"""
    dalians = find_all_dalians(state.board, player)
    trigger_points = set()
    empty_points = set()
    
    for d in dalians:
        trigger_points.add(d.trigger)
        empty_points.add(d.empty)
    
    return trigger_points, empty_points


def print_board(state: GameState, highlight_src=None, highlight_dst=None):
    """打印棋盘"""
    board = state.board
    
    # 获取褡裢点位
    white_triggers, white_empties = get_dalian_points(state, Player.white)
    black_triggers, black_empties = get_dalian_points(state, Player.black)
    
    # 打印图例
    print("  图例: ○=白子 ●=黑子 ·=空")
    print("        ◆=白褡裢游子 ◇=褡裢通道 ▲=黑褡裢游子")
    print("        [新]=刚落子位置")
    print()
    
    # 打印列号
    print("      ", end="")
    for c in range(1, board_size + 1):
        print(f"{c:3d}", end="")
    print()
    
    print("    +" + "---" * board_size + "+")
    
    for r in range(1, board_size + 1):
        print(f" {r:2d} |", end="")
        for c in range(1, board_size + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            
            # 确定显示字符
            is_new = (highlight_dst and pt == highlight_dst)
            is_dalian_trigger_w = pt in white_triggers
            is_dalian_trigger_b = pt in black_triggers
            is_dalian_empty = pt in white_empties or pt in black_empties
            
            if pl == Player.white:
                if is_new:
                    char = "[○]"  # 刚走到的白子
                elif is_dalian_trigger_w:
                    char = " ◆ "  # 白方褡裢游子（菱形实心）
                else:
                    char = " ○ "  # 普通白子
            elif pl == Player.black:
                if is_new:
                    char = "[●]"  # 刚走到的黑子
                elif is_dalian_trigger_b:
                    char = " ▲ "  # 黑方褡裢游子（三角形）
                else:
                    char = " ● "  # 普通黑子
            else:
                if is_dalian_empty:
                    char = " ◇ "  # 褡裢通道（菱形空心）
                elif highlight_src and pt == highlight_src:
                    char = " □ "  # 走出的位置
                else:
                    char = " · "  # 空位
            
            print(char, end="")
        print("|")
    
    print("    +" + "---" * board_size + "+")


def get_move_info(move: Move):
    """获取走法的源点和目标点"""
    src, dst = None, None
    desc = ""
    
    if move.is_put:
        dst = move.point
        desc = f"落子 ({dst.row},{dst.col})"
    elif move.is_go:
        src = move.go_to.go
        dst = move.go_to.to
        desc = f"走子 ({src.row},{src.col})→({dst.row},{dst.col})"
    elif move.is_fly:
        src = move.go_to.go
        dst = move.go_to.to
        desc = f"飞子 ({src.row},{src.col})→({dst.row},{dst.col})"
    elif move.is_skip_eat:
        se = move.skip_eat_points
        src = se.go
        dst = se.to
        desc = f"跳吃 ({src.row},{src.col})→({dst.row},{dst.col}) 吃({se.eat.row},{se.eat.col})"
    elif move.is_skip_eat_seq:
        seq = move.skip_eat_points
        src = seq[0].go
        dst = seq[-1].to
        eaten = len(seq)
        desc = f"连跳 ({src.row},{src.col})→({dst.row},{dst.col}) 吃{eaten}子"
    
    return src, dst, desc


def play_visual(enhanced_agent, expert_agent, enhanced_is_white=True):
    """可视化对战一局"""
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    # 褡裢统计
    enhanced_dalian_formed = 0
    enhanced_dalian_used = 0
    prev_enhanced_dalian = 0
    
    # 时间统计
    enhanced_times = []
    expert_times = []
    
    # 名称
    if enhanced_is_white:
        white_name, black_name = "Enhanced(白○)", "Expert(黑●)"
    else:
        white_name, black_name = "Expert(白○)", "Enhanced(黑●)"
    
    print("\n" + "═" * 70)
    print("🎮 可视化对战")
    print(f"  白方: {white_name}")
    print(f"  黑方: {black_name}")
    print("═" * 70)
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        move_start = time.time()
        
        if (current_player == Player.white and enhanced_is_white) or \
           (current_player == Player.black and not enhanced_is_white):
            move, info = enhanced_agent.select_move(state)
            is_enhanced = True
            agent_name = "Enhanced"
        else:
            move, _ = expert_agent.select_move(state)
            info = {}
            is_enhanced = False
            agent_name = "Expert"
        
        move_time = time.time() - move_start
        
        if move is None:
            print(f"\n⚠️ {agent_name} 无合法走法!")
            break
        
        if is_enhanced:
            enhanced_times.append(move_time)
        else:
            expert_times.append(move_time)
        
        src, dst, move_desc = get_move_info(move)
        state = state.apply_move(move)
        step += 1
        
        w_count = state.board.get_player_total(Player.white)
        b_count = state.board.get_player_total(Player.black)
        
        if state.step <= board_gild:
            phase = "布局"
            in_battle = False
        elif w_count <= 14 or b_count <= 14:
            phase = "飞子"
            in_battle = True
        else:
            phase = "对战"
            in_battle = True
        
        # 褡裢统计
        if in_battle:
            enhanced_player = Player.white if enhanced_is_white else Player.black
            curr_dalian = count_independent_dalians(state.board, enhanced_player)
            
            if is_enhanced:
                if curr_dalian > prev_enhanced_dalian:
                    enhanced_dalian_formed += 1
                if info.get('uses_dalian', False):
                    enhanced_dalian_used += 1
            
            prev_enhanced_dalian = curr_dalian
        
        # 事件
        event = ""
        if info.get('creates_dalian', False):
            event = " 🎯形成褡裢!"
        elif info.get('uses_dalian', False):
            event = " ⚡利用褡裢!"
        elif info.get('will_form_square', False):
            event = " 🔲成方!"
        elif move.is_skip_eat_seq and len(move.skip_eat_points) >= 2:
            event = f" 💥连跳{len(move.skip_eat_points)}!"
        
        # 对战阶段每步显示
        if in_battle:
            player_sym = "○" if current_player == Player.white else "●"
            
            enh_player = Player.white if enhanced_is_white else Player.black
            exp_player = Player.black if enhanced_is_white else Player.white
            enh_dalian = count_independent_dalians(state.board, enh_player)
            exp_dalian = count_independent_dalians(state.board, exp_player)
            
            print(f"\n{'─'*70}")
            print(f"【Step {step}】{phase}阶段 | 白○:{w_count} 黑●:{b_count}{event}")
            print(f"  {player_sym} {agent_name}: {move_desc} (用时{move_time:.2f}秒)")
            print(f"  褡裢: Enhanced有{enh_dalian}个, Expert有{exp_dalian}个")
            print()
            
            print_board(state, src, dst)
            
            avg_e = np.mean(enhanced_times[-10:]) if enhanced_times else 0
            avg_x = np.mean(expert_times[-10:]) if expert_times else 0
            print(f"\n  累计褡裢: 成{enhanced_dalian_formed}次 用{enhanced_dalian_used}次 | "
                  f"平均耗时 E:{avg_e:.2f}s X:{avg_x:.2f}s")
        
        elif step == board_gild:
            print(f"\n{'═'*70}")
            print(f"📍 布局结束 (Step {step}) | 白○:{w_count} 黑●:{b_count}")
            print("═" * 70)
            print()
            print_board(state)
        elif step % 40 == 0:
            print(f"  布局中... Step {step}: 白{w_count} 黑{b_count}")
    
    # 最终结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    w_count = state.board.get_player_total(Player.white)
    b_count = state.board.get_player_total(Player.black)
    
    print("\n" + "═" * 70)
    print("🏁 对战结束")
    print("═" * 70)
    
    print(f"\n最终局面 (Step {step}):")
    print()
    print_board(state)
    
    print(f"\n最终子数: 白○ {w_count} vs 黑● {b_count}")
    
    if winner == Player.white:
        winner_name = white_name
    elif winner == Player.black:
        winner_name = black_name
    else:
        winner_name = "平局"
    
    print(f"\n🎉 获胜者: {winner_name}")
    
    print(f"\n【统计】")
    print(f"  Enhanced 褡裢: 形成 {enhanced_dalian_formed} 次, 利用 {enhanced_dalian_used} 次")
    print(f"  Enhanced 平均走子: {np.mean(enhanced_times):.3f}s")
    print(f"  Expert   平均走子: {np.mean(expert_times):.3f}s")
    
    return winner


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--expert-depth', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("═" * 70)
    print("🎯 Enhanced Agent vs Expert AI 可视化对战")
    print("═" * 70)
    print(f"  Expert 搜索深度: {args.expert_depth}")
    print("═" * 70)
    
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
    print("✅ 加载完成")
    
    play_visual(enhanced_agent, expert_agent, enhanced_is_white=True)


if __name__ == '__main__':
    main()
