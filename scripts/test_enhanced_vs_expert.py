#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced vs Expert 对战测试脚本
- 10局对战，黑白交替
- 每50步输出棋局状态（x=黑棋，o=白棋）
- 记录走子时间
- 褡裢编号和使用次数统计
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import (
    GameState, Move, Player, Board, Point,
    find_all_dalians, Dalian
)
from jiu.jiutypes import Decision, board_gild, board_size
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent
from battle_test import decision_to_move


@dataclass
class DalianInfo:
    """褡裢信息"""
    id: int
    trigger: Point
    empty: Point
    pieces: set
    form_step: int
    use_count: int = 0


def print_board_state(board: Board, step: int, white_name: str, black_name: str):
    """打印棋盘状态（x=黑棋，o=白棋）"""
    print(f"\n--- 第{step}步棋盘 ---")
    print(f"    白方(o): {white_name}  |  黑方(x): {black_name}")
    
    # 列标题
    col_header = "    "
    for c in range(1, board_size + 1):
        col_header += f"{c:2d}"
    print(col_header)
    
    for r in range(1, board_size + 1):
        row_str = f"{r:2d}  "
        for c in range(1, board_size + 1):
            p = board.get(Point(r, c))
            if p == Player.white:
                row_str += " o"
            elif p == Player.black:
                row_str += " x"
            else:
                row_str += " ."
        print(row_str)


def dalian_key(d: Dalian) -> tuple:
    """生成褡裢唯一标识"""
    return (d.trigger.row, d.trigger.col, d.empty.row, d.empty.col)


def run_game(agent_white, agent_black, white_name: str, black_name: str, 
             game_num: int, max_steps: int = 500):
    """运行一局游戏"""
    state = GameState.new_game(board_size)
    step = 0
    
    # 褡裢追踪
    dalian_registry: Dict[tuple, DalianInfo] = {}  # key -> DalianInfo
    dalian_counter = 0
    
    # 时间统计
    move_times = []
    checkpoint_times = []  # 每50步的平均时间
    
    print(f"\n{'#'*70}")
    print(f"#  第{game_num}局: {white_name}(白/o) vs {black_name}(黑/x)")
    print(f"{'#'*70}")
    
    while not state.is_over() and step < max_steps:
        step += 1
        current_player = state.next_player
        agent = agent_white if current_player == Player.white else agent_black
        player_symbol = "o" if current_player == Player.white else "x"
        player_name = white_name if current_player == Player.white else black_name
        
        # 计时
        start_time = time.time()
        
        # 获取走法
        result = agent.select_move(state)
        if isinstance(result, tuple):
            move = result[0]
        else:
            move = result
        
        elapsed = time.time() - start_time
        move_times.append(elapsed)
        
        if move is None:
            print(f"  第{step}步: {player_name}({player_symbol}) 无法走棋")
            break
        
        # 获取当前阶段
        phase_id = 0 if state.step < board_gild else (2 if state.board.get_player_total(state.next_player) <= 14 else 1)
        
        # 执行走法前检测褡裢（只在对战/飞子阶段记录，布局阶段不记录）
        if phase_id > 0:
            white_dalians_before = find_all_dalians(state.board, Player.white)
            black_dalians_before = find_all_dalians(state.board, Player.black)
        else:
            white_dalians_before = []
            black_dalians_before = []
        
        # 执行走法
        try:
            state = state.apply_move(move)
        except Exception as e:
            print(f"  第{step}步: 走法执行失败: {e}")
            break
        
        # 执行走法后检测褡裢（只在对战/飞子阶段）
        # 更新阶段（走法后状态可能变化）
        phase_id_after = 0 if state.step < board_gild else (2 if state.board.get_player_total(Player.white) <= 14 or state.board.get_player_total(Player.black) <= 14 else 1)
        
        if phase_id_after > 0:
            white_dalians_after = find_all_dalians(state.board, Player.white)
            black_dalians_after = find_all_dalians(state.board, Player.black)
        else:
            white_dalians_after = []
            black_dalians_after = []
        
        # 检测新形成的褡裢（只在布局完毕后）
        all_dalians_before = {dalian_key(d) for d in white_dalians_before + black_dalians_before}
        all_dalians_after = white_dalians_after + black_dalians_after
        
        for d in all_dalians_after:
            key = dalian_key(d)
            if key not in dalian_registry:
                # 新褡裢
                dalian_counter += 1
                dalian_registry[key] = DalianInfo(
                    id=dalian_counter,
                    trigger=d.trigger,
                    empty=d.empty,
                    pieces=d.pieces,
                    form_step=step
                )
                owner = "白方" if state.board.get(d.trigger) == Player.white else "黑方"
                print(f"\n  ⭐ 第{step}步: {owner}形成【褡裢{dalian_counter}】")
                print(f"     游子: ({d.trigger.row},{d.trigger.col}), 空位: ({d.empty.row},{d.empty.col})")
        
        # 检测使用褡裢
        for key, info in dalian_registry.items():
            # 检查游子是否移动到了空位
            if hasattr(move, 'go_to'):
                go_to = move.go_to
                if hasattr(go_to, 'go') and hasattr(go_to, 'to'):
                    if (go_to.go.row == info.trigger.row and go_to.go.col == info.trigger.col and
                        go_to.to.row == info.empty.row and go_to.to.col == info.empty.col):
                        info.use_count += 1
                        print(f"  💥 第{step}步: 使用【褡裢{info.id}】(第{info.use_count}次)")
        
        # 每50步输出状态
        if step % 50 == 0:
            avg_time = np.mean(move_times[-50:]) if len(move_times) >= 50 else np.mean(move_times)
            checkpoint_times.append(avg_time)
            
            print_board_state(state.board, step, white_name, black_name)
            
            white_count = state.board.get_player_total(Player.white)
            black_count = state.board.get_player_total(Player.black)
            white_dalian_count = len(find_all_dalians(state.board, Player.white))
            black_dalian_count = len(find_all_dalians(state.board, Player.black))
            
            print(f"\n  统计: 白{white_count}子(褡裢:{white_dalian_count}) vs 黑{black_count}子(褡裢:{black_dalian_count})")
            print(f"  最近50步平均走子时间: {avg_time*1000:.1f}ms")
    
    # 游戏结束
    print(f"\n{'='*70}")
    print(f"  第{game_num}局结束! 共{step}步")
    
    white_count = state.board.get_player_total(Player.white)
    black_count = state.board.get_player_total(Player.black)
    
    print(f"  白方({white_name}): {white_count}子")
    print(f"  黑方({black_name}): {black_count}子")
    
    if white_count > black_count:
        winner = f"白方({white_name})"
        winner_is_enhanced = "Enhanced" in white_name
    elif black_count > white_count:
        winner = f"黑方({black_name})"
        winner_is_enhanced = "Enhanced" in black_name
    else:
        winner = "平局"
        winner_is_enhanced = None
    
    print(f"  胜者: {winner}")
    
    # 褡裢统计
    if dalian_registry:
        print(f"\n  褡裢统计 (共{len(dalian_registry)}个):")
        for key, info in dalian_registry.items():
            print(f"    【褡裢{info.id}】形成于第{info.form_step}步, 使用{info.use_count}次")
    else:
        print(f"\n  本局无褡裢形成")
    
    # 时间统计
    if move_times:
        print(f"\n  走子时间统计:")
        print(f"    总平均: {np.mean(move_times)*1000:.1f}ms")
        if checkpoint_times:
            for i, t in enumerate(checkpoint_times):
                print(f"    第{(i+1)*50}步: {t*1000:.1f}ms")
    
    print(f"{'='*70}\n")
    
    return {
        'winner': winner,
        'winner_is_enhanced': winner_is_enhanced,
        'steps': step,
        'white_count': white_count,
        'black_count': black_count,
        'dalian_count': len(dalian_registry),
        'total_dalian_uses': sum(info.use_count for info in dalian_registry.values())
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced vs Expert 对战测试')
    parser.add_argument('--model', type=str, 
                       default='exp/jcar_sft_2025_balanced/checkpoint_best.pt',
                       help='模型路径')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--depth', type=int, default=2, help='Expert搜索深度')
    parser.add_argument('--games', type=int, default=10, help='对局数')
    args = parser.parse_args()
    
    model_path = Path(__file__).parent.parent / args.model
    
    print(f"模型: {model_path}")
    print(f"Expert深度: {args.depth}")
    
    # 创建agents
    print("\n创建 Agents...")
    enhanced_agent = EnhancedJiuqiNetAgent(str(model_path), device=args.device, verbose=False)
    expert_agent = ExpertAgent(alpha_beta_depth=args.depth)
    
    # 统计
    enhanced_wins = 0
    expert_wins = 0
    draws = 0
    total_dalians = 0
    total_dalian_uses = 0
    
    results = []
    
    for i in range(args.games):
        # 黑白交替
        if i % 2 == 0:
            result = run_game(enhanced_agent, expert_agent, 
                            "Enhanced", f"Expert(d={args.depth})", i+1)
        else:
            result = run_game(expert_agent, enhanced_agent,
                            f"Expert(d={args.depth})", "Enhanced", i+1)
        
        results.append(result)
        
        if result['winner_is_enhanced'] is True:
            enhanced_wins += 1
        elif result['winner_is_enhanced'] is False:
            expert_wins += 1
        else:
            draws += 1
        
        total_dalians += result['dalian_count']
        total_dalian_uses += result['total_dalian_uses']
        
        print(f"\n>>> 当前战绩: Enhanced {enhanced_wins} - {expert_wins} Expert (平局:{draws})")
    
    # 最终统计
    print(f"\n{'#'*70}")
    print(f"#  最终统计 ({args.games}局)")
    print(f"{'#'*70}")
    print(f"  Enhanced胜: {enhanced_wins}局")
    print(f"  Expert胜:   {expert_wins}局")
    print(f"  平局:       {draws}局")
    print(f"  Enhanced胜率: {enhanced_wins/args.games*100:.1f}%")
    print(f"\n  褡裢统计:")
    print(f"    总形成: {total_dalians}个")
    print(f"    总使用: {total_dalian_uses}次")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
