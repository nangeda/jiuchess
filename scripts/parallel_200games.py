#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行对战: Basic Enhanced vs Expert (100局) + Dalian Enhanced vs Expert (100局)
"""

import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))


def play_single_game(args):
    """单局对战 - 在子进程中运行"""
    game_idx, agent_type, is_white, model_path, expert_depth = args
    
    import torch
    from jiu.jiuboard_fast import GameState, Player, count_independent_dalians
    from jiu.jiutypes import board_gild
    from battle_test import decision_to_move
    from baseline.baseline_fxy_expert.adapter import ExpertAgent
    
    # 根据agent类型加载不同的Agent
    if agent_type == 'basic':
        from agent.basic_enhanced_agent import BasicEnhancedAgent
        agent = BasicEnhancedAgent(model_path, device='cuda:0')
    else:
        from agent.dalian_enhanced_agent import DalianEnhancedAgent
        agent = DalianEnhancedAgent(model_path, device='cuda:0')
    
    expert = ExpertAgent(alpha_beta_depth=expert_depth)
    
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    # 统计
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
        
        # 褡裢统计 (仅对dalian agent)
        if agent_type == 'dalian' and state.step > board_gild:
            agent_player = Player.white if is_white else Player.black
            curr_dalian = count_independent_dalians(state.board, agent_player)
            
            if is_agent:
                if curr_dalian > prev_dalian:
                    dalian_formed += 1
                if info.get('uses_dalian', False):
                    dalian_used += 1
            
            prev_dalian = curr_dalian
    
    # 结果
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
    
    return {
        'game_idx': game_idx,
        'agent_type': agent_type,
        'is_white': is_white,
        'winner': winner_code,
        'agent_pieces': agent_count,
        'expert_pieces': expert_count,
        'steps': step,
        'dalian_formed': dalian_formed,
        'dalian_used': dalian_used
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--expert-depth', type=int, default=2)
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='logs/1215_200games.log')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Parallel Battle: Basic/Dalian Enhanced vs Expert")
    print("=" * 70)
    print(f"  Games per agent: {args.num_games}")
    print(f"  Expert depth: {args.expert_depth}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    print("=" * 70)
    
    # 准备所有游戏任务
    tasks = []
    
    # Basic Enhanced vs Expert (100局)
    for i in range(args.num_games):
        is_white = (i % 2 == 0)  # 交替执白
        tasks.append((i + 1, 'basic', is_white, args.model, args.expert_depth))
    
    # Dalian Enhanced vs Expert (100局)
    for i in range(args.num_games):
        is_white = (i % 2 == 0)
        tasks.append((i + 1, 'dalian', is_white, args.model, args.expert_depth))
    
    print(f"\nTotal games: {len(tasks)}")
    print("Starting parallel execution...\n")
    
    results = []
    start_time = time.time()
    
    # 并行执行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(play_single_game, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # 进度报告
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{len(tasks)} ({completed*100//len(tasks)}%) "
                          f"| Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            except Exception as e:
                print(f"  Error in game: {e}")
    
    total_time = time.time() - start_time
    print(f"\nAll games completed in {total_time:.1f}s")
    
    # 分析结果
    basic_results = [r for r in results if r['agent_type'] == 'basic']
    dalian_results = [r for r in results if r['agent_type'] == 'dalian']
    
    def analyze(res_list, name):
        wins = sum(1 for r in res_list if r['winner'] == 'agent')
        losses = sum(1 for r in res_list if r['winner'] == 'expert')
        draws = sum(1 for r in res_list if r['winner'] == 'draw')
        
        total = len(res_list)
        win_rate = wins / total * 100 if total > 0 else 0
        
        avg_agent_pieces = sum(r['agent_pieces'] for r in res_list) / total if total > 0 else 0
        avg_expert_pieces = sum(r['expert_pieces'] for r in res_list) / total if total > 0 else 0
        avg_steps = sum(r['steps'] for r in res_list) / total if total > 0 else 0
        
        # 褡裢统计
        total_formed = sum(r['dalian_formed'] for r in res_list)
        total_used = sum(r['dalian_used'] for r in res_list)
        
        return {
            'name': name,
            'total': total,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'avg_agent_pieces': avg_agent_pieces,
            'avg_expert_pieces': avg_expert_pieces,
            'avg_steps': avg_steps,
            'dalian_formed': total_formed,
            'dalian_used': total_used
        }
    
    basic_stats = analyze(basic_results, 'Basic Enhanced')
    dalian_stats = analyze(dalian_results, 'Dalian Enhanced')
    
    # 输出结果
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("BATTLE RESULTS: Basic/Dalian Enhanced vs Expert (depth=2)")
    output_lines.append("=" * 70)
    output_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Total time: {total_time:.1f}s")
    output_lines.append("")
    
    for stats in [basic_stats, dalian_stats]:
        output_lines.append("-" * 70)
        output_lines.append(f"{stats['name']} vs Expert ({stats['total']} games)")
        output_lines.append("-" * 70)
        output_lines.append(f"  Wins:   {stats['wins']:3d} ({stats['win_rate']:.1f}%)")
        output_lines.append(f"  Losses: {stats['losses']:3d} ({stats['losses']/stats['total']*100:.1f}%)")
        output_lines.append(f"  Draws:  {stats['draws']:3d} ({stats['draws']/stats['total']*100:.1f}%)")
        output_lines.append(f"  Avg pieces - Agent: {stats['avg_agent_pieces']:.1f}, Expert: {stats['avg_expert_pieces']:.1f}")
        output_lines.append(f"  Avg steps: {stats['avg_steps']:.1f}")
        if stats['name'] == 'Dalian Enhanced':
            output_lines.append(f"  Dalian formed: {stats['dalian_formed']} (avg {stats['dalian_formed']/stats['total']:.1f}/game)")
            output_lines.append(f"  Dalian used: {stats['dalian_used']} (avg {stats['dalian_used']/stats['total']:.1f}/game)")
        output_lines.append("")
    
    # 对比分析
    output_lines.append("=" * 70)
    output_lines.append("COMPARISON")
    output_lines.append("=" * 70)
    win_diff = dalian_stats['win_rate'] - basic_stats['win_rate']
    output_lines.append(f"  Basic Enhanced win rate:  {basic_stats['win_rate']:.1f}%")
    output_lines.append(f"  Dalian Enhanced win rate: {dalian_stats['win_rate']:.1f}%")
    output_lines.append(f"  Difference: {win_diff:+.1f}%")
    output_lines.append("")
    
    if win_diff > 0:
        output_lines.append(f"  >>> Dalian Enhanced is BETTER by {win_diff:.1f}% win rate")
    elif win_diff < 0:
        output_lines.append(f"  >>> Basic Enhanced is BETTER by {-win_diff:.1f}% win rate")
    else:
        output_lines.append(f"  >>> Both agents perform equally")
    
    output_lines.append("")
    output_lines.append("=" * 70)
    
    # 保存详细结果
    output_lines.append("\nDETAILED RESULTS:")
    output_lines.append("-" * 70)
    for r in sorted(results, key=lambda x: (x['agent_type'], x['game_idx'])):
        side = 'W' if r['is_white'] else 'B'
        output_lines.append(f"  {r['agent_type']:6s} #{r['game_idx']:3d} ({side}) | "
                           f"Winner: {r['winner']:6s} | "
                           f"Pieces: {r['agent_pieces']:2d} vs {r['expert_pieces']:2d} | "
                           f"Steps: {r['steps']:3d}")
    
    # 保存到文件
    output_text = "\n".join(output_lines)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(output_text)
    
    print(output_text)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
