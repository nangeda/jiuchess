#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic Enhanced Agent vs Expert 可视化对战"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import board_gild, Point, board_size
from agent.basic_enhanced_agent import BasicEnhancedAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent


def print_board(state, highlight_src=None, highlight_dst=None):
    board = state.board
    print('      ', end='')
    for c in range(1, board_size + 1):
        print(f'{c:3d}', end='')
    print()
    print('    +' + '---' * board_size + '+')
    
    for r in range(1, board_size + 1):
        print(f' {r:2d} |', end='')
        for c in range(1, board_size + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            is_new = (highlight_dst and pt == highlight_dst)
            
            if pl == Player.white:
                char = '[O]' if is_new else ' O '
            elif pl == Player.black:
                char = '[X]' if is_new else ' X '
            else:
                if highlight_src and pt == highlight_src:
                    char = ' _ '
                else:
                    char = ' . '
            print(char, end='')
        print('|')
    print('    +' + '---' * board_size + '+')


def get_move_info(move):
    src, dst = None, None
    desc = ''
    if move.is_put:
        dst = move.point
        desc = f'Put ({dst.row},{dst.col})'
    elif move.is_go:
        src, dst = move.go_to.go, move.go_to.to
        desc = f'Go ({src.row},{src.col})->({dst.row},{dst.col})'
    elif move.is_fly:
        src, dst = move.go_to.go, move.go_to.to
        desc = f'Fly ({src.row},{src.col})->({dst.row},{dst.col})'
    elif move.is_skip_eat:
        se = move.skip_eat_points
        src, dst = se.go, se.to
        desc = f'Jump ({src.row},{src.col})->({dst.row},{dst.col}) eat({se.eat.row},{se.eat.col})'
    elif move.is_skip_eat_seq:
        seq = move.skip_eat_points
        src, dst = seq[0].go, seq[-1].to
        desc = f'MultiJump ({src.row},{src.col})->({dst.row},{dst.col}) eat {len(seq)}'
    return src, dst, desc


def main():
    print('=' * 70)
    print('Basic Enhanced Agent vs Expert')
    print('=' * 70)
    
    print('\nLoading agents...')
    basic_agent = BasicEnhancedAgent(
        'exp/jcar_sft_2025_balanced/checkpoint_best.pt',
        device='cuda:0'
    )
    expert_agent = ExpertAgent(alpha_beta_depth=2)
    print('Done!\n')
    
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    basic_is_white = True
    
    print('White: Basic Enhanced  vs  Black: Expert')
    print('=' * 70)
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        
        if (current_player == Player.white and basic_is_white) or \
           (current_player == Player.black and not basic_is_white):
            move, info = basic_agent.select_move(state)
            agent_name = 'Basic'
        else:
            move, _ = expert_agent.select_move(state)
            info = {}
            agent_name = 'Expert'
        
        if move is None:
            break
        
        src, dst, move_desc = get_move_info(move)
        state = state.apply_move(move)
        step += 1
        
        w_count = state.board.get_player_total(Player.white)
        b_count = state.board.get_player_total(Player.black)
        
        if state.step <= board_gild:
            phase = 'Layout'
            in_battle = False
        elif w_count <= 14 or b_count <= 14:
            phase = 'Flying'
            in_battle = True
        else:
            phase = 'Battle'
            in_battle = True
        
        # 对战阶段每15步显示一次棋盘
        if in_battle and step % 15 == 0:
            print(f'\n{"-"*70}')
            print(f'[Step {step}] {phase} | White(O):{w_count} Black(X):{b_count}')
            print(f'  {agent_name}: {move_desc}')
            print()
            print_board(state, src, dst)
        elif step == board_gild:
            print(f'\n{"="*70}')
            print(f'Layout Done (Step {step}) | White(O):{w_count} Black(X):{b_count}')
            print('=' * 70)
            print()
            print_board(state)
        elif step % 40 == 0 and not in_battle:
            print(f'  Layout... Step {step}: W{w_count} B{b_count}')
    
    # 最终结果
    winner = state.winner()
    if winner is None:
        winner = state.winner_by_timeout()
    
    w_count = state.board.get_player_total(Player.white)
    b_count = state.board.get_player_total(Player.black)
    
    print('\n' + '=' * 70)
    print('GAME OVER')
    print('=' * 70)
    print(f'\nFinal Board (Step {step}):')
    print()
    print_board(state)
    print(f'\nFinal pieces: White(O) {w_count} vs Black(X) {b_count}')
    
    if winner == Player.white:
        winner_name = 'Basic Enhanced (White)'
    elif winner == Player.black:
        winner_name = 'Expert (Black)'
    else:
        winner_name = 'Draw'
    
    print(f'\nWinner: {winner_name}')


if __name__ == '__main__':
    main()
