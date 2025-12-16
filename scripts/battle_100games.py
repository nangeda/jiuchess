#!/usr/bin/env python3
"""增强型Agent vs Expert 100局对战 - 胜率统计"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Player
from jiu.jiutypes import board_gild
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from baseline.baseline_fxy_expert.adapter import ExpertAgent
import time

def play_game(agent_black, agent_white, max_steps=800):
    """进行一局对战"""
    state = GameState.new_game(14)
    
    for step in range(max_steps):
        current = state.next_player
        agent = agent_black if current == Player.black else agent_white
        
        result = agent.select_move(state)
        move = result[0] if isinstance(result, tuple) else result
        
        if move is None:
            return current.other, step, '无法走棋'
        
        try:
            state = state.apply_move(move)
        except:
            return current.other, step, '非法动作'
        
        winner = state.winner()
        if winner:
            return winner, step + 1, '胜利'
    
    b = state.board.get_player_total(Player.black)
    w = state.board.get_player_total(Player.white)
    if b > w: 
        return Player.black, max_steps, f'超时({b}>{w})'
    elif w > b: 
        return Player.white, max_steps, f'超时({w}>{b})'
    return None, max_steps, '平局'


def main():
    print('=' * 70, flush=True)
    print('     🎮 增强型Agent vs Expert(深度2) 100局对战 🎮', flush=True)
    print('=' * 70, flush=True)
    
    # 加载Agent
    print('\n🔵 加载 Enhanced JiuqiNet Agent...', flush=True)
    enhanced_agent = EnhancedJiuqiNetAgent(
        'exp/jcar_sft_2025_balanced/checkpoint_best.pt', 
        'cuda:0',
        square_weight=3.0,
        eat_weight=2.5,
        safety_weight=1.0,
        verbose=False
    )
    
    print('🟢 加载 Expert Agent (搜索深度=2)...', flush=True)
    expert_agent = ExpertAgent(alpha_beta_depth=2)
    
    print('\n' + '=' * 70, flush=True)
    print('  开始100局对战...', flush=True)
    print('=' * 70, flush=True)
    
    # 统计
    enhanced_wins, expert_wins, draws = 0, 0, 0
    enhanced_black_wins, enhanced_white_wins = 0, 0
    expert_black_wins, expert_white_wins = 0, 0
    total_steps = []
    
    start_time = time.time()
    
    for i in range(100):
        if i % 2 == 0:
            black, white = enhanced_agent, expert_agent
            bn, wn = 'Enhanced', 'Expert'
        else:
            black, white = expert_agent, enhanced_agent
            bn, wn = 'Expert', 'Enhanced'
        
        game_start = time.time()
        winner, steps, reason = play_game(black, white, 800)
        game_time = time.time() - game_start
        
        total_steps.append(steps)
        
        wname = bn if winner == Player.black else (wn if winner == Player.white else '平局')
        
        if wname == 'Enhanced':
            enhanced_wins += 1
            if bn == 'Enhanced':
                enhanced_black_wins += 1
            else:
                enhanced_white_wins += 1
        elif wname == 'Expert':
            expert_wins += 1
            if bn == 'Expert':
                expert_black_wins += 1
            else:
                expert_white_wins += 1
        else:
            draws += 1
        
        # 每局输出简短结果
        emoji = '🏆' if wname == 'Enhanced' else ('💀' if wname == 'Expert' else '🤝')
        print(f'{emoji} 第{i+1:3d}局: {bn:8s} vs {wn:8s} -> {wname:8s} ({steps:3d}步, {game_time:.0f}s) | 战绩: {enhanced_wins}:{expert_wins}', flush=True)
        
        # 每10局输出详细统计
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            win_rate = enhanced_wins / (i + 1) * 100
            print(f'\n📊 [{i+1}局] Enhanced胜率: {win_rate:.1f}% ({enhanced_wins}/{i+1}) | 用时: {elapsed/60:.1f}分钟\n', flush=True)
    
    total_time = time.time() - start_time
    
    # ==================== 详细报告 ====================
    print('\n')
    print('=' * 70, flush=True)
    print('                   📋 100局对战报告 📋', flush=True)
    print('=' * 70, flush=True)
    
    print('\n【总体战绩】', flush=True)
    print('─' * 50, flush=True)
    print(f'  🔵 Enhanced 胜: {enhanced_wins:3d} 局  ({enhanced_wins:.1f}%)', flush=True)
    print(f'  🟢 Expert 胜:   {expert_wins:3d} 局  ({expert_wins:.1f}%)', flush=True)
    print(f'  ⚖️  平局:        {draws:3d} 局  ({draws:.1f}%)', flush=True)
    
    print('\n【分先战绩】', flush=True)
    print('─' * 50, flush=True)
    print(f'  Enhanced 执黑: {enhanced_black_wins:2d}/50 胜  ({enhanced_black_wins/50*100:.0f}%)', flush=True)
    print(f'  Enhanced 执白: {enhanced_white_wins:2d}/50 胜  ({enhanced_white_wins/50*100:.0f}%)', flush=True)
    print(f'  Expert 执黑:   {expert_black_wins:2d}/50 胜  ({expert_black_wins/50*100:.0f}%)', flush=True)
    print(f'  Expert 执白:   {expert_white_wins:2d}/50 胜  ({expert_white_wins/50*100:.0f}%)', flush=True)
    
    print('\n【对局统计】', flush=True)
    print('─' * 50, flush=True)
    print(f'  平均步数: {sum(total_steps)/len(total_steps):.0f} 步', flush=True)
    print(f'  最短对局: {min(total_steps)} 步', flush=True)
    print(f'  最长对局: {max(total_steps)} 步', flush=True)
    print(f'  总用时:   {total_time/60:.1f} 分钟', flush=True)
    print(f'  平均每局: {total_time/100:.1f} 秒', flush=True)
    
    print('\n【结论】', flush=True)
    print('=' * 70, flush=True)
    if enhanced_wins > expert_wins:
        print(f'  ✅ Enhanced Agent 以 {enhanced_wins}:{expert_wins} 战胜 Expert!', flush=True)
        print(f'  ✅ 胜率: {enhanced_wins}%', flush=True)
    elif expert_wins > enhanced_wins:
        print(f'  ❌ Expert 以 {expert_wins}:{enhanced_wins} 战胜 Enhanced Agent', flush=True)
        print(f'  ❌ Enhanced胜率: {enhanced_wins}%', flush=True)
    else:
        print(f'  ⚖️ 双方战平 {enhanced_wins}:{expert_wins}', flush=True)
    
    print('=' * 70, flush=True)
    print('🎮 100局对战完成!', flush=True)


if __name__ == '__main__':
    main()

