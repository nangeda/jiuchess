#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢特征对战测试 - 详细版

记录详细信息：
- 每20步：白子/黑子数量、褡裢数、褡裢使用次数
- 每步走子时间
- 每局完整统计
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians
from jiu.jiutypes import Decision, board_gild, Point, Go, Skip_eat
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move
from jcar.candidate_features import build_enhanced_features


class BasicJiuqiNetAgent:
    """基础版 JiuqiNet Agent（无特征增强）"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        from jcar.model import JiuqiNet
        from jcar.config import JiuqiNetConfig
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_config' in checkpoint:
            cfg_dict = checkpoint['model_config']
            cfg = JiuqiNetConfig(**cfg_dict)
        else:
            cfg = JiuqiNetConfig()
        
        self.model = JiuqiNet(cfg)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def select_move(self, state: GameState):
        obs = encode_board_state(state)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        phase_id = get_phase_id(state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        legal_decs = state.legal_moves()
        if not legal_decs:
            return None, {}
        
        flying = state.board.get_player_total(state.next_player) <= 14
        cand_dicts = [decision_to_dict(d) for d in legal_decs]
        
        enhanced_feats = build_enhanced_features(cand_dicts, state, phase_id, flying)
        basic_feats = enhanced_feats[:, :14]
        cand_tensor = torch.from_numpy(basic_feats).float().to(self.device)
        
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        logits = logits_list[0].cpu().numpy()
        
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        best_idx = int(np.argmax(probs))
        
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        
        return move, {'value': value.item()}


def play_game_detailed(agent1, agent2, game_idx, max_steps=800, log_interval=20):
    """
    详细记录的对战
    agent1执白(Enhanced), agent2执黑(Basic)
    """
    state = GameState.new_game(14)
    step = 0
    
    # 走子时间统计
    enhanced_move_times = []
    basic_move_times = []
    
    # 褡裢统计
    enhanced_dalian_formed = 0
    enhanced_dalian_used = 0
    basic_dalian_formed = 0
    basic_dalian_used = 0
    prev_enhanced_dalian = 0
    prev_basic_dalian = 0
    
    # 检查点数据
    checkpoints = []
    
    game_start = time.time()
    
    print(f"\n{'─'*70}")
    print(f"📍 第 {game_idx} 局开始")
    print(f"   白方: Enhanced Agent (带褡裢增强)")
    print(f"   黑方: Basic Agent (无增强)")
    print(f"{'─'*70}")
    
    while not state.is_over() and step < max_steps:
        move_start = time.time()
        
        if state.next_player == Player.white:
            current_agent = agent1
            is_enhanced = True
        else:
            current_agent = agent2
            is_enhanced = False
        
        move, info = current_agent.select_move(state)
        
        move_time = (time.time() - move_start) * 1000  # 毫秒
        
        if move is None:
            break
        
        # 记录走子时间
        if is_enhanced:
            enhanced_move_times.append(move_time)
        else:
            basic_move_times.append(move_time)
        
        state = state.apply_move(move)
        step += 1
        
        # 统计褡裢（对战阶段）
        if state.step > board_gild:
            curr_enhanced_dalian = count_independent_dalians(state.board, Player.white)
            curr_basic_dalian = count_independent_dalians(state.board, Player.black)
            
            if state.next_player == Player.black:  # 刚走完的是白方(Enhanced)
                if curr_enhanced_dalian > prev_enhanced_dalian:
                    enhanced_dalian_formed += 1
                if info.get('uses_dalian', False):
                    enhanced_dalian_used += 1
            else:  # 刚走完的是黑方(Basic)
                if curr_basic_dalian > prev_basic_dalian:
                    basic_dalian_formed += 1
            
            prev_enhanced_dalian = curr_enhanced_dalian
            prev_basic_dalian = curr_basic_dalian
        
        # 每log_interval步输出检查点
        if step % log_interval == 0:
            w_count = state.board.get_player_total(Player.white)
            b_count = state.board.get_player_total(Player.black)
            
            if state.step <= board_gild:
                phase = "布局"
                enh_dalian = 0
                bas_dalian = 0
            else:
                phase = "飞子" if (w_count <= 14 or b_count <= 14) else "对战"
                enh_dalian = count_independent_dalians(state.board, Player.white)
                bas_dalian = count_independent_dalians(state.board, Player.black)
            
            # 计算这20步的平均走子时间
            recent_enh_times = enhanced_move_times[-10:] if enhanced_move_times else [0]
            recent_bas_times = basic_move_times[-10:] if basic_move_times else [0]
            avg_enh_time = np.mean(recent_enh_times)
            avg_bas_time = np.mean(recent_bas_times)
            
            checkpoint = {
                'step': step,
                'phase': phase,
                'white': w_count,
                'black': b_count,
                'enh_dalian': enh_dalian,
                'bas_dalian': bas_dalian,
                'enh_formed': enhanced_dalian_formed,
                'enh_used': enhanced_dalian_used,
                'bas_formed': basic_dalian_formed,
                'bas_used': basic_dalian_used,
                'avg_enh_time_ms': avg_enh_time,
                'avg_bas_time_ms': avg_bas_time,
            }
            checkpoints.append(checkpoint)
            
            # 直观输出
            print(f"   Step {step:4d} [{phase:2s}] | "
                  f"白○:{w_count:2d} 黑●:{b_count:2d} | "
                  f"褡裢 白:{enh_dalian} 黑:{bas_dalian} | "
                  f"成褡裢 E:{enhanced_dalian_formed}/B:{basic_dalian_formed} | "
                  f"用褡裢 E:{enhanced_dalian_used}/B:{basic_dalian_used} | "
                  f"走子时间 E:{avg_enh_time:5.1f}ms B:{avg_bas_time:5.1f}ms")
    
    # 最终结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    total_time = time.time() - game_start
    w_final = state.board.get_player_total(Player.white)
    b_final = state.board.get_player_total(Player.black)
    
    if winner == Player.white:
        winner_str = "白方(Enhanced)"
        winner_code = "Enhanced"
    elif winner == Player.black:
        winner_str = "黑方(Basic)"
        winner_code = "Basic"
    else:
        winner_str = "平局"
        winner_code = "Draw"
    
    # 输出本局汇总
    print(f"{'─'*70}")
    print(f"🏁 第 {game_idx} 局结束")
    print(f"   获胜者: {winner_str}")
    print(f"   最终子数: 白○ {w_final} vs 黑● {b_final}")
    print(f"   总步数: {step}步, 用时: {total_time:.1f}秒")
    print(f"   ┌──────────────────────────────────────────────────────────┐")
    print(f"   │ Enhanced(白)  褡裢统计: 形成 {enhanced_dalian_formed:2d} 次, 利用 {enhanced_dalian_used:2d} 次       │")
    print(f"   │ Basic(黑)     褡裢统计: 形成 {basic_dalian_formed:2d} 次, 利用 {basic_dalian_used:2d} 次       │")
    print(f"   ├──────────────────────────────────────────────────────────┤")
    if enhanced_move_times:
        print(f"   │ Enhanced 走子时间: 平均 {np.mean(enhanced_move_times):5.1f}ms, 最大 {np.max(enhanced_move_times):5.1f}ms │")
    if basic_move_times:
        print(f"   │ Basic    走子时间: 平均 {np.mean(basic_move_times):5.1f}ms, 最大 {np.max(basic_move_times):5.1f}ms │")
    print(f"   └──────────────────────────────────────────────────────────┘")
    
    return {
        'game_idx': game_idx,
        'winner': winner_code,
        'total_steps': step,
        'total_time': round(total_time, 2),
        'white_final': w_final,
        'black_final': b_final,
        'enhanced_dalian_formed': enhanced_dalian_formed,
        'enhanced_dalian_used': enhanced_dalian_used,
        'basic_dalian_formed': basic_dalian_formed,
        'basic_dalian_used': basic_dalian_used,
        'enhanced_avg_time_ms': round(np.mean(enhanced_move_times), 2) if enhanced_move_times else 0,
        'basic_avg_time_ms': round(np.mean(basic_move_times), 2) if basic_move_times else 0,
        'checkpoints': checkpoints,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='褡裢特征详细对战')
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='logs/dalian_battle.log')
    args = parser.parse_args()
    
    # 确保日志目录存在
    log_path = Path(args.output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    print("=" * 70)
    print("🎯 褡裢特征对战测试 - 详细版")
    print("=" * 70)
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   对战局数: {args.num_games}")
    print(f"   模型: {args.model}")
    print(f"   日志: {args.output}")
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
    
    basic_agent = BasicJiuqiNetAgent(args.model, args.device)
    print("✅ Agent加载完成")
    
    # 统计
    enhanced_wins, basic_wins, draws = 0, 0, 0
    total_enhanced_dalian_formed = 0
    total_enhanced_dalian_used = 0
    total_basic_dalian_formed = 0
    total_basic_dalian_used = 0
    all_game_stats = []
    
    print("\n" + "=" * 70)
    print("🏟️ 开始对战")
    print("=" * 70)
    
    for game_idx in range(1, args.num_games + 1):
        # 交替执白: 奇数局Enhanced执白，偶数局Enhanced执黑
        if game_idx % 2 == 1:
            white_agent, black_agent = enhanced_agent, basic_agent
            enhanced_is_white = True
        else:
            white_agent, black_agent = basic_agent, enhanced_agent
            enhanced_is_white = False
        
        # 对战
        stats = play_game_detailed(
            white_agent, black_agent, 
            game_idx,
            max_steps=args.max_steps,
            log_interval=args.log_interval
        )
        
        # 调整统计（如果Enhanced执黑）
        if not enhanced_is_white:
            # 交换统计
            stats['enhanced_dalian_formed'], stats['basic_dalian_formed'] = \
                stats['basic_dalian_formed'], stats['enhanced_dalian_formed']
            stats['enhanced_dalian_used'], stats['basic_dalian_used'] = \
                stats['basic_dalian_used'], stats['enhanced_dalian_used']
            stats['enhanced_avg_time_ms'], stats['basic_avg_time_ms'] = \
                stats['basic_avg_time_ms'], stats['enhanced_avg_time_ms']
            if stats['winner'] == 'Enhanced':
                stats['winner'] = 'Basic'
            elif stats['winner'] == 'Basic':
                stats['winner'] = 'Enhanced'
        
        all_game_stats.append(stats)
        
        # 更新总计
        if stats['winner'] == 'Enhanced':
            enhanced_wins += 1
        elif stats['winner'] == 'Basic':
            basic_wins += 1
        else:
            draws += 1
        
        total_enhanced_dalian_formed += stats['enhanced_dalian_formed']
        total_enhanced_dalian_used += stats['enhanced_dalian_used']
        total_basic_dalian_formed += stats['basic_dalian_formed']
        total_basic_dalian_used += stats['basic_dalian_used']
        
        # 每10局打印汇总
        if game_idx % 10 == 0:
            print(f"\n{'═'*70}")
            print(f"📊 前 {game_idx} 局汇总")
            print(f"{'═'*70}")
            print(f"   Enhanced 胜: {enhanced_wins:3d} ({enhanced_wins/game_idx*100:5.1f}%)")
            print(f"   Basic    胜: {basic_wins:3d} ({basic_wins/game_idx*100:5.1f}%)")
            print(f"   平局:        {draws:3d}")
            print(f"   ────────────────────────────────────────")
            print(f"   Enhanced 总褡裢: 形成 {total_enhanced_dalian_formed}, 利用 {total_enhanced_dalian_used}")
            print(f"   Basic    总褡裢: 形成 {total_basic_dalian_formed}, 利用 {total_basic_dalian_used}")
            print(f"{'═'*70}\n")
    
    # 最终汇总
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "═" * 70)
    print("🏆 最终结果")
    print("═" * 70)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                           胜率统计                                  │
├────────────────────────────────────────────────────────────────────┤
│  Enhanced (带褡裢增强):  {enhanced_wins:3d} 胜  ({enhanced_wins/args.num_games*100:5.1f}%)                       │
│  Basic    (无增强):      {basic_wins:3d} 胜  ({basic_wins/args.num_games*100:5.1f}%)                       │
│  平局:                   {draws:3d}     ({draws/args.num_games*100:5.1f}%)                       │
├────────────────────────────────────────────────────────────────────┤
│                           褡裢统计                                  │
├────────────────────────────────────────────────────────────────────┤
│  Enhanced:  形成 {total_enhanced_dalian_formed:4d} 次 (平均 {total_enhanced_dalian_formed/args.num_games:.2f}/局)                        │
│             利用 {total_enhanced_dalian_used:4d} 次 (平均 {total_enhanced_dalian_used/args.num_games:.2f}/局)                        │
│  Basic:     形成 {total_basic_dalian_formed:4d} 次 (平均 {total_basic_dalian_formed/args.num_games:.2f}/局)                        │
│             利用 {total_basic_dalian_used:4d} 次 (平均 {total_basic_dalian_used/args.num_games:.2f}/局)                        │
├────────────────────────────────────────────────────────────────────┤
│                           时间统计                                  │
├────────────────────────────────────────────────────────────────────┤
│  总耗时: {total_time/60:6.1f} 分钟                                            │
│  平均每局: {total_time/args.num_games:5.1f} 秒                                             │
└────────────────────────────────────────────────────────────────────┘
""")
    
    # 保存JSON
    result_json = {
        'summary': {
            'total_games': args.num_games,
            'enhanced_wins': enhanced_wins,
            'basic_wins': basic_wins,
            'draws': draws,
            'enhanced_win_rate': round(enhanced_wins/args.num_games*100, 2),
            'total_enhanced_dalian_formed': total_enhanced_dalian_formed,
            'total_enhanced_dalian_used': total_enhanced_dalian_used,
            'total_basic_dalian_formed': total_basic_dalian_formed,
            'total_basic_dalian_used': total_basic_dalian_used,
            'total_time_seconds': round(total_time, 2),
        },
        'games': all_game_stats
    }
    
    json_path = log_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 详细结果已保存到: {json_path}")
    print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)


if __name__ == '__main__':
    main()


