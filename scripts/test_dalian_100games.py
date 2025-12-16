#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢特征100局对战测试

记录每局情况：走子时间、褡裢数量、使用褡裢次数
每20步记录一次数据
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


def play_game_with_logging(agent1, agent2, game_idx, max_steps=800, log_interval=20):
    """
    对战一局并记录数据
    agent1执白, agent2执黑
    每log_interval步记录一次数据
    """
    state = GameState.new_game(14)
    step = 0
    
    # 统计数据
    stats = {
        'game_idx': game_idx,
        'start_time': datetime.now().strftime('%H:%M:%S'),
        'checkpoints': [],  # 每20步的检查点
        'enhanced_dalian_formed': 0,
        'enhanced_dalian_used': 0,
        'basic_dalian_formed': 0,
        'basic_dalian_used': 0,
    }
    
    prev_enhanced_dalian = 0
    prev_basic_dalian = 0
    game_start = time.time()
    checkpoint_start = time.time()
    
    while not state.is_over() and step < max_steps:
        if state.next_player == Player.white:
            current_agent = agent1
            is_enhanced = True
        else:
            current_agent = agent2
            is_enhanced = False
        
        move, info = current_agent.select_move(state)
        if move is None:
            break
        
        state = state.apply_move(move)
        step += 1
        
        # 统计褡裢（对战阶段）
        if state.step > board_gild:
            curr_enhanced_dalian = count_independent_dalians(state.board, Player.white)
            curr_basic_dalian = count_independent_dalians(state.board, Player.black)
            
            if state.next_player == Player.black:  # 刚走完的是白方
                if curr_enhanced_dalian > prev_enhanced_dalian:
                    stats['enhanced_dalian_formed'] += 1
                if info.get('uses_dalian', False):
                    stats['enhanced_dalian_used'] += 1
            else:
                if curr_basic_dalian > prev_basic_dalian:
                    stats['basic_dalian_formed'] += 1
            
            prev_enhanced_dalian = curr_enhanced_dalian
            prev_basic_dalian = curr_basic_dalian
        
        # 每log_interval步记录检查点
        if step % log_interval == 0:
            checkpoint_time = time.time() - checkpoint_start
            w_count = state.board.get_player_total(Player.white)
            b_count = state.board.get_player_total(Player.black)
            
            phase = "布局" if state.step <= board_gild else ("飞子" if w_count <= 14 or b_count <= 14 else "对战")
            
            checkpoint = {
                'step': step,
                'phase': phase,
                'white_pieces': w_count,
                'black_pieces': b_count,
                'enhanced_dalians': count_independent_dalians(state.board, Player.white) if state.step > board_gild else 0,
                'basic_dalians': count_independent_dalians(state.board, Player.black) if state.step > board_gild else 0,
                'time_20steps': round(checkpoint_time, 2),
            }
            stats['checkpoints'].append(checkpoint)
            checkpoint_start = time.time()
    
    # 最终结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    stats['total_steps'] = step
    stats['total_time'] = round(time.time() - game_start, 2)
    stats['white_pieces'] = state.board.get_player_total(Player.white)
    stats['black_pieces'] = state.board.get_player_total(Player.black)
    
    if winner == Player.white:
        stats['winner'] = 'Enhanced'
    elif winner == Player.black:
        stats['winner'] = 'Basic'
    else:
        stats['winner'] = 'Draw'
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='褡裢特征100局对战')
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='logs/dalian_100games.log')
    args = parser.parse_args()
    
    # 确保日志目录存在
    log_path = Path(args.output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 开始时间
    start_time = datetime.now()
    
    print("=" * 70, flush=True)
    print(f"🎯 褡裢特征100局对战测试", flush=True)
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"   日志文件: {args.output}", flush=True)
    print("=" * 70, flush=True)
    
    # 加载Agent
    print("\n📦 加载Agent...", flush=True)
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
    print("✅ Agent加载完成\n", flush=True)
    
    # 统计
    enhanced_wins, basic_wins, draws = 0, 0, 0
    total_enhanced_dalian_formed = 0
    total_enhanced_dalian_used = 0
    total_basic_dalian_formed = 0
    total_basic_dalian_used = 0
    all_game_stats = []
    
    print("=" * 70, flush=True)
    print("🏟️ 开始对战", flush=True)
    print("=" * 70, flush=True)
    
    for game_idx in range(args.num_games):
        # 交替执白
        if game_idx % 2 == 0:
            white_agent, black_agent = enhanced_agent, basic_agent
        else:
            white_agent, black_agent = basic_agent, enhanced_agent
        
        # 对战
        stats = play_game_with_logging(
            white_agent, black_agent, 
            game_idx + 1,
            max_steps=args.max_steps,
            log_interval=args.log_interval
        )
        
        # 调整统计（考虑交替执白）
        if game_idx % 2 == 1:
            # 这局Enhanced执黑，需要交换统计
            stats['enhanced_dalian_formed'], stats['basic_dalian_formed'] = \
                stats['basic_dalian_formed'], stats['enhanced_dalian_formed']
            stats['enhanced_dalian_used'], stats['basic_dalian_used'] = \
                stats['basic_dalian_used'], stats['enhanced_dalian_used']
            if stats['winner'] == 'Enhanced':
                stats['winner'] = 'Basic'
            elif stats['winner'] == 'Basic':
                stats['winner'] = 'Enhanced'
        
        all_game_stats.append(stats)
        
        # 更新总计
        if stats['winner'] == 'Enhanced':
            enhanced_wins += 1
            result_sym = "✅"
        elif stats['winner'] == 'Basic':
            basic_wins += 1
            result_sym = "❌"
        else:
            draws += 1
            result_sym = "🤝"
        
        total_enhanced_dalian_formed += stats['enhanced_dalian_formed']
        total_enhanced_dalian_used += stats['enhanced_dalian_used']
        total_basic_dalian_formed += stats['basic_dalian_formed']
        total_basic_dalian_used += stats['basic_dalian_used']
        
        # 打印每局结果
        color = "白" if game_idx % 2 == 0 else "黑"
        print(f"第{game_idx+1:3d}局 {result_sym} Enhanced({color}): "
              f"{stats['winner']:8s} | {stats['total_steps']:3d}步 {stats['total_time']:5.1f}s | "
              f"褡裢 E[{stats['enhanced_dalian_formed']}/{stats['enhanced_dalian_used']}] "
              f"B[{stats['basic_dalian_formed']}/{stats['basic_dalian_used']}] | "
              f"子数 {stats['white_pieces']:2d}:{stats['black_pieces']:2d}", flush=True)
        
        # 每10局打印一次汇总
        if (game_idx + 1) % 10 == 0:
            print(f"\n--- 前{game_idx+1}局汇总: Enhanced {enhanced_wins}胜 "
                  f"({enhanced_wins/(game_idx+1)*100:.1f}%) | "
                  f"Basic {basic_wins}胜 | 平局 {draws} ---\n", flush=True)
    
    # 最终汇总
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70, flush=True)
    print("📊 最终结果汇总", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n【胜率统计】", flush=True)
    print(f"  Enhanced (带褡裢增强): {enhanced_wins:3d} 胜 ({enhanced_wins/args.num_games*100:5.1f}%)", flush=True)
    print(f"  Basic    (无增强):     {basic_wins:3d} 胜 ({basic_wins/args.num_games*100:5.1f}%)", flush=True)
    print(f"  平局:                  {draws:3d}    ({draws/args.num_games*100:5.1f}%)", flush=True)
    
    print(f"\n【褡裢统计】", flush=True)
    print(f"  Enhanced:", flush=True)
    print(f"    - 形成褡裢总次数: {total_enhanced_dalian_formed}", flush=True)
    print(f"    - 利用褡裢总次数: {total_enhanced_dalian_used}", flush=True)
    print(f"    - 平均每局形成: {total_enhanced_dalian_formed/args.num_games:.2f}", flush=True)
    print(f"    - 平均每局利用: {total_enhanced_dalian_used/args.num_games:.2f}", flush=True)
    
    print(f"  Basic:", flush=True)
    print(f"    - 形成褡裢总次数: {total_basic_dalian_formed}", flush=True)
    print(f"    - 利用褡裢总次数: {total_basic_dalian_used}", flush=True)
    print(f"    - 平均每局形成: {total_basic_dalian_formed/args.num_games:.2f}", flush=True)
    print(f"    - 平均每局利用: {total_basic_dalian_used/args.num_games:.2f}", flush=True)
    
    print(f"\n【时间统计】", flush=True)
    print(f"  总耗时: {total_time/60:.1f} 分钟", flush=True)
    print(f"  平均每局: {total_time/args.num_games:.1f} 秒", flush=True)
    
    # 保存详细结果到JSON
    result_json = {
        'summary': {
            'total_games': args.num_games,
            'enhanced_wins': enhanced_wins,
            'basic_wins': basic_wins,
            'draws': draws,
            'enhanced_win_rate': enhanced_wins/args.num_games*100,
            'total_enhanced_dalian_formed': total_enhanced_dalian_formed,
            'total_enhanced_dalian_used': total_enhanced_dalian_used,
            'total_basic_dalian_formed': total_basic_dalian_formed,
            'total_basic_dalian_used': total_basic_dalian_used,
            'total_time_seconds': total_time,
        },
        'games': all_game_stats
    }
    
    json_path = log_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 详细结果已保存到: {json_path}", flush=True)
    print(f"✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()


