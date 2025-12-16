#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢特征测试脚本

测试带褡裢特征增强的Agent vs 普通Agent
记录胜率、形成褡裢次数、利用褡裢次数
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians
from jiu.jiutypes import Decision, board_gild, Point, Go, Skip_eat
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move


class BasicJiuqiNetAgent:
    """
    基础版 JiuqiNet Agent（无特征增强）
    只使用神经网络评分，不加规则加成
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        from jcar.model import JiuqiNet
        from jcar.config import JiuqiNetConfig
        from jcar.candidate_features import build_enhanced_features
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.build_enhanced_features = build_enhanced_features
        
        print(f"🔵 Basic JiuqiNet Agent (无增强) on {self.device}")
        
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
        print(f"✅ Model loaded from {model_path}")
    
    def select_move(self, state: GameState):
        obs = encode_board_state(state)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        phase_id = get_phase_id(state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        legal_decs = state.legal_moves()
        if not legal_decs:
            return None, {'error': 'no legal moves'}
        
        flying = state.board.get_player_total(state.next_player) <= 14
        cand_dicts = [decision_to_dict(d) for d in legal_decs]
        
        # 构建特征（但只使用前14维基础特征）
        enhanced_feats = self.build_enhanced_features(cand_dicts, state, phase_id, flying)
        basic_feats = enhanced_feats[:, :14]
        cand_tensor = torch.from_numpy(basic_feats).float().to(self.device)
        
        # 纯模型评分，不加任何规则加成
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        logits = logits_list[0].cpu().numpy()
        
        # 直接选最高分
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        best_idx = int(np.argmax(probs))
        
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        
        info = {
            'value': value.item(),
            'prob': float(probs[best_idx]),
        }
        
        return move, info


class DalianStatsCollector:
    """褡裢统计收集器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dalian_formed = {Player.white: 0, Player.black: 0}  # 形成褡裢次数
        self.dalian_used = {Player.white: 0, Player.black: 0}    # 利用褡裢吃子次数
        self.max_dalians = {Player.white: 0, Player.black: 0}    # 最大褡裢数
        self.prev_dalian_count = {Player.white: 0, Player.black: 0}
    
    def update(self, state: GameState, player: Player, move_info: dict):
        """更新褡裢统计"""
        if state.step < board_gild:
            return  # 布局阶段不统计
        
        # 当前褡裢数量
        current_dalians = count_independent_dalians(state.board, player)
        
        # 检查是否形成了新褡裢
        if current_dalians > self.prev_dalian_count[player]:
            self.dalian_formed[player] += 1
        
        # 更新最大褡裢数
        if current_dalians > self.max_dalians[player]:
            self.max_dalians[player] = current_dalians
        
        # 检查是否利用褡裢吃子
        if move_info.get('uses_dalian', False):
            self.dalian_used[player] += 1
        
        self.prev_dalian_count[player] = current_dalians
    
    def get_stats(self):
        return {
            'dalian_formed': dict(self.dalian_formed),
            'dalian_used': dict(self.dalian_used),
            'max_dalians': dict(self.max_dalians),
        }


def play_game_with_stats(agent1, agent2, max_steps=800, verbose=False):
    """
    进行一局对战，agent1执白，agent2执黑
    同时收集褡裢统计数据
    """
    state = GameState.new_game(14)
    step = 0
    
    stats1 = DalianStatsCollector()  # agent1的统计
    stats2 = DalianStatsCollector()  # agent2的统计
    
    while not state.is_over() and step < max_steps:
        if state.next_player == Player.white:
            current_agent = agent1
            current_stats = stats1
            player = Player.white
        else:
            current_agent = agent2
            current_stats = stats2
            player = Player.black
        
        move, info = current_agent.select_move(state)
        if move is None:
            break
        
        state = state.apply_move(move)
        step += 1
        
        # 更新褡裢统计
        current_stats.update(state, player, info)
        
        if verbose and step % 100 == 0:
            w = state.board.get_player_total(Player.white)
            b = state.board.get_player_total(Player.black)
            print(f"  Step {step}: 白{w} vs 黑{b}")
    
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    return winner, step, stats1.get_stats(), stats2.get_stats()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='褡裢特征测试')
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt',
                        help='模型路径')
    parser.add_argument('--num-games', type=int, default=10, help='对战局数')
    parser.add_argument('--max-steps', type=int, default=800, help='每局最大步数')
    parser.add_argument('--device', default='cuda:0', help='设备')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 褡裢特征增强测试")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"对战局数: {args.num_games}")
    print("=" * 70)
    
    # 创建两个Agent
    print("\n📦 加载Agent...")
    enhanced_agent = EnhancedJiuqiNetAgent(
        args.model, 
        args.device,
        # 褡裢权重
        dalian_create_weight=12.0,
        dalian_use_weight=20.0,
        dalian_break_weight=10.0,
        pre_dalian_weight=5.0,
        verbose=False
    )
    
    basic_agent = BasicJiuqiNetAgent(args.model, args.device)
    
    print("\n" + "=" * 70)
    print("🏟️ 开始对战")
    print("=" * 70)
    
    # 统计
    enhanced_wins, basic_wins, draws = 0, 0, 0
    total_enhanced_dalian_formed = 0
    total_enhanced_dalian_used = 0
    total_basic_dalian_formed = 0
    total_basic_dalian_used = 0
    game_details = []
    
    start_time = time.time()
    
    for game_idx in tqdm(range(args.num_games), desc="对战进度"):
        # 交替执白
        if game_idx % 2 == 0:
            white_agent, black_agent = enhanced_agent, basic_agent
            white_name, black_name = "Enhanced", "Basic"
        else:
            white_agent, black_agent = basic_agent, enhanced_agent
            white_name, black_name = "Basic", "Enhanced"
        
        winner, steps, white_stats, black_stats = play_game_with_stats(
            white_agent, black_agent, 
            max_steps=args.max_steps,
            verbose=False
        )
        
        # 确定胜者
        if winner == Player.white:
            winner_name = white_name
        elif winner == Player.black:
            winner_name = black_name
        else:
            winner_name = "平局"
        
        # 更新胜负统计
        if winner_name == "Enhanced":
            enhanced_wins += 1
            result_sym = "✅"
        elif winner_name == "Basic":
            basic_wins += 1
            result_sym = "❌"
        else:
            draws += 1
            result_sym = "🤝"
        
        # 收集褡裢统计
        if white_name == "Enhanced":
            enh_stats = white_stats
            bas_stats = black_stats
            enh_player = Player.white
            bas_player = Player.black
        else:
            enh_stats = black_stats
            bas_stats = white_stats
            enh_player = Player.black
            bas_player = Player.white
        
        enh_formed = enh_stats['dalian_formed'][enh_player]
        enh_used = enh_stats['dalian_used'][enh_player]
        bas_formed = bas_stats['dalian_formed'][bas_player]
        bas_used = bas_stats['dalian_used'][bas_player]
        
        total_enhanced_dalian_formed += enh_formed
        total_enhanced_dalian_used += enh_used
        total_basic_dalian_formed += bas_formed
        total_basic_dalian_used += bas_used
        
        game_details.append({
            'game': game_idx + 1,
            'winner': winner_name,
            'steps': steps,
            'enhanced_dalian_formed': enh_formed,
            'enhanced_dalian_used': enh_used,
            'basic_dalian_formed': bas_formed,
            'basic_dalian_used': bas_used,
        })
        
        print(f"  第{game_idx+1:2d}局: {result_sym} {winner_name:8s} ({steps}步) | "
              f"Enhanced褡裢: 形成{enh_formed}/利用{enh_used} | "
              f"Basic褡裢: 形成{bas_formed}/利用{bas_used}")
    
    total_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "=" * 70)
    print("📊 对战结果汇总")
    print("=" * 70)
    
    total = args.num_games
    print(f"\n【胜率统计】")
    print(f"  Enhanced (带褡裢增强): {enhanced_wins:2d} 胜 ({enhanced_wins/total*100:5.1f}%)")
    print(f"  Basic    (无增强):     {basic_wins:2d} 胜 ({basic_wins/total*100:5.1f}%)")
    print(f"  平局:                  {draws:2d}    ({draws/total*100:5.1f}%)")
    
    print(f"\n【褡裢统计】")
    print(f"  Enhanced:")
    print(f"    - 形成褡裢总次数: {total_enhanced_dalian_formed}")
    print(f"    - 利用褡裢总次数: {total_enhanced_dalian_used}")
    print(f"    - 平均每局形成: {total_enhanced_dalian_formed/total:.2f}")
    print(f"    - 平均每局利用: {total_enhanced_dalian_used/total:.2f}")
    
    print(f"  Basic:")
    print(f"    - 形成褡裢总次数: {total_basic_dalian_formed}")
    print(f"    - 利用褡裢总次数: {total_basic_dalian_used}")
    print(f"    - 平均每局形成: {total_basic_dalian_formed/total:.2f}")
    print(f"    - 平均每局利用: {total_basic_dalian_used/total:.2f}")
    
    print(f"\n【时间统计】")
    print(f"  总耗时: {total_time:.1f}秒")
    print(f"  平均每局: {total_time/total:.1f}秒")
    
    print("\n" + "=" * 70)
    print("📝 详细对局记录")
    print("=" * 70)
    for g in game_details:
        print(f"  第{g['game']:2d}局: {g['winner']:8s} ({g['steps']}步) | "
              f"Enh褡裢[{g['enhanced_dalian_formed']}/{g['enhanced_dalian_used']}] | "
              f"Bas褡裢[{g['basic_dalian_formed']}/{g['basic_dalian_used']}]")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()


