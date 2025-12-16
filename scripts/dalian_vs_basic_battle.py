#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢增强 vs 基础增强 对战
比较加入褡裢特征前后的效果
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians
from jiu.jiutypes import board_gild
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move
from jcar.candidate_features import build_enhanced_features


class BasicEnhancedAgent:
    """基础增强Agent（无褡裢特征）"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        from jcar.model import JiuqiNet
        from jcar.config import JiuqiNetConfig
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_config' in checkpoint:
            cfg = JiuqiNetConfig(**checkpoint['model_config'])
        else:
            cfg = JiuqiNetConfig()
        
        self.model = JiuqiNet(cfg)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 基础增强权重（无褡裢）
        self.square_weight = 3.0
        self.eat_weight = 2.5
        self.safety_weight = 1.0
        self.triple_weight = 2.0
        self.break_weight = 1.5
        
        print(f"🔵 Basic Enhanced Agent (无褡裢增强)")
        print(f"   权重: 成方={self.square_weight}, 跳吃={self.eat_weight}")
    
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
        
        # 计算基础规则加成（无褡裢）
        bonus = self._compute_basic_bonus(enhanced_feats, phase_id)
        logits = logits + bonus
        
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        best_idx = int(np.argmax(probs))
        
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        
        return move, {'value': value.item()}
    
    def _compute_basic_bonus(self, feats, phase_id):
        """基础规则加成（无褡裢特征）- 正确的特征索引"""
        bonus = np.zeros(len(feats))
        
        # === 成方加成（核心策略）===
        bonus += feats[:, 14] * self.square_weight  # [14] will_form_square
        bonus += feats[:, 15] * 2.0 * 4  # [15] square_count_norm * 4 恢复原值
        
        # === 跳吃加成（关键战术）===
        bonus += feats[:, 16] * self.eat_weight * 8  # [16] eat_count_norm * 8
        
        # === 安全性加成 ===
        bonus += feats[:, 17] * self.safety_weight  # [17] is_safe
        bonus -= (1 - feats[:, 17]) * self.safety_weight * 0.5  # 危险惩罚
        
        # === 准方加成 ===
        bonus += feats[:, 18] * self.triple_weight  # [18] creates_triple
        bonus += feats[:, 19] * self.triple_weight * 2  # [19] triple_count_norm
        
        # === 破坏对方加成 ===
        bonus += feats[:, 20] * self.break_weight  # [20] breaks_opp_potential
        
        # === 吃子走法加成 ===
        bonus += feats[:, 25] * 1.5  # [25] is_capture_move
        
        return bonus


def play_game(dalian_agent, basic_agent, game_idx, dalian_is_white, show_interval=50):
    """对战一局"""
    state = GameState.new_game(14)
    step = 0
    max_steps = 800
    
    dalian_times = []
    basic_times = []
    dalian_dalian_formed = 0
    dalian_dalian_used = 0
    prev_dalian_count = 0
    
    if dalian_is_white:
        white_name, black_name = "褡裢增强(白)", "基础增强(黑)"
    else:
        white_name, black_name = "基础增强(白)", "褡裢增强(黑)"
    
    print(f"\n{'═'*60}")
    print(f"📍 第 {game_idx} 局 | {white_name} vs {black_name}")
    print(f"{'═'*60}")
    
    while not state.is_over() and step < max_steps:
        current_player = state.next_player
        move_start = time.time()
        
        if (current_player == Player.white and dalian_is_white) or \
           (current_player == Player.black and not dalian_is_white):
            move, info = dalian_agent.select_move(state)
            is_dalian = True
        else:
            move, info = basic_agent.select_move(state)
            is_dalian = False
        
        move_time = time.time() - move_start
        
        if move is None:
            break
        
        if is_dalian:
            dalian_times.append(move_time)
        else:
            basic_times.append(move_time)
        
        state = state.apply_move(move)
        step += 1
        
        # 褡裢统计
        if state.step > board_gild:
            dalian_player = Player.white if dalian_is_white else Player.black
            curr_count = count_independent_dalians(state.board, dalian_player)
            
            if is_dalian:
                if curr_count > prev_dalian_count:
                    dalian_dalian_formed += 1
                if info.get('uses_dalian', False):
                    dalian_dalian_used += 1
            
            prev_dalian_count = curr_count
        
        # 阶段
        w_count = state.board.get_player_total(Player.white)
        b_count = state.board.get_player_total(Player.black)
        
        if state.step <= board_gild:
            phase = "布局"
        elif w_count <= 14 or b_count <= 14:
            phase = "飞子"
        else:
            phase = "对战"
        
        # 每show_interval步输出
        if step % show_interval == 0:
            dalian_player = Player.white if dalian_is_white else Player.black
            basic_player = Player.black if dalian_is_white else Player.white
            d_count = w_count if dalian_is_white else b_count
            b_count_now = b_count if dalian_is_white else w_count
            d_dalian = count_independent_dalians(state.board, dalian_player) if state.step > board_gild else 0
            b_dalian = count_independent_dalians(state.board, basic_player) if state.step > board_gild else 0
            
            avg_d = np.mean(dalian_times[-25:]) if dalian_times else 0
            avg_b = np.mean(basic_times[-25:]) if basic_times else 0
            
            print(f"  Step {step:4d} [{phase}] | 褡裢:{d_count:2d} 基础:{b_count_now:2d} | "
                  f"褡裢数 D:{d_dalian} B:{b_dalian} | "
                  f"成褡裢:{dalian_dalian_formed} 用褡裢:{dalian_dalian_used} | "
                  f"时间 D:{avg_d:.2f}s B:{avg_b:.2f}s")
    
    # 结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    w_final = state.board.get_player_total(Player.white)
    b_final = state.board.get_player_total(Player.black)
    
    if winner == Player.white:
        winner_code = "Dalian" if dalian_is_white else "Basic"
    elif winner == Player.black:
        winner_code = "Dalian" if not dalian_is_white else "Basic"
    else:
        winner_code = "Draw"
    
    d_final = w_final if dalian_is_white else b_final
    b_final_count = b_final if dalian_is_white else w_final
    
    print(f"{'─'*60}")
    print(f"🏁 第 {game_idx} 局结束 | 获胜: {winner_code}")
    print(f"   最终子数: 褡裢增强 {d_final} vs 基础增强 {b_final_count}")
    print(f"   褡裢统计: 形成 {dalian_dalian_formed} 次, 利用 {dalian_dalian_used} 次")
    print(f"   平均走子时间: 褡裢 {np.mean(dalian_times):.3f}s, 基础 {np.mean(basic_times):.3f}s")
    
    return winner_code, step, dalian_dalian_formed, dalian_dalian_used


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt')
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--show-interval', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🎯 褡裢增强 vs 基础增强 对战")
    print("═" * 60)
    print(f"  对战局数: {args.num_games}")
    print("═" * 60)
    
    # 加载Agent
    print("\n📦 加载Agent...")
    
    dalian_agent = EnhancedJiuqiNetAgent(
        args.model,
        args.device,
        dalian_create_weight=12.0,
        dalian_use_weight=20.0,
        dalian_break_weight=10.0,
        pre_dalian_weight=5.0,
        verbose=False
    )
    
    basic_agent = BasicEnhancedAgent(args.model, args.device)
    print("✅ 加载完成\n")
    
    # 统计
    dalian_wins, basic_wins, draws = 0, 0, 0
    total_dalian_formed = 0
    total_dalian_used = 0
    
    for i in range(1, args.num_games + 1):
        # 交替执白
        dalian_is_white = (i % 2 == 1)
        
        winner, steps, formed, used = play_game(
            dalian_agent, basic_agent, i, dalian_is_white,
            show_interval=args.show_interval
        )
        
        if winner == 'Dalian':
            dalian_wins += 1
        elif winner == 'Basic':
            basic_wins += 1
        else:
            draws += 1
        
        total_dalian_formed += formed
        total_dalian_used += used
        
        # 每局后显示汇总
        print(f"\n  【当前战绩】褡裢增强 {dalian_wins} 胜 | 基础增强 {basic_wins} 胜 | 平局 {draws}")
    
    # 最终结果
    print("\n" + "═" * 60)
    print("🏆 最终结果")
    print("═" * 60)
    
    print(f"""
┌────────────────────────────────────────────────────────┐
│                      胜率统计                           │
├────────────────────────────────────────────────────────┤
│  褡裢增强:  {dalian_wins:2d} 胜  ({dalian_wins/args.num_games*100:5.1f}%)                      │
│  基础增强:  {basic_wins:2d} 胜  ({basic_wins/args.num_games*100:5.1f}%)                      │
│  平局:      {draws:2d}     ({draws/args.num_games*100:5.1f}%)                      │
├────────────────────────────────────────────────────────┤
│                      褡裢统计                           │
├────────────────────────────────────────────────────────┤
│  总形成: {total_dalian_formed:3d} 次 (平均 {total_dalian_formed/args.num_games:.1f}/局)               │
│  总利用: {total_dalian_used:3d} 次 (平均 {total_dalian_used/args.num_games:.1f}/局)               │
└────────────────────────────────────────────────────────┘
""")
    
    print("═" * 60)


if __name__ == '__main__':
    main()

