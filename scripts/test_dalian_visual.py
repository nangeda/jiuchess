#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢特征可视化对战脚本

一局对战，详细可视化输出
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from jiu.jiuboard_fast import GameState, Move, Player, count_independent_dalians, find_all_dalians
from jiu.jiutypes import Decision, board_gild, Point, Go, Skip_eat, board_size
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move
from jcar.candidate_features import build_enhanced_features


def print_board(state: GameState, highlight_points=None):
    """
    打印棋盘状态
    ○ = 白子, ● = 黑子, · = 空位
    """
    board = state.board
    highlight_points = highlight_points or set()
    
    # 打印列号
    print("    ", end="")
    for c in range(1, board_size + 1):
        print(f"{c:2d}", end="")
    print()
    
    print("   +" + "--" * board_size + "+")
    
    for r in range(1, board_size + 1):
        print(f"{r:2d} |", end="")
        for c in range(1, board_size + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            
            if pt in highlight_points:
                if pl == Player.white:
                    char = "◎"  # 高亮白子
                elif pl == Player.black:
                    char = "◉"  # 高亮黑子
                else:
                    char = "★"  # 高亮空位
            else:
                if pl == Player.white:
                    char = "○"
                elif pl == Player.black:
                    char = "●"
                else:
                    char = "·"
            print(f" {char}", end="")
        print(" |")
    
    print("   +" + "--" * board_size + "+")


def get_move_description(move: Move) -> str:
    """获取走法描述"""
    if move.is_put:
        return f"落子 ({move.point.row},{move.point.col})"
    elif move.is_go:
        return f"走子 ({move.go_to.go.row},{move.go_to.go.col})→({move.go_to.to.row},{move.go_to.to.col})"
    elif move.is_fly:
        return f"飞子 ({move.go_to.go.row},{move.go_to.go.col})→({move.go_to.to.row},{move.go_to.to.col})"
    elif move.is_skip_eat:
        se = move.skip_eat_points
        return f"跳吃 ({se.go.row},{se.go.col})→({se.to.row},{se.to.col}) 吃({se.eat.row},{se.eat.col})"
    elif move.is_skip_eat_seq:
        seq = move.skip_eat_points
        start = seq[0].go
        end = seq[-1].to
        eaten = [f"({s.eat.row},{s.eat.col})" for s in seq]
        return f"连跳 ({start.row},{start.col})→({end.row},{end.col}) 吃{len(seq)}子: {','.join(eaten)}"
    return "未知走法"


def print_dalian_info(state: GameState, player: Player, player_name: str):
    """打印褡裢信息"""
    dalians = find_all_dalians(state.board, player)
    count = count_independent_dalians(state.board, player)
    
    if dalians:
        print(f"  {player_name} 褡裢数: {count} (共发现{len(dalians)}个)")
        for i, d in enumerate(dalians[:3]):  # 最多显示3个
            trigger = d.trigger
            empty = d.empty
            print(f"    褡裢{i+1}: 游子({trigger.row},{trigger.col}) ↔ 空位({empty.row},{empty.col})")
    else:
        print(f"  {player_name} 褡裢数: 0")


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
        
        return move, {'value': value.item(), 'prob': float(probs[best_idx])}


def play_visual_game(enhanced_agent, basic_agent, max_steps=800, show_interval=50):
    """
    可视化对战一局
    enhanced_agent 执白, basic_agent 执黑
    """
    state = GameState.new_game(14)
    step = 0
    
    # 褡裢统计
    enhanced_dalian_formed = 0
    enhanced_dalian_used = 0
    basic_dalian_formed = 0
    basic_dalian_used = 0
    prev_enhanced_dalian = 0
    prev_basic_dalian = 0
    
    print("\n" + "=" * 60)
    print("🎮 可视化对战开始")
    print("  白方: Enhanced Agent (带褡裢增强)")
    print("  黑方: Basic Agent (无增强)")
    print("=" * 60)
    
    while not state.is_over() and step < max_steps:
        if state.next_player == Player.white:
            current_agent = enhanced_agent
            player_name = "白(Enhanced)"
            player_sym = "○"
        else:
            current_agent = basic_agent
            player_name = "黑(Basic)"
            player_sym = "●"
        
        move, info = current_agent.select_move(state)
        if move is None:
            print(f"\n⚠️ {player_name} 无合法走法!")
            break
        
        # 获取落点用于高亮
        highlight = set()
        if move.is_put:
            highlight.add(move.point)
        elif move.is_go or move.is_fly:
            highlight.add(move.go_to.to)
        elif move.is_skip_eat:
            highlight.add(move.skip_eat_points.to)
        elif move.is_skip_eat_seq:
            highlight.add(move.skip_eat_points[-1].to)
        
        state = state.apply_move(move)
        step += 1
        
        # 阶段名称
        if state.step <= board_gild:
            phase_name = "布局阶段"
        elif state.board.get_player_total(state.next_player) <= 14:
            phase_name = "飞子阶段"
        else:
            phase_name = "对战阶段"
        
        # 检查褡裢变化（仅对战阶段）
        if state.step > board_gild:
            curr_enhanced_dalian = count_independent_dalians(state.board, Player.white)
            curr_basic_dalian = count_independent_dalians(state.board, Player.black)
            
            if state.next_player == Player.black:  # 刚走完的是白方
                if curr_enhanced_dalian > prev_enhanced_dalian:
                    enhanced_dalian_formed += 1
                if info.get('uses_dalian', False):
                    enhanced_dalian_used += 1
            else:  # 刚走完的是黑方
                if curr_basic_dalian > prev_basic_dalian:
                    basic_dalian_formed += 1
            
            prev_enhanced_dalian = curr_enhanced_dalian
            prev_basic_dalian = curr_basic_dalian
        
        # 显示关键时刻
        show_board = False
        reason = ""
        
        # 布局结束时显示
        if step == board_gild:
            show_board = True
            reason = "📍 布局阶段结束"
        # 定期显示
        elif step > board_gild and (step - board_gild) % show_interval == 0:
            show_board = True
            reason = f"📍 第{step}步"
        # 形成褡裢时显示
        elif info.get('creates_dalian', False):
            show_board = True
            reason = "🎯 形成褡裢!"
        # 利用褡裢时显示
        elif info.get('uses_dalian', False):
            show_board = True
            reason = "⚡ 利用褡裢吃子!"
        # 成方吃子时显示
        elif info.get('will_form_square', False):
            show_board = True
            reason = "🔲 成方吃子!"
        # 连跳吃子时显示
        elif move.is_skip_eat_seq and len(move.skip_eat_points) >= 3:
            show_board = True
            reason = f"💥 连跳吃{len(move.skip_eat_points)}子!"
        
        if show_board:
            w_count = state.board.get_player_total(Player.white)
            b_count = state.board.get_player_total(Player.black)
            
            print(f"\n{'─' * 60}")
            print(f"{reason} | {phase_name} | 白{w_count} vs 黑{b_count}")
            print(f"  {player_sym} {player_name}: {get_move_description(move)}")
            if 'value' in info:
                print(f"  评估值: {info['value']:.3f}, 置信度: {info.get('prob', 0):.3f}")
            if info.get('rule_bonus', 0) > 0:
                print(f"  规则加成: +{info['rule_bonus']:.2f}")
            print()
            
            print_board(state, highlight)
            
            # 对战阶段显示褡裢信息
            if state.step > board_gild:
                print()
                print_dalian_info(state, Player.white, "白(Enhanced)")
                print_dalian_info(state, Player.black, "黑(Basic)")
        else:
            # 简略输出
            if step <= board_gild:
                pass  # 布局阶段不输出
            elif step % 10 == 0:
                w_count = state.board.get_player_total(Player.white)
                b_count = state.board.get_player_total(Player.black)
                print(f"  Step {step}: 白{w_count} vs 黑{b_count}", end="\r")
    
    print("\n")
    
    # 最终结果
    winner = state.winner()
    if winner is None and step >= max_steps:
        winner = state.winner_by_timeout()
    
    w_count = state.board.get_player_total(Player.white)
    b_count = state.board.get_player_total(Player.black)
    
    print("=" * 60)
    print("🏁 对战结束")
    print("=" * 60)
    print(f"\n最终局面 (第{step}步):")
    print_board(state)
    
    print(f"\n棋子数: 白{w_count} vs 黑{b_count}")
    
    if winner == Player.white:
        print(f"\n🎉 获胜者: 白方 (Enhanced Agent)")
    elif winner == Player.black:
        print(f"\n🎉 获胜者: 黑方 (Basic Agent)")
    else:
        print(f"\n🤝 平局")
    
    print(f"\n【褡裢统计】")
    print(f"  Enhanced (白): 形成{enhanced_dalian_formed}次, 利用{enhanced_dalian_used}次")
    print(f"  Basic (黑):    形成{basic_dalian_formed}次, 利用{basic_dalian_used}次")
    
    return winner, step


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='褡裢特征可视化对战')
    parser.add_argument('--model', default='exp/jcar_sft_2025_balanced/checkpoint_best.pt',
                        help='模型路径')
    parser.add_argument('--max-steps', type=int, default=800, help='最大步数')
    parser.add_argument('--show-interval', type=int, default=30, help='显示间隔')
    parser.add_argument('--device', default='cuda:0', help='设备')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 褡裢特征可视化对战")
    print("=" * 60)
    print(f"模型: {args.model}")
    print("=" * 60)
    
    # 创建Agent
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
    
    # 对战
    play_visual_game(enhanced_agent, basic_agent, 
                     max_steps=args.max_steps, 
                     show_interval=args.show_interval)


if __name__ == '__main__':
    main()


