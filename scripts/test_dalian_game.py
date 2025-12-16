#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
褡裢对局测试脚本

对比带褡裢特征提升和不带褡裢特征提升的agent
重点观察褡裢的形成和使用
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import (
    GameState, Move, Player, Board, Point,
    find_all_dalians, count_independent_dalians, Dalian
)
from jiu.jiutypes import Decision, board_gild, board_size
from agent.enhanced_agent import EnhancedJiuqiNetAgent
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move
from jcar.candidate_features import build_enhanced_features
from dyt.candidate_features import build_features_for_candidates


def print_board(board: Board, title: str = "", dalians: List[Dalian] = None, 
                highlight_move: Tuple[Point, Point] = None):
    """
    打印棋盘，可选高亮褡裢和走法
    
    Args:
        board: 棋盘
        title: 标题
        dalians: 褡裢列表（高亮显示）
        highlight_move: (from_pos, to_pos) 高亮走法
    """
    # 收集褡裢相关位置
    dalian_pieces = set()
    dalian_triggers = set()
    dalian_empties = set()
    
    if dalians:
        for d in dalians:
            for p in d.pieces:
                dalian_pieces.add((p.row, p.col))
            dalian_triggers.add((d.trigger.row, d.trigger.col))
            if d.empty:
                dalian_empties.add((d.empty.row, d.empty.col))
    
    move_from = None
    move_to = None
    if highlight_move:
        move_from = (highlight_move[0].row, highlight_move[0].col) if highlight_move[0] else None
        move_to = (highlight_move[1].row, highlight_move[1].col) if highlight_move[1] else None
    
    print(f"\n{'='*50}")
    if title:
        print(f"  {title}")
    print(f"{'='*50}")
    
    # 列标题
    col_header = "     "
    for c in range(1, board_size + 1):
        col_header += f"{c:2d} "
    print(col_header)
    print("    +" + "---" * board_size + "+")
    
    for r in range(1, board_size + 1):
        row_str = f" {r:2d} |"
        for c in range(1, board_size + 1):
            pt = Point(r, c)
            p = board.get(pt)
            pos = (r, c)
            
            # 确定显示字符
            if p == Player.white:
                char = "●"
            elif p == Player.black:
                char = "○"
            else:
                char = "·"
            
            # 添加标记
            if pos == move_from:
                row_str += f"[{char}]"[1:3]  # 用方括号标记起点
            elif pos == move_to:
                row_str += f"→{char}"  # 用箭头标记终点
            elif pos in dalian_triggers:
                row_str += f"★ "  # 游子用星号
            elif pos in dalian_empties:
                row_str += f"◇ "  # 空位用菱形
            elif pos in dalian_pieces:
                row_str += f"{char}!"[0:2]  # 褡裢棋子加感叹号
            else:
                row_str += f"{char} "
        
        row_str += "|"
        print(row_str)
    
    print("    +" + "---" * board_size + "+")
    
    # 打印褡裢信息
    if dalians:
        print(f"\n  检测到 {len(dalians)} 个褡裢:")
        for i, d in enumerate(dalians):
            print(f"    [{i+1}] 游子: ({d.trigger.row},{d.trigger.col}), "
                  f"空位: ({d.empty.row},{d.empty.col}), "
                  f"棋子数: {len(d.pieces)}")


class BasicJiuqiNetAgent:
    """基础增强Agent（使用26维特征+规则加成，但不含褡裢特征）"""
    
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
        
        # 规则加成权重（与EnhancedAgent相同，但不含褡裢）
        self.square_weight = 3.5
        self.multi_square_weight = 2.0
        self.eat_weight = 2.5
        self.safety_weight = 1.5
        self.triple_weight = 2.5
        self.break_weight = 2.0
        self.capture_weight = 2.0
        self.rule_weight = 0.3
    
    def select_move(self, state: GameState) -> Tuple[Optional[Move], dict]:
        obs = encode_board_state(state)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        phase_id = get_phase_id(state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        legal_decs = state.legal_moves()
        if not legal_decs:
            return None, {}
        
        flying = state.board.get_player_total(state.next_player) <= 14
        cand_dicts = [decision_to_dict(d) for d in legal_decs]
        
        # 使用26维增强特征（不含褡裢的6维）
        enhanced_feats = build_enhanced_features(cand_dicts, state, phase_id, flying)
        cand_feats = enhanced_feats[:, :26]  # 14基础 + 12增强，不含褡裢
        cand_tensor = torch.from_numpy(cand_feats[:, :14]).float().to(self.device)
        
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        base_scores = logits_list[0].cpu().numpy()
        
        # 计算规则加成（不含褡裢）
        rule_bonus = self._compute_rule_bonus(cand_feats)
        
        # 综合分数
        final_scores = base_scores + self.rule_weight * rule_bonus
        
        best_idx = int(np.argmax(final_scores))
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        
        return move, {
            'value': value.item(),
            'score': float(final_scores[best_idx]),
            'decision': best_dec
        }
    
    def _compute_rule_bonus(self, enhanced_feats: np.ndarray) -> np.ndarray:
        """计算规则加成（不含褡裢）"""
        bonus = np.zeros(len(enhanced_feats), dtype=np.float32)
        
        # 成方加成
        bonus += enhanced_feats[:, 14] * self.square_weight
        bonus += enhanced_feats[:, 15] * self.multi_square_weight * 4
        
        # 跳吃加成
        bonus += enhanced_feats[:, 16] * self.eat_weight * 8
        
        # 安全性加成
        bonus += enhanced_feats[:, 17] * self.safety_weight
        bonus -= (1 - enhanced_feats[:, 17]) * self.safety_weight * 0.5
        
        # 准方加成
        bonus += enhanced_feats[:, 18] * self.triple_weight
        bonus += enhanced_feats[:, 19] * self.triple_weight * 2
        
        # 破坏对方加成
        bonus += enhanced_feats[:, 20] * self.break_weight
        bonus += enhanced_feats[:, 21] * self.break_weight * 2
        
        # 吃子走法加成
        bonus += enhanced_feats[:, 25] * self.capture_weight
        
        # 根据局势调整
        piece_diff = enhanced_feats[:, 22]
        leading = piece_diff > 0.1
        bonus[leading] += enhanced_feats[leading, 25] * 1.0
        
        behind = piece_diff < -0.1
        bonus[behind] += enhanced_feats[behind, 17] * 0.5
        
        return bonus


def run_game_with_dalian_tracking(agent_white, agent_black, 
                                   white_name: str, black_name: str,
                                   max_steps: int = 500,
                                   verbose: bool = True):
    """
    运行一局游戏，追踪褡裢的形成和使用
    """
    state = GameState.new_game(board_size)
    step = 0
    
    # 记录褡裢相关事件
    dalian_events = []
    
    # 上一步的褡裢状态
    prev_white_dalians = 0
    prev_black_dalians = 0
    
    print(f"\n{'#'*60}")
    print(f"#  对局: {white_name} (白) vs {black_name} (黑)")
    print(f"{'#'*60}")
    
    while not state.is_over() and step < max_steps:
        step += 1
        current_player = state.next_player
        player_name = white_name if current_player == Player.white else black_name
        agent = agent_white if current_player == Player.white else agent_black
        
        # 检测当前褡裢状态
        white_dalians = find_all_dalians(state.board, Player.white)
        black_dalians = find_all_dalians(state.board, Player.black)
        white_dalian_count = len(white_dalians)
        black_dalian_count = len(black_dalians)
        
        # 检测是否新形成了褡裢
        new_white_dalian = white_dalian_count > prev_white_dalians
        new_black_dalian = black_dalian_count > prev_black_dalians
        
        if new_white_dalian and verbose:
            print(f"\n{'!'*50}")
            print(f"  ⚠️  第{step}步: 白方形成了新褡裢！")
            print_board(state.board, f"白方褡裢形成 (共{white_dalian_count}个)", white_dalians)
            dalian_events.append(('white_form', step, white_dalian_count))
        
        if new_black_dalian and verbose:
            print(f"\n{'!'*50}")
            print(f"  ⚠️  第{step}步: 黑方形成了新褡裢！")
            print_board(state.board, f"黑方褡裢形成 (共{black_dalian_count}个)", black_dalians)
            dalian_events.append(('black_form', step, black_dalian_count))
        
        # 获取走法
        move, info = agent.select_move(state)
        
        if move is None:
            print(f"  第{step}步: {player_name} 无法走棋")
            break
        
        # 检测是否使用了褡裢
        used_dalian = False
        if current_player == Player.white and white_dalians:
            for d in white_dalians:
                if hasattr(move, 'point'):
                    if move.point == d.empty:
                        used_dalian = True
                        break
                elif hasattr(move, 'go_to'):
                    go_to = move.go_to
                    if hasattr(go_to, 'go') and go_to.go == d.trigger:
                        used_dalian = True
                        break
        elif current_player == Player.black and black_dalians:
            for d in black_dalians:
                if hasattr(move, 'point'):
                    if move.point == d.empty:
                        used_dalian = True
                        break
                elif hasattr(move, 'go_to'):
                    go_to = move.go_to
                    if hasattr(go_to, 'go') and go_to.go == d.trigger:
                        used_dalian = True
                        break
        
        # 检测info中的褡裢使用标记
        if info.get('uses_dalian', False):
            used_dalian = True
        
        if used_dalian and verbose:
            print(f"\n{'*'*50}")
            print(f"  ⭐ 第{step}步: {player_name} 使用褡裢!")
            dalians_to_show = white_dalians if current_player == Player.white else black_dalians
            print_board(state.board, f"{player_name} 使用褡裢", dalians_to_show)
            dalian_events.append(('use', step, player_name))
        
        # 执行走法
        try:
            state = state.apply_move(move)
        except Exception as e:
            print(f"  第{step}步: {player_name} 走法执行失败: {e}")
            break
        
        # 每50步或关键时刻打印状态
        if step % 100 == 0 and verbose:
            print(f"\n--- 第{step}步 ---")
            white_count = state.board.get_player_total(Player.white)
            black_count = state.board.get_player_total(Player.black)
            print(f"  白方: {white_count}子, 褡裢: {white_dalian_count}")
            print(f"  黑方: {black_count}子, 褡裢: {black_dalian_count}")
        
        prev_white_dalians = white_dalian_count
        prev_black_dalians = black_dalian_count
    
    # 游戏结束，打印结果
    print(f"\n{'='*60}")
    print(f"  游戏结束! 共{step}步")
    
    white_count = state.board.get_player_total(Player.white)
    black_count = state.board.get_player_total(Player.black)
    
    print(f"  白方({white_name}): {white_count}子")
    print(f"  黑方({black_name}): {black_count}子")
    
    if white_count > black_count:
        winner = f"白方({white_name})"
    elif black_count > white_count:
        winner = f"黑方({black_name})"
    else:
        winner = "平局"
    print(f"  胜者: {winner}")
    
    # 打印褡裢事件统计
    print(f"\n  褡裢事件统计:")
    print(f"    总事件数: {len(dalian_events)}")
    form_events = [e for e in dalian_events if 'form' in e[0]]
    use_events = [e for e in dalian_events if e[0] == 'use']
    print(f"    形成褡裢: {len(form_events)}次")
    print(f"    使用褡裢: {len(use_events)}次")
    
    print(f"{'='*60}\n")
    
    return {
        'winner': winner,
        'steps': step,
        'white_count': white_count,
        'black_count': black_count,
        'dalian_events': dalian_events
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='褡裢对局测试')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/jcar_sft_best.pt',
                       help='模型路径')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--games', type=int, default=1, help='对局数')
    args = parser.parse_args()
    
    model_path = Path(__file__).parent.parent / args.model
    
    if not model_path.exists():
        # 尝试其他路径
        alt_paths = [
            Path(__file__).parent.parent / 'checkpoints' / 'best_model.pt',
            Path(__file__).parent.parent / 'checkpoints' / 'checkpoint_latest.pt',
        ]
        for p in alt_paths:
            if p.exists():
                model_path = p
                break
    
    print(f"使用模型: {model_path}")
    
    # 创建两个agent
    print("\n创建 Agent...")
    print("  1. EnhancedAgent (带褡裢特征加成)")
    enhanced_agent = EnhancedJiuqiNetAgent(
        str(model_path), 
        device=args.device,
        verbose=False
    )
    
    print("  2. BasicAgent (不带褡裢特征加成)")
    basic_agent = BasicJiuqiNetAgent(str(model_path), device=args.device)
    
    # 运行对局
    for game_idx in range(args.games):
        print(f"\n\n{'#'*60}")
        print(f"#  第 {game_idx + 1}/{args.games} 局")
        print(f"{'#'*60}")
        
        # Enhanced vs Basic
        result = run_game_with_dalian_tracking(
            enhanced_agent, basic_agent,
            "Enhanced(褡裢)", "Basic(无褡裢)",
            max_steps=500,
            verbose=True
        )
        
        print(f"\n对局结果: {result['winner']}")
        print(f"步数: {result['steps']}")
        print(f"比分: 白{result['white_count']} - 黑{result['black_count']}")


if __name__ == '__main__':
    main()
