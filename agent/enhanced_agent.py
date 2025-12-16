#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型 Agent

使用增强特征进行决策：
1. 基础评分：使用神经网络对候选动作评分
2. 规则加成：利用增强特征（成方、跳吃、安全性等）调整评分

这样可以在不重新训练模型的情况下，利用规则知识增强决策。
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import Decision, board_gild, Point, Go, Skip_eat
from jcar.candidate_features import build_enhanced_features, ENHANCED_FEAT_DIM
from battle_test import encode_board_state, get_phase_id, decision_to_dict, decision_to_move


class EnhancedJiuqiNetAgent:
    """
    增强型 JiuqiNet Agent
    
    使用神经网络 + 规则加成的混合决策：
    - 神经网络提供基础评分（使用前14维特征）
    - 规则加成利用增强特征调整评分（后18维特征，包含褡裢）
    
    增强特征权重：
    - will_form_square: 成方加成
    - square_count: 多成方额外加成
    - eat_count: 跳吃数量加成
    - is_safe: 安全性加成
    - creates_triple: 准方加成
    - breaks_opp_potential: 破坏对方加成
    
    褡裢特征权重（新增）：
    - creates_dalian: 形成褡裢加成（很高权重）
    - uses_dalian: 利用褡裢吃子加成（极高权重）
    - breaks_opp_dalian: 破坏对方褡裢加成
    - creates_pre_dalian: 形成准褡裢加成（布局阶段）
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda',
        # 规则加成权重
        square_weight: float = 3.0,        # 成方权重
        multi_square_weight: float = 2.0,  # 多成方额外权重
        eat_weight: float = 2.5,           # 跳吃权重
        safety_weight: float = 1.0,        # 安全性权重
        triple_weight: float = 0.8,        # 准方权重
        break_weight: float = 1.2,         # 破坏对方权重
        capture_weight: float = 1.5,       # 吃子走法权重
        # 褡裢相关权重（新增）- 褡裢是必杀技，权重要足够高
        dalian_create_weight: float = 12.0,    # 形成褡裢权重（战略性极高）
        dalian_use_weight: float = 20.0,       # 利用褡裢吃子权重（绝对最高优先级！）
        dalian_break_weight: float = 10.0,     # 破坏对方褡裢权重（生死攸关）
        pre_dalian_weight: float = 5.0,        # 准褡裢权重（布局阶段关键）
        verbose: bool = False
    ):
        from jcar.model import JiuqiNet
        from jcar.config import JiuqiNetConfig
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # 规则加成权重
        self.square_weight = square_weight
        self.multi_square_weight = multi_square_weight
        self.eat_weight = eat_weight
        self.safety_weight = safety_weight
        self.triple_weight = triple_weight
        self.break_weight = break_weight
        self.capture_weight = capture_weight
        
        # 褡裢相关权重
        self.dalian_create_weight = dalian_create_weight
        self.dalian_use_weight = dalian_use_weight
        self.dalian_break_weight = dalian_break_weight
        self.pre_dalian_weight = pre_dalian_weight
        
        print(f"🔵 Enhanced JiuqiNet Agent on {self.device}")
        print(f"   规则权重: 成方={square_weight}, 跳吃={eat_weight}, 安全={safety_weight}")
        print(f"   褡裢权重: 形成={dalian_create_weight}, 利用={dalian_use_weight}, 破坏={dalian_break_weight}")
        
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
    
    def _compute_rule_bonus(self, enhanced_feats: np.ndarray, phase_id: int) -> np.ndarray:
        """
        根据增强特征计算规则加成
        
        增强特征布局 (索引14-31):
        - [14] will_form_square: 是否成方 (0/1)
        - [15] square_count_norm: 成方数量归一化 (0-1)
        - [16] eat_count_norm: 跳吃数量归一化 (0-1)
        - [17] is_safe: 是否安全 (0/1)
        - [18] creates_triple: 是否形成准方 (0/1)
        - [19] triple_count_norm: 准方数量归一化 (0-1)
        - [20] breaks_opp_potential: 是否破坏对方 (0/1)
        - [21] breaks_count_norm: 破坏数量归一化 (0-1)
        - [22] piece_diff_norm: 棋子差距归一化 (-1到1)
        - [23] my_squares_norm: 我方成方数归一化 (0-1)
        - [24] opp_squares_norm: 对方成方数归一化 (0-1)
        - [25] is_capture_move: 是否吃子走法 (0/1)
        --- 褡裢特征 ---
        - [26] my_dalian_count_norm: 我方褡裢数归一化 (0-1)
        - [27] opp_dalian_count_norm: 对方褡裢数归一化 (0-1)
        - [28] creates_dalian: 是否形成褡裢 (0/1)
        - [29] uses_dalian: 是否利用褡裢吃子 (0/1)
        - [30] breaks_opp_dalian: 是否破坏对方褡裢 (0/1)
        - [31] creates_pre_dalian: 是否形成准褡裢 (0/1)
        """
        N = enhanced_feats.shape[0]
        bonus = np.zeros(N, dtype=np.float32)
        
        # 检查特征维度，兼容旧版本（26维）和新版本（32维）
        has_dalian_feats = enhanced_feats.shape[1] >= 32
        
        # 统一逻辑（所有阶段通用）
        
        # === 褡裢加成（最高优先级）===
        if has_dalian_feats:
            # 利用褡裢吃子：最高权重，这是必杀技
            bonus += enhanced_feats[:, 29] * self.dalian_use_weight
            
            # 形成褡裢：很高权重
            bonus += enhanced_feats[:, 28] * self.dalian_create_weight
            
            # 破坏对方褡裢：重要的防守手段
            bonus += enhanced_feats[:, 30] * self.dalian_break_weight
            
            # 如果对方有褡裢而我方没有，提高破坏权重
            opp_has_dalian = enhanced_feats[:, 27] > 0
            my_no_dalian = enhanced_feats[:, 26] == 0
            urgent_break = opp_has_dalian & my_no_dalian
            bonus[urgent_break] += enhanced_feats[urgent_break, 30] * 2.0  # 额外破坏加成
        
        # === 成方加成（核心） ===
        bonus += enhanced_feats[:, 14] * self.square_weight  # 基础成方
        bonus += enhanced_feats[:, 15] * self.multi_square_weight * 4  # 多成方额外加成
        
        # 跳吃加成
        bonus += enhanced_feats[:, 16] * self.eat_weight * 8  # 跳吃数量
        
        # 安全性加成
        bonus += enhanced_feats[:, 17] * self.safety_weight  # 安全走法
        bonus -= (1 - enhanced_feats[:, 17]) * self.safety_weight * 0.5  # 危险走法惩罚
        
        # 准方加成
        bonus += enhanced_feats[:, 18] * self.triple_weight
        bonus += enhanced_feats[:, 19] * self.triple_weight * 2
        
        # 破坏对方加成
        bonus += enhanced_feats[:, 20] * self.break_weight
        bonus += enhanced_feats[:, 21] * self.break_weight * 2
        
        # 吃子走法加成
        bonus += enhanced_feats[:, 25] * self.capture_weight
        
        # 根据局势调整
        piece_diff = enhanced_feats[:, 22]  # 棋子差距
        
        # 领先时：更激进地吃子终结
        leading = piece_diff > 0.1
        bonus[leading] += enhanced_feats[leading, 25] * 1.0  # 额外吃子加成
        
        # 落后时：更重视安全性和褡裢
        behind = piece_diff < -0.1
        bonus[behind] += enhanced_feats[behind, 17] * 0.5  # 额外安全加成
        if has_dalian_feats:
            # 落后时更重视形成褡裢（翻盘机会）
            bonus[behind] += enhanced_feats[behind, 28] * 2.0  # 额外褡裢形成加成
        
        return bonus
    
    def select_move(self, state: GameState) -> Tuple[Optional[Move], dict]:
        obs = encode_board_state(state)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        phase_id = get_phase_id(state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        legal_decs = state.legal_moves()
        if not legal_decs:
            return None, {'error': 'no legal moves'}
        
        flying = state.board.get_player_total(state.next_player) <= 14
        cand_dicts = [decision_to_dict(d) for d in legal_decs]
        
        # 构建增强特征
        enhanced_feats = build_enhanced_features(cand_dicts, state, phase_id, flying)
        
        # 基础特征（前14维）用于模型
        basic_feats = enhanced_feats[:, :14]
        cand_tensor = torch.from_numpy(basic_feats).float().to(self.device)
        
        # 模型评分
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
        logits = logits_list[0].cpu().numpy()
        
        # 计算规则加成
        rule_bonus = self._compute_rule_bonus(enhanced_feats, phase_id)
        
        # 合并评分
        adjusted_logits = logits + rule_bonus
        
        # Softmax选择
        adjusted_probs = np.exp(adjusted_logits - np.max(adjusted_logits))
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        best_idx = int(np.argmax(adjusted_probs))
        
        if self.verbose and phase_id > 0:
            orig_best = int(np.argmax(logits))
            if orig_best != best_idx:
                print(f"  [规则调整] 原选择{orig_best} -> 新选择{best_idx}")
                print(f"    原分数: {logits[orig_best]:.2f}, 新分数: {adjusted_logits[best_idx]:.2f}")
                print(f"    加成: 成方={enhanced_feats[best_idx, 14]:.0f}, "
                      f"跳吃={enhanced_feats[best_idx, 16]*8:.0f}, "
                      f"安全={enhanced_feats[best_idx, 17]:.0f}")
                # 褡裢信息
                if enhanced_feats.shape[1] >= 32:
                    print(f"    褡裢: 形成={enhanced_feats[best_idx, 28]:.0f}, "
                          f"利用={enhanced_feats[best_idx, 29]:.0f}, "
                          f"破坏对方={enhanced_feats[best_idx, 30]:.0f}")
        
        best_dec = legal_decs[best_idx]
        move = decision_to_move(best_dec)
        
        # 构建返回信息
        info = {
            'value': value.item(),
            'prob': float(adjusted_probs[best_idx]),
            'orig_prob': float(np.exp(logits[best_idx] - np.max(logits)) / np.exp(logits - np.max(logits)).sum()),
            'rule_bonus': float(rule_bonus[best_idx]),
            'will_form_square': bool(enhanced_feats[best_idx, 14]),
            'eat_count': int(enhanced_feats[best_idx, 16] * 8),
            'is_safe': bool(enhanced_feats[best_idx, 17]),
        }
        
        # 添加褡裢信息
        if enhanced_feats.shape[1] >= 32:
            info['creates_dalian'] = bool(enhanced_feats[best_idx, 28])
            info['uses_dalian'] = bool(enhanced_feats[best_idx, 29])
            info['breaks_opp_dalian'] = bool(enhanced_feats[best_idx, 30])
            info['my_dalian_count'] = int(enhanced_feats[best_idx, 26] * 4)
            info['opp_dalian_count'] = int(enhanced_feats[best_idx, 27] * 4)
        
        return move, info


def create_enhanced_agent(model_path: str, device: str = 'cuda', **kwargs) -> EnhancedJiuqiNetAgent:
    """创建增强型Agent的工厂函数"""
    return EnhancedJiuqiNetAgent(model_path, device, **kwargs)

