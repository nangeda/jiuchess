#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Value Head 训练效果

测试内容：
1. Value Head 是否输出有效值（不再全是-1或0）
2. Value 输出是否对不同局面有区分度
3. 在多个样本上统计 Value 分布
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from jcar.model import JiuqiNet
from jcar.config import JiuqiNetConfig
from jiu.jiuboard_fast import GameState, Player
from jiu.jiutypes import board_gild, Point
from dyt.candidate_features import build_features_for_candidates
from jiu.jiuboard_fast import Move
from jiu.jiutypes import Decision


def decision_to_move_local(dec: Decision):
    """将Decision转换为Move"""
    if dec.act == 'put_piece':
        return Move.put_piece(dec.points)
    elif dec.act == 'is_go':
        return Move.go_piece(dec.points)
    elif dec.act == 'fly':
        return Move.fly_piece(dec.points)
    elif dec.act == 'skip_move':
        return Move.skip_eat(dec.points)
    elif dec.act == 'skip_eat_seq':
        return Move.skip_eat_seq(dec.points)
    return None


def encode_board_state(state: GameState) -> np.ndarray:
    """编码棋盘状态为 (6, H, W)"""
    from jiu.jiutypes import board_size
    
    board = state.board
    H, W = board_size, board_size
    obs = np.zeros((6, H, W), dtype=np.float32)
    
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            if pl == Player.white:
                obs[0, r - 1, c - 1] = 1.0
            elif pl == Player.black:
                obs[1, r - 1, c - 1] = 1.0
            else:
                obs[2, r - 1, c - 1] = 1.0
    
    return obs


def get_phase_id(state: GameState) -> int:
    """获取阶段ID: 0=布局, 1=对战, 2=飞子"""
    if state.step < board_gild:
        return 0
    elif state.board.get_player_total(state.next_player) <= 14:
        return 2
    else:
        return 1


def load_model(checkpoint_path: str, device: str = 'cuda') -> JiuqiNet:
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        cfg = JiuqiNetConfig()
    
    model = JiuqiNet(cfg)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def test_value_head(model: JiuqiNet, device: str, num_samples: int = 100):
    """测试 Value Head 输出"""
    print("\n" + "=" * 60)
    print("📊 Value Head 输出测试")
    print("=" * 60)
    
    values = []
    phase_values = {0: [], 1: [], 2: []}  # 按阶段分类
    
    # 创建初始状态并进行随机走子
    state = GameState.new_game(14)
    
    for i in range(num_samples):
        # 编码状态
        obs = encode_board_state(state)
        phase_id = get_phase_id(state)
        
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=device)
        
        # 获取 Value 输出
        with torch.no_grad():
            _, value = model(obs_tensor, phase_tensor)
            v = value.item()
            values.append(v)
            phase_values[phase_id].append(v)
        
        # 随机走一步
        legal_moves = state.legal_moves()
        if not legal_moves or state.is_over():
            state = GameState.new_game(14)  # 重新开始
            continue

        import random
        dec = random.choice(legal_moves)
        try:
            move = decision_to_move_local(dec)
            if move:
                state = state.apply_move(move)
        except:
            state = GameState.new_game(14)
    
    # 统计分析
    values = np.array(values)
    print(f"\n📈 Value 统计 (共 {len(values)} 个样本):")
    print(f"   均值:   {values.mean():.4f}")
    print(f"   标准差: {values.std():.4f}")
    print(f"   最小值: {values.min():.4f}")
    print(f"   最大值: {values.max():.4f}")
    print(f"   中位数: {np.median(values):.4f}")
    
    # 检查是否全是相同值
    unique_values = len(np.unique(np.round(values, 4)))
    print(f"\n   唯一值数量: {unique_values}")
    
    if values.std() < 0.01:
        print("\n⚠️  警告: Value 输出方差过小，可能没有被有效训练！")
    else:
        print("\n✅ Value Head 输出正常，有足够的区分度")
    
    # 按阶段分析
    print("\n📊 按阶段分析:")
    phase_names = {0: "布局", 1: "对战", 2: "飞子"}
    for phase_id, phase_vals in phase_values.items():
        if phase_vals:
            pv = np.array(phase_vals)
            print(f"   {phase_names[phase_id]}阶段: 均值={pv.mean():.4f}, 标准差={pv.std():.4f}, 样本数={len(pv)}")
    
    return values


def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证 Value Head 训练效果')
    parser.add_argument('--checkpoint', type=str, 
                        default='exp/jcar_sft_sgf_jiu_full/checkpoint_best.pt',
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--num-samples', type=int, default=200, help='测试样本数')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Value Head 训练效果验证")
    print("=" * 60)
    print(f"模型: {args.checkpoint}")
    print(f"设备: {args.device}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = load_model(args.checkpoint, args.device)
    print("✅ 模型加载成功")
    
    # 测试 Value Head
    values = test_value_head(model, args.device, args.num_samples)
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

