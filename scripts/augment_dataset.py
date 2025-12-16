#!/usr/bin/env python3
"""
数据增强脚本 - 通过棋盘对称变换扩充训练数据8倍

久棋棋盘具有8种对称性（4种旋转 × 2种翻转），
利用这些对称性可以大幅扩充训练数据。

用法：
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --input data/processed/train.pt --output data/processed/train_aug.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def transform_obs(obs: np.ndarray, aug_id: int) -> np.ndarray:
    """
    对观察进行对称变换
    
    Args:
        obs: (C, H, W) 观察张量
        aug_id: 0-7 变换ID
            0: 原始
            1: 旋转90度
            2: 旋转180度
            3: 旋转270度
            4: 水平翻转
            5: 水平翻转 + 旋转90度
            6: 水平翻转 + 旋转180度
            7: 水平翻转 + 旋转270度
    
    Returns:
        变换后的观察
    """
    if aug_id == 0:
        return obs.copy()
    
    if aug_id < 4:
        # 旋转
        return np.rot90(obs, k=aug_id, axes=(1, 2)).copy()
    else:
        # 先翻转，再旋转
        obs_flip = np.flip(obs, axis=2).copy()
        k = aug_id - 4
        if k == 0:
            return obs_flip
        return np.rot90(obs_flip, k=k, axes=(1, 2)).copy()


def transform_point(row: int, col: int, aug_id: int, board_size: int = 14) -> tuple:
    """
    对坐标进行对称变换
    
    Args:
        row, col: 原始坐标 (1-indexed)
        aug_id: 变换ID
        board_size: 棋盘大小
    
    Returns:
        变换后的 (row, col)
    """
    # 转为0-indexed
    r, c = row - 1, col - 1
    n = board_size
    
    if aug_id == 0:
        new_r, new_c = r, c
    elif aug_id == 1:  # 旋转90度
        new_r, new_c = c, n - 1 - r
    elif aug_id == 2:  # 旋转180度
        new_r, new_c = n - 1 - r, n - 1 - c
    elif aug_id == 3:  # 旋转270度
        new_r, new_c = n - 1 - c, r
    elif aug_id == 4:  # 水平翻转
        new_r, new_c = r, n - 1 - c
    elif aug_id == 5:  # 水平翻转 + 旋转90度
        new_r, new_c = n - 1 - c, n - 1 - r
    elif aug_id == 6:  # 水平翻转 + 旋转180度
        new_r, new_c = n - 1 - r, c
    elif aug_id == 7:  # 水平翻转 + 旋转270度
        new_r, new_c = c, r
    else:
        new_r, new_c = r, c
    
    # 转回1-indexed
    return new_r + 1, new_c + 1


def transform_cand_feats(cand_feats: np.ndarray, aug_id: int, board_size: int = 14) -> np.ndarray:
    """
    对候选特征进行对称变换
    
    候选特征格式（14维）：
    - 0-4: act_onehot (5维)
    - 5-6: from坐标 (2维, 归一化)
    - 7-8: to坐标 (2维, 归一化)
    - 9-10: delta (2维)
    - 11: seq_len
    - 12: phase
    - 13: flying
    """
    if aug_id == 0:
        return cand_feats.copy()
    
    new_feats = cand_feats.copy()
    n = board_size
    
    for i in range(len(new_feats)):
        feat = new_feats[i]
        
        # 变换from坐标 (索引5-6, 值在0-1范围)
        from_r = feat[5] * (n - 1) + 1
        from_c = feat[6] * (n - 1) + 1
        new_from_r, new_from_c = transform_point(from_r, from_c, aug_id, n)
        feat[5] = (new_from_r - 1) / (n - 1)
        feat[6] = (new_from_c - 1) / (n - 1)
        
        # 变换to坐标 (索引7-8)
        to_r = feat[7] * (n - 1) + 1
        to_c = feat[8] * (n - 1) + 1
        new_to_r, new_to_c = transform_point(to_r, to_c, aug_id, n)
        feat[7] = (new_to_r - 1) / (n - 1)
        feat[8] = (new_to_c - 1) / (n - 1)
        
        # 重新计算delta (索引9-10)
        feat[9] = feat[7] - feat[5]  # delta_r
        feat[10] = feat[8] - feat[6]  # delta_c
    
    return new_feats


def augment_sample(sample: dict, aug_id: int) -> dict:
    """
    对单个样本进行增强
    
    Args:
        sample: 包含 obs, phase_id, cand_feats, label_idx, value 的字典
        aug_id: 变换ID (0-7)
    
    Returns:
        增强后的样本
    """
    if aug_id == 0:
        return deepcopy(sample)
    
    new_sample = {}
    
    # 变换观察
    obs = sample['obs']
    if isinstance(obs, torch.Tensor):
        obs = obs.numpy()
    new_sample['obs'] = transform_obs(obs, aug_id)
    
    # 保持不变的字段
    new_sample['phase_id'] = sample['phase_id']
    new_sample['label_idx'] = sample['label_idx']  # 索引不变
    new_sample['value'] = sample['value']
    
    # 变换候选特征
    cand_feats = sample['cand_feats']
    if isinstance(cand_feats, torch.Tensor):
        cand_feats = cand_feats.numpy()
    new_sample['cand_feats'] = transform_cand_feats(cand_feats, aug_id)
    
    return new_sample


def augment_dataset(
    input_path: str,
    output_path: str,
    num_augmentations: int = 8,
    include_original: bool = True
):
    """
    增强整个数据集
    
    Args:
        input_path: 输入.pt文件路径
        output_path: 输出.pt文件路径
        num_augmentations: 增强数量 (1-8)
        include_original: 是否包含原始样本
    """
    print(f"📂 加载数据: {input_path}")
    data = torch.load(input_path, weights_only=False)
    print(f"   原始样本数: {len(data)}")
    
    augmented_data = []
    
    # 确定要使用的变换
    aug_ids = list(range(num_augmentations))
    if not include_original and 0 in aug_ids:
        aug_ids = aug_ids[1:]
    
    print(f"🔄 使用 {len(aug_ids)} 种变换: {aug_ids}")
    
    for sample in tqdm(data, desc="增强中"):
        for aug_id in aug_ids:
            try:
                aug_sample = augment_sample(sample, aug_id)
                augmented_data.append(aug_sample)
            except Exception as e:
                print(f"⚠️ 增强失败 (aug_id={aug_id}): {e}")
                continue
    
    print(f"\n📊 增强结果:")
    print(f"   原始: {len(data)}")
    print(f"   增强后: {len(augmented_data)}")
    print(f"   倍数: {len(augmented_data) / len(data):.1f}x")
    
    print(f"\n💾 保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(augmented_data, output_path)
    
    # 验证
    print("\n✅ 验证增强数据...")
    loaded = torch.load(output_path, weights_only=False)
    print(f"   加载样本数: {len(loaded)}")
    
    # 检查第一个样本
    sample = loaded[0]
    print(f"   样本keys: {list(sample.keys())}")
    print(f"   obs shape: {np.array(sample['obs']).shape}")
    print(f"   cand_feats shape: {np.array(sample['cand_feats']).shape}")
    
    print("\n🎉 数据增强完成!")


def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--input', type=str, default='data/processed/train.pt',
                        help='输入数据集路径')
    parser.add_argument('--output', type=str, default='data/processed/train_aug8x.pt',
                        help='输出数据集路径')
    parser.add_argument('--num_aug', type=int, default=8,
                        help='增强数量 (1-8)')
    parser.add_argument('--no_original', action='store_true',
                        help='不包含原始样本（仅增强）')
    args = parser.parse_args()
    
    print("="*60)
    print("🔄 久棋数据增强工具")
    print("="*60)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"增强倍数: {args.num_aug}x")
    print("="*60)
    
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        num_augmentations=args.num_aug,
        include_original=not args.no_original
    )


if __name__ == '__main__':
    main()

