#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet SFT 训练脚本

阶段1: 使用人类对局数据进行监督学习预训练
- 同时训练 Policy Head 和 Value Head
- Policy: 学习人类走法模式
- Value: 使用Expert评估函数生成的局面价值标签
"""

import os
import sys
import json
import random
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from jcar.model import JiuqiNet
from jcar.config import JiuqiNetConfig


def setup_logging(log_file: str) -> logging.Logger:
    """配置日志"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('jiuqi_sft')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


class HumanGamesDataset(Dataset):
    """人类对局数据集"""
    
    def __init__(self, data_path: str, augment: bool = True):
        self.augment = augment
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.data):,} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        
        obs = sample['obs']
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        else:
            obs = obs.float()
        
        cand_feats = sample['cand_feats']
        if isinstance(cand_feats, np.ndarray):
            cand_feats = torch.from_numpy(cand_feats).float()
        else:
            cand_feats = cand_feats.float()
        
        # 注意：数据增强已禁用！
        # 原因：旋转/翻转 obs 后，cand_feats 中的坐标没有同步变换
        # 导致模型看到的棋盘和候选动作坐标不匹配，无法学习
        # 要正确实现数据增强，需要同时变换 obs 和 cand_feats 中的坐标
        # if self.augment and random.random() < 0.5:
        #     aug_type = random.randint(1, 4)
        #     if aug_type == 1:
        #         obs = torch.rot90(obs, k=1, dims=(1, 2))
        #     elif aug_type == 2:
        #         obs = torch.rot90(obs, k=2, dims=(1, 2))
        #     elif aug_type == 3:
        #         obs = torch.flip(obs, dims=[2])
        #     elif aug_type == 4:
        #         obs = torch.flip(obs, dims=[1])
        
        # 获取value标签（Expert评估生成）
        value = sample.get('value', 0.0)
        if isinstance(value, np.ndarray):
            value = float(value)

        return {
            'obs': obs,
            'phase_id': sample['phase_id'],
            'cand_feats': cand_feats,
            'label_idx': sample['label_idx'],
            'value': value,
        }


def collate_fn(batch: List[dict]) -> dict:
    """自定义 collate 函数"""
    obs = torch.stack([item['obs'] for item in batch])
    phase_ids = torch.tensor([item['phase_id'] for item in batch], dtype=torch.long)
    cand_feats_list = [item['cand_feats'] for item in batch]
    label_idx = [item['label_idx'] for item in batch]
    values = torch.tensor([item['value'] for item in batch], dtype=torch.float32)

    return {
        'obs': obs,
        'phase_ids': phase_ids,
        'cand_feats_list': cand_feats_list,
        'label_idx': label_idx,
        'values': values,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, best_loss, cfg, out_dir, suffix=''):
    """保存检查点"""
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': step,
        'best_val_loss': best_loss,
        'model_config': cfg.to_dict(),
    }
    
    path = os.path.join(out_dir, f'checkpoint{suffix}.pt')
    torch.save(checkpoint, path)
    return path


def train():
    import argparse
    
    parser = argparse.ArgumentParser(description='JiuqiNet SFT Training')
    parser.add_argument('--data', type=str, default='data/processed/human_games.pt')
    parser.add_argument('--out-dir', type=str, default='exp/jcar_sft')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--num-blocks', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--value-loss-weight', type=float, default=0.5,
                        help='Weight for value loss (default: 0.5)')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.out_dir, 'train.log'))
    
    # 加载数据
    dataset = HumanGamesDataset(args.data, augment=True)
    
    # 划分训练/验证集
    n_val = min(5000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train samples: {n_train:,}, Val samples: {n_val:,}")
    
    # 创建模型
    cfg = JiuqiNetConfig(
        hidden_channels=args.hidden_channels,
        backbone_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    model = JiuqiNet(cfg).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {num_params:,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    # 训练
    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # 支持两种checkpoint格式
        model_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
        opt_key = 'optimizer_state_dict' if 'optimizer_state_dict' in ckpt else 'optimizer'
        sched_key = 'scheduler_state_dict' if 'scheduler_state_dict' in ckpt else 'scheduler'
        model.load_state_dict(ckpt[model_key])
        optimizer.load_state_dict(ckpt[opt_key])
        scheduler.load_state_dict(ckpt[sched_key])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val_loss = ckpt['best_val_loss']
        if os.path.exists(os.path.join(args.out_dir, 'history.json')):
            with open(os.path.join(args.out_dir, 'history.json')) as f:
                history = json.load(f)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    logger.info("=" * 60)
    logger.info("Starting SFT Training (Policy Only)")
    logger.info("=" * 60)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            obs = batch['obs'].to(device)
            phase_ids = batch['phase_ids'].to(device)
            cand_feats_list = [c.to(device) for c in batch['cand_feats_list']]
            label_idx = batch['label_idx']
            value_targets = batch['values'].to(device)  # (B,)

            optimizer.zero_grad()

            # 前向传播 - 同时获取 logits 和 values
            logits_list, values = model.score_candidates(obs, phase_ids, cand_feats_list)

            # 计算 Policy Loss
            targets = torch.tensor(label_idx, dtype=torch.long, device=device)

            # 将变长 logits 拼接
            max_len = max(l.size(0) for l in logits_list)
            padded_logits = torch.full((len(logits_list), max_len), float('-inf'), device=device)
            for i, l in enumerate(logits_list):
                padded_logits[i, :l.size(0)] = l

            policy_loss = F.cross_entropy(padded_logits, targets)

            # 计算 Value Loss (MSE)
            value_loss = F.mse_loss(values.squeeze(-1), value_targets)

            # 总损失
            loss = policy_loss + args.value_loss_weight * value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # 统计
            preds = padded_logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()

            epoch_loss += loss.item() * obs.size(0)
            epoch_correct += correct
            epoch_total += obs.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'p_loss': f'{policy_loss.item():.4f}',
                'v_loss': f'{value_loss.item():.4f}',
                'acc': f'{correct/obs.size(0)*100:.1f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                phase_ids = batch['phase_ids'].to(device)
                cand_feats_list = [c.to(device) for c in batch['cand_feats_list']]
                label_idx = batch['label_idx']
                value_targets = batch['values'].to(device)

                logits_list, values = model.score_candidates(obs, phase_ids, cand_feats_list)
                targets = torch.tensor(label_idx, dtype=torch.long, device=device)

                max_len = max(l.size(0) for l in logits_list)
                padded_logits = torch.full((len(logits_list), max_len), float('-inf'), device=device)
                for i, l in enumerate(logits_list):
                    padded_logits[i, :l.size(0)] = l

                policy_loss = F.cross_entropy(padded_logits, targets)
                value_loss = F.mse_loss(values.squeeze(-1), value_targets)
                loss = policy_loss + args.value_loss_weight * value_loss

                preds = padded_logits.argmax(dim=-1)

                val_loss += loss.item() * obs.size(0)
                val_policy_loss += policy_loss.item() * obs.size(0)
                val_value_loss += value_loss.item() * obs.size(0)
                val_correct += (preds == targets).sum().item()
                val_total += obs.size(0)
        
        # 记录
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss_avg = val_loss / val_total
        val_policy_loss_avg = val_policy_loss / val_total
        val_value_loss_avg = val_value_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_policy_loss'] = history.get('val_policy_loss', [])
        history['val_policy_loss'].append(val_policy_loss_avg)
        history['val_value_loss'] = history.get('val_value_loss', [])
        history['val_value_loss'].append(val_value_loss_avg)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss_avg:.4f} (P: {val_policy_loss_avg:.4f}, V: {val_value_loss_avg:.4f}), Acc: {val_acc*100:.2f}%"
        )
        
        # 保存历史
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, cfg, args.out_dir, '_best')
            logger.info(f"  New best model saved!")
        
        # 保存周期检查点
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, cfg, args.out_dir, f'_epoch{epoch}')
    
    # 保存最终模型
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, global_step, best_val_loss, cfg, args.out_dir, '_final')
    
    logger.info("=" * 60)
    logger.info(f"Training completed! Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    train()
