#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet 训练脚本

CoordAttention-ResNet 久棋模型训练。

支持：
- 多GPU训练 (DataParallel / DistributedDataParallel)
- 流式分片数据加载
- 数据增强（旋转、翻转）
- 断点续训
- NaN检测和自动恢复
- 梯度裁剪和混合精度训练

Author: JiuqiNet Team
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jcar.config import JiuqiNetConfig, TrainConfig, save_config
    from jcar.model import JiuqiNet
except ImportError:
    from config import JiuqiNetConfig, TrainConfig, save_config
    from model import JiuqiNet


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_file: str, console_level: int = logging.INFO) -> logging.Logger:
    """配置日志"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('jcar')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# 数据集
# ============================================================================

class StreamingShardDataset(Dataset):
    """流式分片数据集"""
    
    def __init__(
        self, 
        shard_paths: List[str], 
        augment: bool = True,
        augment_prob: float = 0.875
    ) -> None:
        self.augment = augment
        self.augment_prob = augment_prob
        self.data: List[dict] = []
        
        print(f"Loading {len(shard_paths)} shards...")
        for shard_path in tqdm(shard_paths, desc='Loading shards'):
            if not os.path.exists(shard_path):
                print(f"Warning: shard not found: {shard_path}")
                continue
            shard_data = torch.load(shard_path, weights_only=False)
            self.data.extend(shard_data)
            gc.collect()
        
        print(f'Loaded {len(self.data):,} samples from {len(shard_paths)} shards.')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        
        if self.augment and random.random() < self.augment_prob:
            aug_type = random.randint(1, 7)
            return self._augment_sample(sample, aug_type)
        else:
            return self._prepare_sample(sample)
    
    def _prepare_sample(self, sample: dict) -> dict:
        """准备样本"""
        obs = sample['obs']
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()
        
        cand_feats = sample['cand_feats']
        if isinstance(cand_feats, np.ndarray):
            cand_feats = torch.from_numpy(cand_feats).float()
        elif not isinstance(cand_feats, torch.Tensor):
            cand_feats = torch.tensor(cand_feats, dtype=torch.float32)
        else:
            cand_feats = cand_feats.float()
        
        value = sample['value']
        if isinstance(value, (int, float)):
            value = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        else:
            value = value.float()
        
        return {
            'obs': obs,
            'cand_feats': cand_feats,
            'phase_id': sample['phase_id'],
            'label_idx': sample['label_idx'],
            'value': value
        }
    
    def _augment_sample(self, sample: dict, aug_type: int) -> dict:
        """数据增强：旋转和翻转"""
        prepared = self._prepare_sample(sample)
        obs = prepared['obs']
        cand_feats = prepared['cand_feats']
        
        # 对 obs 进行变换 (C, H, W)
        if aug_type == 1:  # 旋转90度
            obs = torch.rot90(obs, k=1, dims=(1, 2))
        elif aug_type == 2:  # 旋转180度
            obs = torch.rot90(obs, k=2, dims=(1, 2))
        elif aug_type == 3:  # 旋转270度
            obs = torch.rot90(obs, k=3, dims=(1, 2))
        elif aug_type == 4:  # 水平翻转
            obs = torch.flip(obs, dims=[2])
        elif aug_type == 5:  # 垂直翻转
            obs = torch.flip(obs, dims=[1])
        elif aug_type == 6:  # 对角翻转
            obs = obs.transpose(1, 2)
        elif aug_type == 7:  # 反对角翻转
            obs = torch.flip(obs.transpose(1, 2), dims=[1, 2])
        
        prepared['obs'] = obs
        return prepared


def collate_fn(batch: List[dict]) -> dict:
    """自定义 collate 函数，处理变长候选"""
    obs = torch.stack([item['obs'] for item in batch])
    phase_ids = torch.tensor([item['phase_id'] for item in batch], dtype=torch.long)
    values = torch.stack([item['value'] for item in batch])
    
    cand_feats_list = [item['cand_feats'] for item in batch]
    label_idx = [item['label_idx'] for item in batch]
    
    return {
        'obs': obs,
        'phase_ids': phase_ids,
        'values': values,
        'cand_feats_list': cand_feats_list,
        'label_idx': label_idx,
    }


# ============================================================================
# 训练工具函数
# ============================================================================

def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_ids: List[int]) -> torch.device:
    """获取设备"""
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        return torch.device(f'cuda:{gpu_ids[0]}')
    return torch.device('cpu')


def setup_model_for_training(
    model: JiuqiNet,
    gpu_ids: List[int]
) -> Tuple[nn.Module, torch.device]:
    """配置模型用于训练"""
    device = get_device(gpu_ids)
    model = model.to(device)

    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    return model, device


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    model_cfg: JiuqiNetConfig,
    train_cfg: TrainConfig,
    out_dir: str,
    suffix: str = ''
) -> str:
    """保存检查点"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取原始模型（去除 DataParallel 包装）
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model_config': model_cfg.to_dict(),
        'train_config': train_cfg.to_dict(),
    }
    
    filename = f'checkpoint{suffix}.pt'
    path = os.path.join(out_dir, filename)
    torch.save(checkpoint, path)
    
    return path


def load_checkpoint(
    path: str,
    model: JiuqiNet,
    optimizer: Optional[AdamW] = None,
    scheduler = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None
) -> Tuple[int, int, float]:
    """加载检查点"""
    checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
    
    # 加载模型
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 加载 scaler
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    return epoch, global_step, best_val_loss


def check_nan_in_model(model: nn.Module) -> bool:
    """检查模型参数是否包含 NaN"""
    for name, param in model.named_parameters():
        if param is not None and torch.isnan(param).any():
            return True
    return False


def check_nan_in_grads(model: nn.Module) -> Tuple[bool, Optional[str]]:
    """检查梯度是否包含 NaN"""
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True, name
    return False, None


# ============================================================================
# 训练器
# ============================================================================

class JiuqiNetTrainer:
    """JiuqiNet 训练器"""

    def __init__(
        self,
        model_cfg: JiuqiNetConfig,
        train_cfg: TrainConfig,
        train_shards: List[str],
        val_shards: List[str],
        gpu_ids: List[int],
        resume_from: Optional[str] = None
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.gpu_ids = gpu_ids

        # 设置随机种子
        set_seed(train_cfg.seed)

        # 创建输出目录
        os.makedirs(train_cfg.out_dir, exist_ok=True)

        # 设置日志
        log_file = os.path.join(train_cfg.out_dir, 'train.log')
        self.logger = setup_logging(log_file)

        # 保存配置
        save_config(model_cfg, train_cfg, os.path.join(train_cfg.out_dir, 'config.json'))

        # 创建数据集
        self.logger.info("Creating datasets...")
        self.train_dataset = StreamingShardDataset(
            train_shards,
            augment=True,
            augment_prob=train_cfg.augment_prob
        )
        self.val_dataset = StreamingShardDataset(
            val_shards,
            augment=False
        ) if val_shards else None

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        ) if self.val_dataset else None

        # 创建模型
        self.logger.info("Creating model...")
        self.model = JiuqiNet(model_cfg)
        self.model, self.device = setup_model_for_training(self.model, gpu_ids)
        
        # 统计参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model params: {num_params:,} (trainable: {trainable_params:,})")
        
        # 创建优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay
        )
        
        # 创建学习率调度器（Warmup + Cosine）
        total_steps = len(self.train_loader) * train_cfg.epochs
        warmup_steps = len(self.train_loader) * train_cfg.warmup_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=train_cfg.lr * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # 混合精度训练
        self.scaler = GradScaler('cuda') if train_cfg.use_amp and torch.cuda.is_available() else None
        
        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.nan_count = 0  # NaN 计数器
        self.last_good_checkpoint = None  # 最后一个正常的检查点
        
        # 断点续训
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from {resume_from}")
            self.start_epoch, self.global_step, self.best_val_loss = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.scaler, self.device
            )
            self.start_epoch += 1
            self.logger.info(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
        
        # 训练历史（用于监控）
        self.history = {
            'train_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_times': [],
            'phase_acc': {0: [], 1: [], 2: []},
        }
    
    def compute_loss(
        self,
        batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Dict[int, Tuple[int, int]]]:
        """
        计算损失

        Args:
            batch: 包含 obs, phase_ids, values, cand_feats_list, label_idx 的字典

        Returns:
            total_loss: 总损失
            policy_loss: 策略损失
            value_loss: 价值损失
            accuracy: 准确率
            phase_stats: 各阶段统计 {phase: (correct, total)}
        """
        obs = batch['obs'].to(self.device)
        phase_ids = batch['phase_ids'].to(self.device)
        values = batch['values'].to(self.device)
        cand_feats_list = [cf.to(self.device) for cf in batch['cand_feats_list']]
        label_idx = batch['label_idx']  # List[int]

        # 获取底层模型（处理 DataParallel 包装）
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # 前向传播：使用 score_candidates 获取每个样本的 logits
        logits_list, value_pred = model.score_candidates(obs, phase_ids, cand_feats_list)

        # 计算 Policy Loss（逐样本 Cross Entropy）
        policy_losses = []
        correct = 0
        total = 0
        phase_stats: Dict[int, Tuple[int, int]] = {}

        for i, (logits, target_idx) in enumerate(zip(logits_list, label_idx)):
            if logits.numel() == 0:
                continue

            # 单样本 Cross Entropy
            target = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            policy_losses.append(F.cross_entropy(logits.unsqueeze(0), target))

            # 准确率统计
            pred = logits.argmax().item()
            is_correct = (pred == target_idx)
            correct += int(is_correct)
            total += 1

            # 按阶段统计
            phase = phase_ids[i].item()
            if phase not in phase_stats:
                phase_stats[phase] = (0, 0)
            c, t = phase_stats[phase]
            phase_stats[phase] = (c + int(is_correct), t + 1)

        # 平均 Policy Loss
        if policy_losses:
            policy_loss = torch.stack(policy_losses).mean()
        else:
            policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Value Loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(-1), values)

        # 总损失
        total_loss = policy_loss + self.train_cfg.value_loss_weight * value_loss

        # 准确率
        accuracy = correct / total if total > 0 else 0.0

        return total_loss, policy_loss, value_loss, accuracy, phase_stats
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct = 0
        total_samples = 0
        phase_correct = {0: 0, 1: 0, 2: 0}
        phase_total = {0: 0, 1: 0, 2: 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # 混合精度前向
            if self.scaler:
                with autocast('cuda'):
                    loss, p_loss, v_loss, acc, phase_stats = self.compute_loss(batch)
                
                # 检查 NaN
                if torch.isnan(loss):
                    self.handle_nan(epoch, batch_idx)
                    continue
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_cfg.grad_clip
                )
                
                # 检查梯度 NaN
                has_nan, nan_param = check_nan_in_grads(self.model)
                if has_nan:
                    self.logger.warning(f"NaN gradient detected in {nan_param}")
                    self.handle_nan(epoch, batch_idx)
                    # 重置 scaler 状态并清零梯度
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    continue

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, p_loss, v_loss, acc, phase_stats = self.compute_loss(batch)
                
                if torch.isnan(loss):
                    self.handle_nan(epoch, batch_idx)
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_cfg.grad_clip
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            # 统计
            batch_size = batch['obs'].size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += p_loss.item() * batch_size
            total_value_loss += v_loss.item() * batch_size
            total_correct += int(acc * batch_size)
            total_samples += batch_size
            
            for phase, (c, t) in phase_stats.items():
                phase_correct[phase] += c
                phase_total[phase] += t
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc*100:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 定期保存检查点
            if self.global_step % self.train_cfg.save_every == 0:
                path = save_checkpoint(
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch, self.global_step, self.best_val_loss,
                    self.model_cfg, self.train_cfg, self.train_cfg.out_dir,
                    suffix='_latest'
                )
                self.last_good_checkpoint = path
        
        # 计算平均值
        metrics = {
            'loss': total_loss / total_samples,
            'policy_loss': total_policy_loss / total_samples,
            'value_loss': total_value_loss / total_samples,
            'acc': total_correct / total_samples,
            'lr': self.scheduler.get_last_lr()[0],
        }
        
        # 阶段准确率
        for phase in [0, 1, 2]:
            if phase_total[phase] > 0:
                metrics[f'phase_{phase}_acc'] = phase_correct[phase] / phase_total[phase]
            else:
                metrics[f'phase_{phase}_acc'] = 0.0
        
        return metrics
    
    def handle_nan(self, epoch: int, batch_idx: int) -> None:
        """处理 NaN"""
        self.nan_count += 1
        self.logger.warning(f"NaN detected at epoch {epoch}, batch {batch_idx}. Count: {self.nan_count}")
        
        # 保存紧急检查点
        emergency_path = save_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler,
            epoch, self.global_step, self.best_val_loss,
            self.model_cfg, self.train_cfg, self.train_cfg.out_dir,
            suffix='_emergency'
        )
        self.logger.info(f"Emergency checkpoint saved to {emergency_path}")
        
        # 如果 NaN 次数过多，尝试从上一个好的检查点恢复
        if self.nan_count >= self.train_cfg.max_nan_count:
            self.logger.error(f"Too many NaN ({self.nan_count}). Attempting recovery...")
            
            if self.last_good_checkpoint and os.path.exists(self.last_good_checkpoint):
                self.logger.info(f"Loading last good checkpoint: {self.last_good_checkpoint}")
                load_checkpoint(
                    self.last_good_checkpoint, self.model, 
                    self.optimizer, self.scheduler, self.scaler, self.device
                )
                self.nan_count = 0
                
                # 降低学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.logger.info("Learning rate halved after recovery.")
            else:
                self.logger.error("No good checkpoint available. Training may be unstable.")
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        phase_correct = {0: 0, 1: 0, 2: 0}
        phase_total = {0: 0, 1: 0, 2: 0}
        
        for batch in tqdm(self.val_loader, desc='Validating', leave=False):
            loss, _, _, acc, phase_stats = self.compute_loss(batch)
            
            batch_size = batch['obs'].size(0)
            total_loss += loss.item() * batch_size
            total_correct += int(acc * batch_size)
            total_samples += batch_size
            
            for phase, (c, t) in phase_stats.items():
                phase_correct[phase] += c
                phase_total[phase] += t
        
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples,
        }
        
        for phase in [0, 1, 2]:
            if phase_total[phase] > 0:
                metrics[f'val_phase_{phase}_acc'] = phase_correct[phase] / phase_total[phase]
        
        return metrics
    
    def train(self) -> None:
        """主训练循环"""
        self.logger.info("=" * 60)
        self.logger.info("Starting JiuqiNet Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Model config: {self.model_cfg}")
        self.logger.info(f"Train config: {self.train_cfg}")
        self.logger.info(f"Device: {self.device}, GPUs: {self.gpu_ids}")
        self.logger.info(f"Train samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            self.logger.info(f"Val samples: {len(self.val_dataset):,}")
        self.logger.info("=" * 60)
        
        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.history['train_value_loss'].append(train_metrics['value_loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['lr'].append(train_metrics['lr'])
            self.history['epoch_times'].append(epoch_time)
            
            for phase in [0, 1, 2]:
                self.history['phase_acc'][phase].append(train_metrics.get(f'phase_{phase}_acc', 0))
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_acc'].append(val_metrics['val_acc'])
            
            # 保存历史到文件（供监控器读取）
            history_path = os.path.join(self.train_cfg.out_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # 日志
            log_msg = (
                f"Epoch {epoch} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['acc']*100:.2f}% | "
                f"Phase[0/1/2]: {train_metrics.get('phase_0_acc', 0)*100:.1f}/"
                f"{train_metrics.get('phase_1_acc', 0)*100:.1f}/"
                f"{train_metrics.get('phase_2_acc', 0)*100:.1f}% | "
                f"LR: {train_metrics['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            if val_metrics:
                log_msg += f" | Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']*100:.2f}%"
            
            self.logger.info(log_msg)
            
            # 保存最佳模型
            current_loss = val_metrics.get('val_loss', train_metrics['loss'])
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                best_path = save_checkpoint(
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch, self.global_step, self.best_val_loss,
                    self.model_cfg, self.train_cfg, self.train_cfg.out_dir,
                    suffix='_best'
                )
                self.logger.info(f"New best model saved: {best_path}")
            
            # 保存周期检查点
            epoch_path = save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, self.global_step, self.best_val_loss,
                self.model_cfg, self.train_cfg, self.train_cfg.out_dir,
                suffix=f'_epoch{epoch}'
            )
            self.last_good_checkpoint = epoch_path
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed!")
        self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 60)


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='JiuqiNet Training')

    # 数据参数
    parser.add_argument('--train-shards', type=str, nargs='+', required=True,
                        help='Training shard paths')
    parser.add_argument('--val-shards', type=str, nargs='*', default=[],
                        help='Validation shard paths')

    # 模型参数
    parser.add_argument('--in-channels', type=int, default=6)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--num-blocks', type=int, default=12)
    parser.add_argument('--board-size', type=int, default=14)
    parser.add_argument('--cand-feat-dim', type=int, default=14)
    parser.add_argument('--num-phases', type=int, default=3)
    parser.add_argument('--coord-reduction', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--value-loss-weight', type=float, default=0.5)
    parser.add_argument('--augment-prob', type=float, default=0.875)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # 混合精度
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')

    # 保存参数
    parser.add_argument('--out-dir', type=str, default='exp/jcar',
                        help='Output directory')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--max-nan-count', type=int, default=5,
                        help='Max NaN before recovery')

    # GPU 参数
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs to use, comma-separated (e.g., "0,1,2,3")')

    # 断点续训
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    # 解析 GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else []

    # 创建配置
    model_cfg = JiuqiNetConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        backbone_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        board_size=args.board_size,
        cand_feat_dim=args.cand_feat_dim,
        num_phases=args.num_phases,
        coord_reduction=args.coord_reduction,
        dropout=args.dropout
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        value_loss_weight=args.value_loss_weight,
        augment_prob=args.augment_prob,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=args.use_amp,
        out_dir=args.out_dir,
        save_every=args.save_every,
        max_nan_count=args.max_nan_count
    )

    # 创建训练器并开始训练
    trainer = JiuqiNetTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        gpu_ids=gpu_ids,
        resume_from=args.resume
    )

    trainer.train()


if __name__ == '__main__':
    main()
