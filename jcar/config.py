#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet 模型配置

CoordAttention-ResNet 久棋模型的配置项。
包含模型架构、训练超参数、监控器配置等。

Author: JiuqiNet Team
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os


@dataclass
class JiuqiNetConfig:
    """
    JiuqiNet 模型架构配置

    包含网络结构的所有超参数，支持序列化和反序列化。

    Attributes:
        in_channels: 输入通道数，默认6（棋盘状态编码）
        board_size: 棋盘大小，默认14×14
        backbone_channels: 主干网络通道数
        hidden_channels: 隐藏层通道数（别名，与backbone_channels相同）
        num_blocks: CoordAttention残差块数量
        kernel_size: 卷积核大小
        coord_reduction: CoordAttention通道压缩比
        policy_channels: 策略头中间通道数
        cand_feat_dim: 候选动作特征维度
        cand_hidden_dim: 候选评分隐藏层维度
        value_channels: 价值头中间通道数
        value_hidden_dim: 价值头隐藏层维度
        num_phases: 游戏阶段数（布局/对战/飞子）
        phase_embed_dim: 阶段嵌入维度
        dropout: Dropout比例（保留接口，当前未使用）
    """

    # 输入配置
    in_channels: int = 6          # 输入通道数 (棋盘状态编码)
    board_size: int = 14          # 棋盘大小

    # Backbone 配置
    backbone_channels: int = 128   # 残差块通道数
    hidden_channels: int = 128     # 隐藏层通道数（别名，用于命令行兼容）
    num_blocks: int = 12           # 残差块数量
    kernel_size: int = 3           # 卷积核大小
    dropout: float = 0.0           # Dropout 比例（保留接口）

    # CoordAttention 配置
    coord_reduction: int = 32      # CoordAttention 通道压缩比

    # Policy Head 配置
    policy_channels: int = 32      # Policy head 中间通道数
    cand_feat_dim: int = 14        # 候选动作特征维度
    cand_hidden_dim: int = 128     # 候选评分隐藏层维度

    # Value Head 配置
    value_channels: int = 1        # Value head 中间通道数
    value_hidden_dim: int = 256    # Value head 隐藏层维度

    # Phase Embedding
    num_phases: int = 3            # 游戏阶段数 (布局/对战/飞子)
    phase_embed_dim: int = 32      # 阶段嵌入维度
    
    def __post_init__(self):
        """参数验证和同步"""
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.board_size > 0, "board_size must be positive"
        assert self.backbone_channels > 0, "backbone_channels must be positive"
        assert self.num_blocks > 0, "num_blocks must be positive"
        assert self.kernel_size > 0 and self.kernel_size % 2 == 1, "kernel_size must be positive odd"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

        # 同步 hidden_channels 和 backbone_channels
        if self.hidden_channels != self.backbone_channels:
            self.backbone_channels = self.hidden_channels

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'in_channels': self.in_channels,
            'board_size': self.board_size,
            'backbone_channels': self.backbone_channels,
            'hidden_channels': self.hidden_channels,
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'coord_reduction': self.coord_reduction,
            'policy_channels': self.policy_channels,
            'cand_feat_dim': self.cand_feat_dim,
            'cand_hidden_dim': self.cand_hidden_dim,
            'value_channels': self.value_channels,
            'value_hidden_dim': self.value_hidden_dim,
            'num_phases': self.num_phases,
            'phase_embed_dim': self.phase_embed_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'JiuqiNetConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    """
    训练配置

    包含训练过程中的所有超参数和设置。

    Attributes:
        seed: 随机种子
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        warmup_epochs: Warmup轮数
        warmup_ratio: Warmup占总步数比例（与warmup_epochs二选一）
        value_loss_weight: Value loss权重
        grad_clip: 梯度裁剪阈值
        save_every: 每N步保存检查点
        max_nan_count: 最大NaN次数后触发恢复
        augment: 是否使用数据增强
        out_dir: 输出目录
        gpus: GPU ID列表
    """

    # 随机种子
    seed: int = 42

    # 训练超参数
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05      # warmup占总步数的比例
    warmup_epochs: int = 3          # warmup轮数

    # 损失权重
    lambda_value: float = 0.5       # value loss 权重（别名）
    value_loss_weight: float = 0.5  # value loss 权重

    # 梯度配置
    grad_accum_steps: int = 1       # 梯度累积步数
    clip_grad_norm: float = 1.0     # 梯度裁剪（别名）
    grad_clip: float = 1.0          # 梯度裁剪阈值

    # 混合精度
    use_amp: bool = True

    # 数据增强
    augment: bool = True
    augment_prob: float = 0.875     # 数据增强概率

    # 保存配置
    out_dir: str = "exp/jcar"
    save_steps: int = 2000          # 每N步保存检查点（别名）
    save_every: int = 1000          # 每N步保存检查点
    save_epochs: bool = True        # 每个epoch结束保存

    # GPU配置
    gpus: str = "0"                 # GPU ID，如 "0" 或 "0,1,2,3"

    # 日志配置
    log_interval: int = 100         # 每N步打印日志

    # 容错配置
    max_nan_per_epoch: int = 50     # 每个epoch最大NaN次数（别名）
    max_nan_count: int = 5          # 触发恢复的最大NaN次数

    # Worker配置
    num_workers: int = 4

    def __post_init__(self):
        """参数验证和同步"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.lr > 0, "lr must be positive"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be in [0, 1]"
        assert self.grad_clip > 0, "grad_clip must be positive"
        assert self.seed >= 0, "seed must be non-negative"

        # 同步别名字段
        self.clip_grad_norm = self.grad_clip
        self.lambda_value = self.value_loss_weight
        self.save_steps = self.save_every

    def get_gpu_ids(self) -> List[int]:
        """解析GPU ID列表"""
        return [int(x.strip()) for x in self.gpus.split(',') if x.strip()]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'seed': self.seed,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'warmup_epochs': self.warmup_epochs,
            'value_loss_weight': self.value_loss_weight,
            'grad_accum_steps': self.grad_accum_steps,
            'grad_clip': self.grad_clip,
            'use_amp': self.use_amp,
            'augment': self.augment,
            'augment_prob': self.augment_prob,
            'out_dir': self.out_dir,
            'save_every': self.save_every,
            'save_epochs': self.save_epochs,
            'gpus': self.gpus,
            'log_interval': self.log_interval,
            'max_nan_count': self.max_nan_count,
            'num_workers': self.num_workers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RLConfig:
    """
    强化学习训练配置

    Attributes:
        lr: 学习率
        weight_decay: 权重衰减
        value_loss_weight: Value loss 权重
        batch_size: 批大小
        train_epochs: 每次迭代训练轮数
        steps_per_epoch: 每轮训练步数
        lr_restart_period: 学习率重启周期
        use_amp: 是否使用混合精度
        mcts_simulations: MCTS 模拟次数 (Gumbel MCTS 只需 16-64)
        mcts_sampled_actions: Gumbel Top-k 采样数
        mcts_c_puct: 探索常数
        dirichlet_alpha: Dirichlet 噪声参数
        noise_frac: 噪声比例
        games_per_iteration: 每次迭代自我对弈局数
        buffer_capacity: 经验回放缓冲区大小
        out_dir: 输出目录
    """

    # 训练参数
    lr: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.5
    batch_size: int = 256
    train_epochs: int = 3
    steps_per_epoch: int = 100
    lr_restart_period: int = 10
    use_amp: bool = True

    # MCTS 参数 (Gumbel MCTS)
    mcts_simulations: int = 32        # Gumbel MCTS 只需较少模拟
    mcts_sampled_actions: int = 16    # Top-k 采样
    mcts_c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    noise_frac: float = 0.25

    # 自我对弈
    games_per_iteration: int = 10
    buffer_capacity: int = 100000
    temperature_schedule: str = "30:1.0,100:0.1,200:0.0"  # step:temp 格式

    # 输出
    out_dir: str = "exp/jcar_rl"

    def __post_init__(self):
        """参数验证"""
        assert self.lr > 0, "lr must be positive"
        assert self.mcts_simulations > 0, "mcts_simulations must be positive"
        assert 0 < self.mcts_c_puct < 10, "mcts_c_puct should be in (0, 10)"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'value_loss_weight': self.value_loss_weight,
            'batch_size': self.batch_size,
            'train_epochs': self.train_epochs,
            'steps_per_epoch': self.steps_per_epoch,
            'lr_restart_period': self.lr_restart_period,
            'use_amp': self.use_amp,
            'mcts_simulations': self.mcts_simulations,
            'mcts_sampled_actions': self.mcts_sampled_actions,
            'mcts_c_puct': self.mcts_c_puct,
            'dirichlet_alpha': self.dirichlet_alpha,
            'noise_frac': self.noise_frac,
            'games_per_iteration': self.games_per_iteration,
            'buffer_capacity': self.buffer_capacity,
            'temperature_schedule': self.temperature_schedule,
            'out_dir': self.out_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RLConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_temperature_schedule(self) -> dict:
        """解析温度调度"""
        schedule = {}
        for item in self.temperature_schedule.split(','):
            step, temp = item.split(':')
            schedule[int(step)] = float(temp)
        return schedule


@dataclass
class MonitorConfig:
    """
    训练监控器配置

    Attributes:
        host: 服务器绑定地址
        port: 服务器端口
        log_dir: 日志目录（包含 train.log 和 history.json）
        refresh_interval: 页面自动刷新间隔（秒）
        max_log_lines: 最大显示日志行数
    """

    host: str = "0.0.0.0"
    port: int = 8889
    log_dir: str = "exp/jcar"
    refresh_interval: int = 5       # 页面刷新间隔（秒）
    max_log_lines: int = 1000       # 最大显示日志行数


# 为了兼容性，保留 ModelConfig 别名
ModelConfig = JiuqiNetConfig


def save_config(model_cfg: JiuqiNetConfig, train_cfg: TrainConfig, path: str) -> None:
    """
    保存配置到JSON文件

    Args:
        model_cfg: 模型配置
        train_cfg: 训练配置
        path: 保存路径
    """
    config = {
        'model': model_cfg.to_dict(),
        'train': train_cfg.to_dict(),
    }
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(path: str) -> Tuple[JiuqiNetConfig, TrainConfig]:
    """
    从JSON文件加载配置

    Args:
        path: 配置文件路径

    Returns:
        (model_cfg, train_cfg) 元组
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_cfg = JiuqiNetConfig.from_dict(config.get('model', {}))
    train_cfg = TrainConfig.from_dict(config.get('train', {}))

    return model_cfg, train_cfg


# 预定义配置
def get_small_config() -> Tuple[JiuqiNetConfig, TrainConfig]:
    """
    小型配置，用于快速测试

    - 64通道，6个残差块
    - 约1.5M参数
    """
    model_cfg = JiuqiNetConfig(
        backbone_channels=64,
        hidden_channels=64,
        num_blocks=6,
    )
    train_cfg = TrainConfig(
        batch_size=128,
        epochs=10,
    )
    return model_cfg, train_cfg


def get_base_config() -> Tuple[JiuqiNetConfig, TrainConfig]:
    """
    基础配置

    - 128通道，12个残差块
    - 约6M参数
    """
    return JiuqiNetConfig(), TrainConfig()


def get_large_config() -> Tuple[JiuqiNetConfig, TrainConfig]:
    """
    大型配置，追求更高性能

    - 256通道，20个残差块
    - 约25M参数
    """
    model_cfg = JiuqiNetConfig(
        backbone_channels=256,
        hidden_channels=256,
        num_blocks=20,
        cand_hidden_dim=256,
        value_hidden_dim=512,
    )
    train_cfg = TrainConfig(
        batch_size=128,
        epochs=100,
        lr=5e-4,
    )
    return model_cfg, train_cfg


if __name__ == '__main__':
    # 测试配置
    print("=" * 60)
    print("Testing JiuqiNet Configuration")
    print("=" * 60)

    model_cfg, train_cfg = get_base_config()
    print("\nModel Config:")
    print(json.dumps(model_cfg.to_dict(), indent=2))
    print("\nTrain Config:")
    print(json.dumps(train_cfg.to_dict(), indent=2))

    # 测试保存/加载
    test_path = '/tmp/test_jiuqi_config.json'
    save_config(model_cfg, train_cfg, test_path)
    loaded_model, loaded_train = load_config(test_path)

    assert model_cfg.to_dict() == loaded_model.to_dict(), "Model config mismatch!"
    assert train_cfg.to_dict() == loaded_train.to_dict(), "Train config mismatch!"

    print(f"\n✓ Config save/load test passed! (saved to {test_path})")

    # 测试预定义配置
    print("\nPredefined Configurations:")
    for name, func in [('small', get_small_config), ('base', get_base_config), ('large', get_large_config)]:
        cfg, _ = func()
        params = cfg.backbone_channels * cfg.backbone_channels * 9 * cfg.num_blocks * 2  # 粗略估计
        print(f"  {name}: {cfg.backbone_channels} channels, {cfg.num_blocks} blocks, ~{params/1e6:.1f}M params")

    print("\n✓ All tests passed!")
