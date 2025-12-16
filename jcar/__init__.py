#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet: CoordAttention-ResNet 久棋模型

基于 CoordAttention 增强的 ResNet 架构，专为藏族久棋（Tibetan Jiu Chess）设计。

主要特性：
- CoordAttention 机制：行列分解的位置感知注意力，完美匹配棋盘结构
- 三阶段支持：布局（placement）、对战（battle）、飞子（endgame）
- 变长候选评分：支持不同数量的合法走法评估
- 混合精度训练：FP16 加速，自动 NaN 检测和恢复

Author: JiuqiNet Team
"""

__version__ = "1.0.0"
__author__ = "JiuqiNet Team"

# 配置类
from .config import (
    JiuqiNetConfig,
    TrainConfig,
    MonitorConfig,
    RLConfig,
    ModelConfig,  # 别名，兼容旧代码
    save_config,
    load_config,
    get_small_config,
    get_base_config,
    get_large_config,
)

# 注意力模块
from .coord_attention import (
    CoordAttention,
    CoordAttentionBlock,
)

# 模型
from .model import (
    JiuqiNet,
    JiuqiNetBackbone,
    PolicyHead,
    CandidateScorer,
    ValueHead,
    create_model,
    CZero,  # 别名，兼容旧代码
    CZeroBackbone,  # 别名，兼容旧代码
)

# 训练器
from .train import (
    JiuqiNetTrainer,
    StreamingShardDataset,
    collate_fn,
    set_seed,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
)

# 监控器
from .monitor import (
    TrainingMonitor,
)

# Gumbel MCTS
from .gumbel_mcts import (
    MCTSConfig,
    MCTSNode,
    GumbelMCTS,
    SimpleMCTS,
    create_mcts,
)

# 自我对弈
from .self_play import (
    GameSample,
    GameRecord,
    SelfPlayWorker,
    ReplayBuffer,
    SelfPlayManager,
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    # 配置
    "JiuqiNetConfig",
    "TrainConfig",
    "MonitorConfig",
    "RLConfig",
    "ModelConfig",
    "save_config",
    "load_config",
    "get_small_config",
    "get_base_config",
    "get_large_config",
    # 注意力
    "CoordAttention",
    "CoordAttentionBlock",
    # 模型
    "JiuqiNet",
    "JiuqiNetBackbone",
    "PolicyHead",
    "CandidateScorer",
    "ValueHead",
    "create_model",
    "CZero",
    "CZeroBackbone",
    # 训练
    "JiuqiNetTrainer",
    "StreamingShardDataset",
    "collate_fn",
    "set_seed",
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint",
    # 监控
    "TrainingMonitor",
    # MCTS
    "MCTSConfig",
    "MCTSNode",
    "GumbelMCTS",
    "SimpleMCTS",
    "create_mcts",
    # 自我对弈
    "GameSample",
    "GameRecord",
    "SelfPlayWorker",
    "ReplayBuffer",
    "SelfPlayManager",
]

