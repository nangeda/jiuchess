# JCAR: Jiu Chess Attention ResNet

基于 CoordAttention 增强的 ResNet 架构，专为藏族久棋（Tibetan Jiu Chess）设计的策略-价值网络。

## 特性

- **CoordAttention 机制**：行列分解的位置感知注意力，完美匹配 14×14 棋盘结构
- **三阶段支持**：布局（placement）、对战（battle）、飞子（endgame）
- **变长候选评分**：支持不同数量的合法走法评估
- **混合精度训练**：FP16 加速，自动 NaN 检测和恢复
- **Web 监控**：实时训练可视化、系统资源监控

## 架构

```
输入 (B, 6, 14, 14)
        ↓
┌─────────────────────────────────────┐
│  JiuqiNetBackbone                   │
│  ├─ Conv 3×3 (6 → 128)              │
│  ├─ CoordAttentionBlock × 8         │
│  │   ├─ Conv 3×3                    │
│  │   ├─ CoordAttention (H/W 分解)   │
│  │   └─ Residual Connection         │
│  └─ Global Features (B, 128)        │
└─────────────────────────────────────┘
        ↓
   ┌────┴────┐
   ↓         ↓
┌──────┐  ┌──────┐
│Policy│  │Value │
│Head  │  │Head  │
└──────┘  └──────┘
```

## 快速开始

### 安装依赖

```bash
pip install torch flask
```

### 训练

```bash
python jcar/train.py \
    --train-shards data/processed/train.pt \
    --val-shards data/processed/val.pt \
    --out-dir exp/jcar \
    --gpus 0 \
    --epochs 50 \
    --batch-size 256
```

### 监控

```bash
python jcar/monitor.py --log-dir exp/jcar --port 8889
```

然后访问 `http://localhost:8889` 查看训练状态。

## 配置

### 模型配置 (JiuqiNetConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_channels` | 6 | 输入通道数 |
| `backbone_channels` | 128 | 骨干网络通道数 |
| `num_blocks` | 8 | CoordAttention 块数量 |
| `reduction` | 16 | 注意力降维比例 |
| `dropout` | 0.0 | Dropout 比例 |

### 训练配置 (TrainConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 256 | 批次大小 |
| `epochs` | 50 | 训练轮数 |
| `lr` | 1e-3 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减 |
| `warmup_epochs` | 3 | 预热轮数 |
| `value_loss_weight` | 0.5 | 价值损失权重 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

## 文件结构

```
jcar/
├── __init__.py          # 模块导出
├── config.py            # 配置类
├── coord_attention.py   # CoordAttention 模块
├── model.py             # JiuqiNet 模型
├── train.py             # 训练脚本
├── monitor.py           # Web 监控器
└── README.md            # 文档
```

## 测试

```bash
# 测试配置
python jcar/config.py

# 测试注意力模块
python jcar/coord_attention.py

# 测试模型
python jcar/model.py
```

## 与 MoE-DyT 对比

| 特性 | MoE-DyT | JiuqiNet |
|------|---------|----------|
| 稳定性 | ⚠️ NaN 问题 | ✅ 稳定 |
| 训练速度 | 中等 | 快 |
| 参数量 | 大 | 中等 |
| 位置感知 | 弱 | 强（CoordAttention）|
| 实现复杂度 | 高 | 低 |

## 后续计划

- [ ] 集成 GNN 模块处理斜向成方规则
- [ ] 实现 Gumbel MCTS 自对弈
- [ ] 添加 ELO 评估系统
- [ ] 支持分布式训练

## 许可证

MIT License

