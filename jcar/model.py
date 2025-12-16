#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JiuqiNet: CoordAttention-ResNet 久棋模型

基于 CoordAttention 增强的 ResNet 架构，专为藏族久棋设计。
支持变长候选动作评分和三阶段游戏（布局/对战/飞子）。

Author: JiuqiNet Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    from .coord_attention import CoordAttention, CoordAttentionBlock
    from .config import JiuqiNetConfig
except ImportError:
    from coord_attention import CoordAttention, CoordAttentionBlock
    from config import JiuqiNetConfig


class JiuqiNetBackbone(nn.Module):
    """
    JiuqiNet 主干网络

    使用 CoordAttention 增强的 ResNet 结构提取棋盘特征。

    Args:
        cfg: 模型配置
    """

    def __init__(self, cfg: JiuqiNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # 输入投影: (B, in_channels, H, W) -> (B, backbone_channels, H, W)
        self.input_conv = nn.Conv2d(
            cfg.in_channels, 
            cfg.backbone_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(cfg.backbone_channels)
        
        # Phase 嵌入
        self.phase_embed = nn.Embedding(cfg.num_phases, cfg.phase_embed_dim)
        
        # Phase 融合层
        self.phase_proj = nn.Conv2d(
            cfg.backbone_channels + cfg.phase_embed_dim,
            cfg.backbone_channels,
            kernel_size=1,
            bias=False
        )
        self.phase_bn = nn.BatchNorm2d(cfg.backbone_channels)
        
        # 残差塔
        self.blocks = nn.ModuleList([
            CoordAttentionBlock(
                channels=cfg.backbone_channels,
                kernel_size=cfg.kernel_size,
                reduction=cfg.coord_reduction
            )
            for _ in range(cfg.num_blocks)
        ])
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.input_bn.weight)
        nn.init.zeros_(self.input_bn.bias)
        nn.init.kaiming_normal_(self.phase_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.phase_bn.weight)
        nn.init.zeros_(self.phase_bn.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        phase_id: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 棋盘状态 (B, C, H, W)
            phase_id: 游戏阶段 (B,)
        
        Returns:
            特征张量 (B, backbone_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 输入投影
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Phase 嵌入并融合
        phase_emb = self.phase_embed(phase_id)  # (B, phase_embed_dim)
        phase_emb = phase_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, phase_embed_dim, H, W)
        x = torch.cat([x, phase_emb], dim=1)  # (B, backbone_channels + phase_embed_dim, H, W)
        x = F.relu(self.phase_bn(self.phase_proj(x)))
        
        # 残差塔
        for block in self.blocks:
            x = block(x)
        
        return x


class PolicyHead(nn.Module):
    """
    策略头
    
    将棋盘特征转换为全局表示，用于候选动作评分。
    
    Args:
        cfg: 模型配置
    """
    
    def __init__(self, cfg: JiuqiNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # 特征压缩
        self.conv = nn.Conv2d(cfg.backbone_channels, cfg.policy_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.policy_channels)
        
        # 全局特征维度
        self.global_dim = cfg.policy_channels * cfg.board_size * cfg.board_size
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor, return_map: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 特征张量 (B, backbone_channels, H, W)
            return_map: 是否返回特征图而不是 flatten 的特征

        Returns:
            如果 return_map=True: 特征图 (B, policy_channels, H, W)
            否则: 全局特征 (B, global_dim)
        """
        x = F.relu(self.bn(self.conv(x)))
        if return_map:
            return x  # (B, policy_channels, H, W)
        return x.flatten(1)  # (B, policy_channels * H * W)


class CandidateScorer(nn.Module):
    """
    候选动作评分器（位置感知版本）

    利用候选位置的局部特征和全局特征来评分候选动作。

    改进：
    - 从 policy_feat_map 中提取候选位置的局部特征
    - 将局部特征与候选特征结合
    - 使用全局上下文增强评分

    Args:
        cfg: 模型配置
    """

    def __init__(self, cfg: JiuqiNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.board_size = cfg.board_size

        # 局部特征维度 = policy_channels（从位置提取）
        # 候选特征维度 = cand_feat_dim
        # 全局特征维度 = policy_channels（全局池化）
        local_dim = cfg.policy_channels * 2  # go 和 to 两个位置
        global_pool_dim = cfg.policy_channels

        # 输入维度 = 局部特征 + 全局池化特征 + 候选特征
        input_dim = local_dim + global_pool_dim + cfg.cand_feat_dim

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, cfg.cand_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.cand_hidden_dim, cfg.cand_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.cand_hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        policy_feat_map: torch.Tensor,
        cand_feats: torch.Tensor,
        global_pool: torch.Tensor = None
    ) -> torch.Tensor:
        """
        评分候选动作

        Args:
            policy_feat_map: 策略特征图 (C, H, W) 或 (B, C, H, W)
            cand_feats: 候选特征 (N, cand_feat_dim)
                - 特征格式: [act_onehot(5), go_row, go_col, to_row, to_col, ...]
                - go_row, go_col, to_row, to_col 是归一化坐标 [0, 1]
            global_pool: 全局池化特征 (C,)，可选

        Returns:
            候选分数 (N,)
        """
        # 确保 policy_feat_map 是 3D (C, H, W)
        if policy_feat_map.dim() == 4:
            policy_feat_map = policy_feat_map.squeeze(0)

        C, H, W = policy_feat_map.shape
        N = cand_feats.size(0)
        device = policy_feat_map.device

        # 从候选特征中提取坐标（索引 5-8: go_row, go_col, to_row, to_col）
        # 坐标是归一化的 [0, 1]，需要转换为像素坐标
        go_row = (cand_feats[:, 5] * (H - 1)).long().clamp(0, H - 1)  # (N,)
        go_col = (cand_feats[:, 6] * (W - 1)).long().clamp(0, W - 1)  # (N,)
        to_row = (cand_feats[:, 7] * (H - 1)).long().clamp(0, H - 1)  # (N,)
        to_col = (cand_feats[:, 8] * (W - 1)).long().clamp(0, W - 1)  # (N,)

        # 提取 go 位置的局部特征
        go_feats = policy_feat_map[:, go_row, go_col].T  # (N, C)

        # 提取 to 位置的局部特征
        to_feats = policy_feat_map[:, to_row, to_col].T  # (N, C)

        # 计算全局池化特征
        if global_pool is None:
            global_pool = policy_feat_map.mean(dim=(1, 2))  # (C,)
        global_pool_expanded = global_pool.unsqueeze(0).expand(N, -1)  # (N, C)

        # 拼接所有特征
        combined = torch.cat([go_feats, to_feats, global_pool_expanded, cand_feats], dim=1)

        # 评分
        scores = self.scorer(combined).squeeze(-1)  # (N,)

        return scores


class ValueHead(nn.Module):
    """
    价值头
    
    预测当前局面的胜率。
    
    Args:
        cfg: 模型配置
    """
    
    def __init__(self, cfg: JiuqiNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # 特征压缩
        self.conv = nn.Conv2d(cfg.backbone_channels, cfg.value_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.value_channels)
        
        # 全连接层
        fc_input_dim = cfg.value_channels * cfg.board_size * cfg.board_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, cfg.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden_dim, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征张量 (B, backbone_channels, H, W)
        
        Returns:
            价值预测 (B, 1)
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = self.fc(x)
        return x


class JiuqiNet(nn.Module):
    """
    JiuqiNet: CoordAttention-ResNet 久棋模型

    完整的策略-价值网络，包含：
    - CoordAttention 增强的 ResNet 主干
    - 候选动作评分的策略头
    - 胜率预测的价值头

    Args:
        cfg: 模型配置，默认使用 JiuqiNetConfig()

    Example:
        >>> model = JiuqiNet()
        >>> obs = torch.randn(2, 6, 14, 14)
        >>> phase_ids = torch.tensor([0, 1])
        >>> cand_feats_list = [torch.randn(50, 14), torch.randn(30, 14)]
        >>> logits_list, values = model.score_candidates(obs, phase_ids, cand_feats_list)
    """

    def __init__(self, cfg: Optional[JiuqiNetConfig] = None) -> None:
        super().__init__()

        self.cfg = cfg or JiuqiNetConfig()

        # 主干网络
        self.backbone = JiuqiNetBackbone(self.cfg)

        # 策略头
        self.policy_head = PolicyHead(self.cfg)

        # 候选评分器
        self.cand_scorer = CandidateScorer(self.cfg)

        # 价值头
        self.value_head = ValueHead(self.cfg)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        phase_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基础前向传播
        
        Args:
            obs: 棋盘状态 (B, 6, H, W)
            phase_id: 游戏阶段 (B,)
        
        Returns:
            policy_feat: 策略特征 (B, global_dim)
            value: 价值预测 (B, 1)
        """
        # 提取特征
        feat = self.backbone(obs, phase_id)
        
        # 策略特征
        policy_feat = self.policy_head(feat)
        
        # 价值预测
        value = self.value_head(feat)
        
        return policy_feat, value
    
    def score_candidates(
        self,
        obs: torch.Tensor,
        phase_id: torch.Tensor,
        cand_feats_list: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        评分候选动作

        Args:
            obs: 棋盘状态 (B, 6, H, W)
            phase_id: 游戏阶段 (B,)
            cand_feats_list: 每个样本的候选特征列表，每个元素 (N_i, cand_feat_dim)

        Returns:
            logits_list: 每个样本的候选 logits 列表
            values: 价值预测 (B, 1)
        """
        B = obs.size(0)

        # 提取特征
        feat = self.backbone(obs, phase_id)

        # 获取策略特征图（保留空间结构）
        policy_feat_map = self.policy_head(feat, return_map=True)  # (B, C, H, W)

        # 计算全局池化特征
        global_pool = policy_feat_map.mean(dim=(2, 3))  # (B, C)

        # 价值预测
        values = self.value_head(feat)

        # 逐样本评分候选
        logits_list = []
        for i in range(B):
            cand_feats = cand_feats_list[i]
            if cand_feats.numel() == 0:
                logits_list.append(torch.empty(0, device=obs.device, dtype=obs.dtype))
            else:
                logits = self.cand_scorer(policy_feat_map[i], cand_feats, global_pool[i])
                logits_list.append(logits)

        return logits_list, values
    
    def get_param_count(self) -> dict:
        """获取各部分参数量"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'backbone': count_params(self.backbone),
            'policy_head': count_params(self.policy_head),
            'cand_scorer': count_params(self.cand_scorer),
            'value_head': count_params(self.value_head),
            'total': count_params(self),
        }


def create_model(cfg: Optional[JiuqiNetConfig] = None) -> JiuqiNet:
    """
    创建 JiuqiNet 模型

    Args:
        cfg: 模型配置，None 使用默认配置

    Returns:
        JiuqiNet 模型实例
    """
    return JiuqiNet(cfg)


# 为了兼容性，保留 CZero 别名
CZero = JiuqiNet
CZeroBackbone = JiuqiNetBackbone


if __name__ == '__main__':
    # 测试模型
    print("=" * 60)
    print("Testing JiuqiNet model...")
    print("=" * 60)

    cfg = JiuqiNetConfig()
    model = JiuqiNet(cfg)

    # 打印参数量
    params = model.get_param_count()
    print(f"\nParameter counts:")
    total = 0
    for name, count in params.items():
        print(f"  {name}: {count:,}")
        total += count
    print(f"  TOTAL: {total:,} ({total/1e6:.2f}M)")

    # 测试前向传播
    B = 4
    obs = torch.randn(B, 6, 14, 14)
    phase_ids = torch.tensor([0, 0, 1, 2])

    policy_feat, values = model(obs, phase_ids)
    print(f"\nForward pass:")
    print(f"  obs: {obs.shape}")
    print(f"  policy_feat: {policy_feat.shape}")
    print(f"  values: {values.shape}")

    # 测试候选评分
    cand_feats_list = [
        torch.randn(50, 14),
        torch.randn(100, 14),
        torch.randn(30, 14),
        torch.randn(80, 14),
    ]

    logits_list, values = model.score_candidates(obs, phase_ids, cand_feats_list)
    print(f"\nCandidate scoring:")
    for i, logits in enumerate(logits_list):
        print(f"  Sample {i}: {cand_feats_list[i].shape[0]} candidates -> logits {logits.shape}")

    # 测试 GPU
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        model = model.cuda()
        obs = obs.cuda()
        phase_ids = phase_ids.cuda()
        cand_feats_list = [c.cuda() for c in cand_feats_list]

        logits_list, values = model.score_candidates(obs, phase_ids, cand_feats_list)
        print(f"  ✓ GPU test passed!")

    print("\n✓ All tests passed!")
