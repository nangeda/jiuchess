#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoordAttention 模块

坐标注意力机制，通过分离的水平和垂直池化显式编码位置信息。
特别适合棋类游戏中的位置敏感特征提取。

Reference: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CoordAttention(nn.Module):
    """
    坐标注意力模块
    
    通过在H和W方向分别进行全局池化，保留精确的位置信息。
    相比CBAM的空间注意力，CoordAttention能够捕获长程依赖的同时保留位置精度。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（通常与in_channels相同）
        reduction: 通道压缩比例
    
    Example:
        >>> attn = CoordAttention(128, 128, reduction=32)
        >>> x = torch.randn(2, 128, 14, 14)
        >>> out = attn(x)  # (2, 128, 14, 14)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        reduction: int = 32
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 中间通道数，确保至少为8
        mid_channels = max(8, in_channels // reduction)
        
        # 共享的1x1卷积，用于降维
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        # 分离的H和W方向的1x1卷积，用于生成注意力权重
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            注意力加权后的张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # H方向全局平均池化: (B, C, H, W) -> (B, C, H, 1)
        x_h = x.mean(dim=3, keepdim=True)
        
        # W方向全局平均池化: (B, C, H, W) -> (B, C, 1, W) -> (B, C, W, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        
        # 拼接: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        
        # 降维 + BN + ReLU
        y = self.act(self.bn(self.conv_reduce(y)))
        
        # 分离H和W
        y_h, y_w = torch.split(y, [H, W], dim=2)
        
        # 恢复W的形状: (B, mid, W, 1) -> (B, mid, 1, W)
        y_w = y_w.permute(0, 1, 3, 2)
        
        # 生成注意力权重
        attn_h = torch.sigmoid(self.conv_h(y_h))  # (B, C, H, 1)
        attn_w = torch.sigmoid(self.conv_w(y_w))  # (B, C, 1, W)
        
        # 应用注意力（广播乘法）
        out = x * attn_h * attn_w
        
        return out


class CoordAttentionBlock(nn.Module):
    """
    带有CoordAttention的残差块
    
    结构: Conv -> BN -> ReLU -> Conv -> BN -> CoordAttn -> ReLU
    
    Args:
        channels: 通道数
        kernel_size: 卷积核大小
        reduction: CoordAttention的通道压缩比
    """
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3,
        reduction: int = 32
    ) -> None:
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.coord_attn = CoordAttention(channels, channels, reduction=reduction)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in [self.bn1, self.bn2]:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            残差连接后的张量 (B, C, H, W)
        """
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.coord_attn(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


if __name__ == '__main__':
    # 测试代码
    print("Testing CoordAttention...")
    
    # 测试 CoordAttention
    attn = CoordAttention(128, 128, reduction=32)
    x = torch.randn(2, 128, 14, 14)
    out = attn(x)
    print(f"CoordAttention: {x.shape} -> {out.shape}")
    assert out.shape == x.shape
    
    # 测试 CoordAttentionBlock
    block = CoordAttentionBlock(128, kernel_size=3, reduction=32)
    out = block(x)
    print(f"CoordAttentionBlock: {x.shape} -> {out.shape}")
    assert out.shape == x.shape
    
    # 参数量统计
    params = sum(p.numel() for p in attn.parameters())
    print(f"CoordAttention params: {params:,}")
    
    params = sum(p.numel() for p in block.parameters())
    print(f"CoordAttentionBlock params: {params:,}")
    
    print("All tests passed!")
