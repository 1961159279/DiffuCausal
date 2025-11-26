# --------------------------------------------------------
# DiffuCausal Event Head - 基于MIDI的多实例扩散实现
# 完整实现多实例注意力、3D场景生成和事件推理
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from einops import rearrange, repeat

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_util import RMSNorm


class MIDIMultiInstanceAttention(nn.Module):
    """
    MIDI风格的多实例注意力处理器
    基于MIAttnProcessor2_0实现，支持动态实例数量处理
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_instances: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_instances = max_instances
        
        # 标准注意力投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        # 实例特定的投影
        self.instance_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.instance_k = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 实例嵌入
        self.instance_embed = nn.Parameter(torch.zeros(1, max_instances, dim))
        nn.init.normal_(self.instance_embed, std=0.02)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # QK归一化
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
        num_instances: Optional[int] = None,
        num_instances_per_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        MIDI风格的多实例注意力前向传播
        
        Args:
            x: [B, N, D] 输入特征
            instance_mask: [B, N, num_instances] 软实例分配
            num_instances: 实例数量
            num_instances_per_batch: 每批实例数量（用于训练）
        
        Returns:
            [B, N, D] 实例感知特征
        """
        B, N, D = x.shape
        
        if num_instances is None:
            num_instances = self.max_instances
        
        # 标准自注意力
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # MIDI风格的多实例处理
        if num_instances_per_batch is not None:
            # 训练模式：处理多实例注意力
            patch_hidden_states = []
            start_idx = 0
            
            while start_idx < B:
                for num in [num_instances]:
                    # 多对象自注意力
                    query_ = q[start_idx:start_idx + num]
                    key_ = rearrange(
                        k[start_idx:start_idx + num],
                        "(b ni) h nt c -> b h (ni nt) c", 
                        ni=num
                    ).repeat_interleave(num, dim=0)
                    value_ = rearrange(
                        v[start_idx:start_idx + num],
                        "(b ni) h nt c -> b h (ni nt) c", 
                        ni=num
                    ).repeat_interleave(num, dim=0)
                    
                    patch_hidden_states.append(
                        F.scaled_dot_product_attention(
                            query_, key_, value_, 
                            dropout_p=0.0, is_causal=False
                        )
                    )
                    
                    # 单对象自注意力（用于填充和正则化）
                    query_ = q[start_idx + num:start_idx + num_instances_per_batch]
                    key_ = k[start_idx + num:start_idx + num_instances_per_batch]
                    value_ = v[start_idx + num:start_idx + num_instances_per_batch]
                    
                    if query_.shape[0] > 0:
                        patch_hidden_states.append(
                            F.scaled_dot_product_attention(
                                query_, key_, value_,
                                dropout_p=0.0, is_causal=False
                            )
                        )
                    
                    start_idx += num_instances_per_batch
            
            out = torch.cat(patch_hidden_states, dim=0)
            out = out.transpose(1, 2).reshape(B, N, D)
        
        # 实例感知调制
        if instance_mask is not None:
            # 获取实例嵌入
            inst_embed = self.instance_embed[:, :num_instances]  # [1, K, D]
            
            # 计算实例加权特征
            inst_features = torch.einsum('bnk,lkd->bnd', instance_mask, inst_embed)
            
            # 实例交叉注意力
            inst_q = self.instance_q(out).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            inst_k = self.instance_k(inst_features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            inst_attn = (inst_q @ inst_k.transpose(-2, -1)) * self.scale
            inst_attn = torch.sigmoid(inst_attn.mean(dim=1, keepdim=True))
            
            # 调制输出
            out = out * (1 + inst_attn.squeeze(1).mean(dim=-1, keepdim=True))
        
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class MIDIDepthDecoder(nn.Module):
    """
    MIDI风格的深度解码器
    支持多尺度特征处理和不确定性估计
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 多尺度特征提取
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1),
            nn.ReLU(inplace=True)
        )
        
        # 深度回归头
        self.depth_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, 1)
        )
        
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        多尺度深度预测
        
        Args:
            x: [B, D, H, W] 输入特征
            return_uncertainty: 是否返回不确定性估计
        
        Returns:
            depth: [B, 1, H, W] 深度预测
            uncertainty: [B, 1, H, W] 不确定性（可选）
        """
        B, D, H, W = x.shape
        
        # 多尺度特征提取
        scale_features = []
        for i, encoder in enumerate(self.scale_encoders):
            scale_factor = 2 ** i
            if scale_factor > 1:
                # 下采样
                x_down = F.interpolate(x, scale_factor=1/scale_factor, mode='bilinear')
                feat = encoder(x_down)
                # 上采样回原尺寸
                feat = F.interpolate(feat, size=(H, W), mode='bilinear')
            else:
                feat = encoder(x)
            scale_features.append(feat)
        
        # 特征融合
        fused_features = torch.cat(scale_features, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 深度预测
        depth = torch.sigmoid(self.depth_head(fused_features))
        
        if return_uncertainty:
            uncertainty = F.softplus(self.uncertainty_head(fused_features))
            return depth, uncertainty
        
        return depth, None


class MIDIInstanceSegmentationHead(nn.Module):
    """
    MIDI风格的实例分割头
    支持动态实例数量和类别预测
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        max_instances: int = 16,
        num_classes: int = 100
    ):
        super().__init__()
        
        self.max_instances = max_instances
        self.num_classes = num_classes
        
        # 实例掩码预测
        self.mask_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, max_instances, 1)
        )
        
        # 类别预测（每个实例）
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes * max_instances)
        )
        
        # 实例存在性预测
        self.existence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, max_instances)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        实例分割预测
        
        Args:
            x: [B, D, H, W] 输入特征
        
        Returns:
            masks: [B, max_instances, H, W] 实例掩码
            classes: [B, max_instances, num_classes] 类别概率
            existence: [B, max_instances] 实例存在性分数
        """
        B, D, H, W = x.shape
        
        # 预测实例掩码
        masks = self.mask_head(x)  # [B, max_instances, H, W]
        masks = F.softmax(masks, dim=1)
        
        # 预测类别
        class_logits = self.class_head(x)  # [B, max_instances * num_classes]
        classes = class_logits.view(B, self.max_instances, self.num_classes)
        
        # 预测实例存在性
        existence = torch.sigmoid(self.existence_head(x))  # [B, max_instances]
        
        return masks, classes, existence


class MIDIEventBlock(nn.Module):
    """
    MIDI风格的事件处理块
    结合多实例注意力和前馈处理
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_instances: int = 16,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.attn = MIDIMultiInstanceAttention(
            dim=dim,
            num_heads=num_heads,
            max_instances=max_instances,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(proj_drop)
        )

    def forward(
        self,
        x: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
        num_instances: Optional[int] = None,
        num_instances_per_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        事件块前向传播
        
        Args:
            x: [B, N, D] 输入特征
            instance_mask: [B, N, K] 软实例分配
            num_instances: 实例数量
            num_instances_per_batch: 每批实例数量
        
        Returns:
            [B, N, D] 处理后的特征
        """
        # 残差连接的多实例注意力
        x = x + self.attn(
            self.norm1(x), 
            instance_mask, 
            num_instances, 
            num_instances_per_batch
        )
        
        # 残差连接的前馈网络
        x = x + self.mlp(self.norm2(x))
        
        return x


class MIDI3DSceneGenerator(nn.Module):
    """
    MIDI风格的3D场景生成器
    基于深度预测生成3D点云和网格
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 点云解码器
        self.point_cloud_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 3)  # 输出3D坐标 (x, y, z)
        )
        
        # 场景组合器
        self.scene_composer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, depth_map, features, intrinsic_params=None):
        """
        从深度图和特征生成3D场景
        
        Args:
            depth_map: [B, 1, H, W] 深度图
            features: [B, D, H, W] 特征图
            intrinsic_params: 相机内参（可选）
        
        Returns:
            points_3d: [B, H, W, 3] 3D点云
        """
        B, _, H, W = depth_map.shape
        
        # 生成网格坐标
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=depth_map.device),
            torch.linspace(-1, 1, W, device=depth_map.device),
            indexing='ij'
        )
        
        # 转换为3D点云
        points_3d = torch.stack([
            x_coords * depth_map.squeeze(1),
            y_coords * depth_map.squeeze(1), 
            depth_map.squeeze(1)
        ], dim=-1)  # [B, H, W, 3]
        
        # 使用特征增强点云
        point_features = self.point_cloud_decoder(
            features.permute(0, 2, 3, 1)  # [B, H, W, D]
        )
        enhanced_points = points_3d + point_features
        
        return enhanced_points


class EventHead(nn.Module):
    """
    完整的事件头实现 - 基于MIDI的多实例扩散
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        max_instances: int = 16,
        num_classes: int = 100,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        image_drop_prob: float = 0.1,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_instances = max_instances
        self.image_drop_prob = image_drop_prob
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # MIDI风格的事件处理块
        self.blocks = nn.ModuleList([
            MIDIEventBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                max_instances=max_instances,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for _ in range(num_layers)
        ])
        
        # MIDI实例分割头
        self.instance_head = MIDIInstanceSegmentationHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            max_instances=max_instances,
            num_classes=num_classes
        )
        
        # MIDI深度解码器
        self.depth_decoder = MIDIDepthDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=1,
            num_scales=num_scales
        )
        
        # MIDI 3D场景生成器
        self.scene_3d_generator = MIDI3DSceneGenerator(hidden_dim=hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 位置嵌入
        self.max_seq_len = 64
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))
        nn.init.normal_(self.temporal_pos, std=0.02)
        
        self.max_spatial_len = 256
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.max_spatial_len, hidden_dim))
        nn.init.normal_(self.spatial_pos, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        num_instances: Optional[int] = None,
        num_instances_per_batch: Optional[int] = None,
        return_depth: bool = True,
        return_instances: bool = True,
        return_3d_scene: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整的事件头前向传播
        
        Args:
            x: [B, T, D] 或 [B, T, N, D] 输入特征
            num_instances: 实例数量
            num_instances_per_batch: 每批实例数量
            return_depth: 是否返回深度预测
            return_instances: 是否返回实例预测
            return_3d_scene: 是否返回3D场景
        
        Returns:
            包含各种输出的字典
        """
        # 处理不同输入形状
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.unsqueeze(2)  # [B, T, 1, D]
            N = 1
        else:
            B, T, N, D = x.shape
        
        # 重塑用于处理
        x_reshaped = x.view(B * T, N, D)
        
        # 输入投影
        x_proj = self.input_proj(x_reshaped)
        
        # 添加空间位置嵌入
        x_proj = x_proj + self.spatial_pos[:, :N, :]
        
        # 训练时的图像dropout
        if self.training and self.image_drop_prob > 0:
            drop_mask = torch.rand(B * T, 1, 1, device=x_proj.device) < self.image_drop_prob
            x_proj = x_proj * (~drop_mask).float()
        
        # 转换为图像格式用于实例分割和深度预测
        patch_size = int(math.sqrt(N)) if math.sqrt(N).is_integer() else 1
        H = W = patch_size
        
        if patch_size > 1:
            x_img = x_proj.view(B * T, patch_size, patch_size, -1).permute(0, 3, 1, 2)
        else:
            x_img = x_proj.unsqueeze(-1).unsqueeze(-1)  # [B*T, D, 1, 1]
        
        # 获取初始实例预测
        masks, classes, existence = self.instance_head(x_img)
        
        # 重塑回序列格式用于事件块处理
        if patch_size > 1:
            instance_mask = masks.view(B * T, -1, self.max_instances)
        else:
            instance_mask = masks.view(B * T, 1, self.max_instances)
        
        # 通过事件块处理
        for block in self.blocks:
            x_proj = block(
                x_proj, 
                instance_mask=instance_mask,
                num_instances=num_instances,
                num_instances_per_batch=num_instances_per_batch
            )
        
        # 准备输出字典
        outputs = {}
        
        # 深度预测
        if return_depth:
            depth, uncertainty = self.depth_decoder(
                x_img if patch_size > 1 else x_proj.unsqueeze(-1).unsqueeze(-1),
                return_uncertainty=True
            )
            outputs['depth'] = depth.view(B, T, 1, H, W)
            outputs['depth_uncertainty'] = uncertainty.view(B, T, 1, H, W) if uncertainty is not None else None
        
        # 实例预测
        if return_instances:
            masks, classes, existence = self.instance_head(
                x_img if patch_size > 1 else x_proj.unsqueeze(-1).unsqueeze(-1)
            )
            outputs['instance_masks'] = masks.view(B, T, self.max_instances, H, W)
            outputs['instance_classes'] = classes.view(B, T, self.max_instances, -1)
            outputs['instance_existence'] = existence.view(B, T, self.max_instances)
        
        # 3D场景生成
        if return_3d_scene and return_depth:
            depth_map = outputs['depth'].view(B * T, 1, H, W)
            features_3d = x_proj.view(B * T, -1, H, W) if patch_size > 1 else x_proj.unsqueeze(-1).unsqueeze(-1)
            
            scene_3d = self.scene_3d_generator(depth_map, features_3d)
            outputs['scene_3d'] = scene_3d.view(B, T, H, W, 3)
        
        # 池化空间维度并投影输出
        x_pooled = x_proj.mean(dim=1)  # [B*T, D]
        x_pooled = x_pooled.view(B, T, -1)
        
        # 添加时间位置嵌入
        x_pooled = x_pooled + self.temporal_pos[:, :T, :]
        
        # 输出投影
        event_features = self.output_proj(x_pooled)
        outputs['event_features'] = event_features
        
        return outputs

    def compute_losses(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 深度损失
        if 'depth' in predictions and 'depth' in targets:
            depth_pred = predictions['depth']
            depth_gt = targets['depth']
            losses['depth_mse'] = F.mse_loss(depth_pred, depth_gt)
            losses['depth_smooth'] = self._compute_depth_smoothness(depth_pred)
        
        # 实例分割损失
        if 'instance_masks' in predictions and 'instance_masks' in targets:
            mask_pred = predictions['instance_masks']
            mask_gt = targets['instance_masks']
            losses['mask_bce'] = F.binary_cross_entropy(mask_pred, mask_gt)
        
        if 'instance_classes' in predictions and 'instance_classes' in targets:
            class_pred = predictions['instance_classes']
            class_gt = targets['instance_classes']
            losses['class_ce'] = F.cross_entropy(
                class_pred.view(-1, class_pred.size(-1)), 
                class_gt.view(-1)
            )
        
        # 特征一致性损失
        if 'event_features' in predictions:
            features = predictions['event_features']
            if features.shape[1] > 1:  # 多帧情况
                losses['temporal_consistency'] = self._compute_temporal_consistency(features)
        
        return losses

    def _compute_depth_smoothness(self, depth: torch.Tensor) -> torch.Tensor:
        """计算深度平滑性损失"""
        depth_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        depth_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        return torch.mean(depth_dx**2) + torch.mean(depth_dy**2)

    def _compute_temporal_consistency(self, features: torch.Tensor) -> torch.Tensor:
        """计算时间一致性损失"""
        diff = features[:, 1:] - features[:, :-1]
        return torch.mean(torch.norm(diff, dim=-1))

    def get_instance_features(self, x: torch.Tensor, instance_idx: int) -> torch.Tensor:
        """
        获取特定实例的特征
        
        Args:
            x: 输入特征
            instance_idx: 实例索引
        
        Returns:
            实例特定特征
        """
        outputs = self.forward(x, return_instances=True)
        
        if 'instance_masks' not in outputs:
            return outputs['event_features']
        
        masks = outputs['instance_masks']
        
        # 检查实例是否存在
        existence = outputs['instance_existence']
        if instance_idx >= self.max_instances or existence[:, :, instance_idx].mean() < 0.1:
            return outputs['event_features']
        
        # 使用实例掩码加权特征
        instance_mask = masks[:, :, instance_idx]  # [B, T, H, W]
        
        # 这里需要根据具体特征形状进行调整
        # 简化版本：返回事件特征
        return outputs['event_features']
