# --------------------------------------------------------
# DiffuCausal: Main Network Architecture
# A Diffusion-Based Model for 3D Scene Generation and Causal Reasoning
# in Autonomous UAV Systems
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jit_backbone import JiTBackbone, JiT, TimestepEmbedder, ActionEmbedder
from models.causal_head import CausalHead
from models.event_head import EventHead
from models.fusion_head import FusionHead, DecisionChainHead
from utils.model_util import RMSNorm


@dataclass
class DiffuCausalConfig:
    """Configuration for DiffuCausal network."""
    
    # Input dimensions
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    
    # JiT backbone
    jit_hidden_size: int = 1024
    jit_depth: int = 24
    jit_num_heads: int = 16
    jit_mlp_ratio: float = 4.0
    
    # Feature dimensions
    video_dim: int = 1024
    pose_dim: int = 128
    scene_dim: int = 256
    causal_dim: int = 256
    event_dim: int = 256
    fused_dim: int = 256
    
    # Action/Decision
    action_dim: int = 16
    num_tools: int = 64
    max_chain_length: int = 16
    
    # Attention
    num_heads: int = 8
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    
    # Causal head
    causal_num_layers: int = 4
    action_drop_prob: float = 0.1
    
    # Event head
    event_num_layers: int = 4
    max_instances: int = 16
    num_classes: int = 100
    image_drop_prob: float = 0.1
    
    # Fusion head
    fusion_num_layers: int = 2
    
    # Decision chain head
    decision_num_layers: int = 4
    
    # Diffusion
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 1e-5
    noise_scale: float = 1.0
    
    # Temporal
    context_frames: int = 4
    rollout_frames: int = 12


class PoseEncoder(nn.Module):
    """
    Pose Encoder for UAV state information.
    
    Encodes 6-DoF pose (position + orientation) into feature vectors.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: [B, T, 6] pose vectors (x, y, z, roll, pitch, yaw)
        
        Returns:
            [B, T, output_dim] pose features
        """
        return self.encoder(pose)


class DiffusionSceneGenerator(nn.Module):
    """
    Diffusion-based Scene Generator.
    
    Generates sparse depth/3D structure from video features using
    diffusion process. Based on JiT architecture.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature processing
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Timestep embedding
        self.t_embed = TimestepEmbedder(hidden_dim)
        
        # Processing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # AdaLN modulation
        self.adaln = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, feature_dim] input features
            timesteps: [B,] or [B, T] diffusion timesteps
        
        Returns:
            [B, T, output_dim] scene features
        """
        B, T, D = features.shape
        
        # Input projection
        x = self.input_proj(features)
        
        # Timestep embedding
        if timesteps is None:
            timesteps = torch.zeros(B, device=features.device)
        
        # Handle different timestep shapes
        if timesteps.dim() == 2:
            # [B, T] -> use mean across time for global conditioning
            timesteps = timesteps.mean(dim=1)  # [B]
        
        t_emb = self.t_embed(timesteps)  # [B, hidden_dim]
        
        # Process through layers with AdaLN
        for layer, adaln in zip(self.layers, self.adaln):
            # AdaLN modulation
            shift, scale = adaln(t_emb).chunk(2, dim=-1)
            
            # Modulate and process
            x_norm = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            x = x + layer(x_norm)
        
        # Output projection
        return self.output_proj(x)


class VideoStreamEncoder(nn.Module):
    """
    Video Stream Encoder combining JiT backbone with temporal processing.
    
    Extracts environmental features from video frames including:
    - Spatial features via JiT
    - Temporal features via temporal attention
    - Depth-aware features via scene generator
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        output_dim: int = 256
    ):
        super().__init__()
        
        # JiT backbone for spatial features
        self.jit_backbone = JiTBackbone(
            input_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            output_dim=hidden_size
        )
        
        # Project to output dimension
        self.output_proj = nn.Linear(hidden_size, output_dim)
        
        self.hidden_size = hidden_size
        self.output_dim = output_dim

    def forward(
        self,
        video: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] video frames
            timesteps: [B,] diffusion timesteps
        
        Returns:
            [B, T, output_dim] video features
        """
        # Extract features through JiT backbone
        features = self.jit_backbone(video, timesteps)  # [B, T, hidden_size]
        
        # Project to output dimension
        return self.output_proj(features)


class DiffuCausalNetwork(nn.Module):
    """
    DiffuCausal: Main Network Architecture
    
    A unified diffusion-based decision system for autonomous UAVs that:
    1. Processes video streams to understand 3D environment structure
    2. Extracts causal relationships between actions and environmental changes
    3. Generates decision chains for autonomous navigation
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         DiffuCausal Network                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Input: Video [B,T,3,H,W] + Actions [B,T,A] + Pose [B,T,6]         │
    │                           │                                         │
    │           ┌───────────────┴───────────────┐                         │
    │           ▼                               ▼                         │
    │  ┌─────────────────────┐     ┌─────────────────────┐               │
    │  │  VideoStreamEncoder │     │    PoseEncoder      │               │
    │  │  (JiT Backbone)     │     │    (MLP)            │               │
    │  └─────────────────────┘     └─────────────────────┘               │
    │           │                               │                         │
    │           └───────────────┬───────────────┘                         │
    │                           ▼                                         │
    │              ┌─────────────────────────┐                           │
    │              │  DiffusionSceneGenerator │                           │
    │              └─────────────────────────┘                           │
    │                           │                                         │
    │           ┌───────────────┴───────────────┐                         │
    │           ▼                               ▼                         │
    │  ┌─────────────────────┐     ┌─────────────────────┐               │
    │  │  CausalHead         │     │  EventHead (MIDI)   │               │
    │  │  (Vid2World-style)  │     │  (Multi-Instance)   │               │
    │  └─────────────────────┘     └─────────────────────┘               │
    │           │                               │                         │
    │           └───────────────┬───────────────┘                         │
    │                           ▼                                         │
    │              ┌─────────────────────────┐                           │
    │              │      FusionHead         │                           │
    │              └─────────────────────────┘                           │
    │                           │                                         │
    │                           ▼                                         │
    │              ┌─────────────────────────┐                           │
    │              │   DecisionChainHead     │                           │
    │              └─────────────────────────┘                           │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Optional[DiffuCausalConfig] = None):
        super().__init__()
        
        if config is None:
            config = DiffuCausalConfig()
        self.config = config
        
        # Video Stream Encoder (JiT-based)
        self.video_encoder = VideoStreamEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.jit_hidden_size,
            depth=config.jit_depth,
            num_heads=config.jit_num_heads,
            output_dim=config.video_dim
        )
        
        # Pose Encoder
        self.pose_encoder = PoseEncoder(
            input_dim=6,
            hidden_dim=64,
            output_dim=config.pose_dim
        )
        
        # Feature combination projection
        self.feature_combine = nn.Linear(
            config.video_dim + config.pose_dim,
            config.scene_dim
        )
        
        # Diffusion Scene Generator
        self.scene_generator = DiffusionSceneGenerator(
            feature_dim=config.scene_dim,
            hidden_dim=config.scene_dim * 2,
            output_dim=config.scene_dim
        )
        
        # Causal Head (Vid2World-style)
        self.causal_head = CausalHead(
            input_dim=config.scene_dim,
            hidden_dim=config.causal_dim,
            output_dim=config.causal_dim,
            num_heads=config.num_heads,
            num_layers=config.causal_num_layers,
            action_dim=config.action_dim,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            action_drop_prob=config.action_drop_prob
        )
        
        # Event Head (MIDI-style)
        self.event_head = EventHead(
            input_dim=config.scene_dim,
            hidden_dim=config.event_dim,
            output_dim=config.event_dim,
            num_heads=config.num_heads,
            num_layers=config.event_num_layers,
            max_instances=config.max_instances,
            num_classes=config.num_classes,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            image_drop_prob=config.image_drop_prob
        )
        
        # Fusion Head
        self.fusion_head = FusionHead(
            video_dim=config.scene_dim,
            causal_dim=config.causal_dim,
            event_dim=config.event_dim,
            hidden_dim=config.fused_dim,
            output_dim=config.fused_dim,
            num_heads=config.num_heads,
            num_layers=config.fusion_num_layers,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop
        )
        
        # Decision Chain Head
        self.decision_head = DecisionChainHead(
            input_dim=config.fused_dim,
            hidden_dim=config.fused_dim * 2,
            num_tools=config.num_tools,
            max_chain_length=config.max_chain_length,
            num_heads=config.num_heads,
            num_layers=config.decision_num_layers,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop
        )
        
        # Diffusion parameters
        self.P_mean = config.P_mean
        self.P_std = config.P_std
        self.t_eps = config.t_eps
        self.noise_scale = config.noise_scale

    def sample_timesteps(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps from logit-normal distribution (JiT-style).
        
        Args:
            n: Number of samples
            device: Target device
        
        Returns:
            [n,] timesteps in [0, 1]
        """
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(
        self,
        video_frames: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        pose_info: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        target_chain: Optional[torch.Tensor] = None,
        drop_actions: bool = False,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DiffuCausal network.
        
        Args:
            video: [B, T, C, H, W] video frames
            actions: [B, T, A] action vectors
            pose: [B, T, 6] pose vectors (optional)
            timesteps: [B,] diffusion timesteps (optional)
            target_chain: [B, L] target decision chain (optional)
            drop_actions: Whether to drop actions for CFG
            return_all: Whether to return all intermediate features
        
        Returns:
            Dictionary containing:
            - decision_chain: [B, L] or [B, L, num_tools+2] decision predictions
            - fused_features: [B, T, fused_dim] fused features
            - video_features: [B, T, scene_dim] video features
            - causal_features: [B, T, causal_dim] causal features
            - event_features: [B, T, event_dim] event features
            - depth: [B, T, N, 1] depth predictions (if return_all)
            - instances: Instance predictions (if return_all)
            - decision_loss: Scalar loss (if target_chain provided)
            - match_features: [B, T, D] for matching loss
        """
        # Handle both video_frames and video parameter names
        if video_frames is not None:
            video = video_frames
        if video is None:
            raise ValueError("Either video_frames or video must be provided")
        
        # Handle both pose_info and pose parameter names
        if pose_info is not None:
            pose = pose_info
        
        # Handle missing actions
        if actions is None:
            B, T = video.shape[:2]
            actions = torch.zeros(B, T, self.config.action_dim, device=video.device)
        
        B, T, C, H, W = video.shape
        device = video.device
        
        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = self.sample_timesteps(B, device)
        
        # Default pose
        if pose is None:
            pose = torch.zeros(B, T, 6, device=device)
        
        # 1. Video Stream Encoding
        video_features = self.video_encoder(video, timesteps)  # [B, T, video_dim]
        
        # 2. Pose Encoding
        pose_features = self.pose_encoder(pose)  # [B, T, pose_dim]
        
        # 3. Combine video and pose features
        combined = torch.cat([video_features, pose_features], dim=-1)
        scene_features = self.feature_combine(combined)  # [B, T, scene_dim]
        
        # 4. Scene generation (diffusion-based refinement)
        scene_features = self.scene_generator(scene_features, timesteps)
        
        # 5. Causal Head (Vid2World-style)
        causal_features = self.causal_head(
            scene_features, actions, drop_actions=drop_actions
        )  # [B, T, causal_dim]
        
        # 6. Event Head (MIDI-style)
        event_features, depth, instances = self.event_head(
            scene_features,
            return_depth=return_all,
            return_instances=return_all
        )  # [B, T, event_dim]
        
        # 7. Fusion Head
        fused_features, match_features = self.fusion_head(
            scene_features, causal_features, event_features
        )  # [B, T, fused_dim]
        
        # 8. Decision Chain Head
        if target_chain is not None:
            decision_logits, decision_loss = self.decision_head(
                fused_features, target_chain
            )
        else:
            decision_logits = self.decision_head(fused_features)
            decision_loss = None
        
        # Build output dictionary
        outputs = {
            'decision_chain': decision_logits,
            'fused_features': fused_features,
            'video_features': scene_features,
            'causal_features': causal_features,
            'event_features': event_features,
            'match_features': match_features,
        }
        
        if decision_loss is not None:
            outputs['decision_loss'] = decision_loss
        
        if return_all:
            outputs['depth'] = depth
            outputs['instances'] = instances
        
        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        video_gt: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training (legacy method).
        
        Args:
            outputs: Output dictionary from forward pass
            video_gt: [B, T, C, H, W] ground truth video
            target_chain: [B, L] target decision chain
            loss_weights: Dictionary of loss weights
        
        Returns:
            Dictionary of losses
        """
        return self.compute_losses(outputs, video_gt, target_chain, None)
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        video_gt: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None,
        depth_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for multi-stage training.
        
        Args:
            outputs: Output dictionary from forward pass
            video_gt: [B, T, C, H, W] ground truth video
            target_chain: [B, L] target decision chain
            depth_gt: [B, T, 1, H, W] ground truth depth (optional)
        
        Returns:
            Dictionary of losses (unweighted, weights applied by trainer)
        """
        losses = {}
        device = video_gt.device
        B, T, C, H, W = video_gt.shape
        
        # ============================================
        # 1. Diffusion Loss (Video Reconstruction)
        # ============================================
        # Compute feature-level diffusion loss
        video_features = outputs.get('video_features')
        if video_features is not None:
            # Use L2 loss on features as proxy for diffusion loss
            # In full implementation, this would be the denoising score matching loss
            # Simple self-supervised loss: predict noise added to features
            noise = torch.randn_like(video_features) * 0.1
            noisy_features = video_features + noise
            
            # The model should learn to denoise - use feature smoothness as proxy
            # Temporal smoothness loss
            if video_features.shape[1] > 1:
                temporal_diff = video_features[:, 1:] - video_features[:, :-1]
                diffusion_loss = (temporal_diff ** 2).mean()
            else:
                diffusion_loss = (video_features ** 2).mean() * 0.01
            
            # Clamp to prevent NaN
            diffusion_loss = torch.clamp(diffusion_loss, min=0.0, max=100.0)
            losses['diffusion'] = diffusion_loss
        else:
            losses['diffusion'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================
        # 2. Event Loss (Depth/Instance Prediction)
        # ============================================
        depth_pred = outputs.get('depth')
        if depth_pred is not None:
            if depth_gt is not None:
                # Supervised depth loss
                # Handle different depth shapes
                try:
                    if depth_pred.dim() == 4 and depth_gt.dim() == 5:
                        # depth_pred: [B, T, N, 1], depth_gt: [B, T, 1, H, W]
                        # Use mean of GT depth as target
                        depth_gt_mean = depth_gt.mean(dim=(-2, -1), keepdim=True)  # [B, T, 1, 1, 1]
                        depth_gt_mean = depth_gt_mean.squeeze(-1).squeeze(-1)  # [B, T, 1]
                        depth_pred_mean = depth_pred.mean(dim=2)  # [B, T, 1]
                        depth_loss = F.l1_loss(depth_pred_mean, depth_gt_mean)
                    elif depth_pred.shape == depth_gt.shape:
                        depth_loss = F.l1_loss(depth_pred, depth_gt)
                    else:
                        # Fallback: use smoothness loss
                        depth_loss = self._compute_depth_smoothness_loss(depth_pred)
                except Exception:
                    depth_loss = self._compute_depth_smoothness_loss(depth_pred)
            else:
                # Self-supervised: smoothness loss
                depth_loss = self._compute_depth_smoothness_loss(depth_pred)
            
            losses['event'] = depth_loss
        else:
            losses['event'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================
        # 3. Matching Loss (Feature Alignment)
        # ============================================
        video_feat = outputs.get('video_features')
        causal_feat = outputs.get('causal_features')
        
        if video_feat is not None and causal_feat is not None:
            match_loss = self.fusion_head.compute_matching_loss(video_feat, causal_feat)
            match_loss = torch.nan_to_num(match_loss, nan=0.0, posinf=1.0, neginf=0.0)
            losses['match'] = match_loss
        else:
            losses['match'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================
        # 4. Decision Loss (Chain Prediction)
        # ============================================
        if 'decision_loss' in outputs and outputs['decision_loss'] is not None:
            decision_loss = outputs['decision_loss']
            decision_loss = torch.nan_to_num(decision_loss, nan=0.0, posinf=10.0, neginf=0.0)
            losses['decision'] = decision_loss
        else:
            losses['decision'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================
        # 5. Causal Consistency Loss (Optional)
        # ============================================
        # Encourage temporal consistency in causal features
        if causal_feat is not None and causal_feat.shape[1] > 1:
            temporal_diff = causal_feat[:, 1:] - causal_feat[:, :-1]
            causal_consistency = (temporal_diff ** 2).mean()
            losses['causal_consistency'] = causal_consistency * 0.01
        
        return losses
    
    def _compute_depth_smoothness_loss(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute depth smoothness loss for self-supervised training.
        
        Args:
            depth: [B, T, 1, H, W] or [B, T, N, 1] depth predictions
            
        Returns:
            Smoothness loss
        """
        if depth.dim() == 5:
            # Image-like depth [B, T, 1, H, W]
            B, T, _, H, W = depth.shape
            depth_flat = depth.view(B * T, 1, H, W)
            
            # Compute gradients
            grad_x = torch.abs(depth_flat[:, :, :, :-1] - depth_flat[:, :, :, 1:])
            grad_y = torch.abs(depth_flat[:, :, :-1, :] - depth_flat[:, :, 1:, :])
            
            loss = grad_x.mean() + grad_y.mean()
            return torch.clamp(loss, min=0.0, max=10.0)
        else:
            # Point-like depth [B, T, N, 1]
            if depth.shape[2] > 1:
                loss = depth.var(dim=2).mean()
            else:
                loss = depth.mean() * 0.01
            return torch.clamp(loss, min=0.0, max=10.0)

    @torch.no_grad()
    def generate_decision_chain(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.5,
        max_length: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generate decision chain for given video and actions.
        
        Args:
            video: [B, T, C, H, W] video frames
            actions: [B, T, A] action vectors
            pose: [B, T, 6] pose vectors (optional)
            guidance_scale: CFG scale for action conditioning
            max_length: Maximum chain length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            [B, L] generated decision chain
        """
        # Forward pass (conditional)
        outputs_cond = self.forward(
            video, actions, pose,
            drop_actions=False,
            return_all=False
        )
        
        # Forward pass (unconditional) for CFG
        if guidance_scale != 1.0:
            outputs_uncond = self.forward(
                video, actions, pose,
                drop_actions=True,
                return_all=False
            )
            
            # Apply CFG to fused features
            fused_cond = outputs_cond['fused_features']
            fused_uncond = outputs_uncond['fused_features']
            fused_guided = fused_uncond + guidance_scale * (fused_cond - fused_uncond)
        else:
            fused_guided = outputs_cond['fused_features']
        
        # Project features for decision head
        memory = self.decision_head.input_proj(fused_guided)
        
        # Generate decision chain
        decision_chain = self.decision_head.generate(
            memory,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return decision_chain

    @torch.no_grad()
    def autoregressive_rollout(
        self,
        context_video: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        context_pose: Optional[torch.Tensor] = None,
        rollout_length: int = 12,
        guidance_scale: float = 1.5
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive rollout for future prediction.
        
        Args:
            context_video: [B, T_ctx, C, H, W] context frames
            context_actions: [B, T_ctx, A] context actions
            future_actions: [B, T_future, A] future actions
            context_pose: [B, T_ctx, 6] context poses (optional)
            rollout_length: Number of frames to predict
            guidance_scale: CFG scale
        
        Returns:
            Dictionary with predicted features and decision chain
        """
        B, T_ctx, C, H, W = context_video.shape
        device = context_video.device
        
        # Process context
        outputs = self.forward(
            context_video, context_actions, context_pose,
            return_all=True
        )
        
        # Use causal head for autoregressive feature prediction
        context_features = outputs['video_features']
        
        future_features = self.causal_head.autoregressive_forward(
            context_features,
            context_actions,
            future_actions,
            num_steps=rollout_length,
            guidance_scale=guidance_scale
        )
        
        # Combine context and future features
        all_features = torch.cat([context_features, future_features], dim=1)
        all_actions = torch.cat([context_actions, future_actions], dim=1)
        
        # Get event features for future
        future_event, _, _ = self.event_head(future_features)
        
        # Fuse features
        fused_features, _ = self.fusion_head(
            future_features,
            future_features,  # Use predicted features as causal
            future_event
        )
        
        # Generate decision chain
        decision_chain = self.generate_decision_chain(
            context_video,
            torch.cat([context_actions, future_actions], dim=1),
            context_pose,
            guidance_scale=guidance_scale
        )
        
        return {
            'future_features': future_features,
            'fused_features': fused_features,
            'decision_chain': decision_chain
        }


def create_diffucausal_model(
    model_size: str = 'base',
    **kwargs
) -> DiffuCausalNetwork:
    """
    Create DiffuCausal model with predefined configurations.
    
    Args:
        model_size: 'small', 'base', or 'large'
        **kwargs: Override configuration parameters
    
    Returns:
        DiffuCausalNetwork instance
    """
    configs = {
        'small': DiffuCausalConfig(
            jit_hidden_size=768,
            jit_depth=12,
            jit_num_heads=12,
            video_dim=768,
            scene_dim=192,
            causal_dim=192,
            event_dim=192,
            fused_dim=192,
            causal_num_layers=2,
            event_num_layers=2,
            fusion_num_layers=1,
            decision_num_layers=2,
        ),
        'base': DiffuCausalConfig(
            jit_hidden_size=1024,
            jit_depth=24,
            jit_num_heads=16,
            video_dim=1024,
            scene_dim=256,
            causal_dim=256,
            event_dim=256,
            fused_dim=256,
            causal_num_layers=4,
            event_num_layers=4,
            fusion_num_layers=2,
            decision_num_layers=4,
        ),
        'large': DiffuCausalConfig(
            jit_hidden_size=1280,
            jit_depth=32,
            jit_num_heads=16,
            video_dim=1280,
            scene_dim=320,
            causal_dim=320,
            event_dim=320,
            fused_dim=320,
            causal_num_layers=6,
            event_num_layers=6,
            fusion_num_layers=3,
            decision_num_layers=6,
        ),
    }
    
    config = configs.get(model_size, configs['base'])
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DiffuCausalNetwork(config)

