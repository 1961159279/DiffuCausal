# --------------------------------------------------------
# DiffuCausal Causal Head
# Vid2World-style causal attention for action-conditioned prediction
# References:
# Vid2World: Crafting Video Diffusion Models to Interactive World Models
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_util import RMSNorm


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Target device
    
    Returns:
        [seq_len, seq_len] boolean mask where True indicates positions to mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


class CausalTemporalAttention(nn.Module):
    """
    Causal Temporal Attention from Vid2World.
    
    Implements temporal attention with causal masking where each frame
    can only attend to past frames (including itself).
    
    Key formula:
        Attn(Q, K, V) = softmax(QK^T / sqrt(d) + M_causal) V
        where M_causal[i,j] = -inf if j > i
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # QK normalization for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input tensor
            causal_mask: Optional pre-computed causal mask [T, T]
        
        Returns:
            [B, T, D] output tensor
        """
        B, T, D = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Normalize Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if causal_mask is None:
            causal_mask = create_causal_mask(T, device=x.device)
        
        # Expand mask for batch and heads
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class ActionConditioningModule(nn.Module):
    """
    Action Conditioning Module from Vid2World.
    
    Injects action information into temporal features using:
    1. Action embedding via MLP
    2. AdaLN-style modulation
    3. Optional action dropout for CFG training
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        action_drop_prob: float = 0.1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_drop_prob = action_drop_prob
        
        # Action embedding MLP
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # AdaLN modulation parameters
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        
        # Null action embedding for CFG
        self.null_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.null_action, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        drop_actions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] input features
            actions: [B, T, A] action vectors
            drop_actions: Whether to drop actions (for CFG unconditional)
        
        Returns:
            modulated_x: [B, T, D] action-conditioned features
            action_embed: [B, T, D] action embeddings
        """
        B, T, D = x.shape
        
        # Embed actions
        action_embed = self.action_embed(actions)  # [B, T, D]
        
        # Action dropout during training
        if self.training and self.action_drop_prob > 0:
            drop_mask = torch.rand(B, 1, 1, device=x.device) < self.action_drop_prob
            action_embed = torch.where(
                drop_mask,
                self.null_action.expand(B, T, -1),
                action_embed
            )
        
        # Force dropout for unconditional generation
        if drop_actions:
            action_embed = self.null_action.expand(B, T, -1)
        
        # Compute modulation parameters
        mod_params = self.modulation(action_embed)
        shift1, scale1, shift2, scale2 = mod_params.chunk(4, dim=-1)
        
        # Apply modulation (simplified AdaLN)
        x = x * (1 + scale1) + shift1
        
        return x, action_embed


class CausalBlock(nn.Module):
    """
    Causal Transformer Block for Vid2World-style processing.
    
    Each block contains:
    - Causal temporal attention
    - Action conditioning via AdaLN
    - Feed-forward network
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        action_dim: int = 16,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        action_drop_prob: float = 0.1
    ):
        super().__init__()
        
        # Causal attention
        self.norm1 = RMSNorm(dim)
        self.attn = CausalTemporalAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Action conditioning
        self.action_cond = ActionConditioningModule(
            action_dim=action_dim,
            hidden_dim=dim,
            action_drop_prob=action_drop_prob
        )
        
        # FFN
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
        actions: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        drop_actions: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input features
            actions: [B, T, A] action vectors
            causal_mask: Optional causal attention mask
            drop_actions: Whether to drop actions for CFG
        
        Returns:
            [B, T, D] output features
        """
        # Causal attention with residual
        x = x + self.attn(self.norm1(x), causal_mask)
        
        # Action conditioning
        x, _ = self.action_cond(x, actions, drop_actions)
        
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class GRUCell(nn.Module):
    """
    GRU Cell for recurrent state tracking in causal prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, D] input
            h: [B, D] hidden state (optional)
        
        Returns:
            [B, D] new hidden state
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h], dim=-1)
        
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_candidate = torch.tanh(self.candidate(combined_reset))
        
        h_new = (1 - z) * h + z * h_candidate
        return h_new


class CausalHead(nn.Module):
    """
    Causal Head for DiffuCausal.
    
    Implements Vid2World-style causal reasoning:
    - Causal temporal attention (each frame attends only to past)
    - Action conditioning with dropout for CFG
    - Diffusion forcing (independent noise per frame)
    - GRU for recurrent state tracking
    
    Key features:
    1. Causal Mask: Strictly enforce mask[i,j] = 0 if j > i
    2. Action Dropout: 10% probability to drop action conditioning
    3. Diffusion Forcing: Sample noise level independently for each frame
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        action_dim: int = 16,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        action_drop_prob: float = 0.1,
        use_gru: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.use_gru = use_gru
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            CausalBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                action_dim=action_dim,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                action_drop_prob=action_drop_prob
            )
            for _ in range(num_layers)
        ])
        
        # GRU for recurrent state
        if use_gru:
            self.gru = GRUCell(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Temporal position embedding
        self.max_seq_len = 64
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))
        nn.init.normal_(self.temporal_pos, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        drop_actions: bool = False,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.
        
        Args:
            x: [B, T, D] input features from video encoder
            actions: [B, T, A] action vectors
            drop_actions: Whether to drop actions for CFG unconditional
            return_hidden: Whether to return hidden states
        
        Returns:
            [B, T, output_dim] causal features
        """
        B, T, D = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add temporal position embedding
        x = x + self.temporal_pos[:, :T, :]
        
        # Create causal mask
        causal_mask = create_causal_mask(T, device=x.device)
        
        # Process through causal blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, actions, causal_mask, drop_actions)
            if return_hidden:
                hidden_states.append(x)
        
        # Apply GRU for recurrent state refinement
        if self.use_gru:
            gru_outputs = []
            h = None
            for t in range(T):
                h = self.gru(x[:, t], h)
                gru_outputs.append(h)
            x = torch.stack(gru_outputs, dim=1)
        
        # Output projection
        output = self.output_proj(x)
        
        if return_hidden:
            return output, hidden_states
        return output

    def autoregressive_forward(
        self,
        context_features: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        num_steps: int,
        guidance_scale: float = 1.5
    ) -> torch.Tensor:
        """
        Autoregressive generation with CFG.
        
        Args:
            context_features: [B, T_ctx, D] context frame features
            context_actions: [B, T_ctx, A] context actions
            future_actions: [B, T_future, A] future actions
            num_steps: Number of denoising steps per frame
            guidance_scale: CFG scale for action conditioning
        
        Returns:
            [B, T_future, D] generated features
        """
        B, T_ctx, D = context_features.shape
        T_future = future_actions.shape[1]
        device = context_features.device
        
        # Initialize with context
        all_features = [context_features]
        all_actions = [context_actions]
        
        # GRU hidden state
        h = None
        if self.use_gru:
            # Warm up GRU with context
            x = self.input_proj(context_features)
            for t in range(T_ctx):
                h = self.gru(x[:, t], h)
        
        # Generate future frames autoregressively
        for t in range(T_future):
            # Current action
            action_t = future_actions[:, t:t+1]
            
            # Concatenate all features so far
            features_so_far = torch.cat(all_features, dim=1)
            actions_so_far = torch.cat(all_actions + [action_t], dim=1)
            
            # Conditional forward
            cond_out = self.forward(features_so_far, actions_so_far, drop_actions=False)
            cond_feat = cond_out[:, -1:]
            
            # Unconditional forward (for CFG)
            if guidance_scale != 1.0:
                uncond_out = self.forward(features_so_far, actions_so_far, drop_actions=True)
                uncond_feat = uncond_out[:, -1:]
                
                # Apply CFG
                feat_t = uncond_feat + guidance_scale * (cond_feat - uncond_feat)
            else:
                feat_t = cond_feat
            
            all_features.append(feat_t)
            all_actions.append(action_t)
        
        # Return only future features
        future_features = torch.cat(all_features[1:], dim=1)
        return future_features

