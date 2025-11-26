# --------------------------------------------------------
# DiffuCausal Fusion Head
# Feature fusion and decision chain generation
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_util import RMSNorm


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing video and causal features.
    
    Enables bidirectional attention between:
    - Video stream features (environmental information)
    - Causal/event stream features (action-effect relationships)
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
        
        # Query from one modality
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # Key and Value from another modality
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # QK normalization
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T, D] query features (from one modality)
            key_value: [B, T, D] key/value features (from another modality)
            attn_mask: Optional attention mask
        
        Returns:
            [B, T, D] cross-attended features
        """
        B, T, D = query.shape
        
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class FusionBlock(nn.Module):
    """
    Fusion Block for combining multi-modal features.
    
    Architecture:
    1. Self-attention on concatenated features
    2. Cross-attention between modalities
    3. Feed-forward network
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        # Cross-attention (video -> causal)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossModalAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # FFN
        self.norm3 = RMSNorm(dim)
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
        cross_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] main features
            cross_input: [B, T, D] features for cross-attention (optional)
        
        Returns:
            [B, T, D] fused features
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention (if cross_input provided)
        if cross_input is not None:
            x = x + self.cross_attn(self.norm2(x), cross_input)
        
        # FFN
        x = x + self.mlp(self.norm3(x))
        
        return x


class GatedFusion(nn.Module):
    """
    Gated Fusion Module for adaptive feature combination.
    
    Uses learned gates to control the contribution of each modality.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.transform = nn.Linear(dim * 2, dim)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat1: [B, T, D] first modality features
            feat2: [B, T, D] second modality features
        
        Returns:
            [B, T, D] gated fusion output
        """
        concat = torch.cat([feat1, feat2], dim=-1)
        
        # Compute gate
        gate = self.gate_proj(concat)
        
        # Gated combination
        fused = gate * feat1 + (1 - gate) * feat2
        
        # Additional transformation
        fused = fused + self.transform(concat)
        
        return fused


class FusionHead(nn.Module):
    """
    Fusion Head for DiffuCausal.
    
    Combines video stream features and causal/event features to produce
    unified representations for decision making.
    
    Architecture:
    1. Input projection for each modality
    2. Multi-layer fusion blocks with cross-attention
    3. Gated fusion for final combination
    4. Output projection
    
    Supports:
    - Video features (environmental information)
    - Causal features (action-effect relationships)
    - Event features (scene decomposition)
    """
    
    def __init__(
        self,
        video_dim: int = 256,
        causal_dim: int = 256,
        event_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.causal_dim = causal_dim
        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projections
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.causal_proj = nn.Linear(causal_dim, hidden_dim)
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        
        # Fusion blocks for video-causal fusion
        self.vc_fusion_blocks = nn.ModuleList([
            FusionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for _ in range(num_layers)
        ])
        
        # Fusion blocks for event integration
        self.event_fusion_blocks = nn.ModuleList([
            FusionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for _ in range(num_layers)
        ])
        
        # Gated fusion modules
        self.video_causal_gate = GatedFusion(hidden_dim)
        self.final_gate = GatedFusion(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Matching loss projection (for feature alignment)
        self.match_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        video_features: torch.Tensor,
        causal_features: torch.Tensor,
        event_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Fusion Head.
        
        Args:
            video_features: [B, T, video_dim] video stream features
            causal_features: [B, T, causal_dim] causal head features
            event_features: [B, T, event_dim] event head features (optional)
        
        Returns:
            fused_features: [B, T, output_dim] fused features for decision
            match_features: [B, T, hidden_dim] features for matching loss
        """
        # Project inputs to hidden dimension
        video_h = self.video_proj(video_features)
        causal_h = self.causal_proj(causal_features)
        
        # Video-Causal fusion with cross-attention
        vc_fused = video_h
        for block in self.vc_fusion_blocks:
            vc_fused = block(vc_fused, cross_input=causal_h)
        
        # Gated combination of video and causal
        vc_combined = self.video_causal_gate(vc_fused, causal_h)
        
        # Integrate event features if provided
        if event_features is not None:
            event_h = self.event_proj(event_features)
            
            # Event fusion with cross-attention
            event_fused = vc_combined
            for block in self.event_fusion_blocks:
                event_fused = block(event_fused, cross_input=event_h)
            
            # Final gated fusion
            final_fused = self.final_gate(event_fused, event_h)
        else:
            final_fused = vc_combined
        
        # Output projection
        output = self.output_proj(final_fused)
        
        # Match features for alignment loss
        match_features = self.match_proj(final_fused)
        
        return output, match_features

    def compute_matching_loss(
        self,
        video_features: torch.Tensor,
        causal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute matching loss for feature alignment.
        
        Uses cosine similarity to encourage alignment between
        video and causal feature spaces.
        
        Args:
            video_features: [B, T, D] video features
            causal_features: [B, T, D] causal features
        
        Returns:
            Scalar matching loss
        """
        # Normalize features
        video_norm = F.normalize(video_features, dim=-1)
        causal_norm = F.normalize(causal_features, dim=-1)
        
        # Cosine similarity
        similarity = (video_norm * causal_norm).sum(dim=-1)
        
        # Loss: 1 - similarity (minimize distance)
        loss = (1 - similarity).mean()
        
        return loss


class DecisionChainHead(nn.Module):
    """
    Decision Chain Head for generating tool call sequences.
    
    Converts fused features into a sequence of decision/action tokens
    representing the UAV's action plan.
    
    Architecture:
    - Transformer decoder for autoregressive generation
    - Tool vocabulary embedding
    - Sequence prediction with causal masking
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_tools: int = 64,
        max_chain_length: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        self.max_chain_length = max_chain_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Tool embeddings
        self.tool_embed = nn.Embedding(num_tools + 2, hidden_dim)  # +2 for BOS and EOS
        self.BOS_TOKEN = num_tools
        self.EOS_TOKEN = num_tools + 1
        
        # Position embeddings for decision chain
        self.chain_pos_embed = nn.Parameter(torch.zeros(1, max_chain_length, hidden_dim))
        nn.init.normal_(self.chain_pos_embed, std=0.02)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=proj_drop,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to tool logits
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, num_tools + 2)  # Include BOS and EOS
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for decision chain generation.
        
        Args:
            fused_features: [B, T, input_dim] fused features
            target_chain: [B, L] target tool indices for teacher forcing
        
        Returns:
            logits: [B, L, num_tools+2] tool prediction logits
            loss: Cross-entropy loss if target_chain provided
        """
        B, T, D = fused_features.shape
        
        # Project fused features (memory for decoder)
        memory = self.input_proj(fused_features)  # [B, T, hidden_dim]
        
        if target_chain is not None:
            # Teacher forcing mode
            L = target_chain.shape[1]
            
            # Embed target tokens (shift right, prepend BOS)
            bos = torch.full((B, 1), self.BOS_TOKEN, device=target_chain.device, dtype=torch.long)
            target_input = torch.cat([bos, target_chain[:, :-1]], dim=1)
            
            # Clamp to valid range (replace -1 padding with BOS token for embedding)
            target_input_clamped = target_input.clamp(min=0, max=self.num_tools + 1)
            tgt = self.tool_embed(target_input_clamped)  # [B, L, hidden_dim]
            
            # Add position embedding
            tgt = tgt + self.chain_pos_embed[:, :L, :]
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
            
            # Decode
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            
            # Project to logits
            logits = self.output_proj(output)  # [B, L, num_tools+2]
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, self.num_tools + 2),
                target_chain.view(-1),
                ignore_index=-1  # Ignore padding
            )
            
            return logits, loss
        else:
            # Inference mode (autoregressive generation)
            return self.generate(memory)

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive generation of decision chain.
        
        Args:
            memory: [B, T, hidden_dim] encoded features
            max_length: Maximum chain length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 for greedy)
            top_p: Nucleus sampling threshold
        
        Returns:
            [B, L] generated tool indices
        """
        B = memory.shape[0]
        device = memory.device
        
        if max_length is None:
            max_length = self.max_chain_length
        
        # Start with BOS token
        generated = torch.full((B, 1), self.BOS_TOKEN, device=device, dtype=torch.long)
        
        for i in range(max_length):
            # Embed current sequence
            tgt = self.tool_embed(generated)
            tgt = tgt + self.chain_pos_embed[:, :generated.shape[1], :]
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.shape[1], device=device
            )
            
            # Decode
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            
            # Get logits for last position
            logits = self.output_proj(output[:, -1, :])  # [B, num_tools+2]
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample next token
            if top_k > 0:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            elif top_p < 1.0:
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                probs = F.softmax(sorted_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = sorted_indices.gather(-1, next_token_idx)
            else:
                # Greedy sampling
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == self.EOS_TOKEN).all():
                break
        
        # Remove BOS token
        return generated[:, 1:]

