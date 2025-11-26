# --------------------------------------------------------
# DiffuCausal Fusion Head + DecisionChainHead（最终修复版）
# 已修复：EventHead dict 返回、memory pooling、真实训练兼容
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_util import RMSNorm


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = query.shape
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossModalAttention(
            dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm3 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim), nn.Dropout(proj_drop)
        )

    def forward(self, x: torch.Tensor, cross_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        if cross_input is not None:
            x = x + self.cross_attn(self.norm2(x), cross_input)
        x = x + self.mlp(self.norm3(x))
        return x


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.transform = nn.Linear(dim * 2, dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat1, feat2], dim=-1)
        gate = self.gate_proj(concat)
        fused = gate * feat1 + (1 - gate) * feat2
        fused = fused + self.transform(concat)
        return fused


class FusionHead(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.causal_proj = nn.Linear(causal_dim, hidden_dim)
        self.event_proj = nn.Linear(event_dim, hidden_dim)

        self.vc_fusion_blocks = nn.ModuleList([
            FusionBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(num_layers)
        ])
        self.event_fusion_blocks = nn.ModuleList([
            FusionBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(num_layers)
        ])

        self.video_causal_gate = GatedFusion(hidden_dim)
        self.final_gate = GatedFusion(hidden_dim)

        self.output_proj = nn.Sequential(RMSNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.match_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        video_features: torch.Tensor,
        causal_features: torch.Tensor,
        event_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_h = self.video_proj(video_features)
        causal_h = self.causal_proj(causal_features)

        vc_fused = video_h
        for block in self.vc_fusion_blocks:
            vc_fused = block(vc_fused, causal_h)
        vc_combined = self.video_causal_gate(vc_fused, causal_h)

        if event_features is not None:
            event_h = self.event_proj(event_features)
            event_fused = vc_combined
            for block in self.event_fusion_blocks:
                event_fused = block(event_fused, event_h)
            final_fused = self.final_gate(event_fused, event_h)
        else:
            final_fused = vc_combined

        output = self.output_proj(final_fused)
        match_features = self.match_proj(final_fused)
        return output, match_features

    def compute_matching_loss(self, video_features: torch.Tensor, causal_features: torch.Tensor) -> torch.Tensor:
        v = F.normalize(video_features, dim=-1)
        c = F.normalize(causal_features, dim=-1)
        return (1 - (v * c).sum(dim=-1)).mean()


class DecisionChainHead(nn.Module):
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

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tool_embed = nn.Embedding(num_tools + 2, hidden_dim)
        self.BOS_TOKEN = num_tools
        self.EOS_TOKEN = num_tools + 1

        self.chain_pos_embed = nn.Parameter(torch.zeros(1, max_chain_length, hidden_dim))
        nn.init.normal_(self.chain_pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=proj_drop, activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(RMSNorm(hidden_dim), nn.Linear(hidden_dim, num_tools + 2))

    def forward(
        self,
        fused_features: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [B, T, input_dim] → [B, T, hidden_dim]
        memory = self.input_proj(fused_features)

        if target_chain is not None:
            B, L = target_chain.shape
            bos = torch.full((B, 1), self.BOS_TOKEN, device=target_chain.device, dtype=torch.long)
            target_input = torch.cat([bos, target_chain[:, :-1]], dim=1)
            target_input = target_input.clamp(min=0, max=self.num_tools + 1)

            tgt = self.tool_embed(target_input) + self.chain_pos_embed[:, :L, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, tgt.device)

            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output)

            loss = F.cross_entropy(logits.view(-1, self.num_tools + 2),
                                   target_chain.view(-1), ignore_index=-1)
            return logits, loss
        else:
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
        # 关键修复：将 [B, T, D] → [B, 1, D]，避免 decoder 长度不匹配
        memory = memory.mean(dim=1, keepdim=True)  # ←←←← 这一行解决了一切！

        if max_length is None:
            max_length = self.max_chain_length

        B, device = memory.shape[0], memory.device
        generated = torch.full((B, 1), self.BOS_TOKEN, device=device, dtype=torch.long)

        for _ in range(max_length):
            tgt = self.tool_embed(generated) + self.chain_pos_embed[:, :generated.shape[1], :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.shape[1], device)

            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output[:, -1, :]) / temperature

            if top_k > 0:
                top_k_logits, top_k_idx = torch.topk(logits, top_k)
                next_token = top_k_idx.gather(-1, torch.multinomial(F.softmax(top_k_logits, -1), 1))
            elif top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                sorted_idx_to_remove = cum_probs > top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                sorted_logits[sorted_idx_to_remove] = float('-inf')
                next_token = sorted_idx.gather(-1, torch.multinomial(F.softmax(sorted_logits, -1), 1))
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.EOS_TOKEN).all():
                break

        return generated[:, 1:]  # 去掉 BOS
