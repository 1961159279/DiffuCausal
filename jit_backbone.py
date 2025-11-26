# --------------------------------------------------------
# DiffuCausal JiT Backbone
# Just image Transformer - A minimalist pixel-space diffusion architecture
# References:
# JiT: https://github.com/LTH14/JiT
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

import sys
import os
# 将当前文件的上级目录加入系统路径，以便导入utils包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_util import (
    VisionRotaryEmbeddingFast,  # 快速视觉旋转位置编码
    RMSNorm,                    # 均方根归一化层
    get_2d_sincos_pos_embed,    # 获取2D正弦余弦位置嵌入
)


def modulate(x, shift, scale):
    """
    AdaLN(Adaptive Layer Normalization)调制函数，用于条件调整模型行为
    
    Args:
        x: 输入张量 [B, N, D]，其中B是批次大小，N是序列长度，D是特征维度
        shift: 偏移参数 [B, D]，用于调整每个通道的偏移量
        scale: 缩放参数 [B, D]，用于调整每个通道的缩放比例
    
    Returns:
        调制后的张量 [B, N, D]
    """
    # 对输入进行缩放和平移操作：x * (1 + scale) + shift
    # unsqueeze(1)在位置1增加一个维度，使scale和shift从[B,D]变为[B,1,D]以支持广播
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """
    图像到补丁嵌入模块，采用瓶颈设计以提高效率
    
    使用两阶段卷积进行有效的补丁嵌入:
    1. 第一个卷积将通道数减少到瓶颈维度
    2. 第二个卷积将瓶颈维度扩展到嵌入维度
    """
    
    def __init__(
        self,
        img_size: int = 224,           # 输入图像大小，默认224x224
        patch_size: int = 16,          # 补丁大小，默认16x16
        in_chans: int = 3,             # 输入通道数，默认RGB三通道
        pca_dim: int = 768,            # 瓶颈维度，中间压缩的特征维度
        embed_dim: int = 768,          # 最终嵌入维度
        bias: bool = True              # 是否使用偏置项
    ):
        super().__init__()
        # 将图像尺寸和补丁尺寸转换为元组格式
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        # 计算补丁数量 = (高/补丁高) * (宽/补丁宽)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size       # 存储图像尺寸
        self.patch_size = patch_size   # 存储补丁尺寸
        self.num_patches = num_patches # 存储补丁总数

        # 第一阶段卷积：降低通道维度到瓶颈维度，步长等于补丁大小以实现分块
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # 第二阶段卷积：从瓶颈维度扩展到最终嵌入维度，1x1卷积不改变空间尺寸
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # 获取输入张量形状：B(批次大小), C(通道数), H(高度), W(宽度)
        B, C, H, W = x.shape
        # 验证输入图像尺寸是否与模型配置匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 执行两阶段投影，然后展平空间维度并转置得到[B, N, D]格式
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器，将标量时间步转换为向量表示，使用正弦嵌入方法
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        # 多层感知机用于将频率嵌入映射到隐藏状态维度
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # 第一层线性变换
            nn.SiLU(),                                                    # SiLU激活函数
            nn.Linear(hidden_size, hidden_size, bias=True),               # 第二层线性变换
        )
        self.frequency_embedding_size = frequency_embedding_size          # 存储频率嵌入维度

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入，这是扩散模型中的标准做法
        
        Args:
            t: N个索引的一维张量，每个批次元素一个索引(可能是小数)
            dim: 输出的维度
            max_period: 控制嵌入的最小频率
        
        Returns:
            (N, D)形状的位置嵌入张量
        """
        # 计算一半维度，因为正弦和余弦各占一半
        half = dim // 2
        # 计算不同频率的参数，使用指数函数创建几何级数
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 计算角度参数：t[:, None]将[N]变成[N,1]，freqs[None]将[half]变成[1,half]
        args = t[:, None].float() * freqs[None]
        # 拼接正弦和余弦值形成完整嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，添加一个零填充
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # 先进行时间步频率嵌入，再通过MLP映射到隐藏状态空间
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入器，将类别标签转换为向量表示
    同时处理无分类器指导(classifier-free guidance)的标签dropout
    """
    
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        # 创建嵌入表，额外增加一个类别用于无条件生成(标签dropout)
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes  # 存储类别数量

    def forward(self, labels):
        # 通过嵌入表查找对应标签的嵌入向量
        embeddings = self.embedding_table(labels)
        return embeddings


class ActionEmbedder(nn.Module):
    """
    动作嵌入器，将连续动作向量嵌入到隐藏表示中
    用于因果世界模型中的动作条件化
    """
    
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        # 多层感知机用于将动作向量映射到隐藏状态维度
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size, bias=True),   # 第一层线性变换
            nn.SiLU(),                                       # SiLU激活函数
            nn.Linear(hidden_size, hidden_size, bias=True),  # 第二层线性变换
        )
        self.action_dim = action_dim  # 存储动作维度

    def forward(self, actions):
        """
        Args:
            actions: [B, A] 或 [B, T, A] 的动作向量，B是批次大小，T是时间步，A是动作维度
        
        Returns:
            [B, D] 或 [B, T, D] 的动作嵌入，D是隐藏维度
        """
        # 通过MLP将动作映射到隐藏空间
        return self.mlp(actions)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    缩放点积注意力机制，可选因果掩码
    
    Args:
        query: [B, H, L, D] 查询张量，B是批次大小，H是头数，L是查询序列长度，D是维度
        key: [B, H, S, D] 键张量，S是键序列长度
        value: [B, H, S, D] 值张量
        dropout_p: Dropout概率
        attn_mask: 可选注意力掩码 [L, S] 或 [B, H, L, S]
    
    Returns:
        [B, H, L, D] 注意力输出
    """
    # 获取查询和键的序列长度
    L, S = query.size(-2), key.size(-2)
    # 计算缩放因子，防止点积值过大导致梯度消失
    scale_factor = 1 / math.sqrt(query.size(-1))
    
    # 初始化注意力偏置为零
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)
    
    # 如果提供了注意力掩码，则应用它
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 布尔类型掩码：将True位置设为负无穷
            attn_bias = attn_bias.masked_fill(attn_mask, float('-inf'))
        else:
            # 数值类型掩码：直接加到偏置上
            attn_bias = attn_bias + attn_mask

    # 在自动混合精度训练中禁用AMP以保证数值稳定性
    with torch.cuda.amp.autocast(enabled=False):
        # 计算查询和键的点积，然后乘以缩放因子
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    # 加上注意力偏置
    attn_weight = attn_weight + attn_bias
    # 对最后一个维度进行softmax归一化
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # 应用dropout（仅在训练时）
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # 将注意力权重与值相乘得到最终输出
    return attn_weight @ value


class Attention(nn.Module):
    """
    多头自注意力机制，支持RoPE和可选的因果掩码
    """
    
    def __init__(
        self,
        dim: int,              # 输入维度
        num_heads: int = 8,    # 注意力头数
        qkv_bias: bool = True, # 是否在qkv线性变换中使用偏置
        qk_norm: bool = True,  # 是否对q和k进行归一化
        attn_drop: float = 0., # 注意力dropout率
        proj_drop: float = 0.  # 投影dropout率
    ):
        super().__init__()
        self.num_heads = num_heads                    # 存储头数
        head_dim = dim // num_heads                   # 计算每个头的维度

        # 如果启用qk归一化，则使用RMSNorm，否则使用恒等映射
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        # 一次性计算QKV的线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        # 投影dropout层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,                              # 输入张量 [B, N, C]
        rope: Optional[VisionRotaryEmbeddingFast] = None,  # 可选的旋转位置嵌入
        attn_mask: Optional[torch.Tensor] = None      # 可选的注意力掩码
    ):
        """
        Args:
            x: [B, N, C] 输入张量
            rope: 可选的旋转位置嵌入
            attn_mask: 可选的注意力掩码，用于因果注意力
        
        Returns:
            [B, N, C] 输出张量
        """
        B, N, C = x.shape
        # 计算QKV并重新排列维度为[3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分离出Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 对Q和K进行归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 如果提供了RoPE，则应用于Q和K
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # 执行缩放点积注意力
        x = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,  # 训练时使用dropout
            attn_mask=attn_mask
        )

        # 重新排列张量形状并合并头
        x = x.transpose(1, 2).reshape(B, N, C)
        # 通过输出投影层
        x = self.proj(x)
        # 应用投影dropout
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU前馈网络
    
    使用SiLU激活函数和门控线性单元以提高性能
    """
    
    def __init__(
        self,
        dim: int,         # 输入维度
        hidden_dim: int,  # 隐藏维度
        drop: float = 0.0, # Dropout率
        bias: bool = True  # 是否使用偏置
    ):
        super().__init__()
        # 将隐藏维度调整为2/3以适应SwiGLU机制
        hidden_dim = int(hidden_dim * 2 / 3)
        # 第一层线性变换，同时计算门控和值
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        # 第二层线性变换
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        # 前馈网络dropout
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        # 同时计算门控和值部分
        x12 = self.w12(x)
        # 将结果分为两个部分
        x1, x2 = x12.chunk(2, dim=-1)
        # SwiGLU操作：SiLU(x1) * x2
        hidden = F.silu(x1) * x2
        # 通过第二层变换和dropout
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    JiT的最终层，带有AdaLN调制
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # 最终归一化层
        self.norm_final = RMSNorm(hidden_size)
        # 线性层将隐藏状态映射到补丁像素值
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # AdaLN调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),                                # SiLU激活函数
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)  # 线性变换生成shift和scale参数
        )

    def forward(self, x, c):
        # 通过AdaLN调制模块生成shift和scale参数
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # 使用modulate函数对归一化的特征进行调制
        x = modulate(self.norm_final(x), shift, scale)
        # 通过线性层得到最终输出
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    """
    JiT Transformer块，带有AdaLN-Zero调制
    
    每个块包含:
    - RMSNorm + 多头注意力 + RoPE
    - RMSNorm + SwiGLU前馈网络
    - AdaLN调制用于时间和类别条件化
    """
    
    def __init__(
        self,
        hidden_size: int,     # 隐藏维度
        num_heads: int,       # 注意力头数
        mlp_ratio: float = 4.0, # MLP隐藏维度与嵌入维度的比例
        attn_drop: float = 0.0, # 注意力dropout率
        proj_drop: float = 0.0  # 投影dropout率
    ):
        super().__init__()
        # 第一个归一化层用于注意力子层
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        # 注意力层
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        # 第二个归一化层用于前馈网络子层
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        # 计算MLP隐藏维度
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # SwiGLU前馈网络
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        # AdaLN调制模块，生成6个参数用于两个子层的调制
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,      # 输入特征 [B, N, D]
        c: torch.Tensor,      # 条件信息(时间+类别/动作嵌入) [B, D]
        feat_rope: Optional[VisionRotaryEmbeddingFast] = None,  # 可选的旋转位置嵌入
        attn_mask: Optional[torch.Tensor] = None      # 可选的注意力掩码
    ):
        """
        Args:
            x: [B, N, D] 输入特征
            c: [B, D] 条件信息(时间+类别/动作嵌入)
            feat_rope: 可选的旋转位置嵌入
            attn_mask: 可选的注意力掩码用于因果注意力
        
        Returns:
            [B, N, D] 输出特征
        """
        # 通过AdaLN调制模块生成6个调制参数，分别用于两个子层的shift、scale和gate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        # 注意力子层：残差连接 + 门控调制
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),  # 对归一化后的输入进行调制
            rope=feat_rope,     # 旋转位置嵌入
            attn_mask=attn_mask # 注意力掩码
        )
        # 前馈网络子层：残差连接 + 门控调制
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer - 一个极简的像素空间扩散架构
    
    主要特性:
    - 纯Transformer块，没有U-Net的归纳偏置
    - AdaLN(自适应层归一化)用于时间和类别条件化
    - 视觉旋转嵌入(RoPE)用于空间局部性
    - 上下文token用于无分类器指导
    - 瓶颈补丁嵌入提高效率
    """
    
    def __init__(
        self,
        input_size: int = 256,      # 输入图像大小
        patch_size: int = 16,       # 补丁大小
        in_channels: int = 3,       # 输入通道数
        hidden_size: int = 1024,    # 隐藏维度
        depth: int = 24,            # Transformer块深度
        num_heads: int = 16,        # 注意力头数
        mlp_ratio: float = 4.0,     # MLP比率
        attn_drop: float = 0.0,     # 注意力dropout
        proj_drop: float = 0.0,     # 投影dropout
        num_classes: int = 1000,    # 类别数
        bottleneck_dim: int = 128,  # 瓶颈维度
        in_context_len: int = 32,   # 上下文长度
        in_context_start: int = 8   # 上下文开始位置
    ):
        super().__init__()
        self.in_channels = in_channels      # 输入通道数
        self.out_channels = in_channels     # 输出通道数
        self.patch_size = patch_size        # 补丁大小
        self.num_heads = num_heads          # 注意力头数
        self.hidden_size = hidden_size      # 隐藏维度
        self.input_size = input_size        # 输入尺寸
        self.in_context_len = in_context_len  # 上下文长度
        self.in_context_start = in_context_start  # 上下文开始位置
        self.num_classes = num_classes      # 类别数
        self.depth = depth                  # 网络深度

        # 时间和类别嵌入器
        self.t_embedder = TimestepEmbedder(hidden_size)    # 时间嵌入器
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)  # 类别嵌入器

        # 补丁嵌入器
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # 固定的sin-cos位置嵌入
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # 用于CFG的上下文分类token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size),
                requires_grad=True
            )
            # 使用正态分布初始化上下文位置嵌入
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # 用于空间注意力的RoPE
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        # 标准特征RoPE（不含上下文token）
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        # 包含上下文token的RoPE
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # Transformer块，带有选择性dropout
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,  # 中间层使用dropout
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
            )
            for i in range(depth)
        ])

        # 最终预测层
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型权重"""
        # 基本初始化函数：对线性层使用Xavier均匀初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 使用sin-cos嵌入初始化pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化patch_embed类似nn.Linear
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # 初始化标签嵌入
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # 初始化时间步嵌入
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 零初始化adaLN调制层（AdaLN-Zero技巧）
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 零初始化输出层
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        将补丁token转换回图像
        
        Args:
            x: [N, T, patch_size**2 * C] 补丁序列
            p: 补丁大小
        
        Returns:
            imgs: [N, C, H, W] 图像
        """
        c = self.out_channels  # 输出通道数
        # 计算高度和宽度
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]  # 验证补丁数量正确

        # 重塑张量并重新排列维度以恢复图像
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)  # 使用爱因斯坦求和约定重新排列
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))  # 展平为空间维度
        return imgs

    def forward(self, x, t, y):
        """
        JiT前向传播
        
        Args:
            x: [N, C, H, W] 输入图像
            t: [N,] 时间步
            y: [N,] 类别标签
        
        Returns:
            [N, C, H, W] 预测图像
        """
        # 类别和时间嵌入
        t_emb = self.t_embedder(t)   # 时间嵌入
        y_emb = self.y_embedder(y)   # 类别嵌入
        c = t_emb + y_emb            # 合并条件信息

        # 补丁嵌入
        x = self.x_embedder(x)       # 将图像转换为补丁序列
        x = x + self.pos_embed       # 添加位置嵌入

        # Transformer块
        for i, block in enumerate(self.blocks):
            # 插入上下文token
            if self.in_context_len > 0 and i == self.in_context_start:
                # 重复类别嵌入作为上下文token并添加位置嵌入
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)  # 拼接上下文token和补丁特征
            
            # 根据是否插入上下文token选择相应的RoPE
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)  # 通过Transformer块

        # 移除上下文token
        x = x[:, self.in_context_len:]

        # 最终预测
        x = self.final_layer(x, c)   # 通过最终层
        output = self.unpatchify(x, self.patch_size)  # 转换回图像格式

        return output

    def get_features(self, x, t, y, return_all_layers: bool = False):
        """
        从JiT提取中间特征
        
        Args:
            x: [N, C, H, W] 输入图像
            t: [N,] 时间步
            y: [N,] 类别标签
            return_all_layers: 如果为True，返回所有层的特征
        
        Returns:
            features: [N, num_patches, hidden_size] 或特征列表
        """
        # 类别和时间嵌入
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # 补丁嵌入和位置嵌入
        x = self.x_embedder(x)
        x = x + self.pos_embed

        all_features = []  # 存储所有层特征（如果需要）
        # 遍历所有Transformer块
        for i, block in enumerate(self.blocks):
            # 插入上下文token
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            
            # 选择适当的RoPE
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)
            
            # 如果需要返回所有层特征，则保存当前层输出
            if return_all_layers:
                # 如果已插入上下文token，则去除上下文部分
                all_features.append(x[:, self.in_context_len:] if i >= self.in_context_start else x)

        # 移除上下文token
        x = x[:, self.in_context_len:]
        
        # 返回特征
        if return_all_layers:
            return all_features
        return x


class JiTBackbone(nn.Module):
    """
    JiT主干网络，用于DiffuCausal视频编码
    
    封装JiT以处理视频序列并提取时序特征
    支持逐帧处理和时序注意力
    """
    
    def __init__(
        self,
        input_size: int = 256,        # 输入尺寸
        patch_size: int = 16,         # 补丁大小
        in_channels: int = 3,         # 输入通道数
        hidden_size: int = 1024,      # 隐藏维度
        depth: int = 24,              # 网络深度
        num_heads: int = 16,          # 注意力头数
        mlp_ratio: float = 4.0,       # MLP比率
        num_classes: int = 1000,      # 类别数
        output_dim: int = 1024,       # 输出维度
        temporal_pooling: str = "mean" # 时序池化方式
    ):
        super().__init__()
        
        # 创建JiT模型，禁用上下文token用于视频处理
        self.jit = JiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            in_context_len=0,  # 禁用上下文token用于视频处理
        )
        
        self.hidden_size = hidden_size      # 隐藏维度
        self.output_dim = output_dim        # 输出维度
        self.temporal_pooling = temporal_pooling  # 时序池化方式
        
        # 如果隐藏维度与输出维度不同，则添加投影层
        if hidden_size != output_dim:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = nn.Identity()  # 否则使用恒等映射
        
        # 时序位置嵌入
        self.max_temporal_len = 64
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_temporal_len, hidden_size)
        )
        # 使用正态分布初始化时序位置嵌入
        nn.init.normal_(self.temporal_pos_embed, std=0.02)

    def forward(
        self,
        video: torch.Tensor,                    # 视频帧 [B, T, C, H, W]
        timesteps: Optional[torch.Tensor] = None,  # 扩散时间步
        labels: Optional[torch.Tensor] = None      # 类别标签
    ) -> torch.Tensor:
        """
        通过JiT主干网络处理视频序列
        
        Args:
            video: [B, T, C, H, W] 视频帧，B是批次大小，T是时间步数
            timesteps: [B,] 或 [B, T] 扩散时间步（默认: 0）
            labels: [B,] 类别标签（默认: 0）
        
        Returns:
            [B, T, output_dim] 时序特征
        """
        B, T, C, H, W = video.shape  # 获取视频张量形状
        
        # 默认时间步和标签
        if timesteps is None:
            timesteps = torch.zeros(B, device=video.device, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(B, device=video.device, dtype=torch.long)
        
        # 确保时间步为浮点型
        timesteps = timesteps.float()
        
        # 逐帧处理（使用reshape处理非连续张量）
        video_flat = video.reshape(B * T, C, H, W)  # 将[B,T,C,H,W]展平为[B*T,C,H,W]
        
        # 扩展时间步和标签到所有帧
        if timesteps.dim() == 1:  # 如果时间步是[B,]形状
            timesteps_flat = timesteps.unsqueeze(1).expand(B, T).reshape(B * T)
        else:  # 如果时间步是[B,T]形状
            timesteps_flat = timesteps.reshape(B * T)
        
        # 扩展标签到所有帧
        labels_flat = labels.unsqueeze(1).expand(B, T).reshape(B * T)
        
        # 从JiT获取特征
        features = self.jit.get_features(video_flat, timesteps_flat, labels_flat)
        
        # 池化空间特征
        if self.temporal_pooling == "mean":
            features = features.mean(dim=1)  # 平均池化所有补丁 [B*T, D]
        elif self.temporal_pooling == "cls":
            features = features[:, 0]  # 使用第一个补丁(类似CLS token) [B*T, D]
        else:
            features = features.mean(dim=1)
        
        # 重塑为时序序列
        features = features.view(B, T, -1)  # [B, T, D]
        
        # 添加时序位置嵌入
        features = features + self.temporal_pos_embed[:, :T, :]
        
        # 投影到输出维度
        features = self.proj(features)
        
        return features


# 模型工厂函数
def JiT_B_16(**kwargs):
    """JiT-Base模型，补丁大小16"""
    return JiT(patch_size=16, hidden_size=768, depth=12, num_heads=12, **kwargs)


def JiT_B_32(**kwargs):
    """JiT-Base模型，补丁大小32"""
    return JiT(patch_size=32, hidden_size=768, depth=12, num_heads=12, **kwargs)


def JiT_L_16(**kwargs):
    """JiT-Large模型，补丁大小16"""
    return JiT(patch_size=16, hidden_size=1024, depth=24, num_heads=16, **kwargs)


def JiT_L_32(**kwargs):
    """JiT-Large模型，补丁大小32"""
    return JiT(patch_size=32, hidden_size=1024, depth=24, num_heads=16, **kwargs)


def JiT_H_16(**kwargs):
    """JiT-Huge模型，补丁大小16"""
    return JiT(patch_size=16, hidden_size=1280, depth=32, num_heads=16, **kwargs)


def JiT_H_32(**kwargs):
    """JiT-Huge模型，补丁大小32"""
    return JiT(patch_size=32, hidden_size=1280, depth=32, num_heads=16, **kwargs)


# 模型字典，便于按名称创建模型
JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}