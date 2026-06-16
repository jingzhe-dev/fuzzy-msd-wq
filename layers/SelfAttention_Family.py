# -*- coding: utf-8 -*-
"""Attention modules.

LogicalFuzzyAttention adds query-key fuzzy membership similarity to the
scaled dot-product attention score.
"""

from __future__ import annotations
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TriangularCausalMask:
    """Upper-triangular causal mask for autoregressive decoding."""
    
    def __init__(self, B, L, device="cpu"):
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        self._mask = mask.unsqueeze(0).expand(B, 1, L, L)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Sparse attention mask for ProbSparse self-attention."""
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device)
        _mask = _mask.triu(1)
        _mask = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask[torch.arange(B)[:, None, None],
                         torch.arange(H)[None, :, None], index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


# =============================================================================
# Fuzzy Membership Modules
# =============================================================================

class FuzzyMembership(nn.Module):
    """Static Gaussian fuzzy membership."""
    
    def __init__(self, x_min, x_max, N, d_model, epsilon=1e-6):
        super().__init__()
        self.N, self.epsilon = N, epsilon
        
        delta = (x_max - x_min) / N
        centers = torch.linspace(x_min + delta/2, x_max - delta/2, N)
        self.register_buffer('centers', centers)
        self.sigma = nn.Parameter(torch.full((N,), delta / 2.5))
        self.proj = nn.Linear(N, d_model)
    
    def forward(self, x):
        """x: [B, L, C] -> (proj [B, L, d], raw [B, L, N])"""
        x_exp = x.unsqueeze(-1)
        c = self.centers.view(1, 1, 1, -1)
        s = self.sigma.view(1, 1, 1, -1).clamp(min=self.epsilon)
        mu = torch.exp(-0.5 * ((x_exp - c) / s) ** 2)
        mu_mean = mu.mean(dim=2)
        return self.proj(mu_mean), mu_mean


class AdaptiveFuzzyMembership(nn.Module):
    """Adaptive Gaussian fuzzy membership."""
    
    def __init__(self, x_min, x_max, N, d_model, beta=0.5, epsilon=1e-6):
        super().__init__()
        self.N, self.d_model = N, d_model
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.epsilon = epsilon
        
        # 模糊规则中心，定义在归一化空间。
        delta = (x_max - x_min) / N
        self.centers = nn.Parameter(torch.linspace(x_min + delta/2, x_max - delta/2, N))
        
        # 基础规则宽度，可训练。
        self.sigma_base = nn.Parameter(torch.full((N,), delta / 2.0))
        
        # 将 N 维隶属度映射回 d_model，方便和 Transformer 表征对齐。
        self.proj = nn.Linear(N, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, context=None):
        """
        Compute fuzzy membership from embedding.
        
        Args:
            x: [B, L, d_model] - Transformer embeddings
        Returns:
            proj: [B, L, d_model] - Projected membership for attention
            mu: [B, L, N] - Raw membership grades
        """
        B, L, D = x.shape
        
        # 用 L2 范数压缩 d_model 维 embedding，得到每个 token 的标量强度。
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [B, L, 1]
        x_norm_scaled = (
            x_norm - x_norm.mean(dim=1, keepdim=True)
        ) / (x_norm.std(dim=1, keepdim=True, unbiased=False) + self.epsilon)
        
        # 不确定性代理：局部特征方差越大，规则宽度越大。
        omega = torch.log1p(x.var(dim=-1, keepdim=True, unbiased=False))  # [B, L, 1]
        omega = (
            omega - omega.mean(dim=1, keepdim=True)
        ) / (omega.std(dim=1, keepdim=True, unbiased=False) + self.epsilon)
        omega = torch.sigmoid(omega)
        
        # 自适应规则宽度：beta 是可学习强度系数。
        beta = F.softplus(self.beta)
        sigma = F.softplus(self.sigma_base.view(1, 1, -1) + beta * omega) + self.epsilon
        
        # 计算每个 token 对各条模糊规则的隶属度。
        c = self.centers.view(1, 1, -1)  # [1, 1, N]
        mu = torch.exp(-0.5 * (x_norm_scaled - c) ** 2 / sigma ** 2)  # [B, L, N]
        
        # 归一化后可直接用于 query-key 模糊相似度计算。
        mu = mu / (mu.sum(dim=-1, keepdim=True) + self.epsilon)
        
        return self.layer_norm(self.proj(mu)), mu


# =============================================================================
# Standard Attention Mechanisms
# =============================================================================

class FullAttention(nn.Module):
    """Standard scaled dot-product attention: A = softmax(QK^T/√d_k)V"""
    
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A if self.output_attention else None


class DSAttention(nn.Module):
    """De-stationary Attention: A = softmax(QK^T/(√d_k·τ) + δ)V"""
    
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale / tau + delta
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A if self.output_attention else None


class ProbAttention(nn.Module):
    """ProbSparse Self-Attention with O(L·log(L)) complexity."""
    
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        idx_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), idx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None], M_top, :]
        return torch.matmul(Q_reduce, K.transpose(-2, -1)), M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        return V.mean(dim=-2).unsqueeze(-2).expand(B, H, L_Q, D).clone()

    def _update_context(self, ctx, V, scores, idx, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, idx, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        ctx[torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None], idx, :] = torch.matmul(attn, V).type_as(ctx)
        return ctx, attn if self.output_attention else None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = min(self.factor * np.ceil(np.log(L_K)).astype('int').item(), L_K)
        u = min(self.factor * np.ceil(np.log(L_Q)).astype('int').item(), L_Q)

        scores_top, idx = self._prob_QK(queries, keys, U_part, u)
        scale = self.scale or 1. / sqrt(D)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, idx, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn


# =============================================================================
# Fuzzy Logic-Enhanced Attention
# =============================================================================

class LogicalFuzzyAttention(nn.Module):
    """模糊逻辑增强注意力。

    公式：
        A = softmax(QK^T/sqrt(d_k) + alpha * log(mu_q @ mu_k^T))

    模糊项同时依赖 query 和 key，避免被 softmax 抵消。
    """

    # 这里使用 query-key 隶属度相似度，因此每个 key 的 bias 都不同。
    def __init__(self, x_min, x_max, N, alpha, d_model, n_heads,
                 mask_flag=True, attention_dropout=0.1, output_attention=False,
                 device="cuda", use_adaptive_membership=True, membership_beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.epsilon = 1e-6
        
        if use_adaptive_membership:
            self.fuzzy = AdaptiveFuzzyMembership(x_min, x_max, N, d_model, beta=membership_beta)
        else:
            self.fuzzy = FuzzyMembership(x_min, x_max, N, d_model)
        
        self.head_scale = nn.Parameter(torch.ones(n_heads))
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1. / sqrt(self.d_k)

    def forward(
        self, queries, keys, values, attn_mask, tau=None, delta=None,
        x_raw=None, queries_raw=None, keys_raw=None
    ):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # 标准 scaled dot-product attention score。
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * self.scale

        # 添加模糊隶属度偏置：bias 的形状为 [B, 1, L, S]，
        # 其中 S 是 key 长度，因此它会真实改变 softmax 权重。
        queries_raw = queries_raw if queries_raw is not None else x_raw
        keys_raw = keys_raw if keys_raw is not None else x_raw
        if queries_raw is not None and keys_raw is not None:
            _, mu_q = self.fuzzy(queries_raw)  # [B, L, N]
            _, mu_k = self.fuzzy(keys_raw)     # [B, S, N]
            # mu_q @ mu_k^T 表示 query 与每个 key 在模糊规则空间的相似度。
            fuzzy_sim = torch.einsum("bln,bsn->bls", mu_q, mu_k).clamp_min(self.epsilon)
            fuzzy_bias = torch.log(fuzzy_sim).unsqueeze(1)  # [B, 1, L, S]
            head_scale = self.head_scale.view(1, H, 1, 1)
            scores = scores + self.alpha * head_scale * fuzzy_bias

        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A if self.output_attention else None


# =============================================================================
# Attention Wrapper Layer
# =============================================================================

class AttentionLayer(nn.Module):
    """多头注意力封装层，负责 Q/K/V 投影和输出投影。"""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 保存投影前 embedding，用于计算模糊隶属度。
        queries_raw = queries
        keys_raw = keys
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if hasattr(self.inner_attention, 'fuzzy'):
            out, attn = self.inner_attention(
                queries, keys, values, attn_mask,
                tau=tau, delta=delta,
                queries_raw=queries_raw, keys_raw=keys_raw
            )
        else:
            out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)

        return self.out_projection(out.view(B, L, -1)), attn


# =============================================================================
# Specialized Layers
# =============================================================================

class ReformerLayer(nn.Module):
    """LSH-based attention layer placeholder."""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = None  # Requires reformer-pytorch

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        pad_len = (self.bucket_size - L % self.bucket_size) % self.bucket_size
        if pad_len > 0:
            queries = F.pad(queries, (0, 0, 0, pad_len))
        out = self.attn(queries) if self.attn else queries
        return out[:, :L, :], None


class TwoStageAttentionLayer(nn.Module):
    """Cross-dimension two-stage attention for Crossformer."""
    
    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.seg_attn = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.cross_attn = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        
        self.dim_pos = nn.Parameter(torch.randn(1, seg_num, d_model) * 0.02)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        x = self.norm1(x + self.dropout(self.seg_attn(x, x, x, attn_mask)[0]))
        x = x + self.dim_pos
        x_t = x.transpose(0, 1)
        x = self.norm2((x_t + self.dropout(self.cross_attn(x_t, x_t, x_t, None)[0])).transpose(0, 1))
        return self.norm3(x + self.ffn(x))
