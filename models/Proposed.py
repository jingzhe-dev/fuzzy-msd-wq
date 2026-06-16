# -*- coding: utf-8 -*-
"""Fuzzy logic-enhanced Transformer for multivariate forecasting.

Architecture:
    Input -> PatchEmbedding -> Fuzzy Transformer Encoder -> Defuzzification -> Output

Core modules:
    - AdaptiveFuzzyMembership;
    - LogicalFuzzyAttention;
    - DifferentiableDefuzzification;
    - Residual Forecast Anchor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, LogicalFuzzyAttention
from layers.Embed import PatchEmbedding


class StationEmbedding(nn.Module):
    """Station embedding for multi-site forecasting."""
    
    def __init__(self, num_stations: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_stations, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, station_ids=None):
        B, L, D = x.shape
        if station_ids is None:
            station_ids = torch.zeros(B, dtype=torch.long, device=x.device)
        e_s = self.embedding(station_ids).unsqueeze(1).expand(-1, L, -1)
        return self.dropout(self.proj(torch.cat([x, e_s], dim=-1)))


class DifferentiableDefuzzification(nn.Module):
    """Differentiable weighted-centroid defuzzification."""
    
    def __init__(
        self, d_model: int, pred_len: int, N: int, 
        x_min: float, x_max: float, dropout: float = 0.1
    ):
        super().__init__()
        self.N, self.pred_len = N, pred_len
        self.x_min, self.x_max = x_min, x_max
        
        # 规则权重投影：每个预测步会得到 N 个模糊规则权重。
        self.weight_proj = nn.Linear(d_model, N)
        
        # 模糊规则中心初始化在归一化区间内均匀分布。
        delta = (x_max - x_min) / N
        self.centers = nn.Parameter(
            torch.linspace(x_min + delta/2, x_max - delta/2, N)
        )
        
        # 动态中心偏移：让规则中心具备小范围自适应能力。
        self.center_offset = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, N),
            nn.Tanh()
        )
        # 限制偏移幅度，避免解模糊中心漂移过大。
        self.offset_scale = nn.Parameter(torch.tensor((x_max - x_min) * 0.2 / N))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        """
        Args: 
            z: [B, L, d_model] - Encoder output
        Returns: 
            predictions: [B, L, 1]
            weights: [B, L, N]
        """
        # Dropout 放在 softmax 前，保证规则权重归一化后仍然和为 1。
        logits = self.dropout(self.weight_proj(z))
        w = F.softmax(logits, dim=-1)
        
        # 为每个样本、每个预测步生成动态规则中心。
        offset = self.center_offset(z) * self.offset_scale
        c_dynamic = self.centers.view(1, 1, -1) + offset  # [B, L, N]
        
        # 中心值限制在归一化预测区间内。
        c_dynamic = torch.clamp(c_dynamic, self.x_min, self.x_max)
        
        # 加权质心解模糊。
        x_pred = torch.sum(w * c_dynamic, dim=-1, keepdim=True)
        return x_pred, w


class FlattenHead(nn.Module):
    """Standard flatten-project prediction head."""
    
    def __init__(self, n_vars: int, nf: int, pred_len: int, dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """Multivariate forecasting model.

    Pipeline:
        1. 输入先做 instance normalization，降低不同指标量纲影响；
        2. PatchEmbedding 把短历史窗口切成 token；
        3. Encoder 内部使用 LogicalFuzzyAttention；
        4. 预测头可选择普通线性头或可微解模糊头；
        5. residual anchor 在预测残差外保留最近观测基线。
    """
    
    def __init__(self, configs, patch_len: int = 16, stride: int = 8):
        super().__init__()
        
        self.task_name = configs.task_name
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.x_min, self.x_max = configs.x_min, configs.x_max
        self.N, self.alpha = configs.fuzzy_N, configs.fuzzy_alpha
        self.d_model = configs.d_model
        patch_len = getattr(configs, 'patch_len', patch_len)
        stride = getattr(configs, 'stride', stride)
        self.patch_len, self.stride = patch_len, stride
        self.use_residual_forecast = getattr(configs, 'use_residual_forecast', True)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(getattr(configs, 'residual_scale_init', 0.1)))
        )
        # residual anchor 用于预测相对最近观测值的变化量。
        self.use_local_residual_anchor = getattr(configs, 'use_local_residual_anchor', True)
        self.local_residual = nn.Linear(self.seq_len, self.pred_len)
        nn.init.zeros_(self.local_residual.weight)
        nn.init.zeros_(self.local_residual.bias)
        
        self.use_station_embedding = getattr(configs, 'use_station_embedding', False)
        self.use_defuzzification = getattr(configs, 'use_defuzzification', False)
        
        if self.use_station_embedding:
            self.station_embedding = StationEmbedding(
                getattr(configs, 'num_stations', 200),
                configs.d_model, configs.dropout
            )
        
        # Patch embedding：把 [B, C, L] 转成 Transformer token。
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, stride, configs.dropout
        )

        # Transformer encoder：核心 attention 换成 LogicalFuzzyAttention。
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    LogicalFuzzyAttention(
                        x_min=self.x_min, x_max=self.x_max,
                        N=self.N, alpha=self.alpha,
                        d_model=configs.d_model, n_heads=configs.n_heads,
                        mask_flag=False, attention_dropout=configs.dropout,
                        output_attention=configs.output_attention,
                        device=configs.devices, use_adaptive_membership=True,
                        membership_beta=getattr(configs, 'membership_beta', 0.5)
                    ),
                    configs.d_model, configs.n_heads
                ),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 预测头：use_defuzzification=True 时走可微解模糊，否则走线性头。
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        if self.use_defuzzification:
            self.flatten_proj = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.head_nf, configs.pred_len * configs.d_model),
                nn.LayerNorm(configs.pred_len * configs.d_model),
                nn.GELU()
            )
            self.defuzzification = DifferentiableDefuzzification(
                configs.d_model, configs.pred_len, self.N,
                self.x_min, self.x_max, configs.dropout
            )
        else:
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.pred_len, configs.dropout
            )

    def _normalize(self, x):
        """Instance normalization: x̃ = (x - μ) / σ"""
        mu = x.mean(dim=1, keepdim=True).detach()
        sigma = torch.sqrt(torch.var(x - mu, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - mu) / sigma, mu, sigma
    
    def _denormalize(self, x, mu, sigma):
        """Reverse instance normalization."""
        return x * sigma[:, 0, :].unsqueeze(1) + mu[:, 0, :].unsqueeze(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, station_ids=None):
        """生成 pred_len 步预测。"""
        # 先归一化，再让模型学习归一化空间中的变化模式。
        x_enc, mu, sigma = self._normalize(x_enc)
        # 最近一个观测值作为预测基线；局部残差分支学习未来短期偏移。
        residual_base = x_enc[:, -1:, :].expand(-1, self.pred_len, -1)
        if self.use_local_residual_anchor:
            local_delta = self.local_residual(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
            residual_base = residual_base + local_delta

        # Patch embedding: [B, L, C] -> [B*C, n_patches, d_model]
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        if self.use_station_embedding and station_ids is not None:
            enc_out = self.station_embedding(
                enc_out, station_ids.repeat_interleave(n_vars)
            )

        # 模糊增强 Transformer 编码。
        enc_out, _ = self.encoder(enc_out)

        # Reshape: [B*C, n_patches, d] -> [B, C, d, n_patches]
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # 解码预测：每个变量独立经过预测头，再拼回多变量输出。
        if self.use_defuzzification:
            B, C = enc_out.shape[0], enc_out.shape[1]
            enc_flat = self.flatten_proj(enc_out).view(B, C, self.pred_len, self.d_model)
            dec_out = torch.cat([
                self.defuzzification(enc_flat[:, v])[0] for v in range(C)
            ], dim=-1)
        else:
            dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_residual_forecast:
            # 模型输出解释为残差，乘以可学习尺度后加回预测基线。
            dec_out = residual_base + self.residual_scale * dec_out

        return self._denormalize(dec_out, mu, sigma)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return None
