# -*- coding: utf-8 -*-
"""
Multiscale Decomposition (MSD) via MSTE-Weighted MSSA.

Decomposes multivariate series: Ξ(t) = Ξ_trend + Σ_k Ξ_seasonal^(k) + Ξ_residual

Key Algorithms:
    - MSTE: Multiscale Short-Term Entropy via coarse-grained Sample Entropy
    - MSSA: Multi-channel SSA with entropy-derived diagonal weighting
    - Weight mapping: w_i(t) = exp(α·z_i(t)) where z_i is z-normalized entropy
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import svd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MSDConfig:
    """Configuration parameters for MSD algorithm."""
    embedding_dim: int = 2              # m: SampEn embedding dimension
    tolerance_ratio: float = 0.2        # r = ratio × σ_local
    window_length: int = 52             # W: MSTE sliding window
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 13, 26])
    trajectory_window: int = 104        # p: Hankel matrix window
    n_trend_components: int = 1         # Leading SVD components for trend
    auto_seasonal: bool = True          # Auto-determine seasonal groups
    min_seasonal_groups: int = 2
    max_seasonal_groups: int = 3
    seasonal_threshold: float = 0.03    # Variance contribution threshold
    entropy_alpha: float = 0.5          # α: weight mapping sensitivity
    epsilon: float = 1e-10


class MultiscaleDecomposition:
    """
    MSD: Integrates MSTE weighting with MSSA decomposition.
    
    Pipeline:
        1. Compute MSTE for each channel via sliding window SampEn
        2. Z-normalize and map to weights: w_i(t) = exp(α·z_i(t))
        3. Apply diagonal weighting: H̃^i = W_i·H^i
        4. Concatenate and perform SVD: H̃ = [H̃^1,...,H̃^N]
        5. Group eigentriples and reconstruct components
    """
    
    def __init__(self, config: Optional[MSDConfig] = None):
        self.config = config or MSDConfig()
        self._cache = {}
    
    # =========================================================================
    # MSTE Computation
    # =========================================================================
    
    def _coarse_grain(self, x: np.ndarray, tau: int) -> np.ndarray:
        """Construct coarse-grained series x^(τ) at scale τ."""
        n = len(x) // tau
        return np.array([np.mean(x[j*tau:(j+1)*tau]) for j in range(n)])
    
    def _delay_embedding(self, x: np.ndarray, m: int) -> np.ndarray:
        """Construct m-dimensional delay embedding vectors."""
        T = len(x)
        return np.lib.stride_tricks.sliding_window_view(x, m)
    
    def _sample_entropy(self, x: np.ndarray, m: int, r: float) -> float:
        """
        Compute SampEn(m, r) = -ln(A/B)
        where A, B are match probabilities for (m+1) and m-dim embeddings.
        """
        eps = self.config.epsilon
        
        # m-dim embedding matches
        Γ_m = self._delay_embedding(x, m)
        D_m = np.max(np.abs(Γ_m[:, None] - Γ_m[None, :]), axis=2)
        np.fill_diagonal(D_m, np.inf)
        B = np.sum(D_m <= r) / (len(Γ_m) * (len(Γ_m) - 1) + eps)
        
        # (m+1)-dim embedding matches
        Γ_m1 = self._delay_embedding(x, m + 1)
        D_m1 = np.max(np.abs(Γ_m1[:, None] - Γ_m1[None, :]), axis=2)
        np.fill_diagonal(D_m1, np.inf)
        A = np.sum(D_m1 <= r) / (len(Γ_m1) * (len(Γ_m1) - 1) + eps)
        
        return -np.log((A + eps) / (B + eps))
    
    def compute_mste(self, x: np.ndarray) -> np.ndarray:
        """
        Compute MSTE: aggregate SampEn across scales τ ∈ S for sliding windows.
        """
        m, W = self.config.embedding_dim, self.config.window_length
        scales, T = self.config.scales, len(x)
        hop = W // 2  # 50% overlap
        
        n_windows = max(1, (T - W) // hop + 1)
        mste = np.zeros(n_windows)
        
        for w in range(n_windows):
            x_w = x[w*hop : min(w*hop + W, T)]
            if len(x_w) < m + 2:
                continue
            
            r = self.config.tolerance_ratio * np.std(x_w)
            entropies = []
            
            for τ in scales:
                x_τ = self._coarse_grain(x_w, τ)
                if len(x_τ) >= m + 2:
                    se = self._sample_entropy(x_τ, m, r)
                    if np.isfinite(se):
                        entropies.append(se)
            
            mste[w] = np.mean(entropies) if entropies else 0
        
        return mste
    
    def compute_entropy_weights(
        self, Xi: np.ndarray, target_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MSTE-derived adaptive weights.
        
        Formulation:
            z_i(t) = (STEM_i(t) - μ) / σ  (z-normalize)
            w_i(t) = exp(α·z_i(t))         (exponential mapping)
            w_i ← w_i / mean(w_i)          (rescale to unit mean)
        
        Returns:
            (channel_weights [N], time_weights [N, target_length])
        """
        N, T = Xi.shape
        α = self.config.entropy_alpha
        eps = self.config.epsilon
        
        # Compute MSTE per channel
        mste_list = [self.compute_mste(Xi[i]) for i in range(N)]
        
        if target_length is None:
            target_length = max(len(m) for m in mste_list)
        
        time_weights = np.zeros((N, target_length))
        
        for i in range(N):
            mste = mste_list[i]
            if len(mste) == 0:
                time_weights[i] = np.ones(target_length)
                continue
            
            # Interpolate to target_length
            if len(mste) != target_length:
                mste = np.interp(
                    np.linspace(0, 1, target_length),
                    np.linspace(0, 1, len(mste)), mste
                )
            
            # Z-normalize → exp mapping → unit mean
            z = (mste - np.mean(mste)) / (np.std(mste) + eps)
            w = np.exp(α * z)
            time_weights[i] = w / (np.mean(w) + eps)
        
        # Channel-level mean weights (L1 normalized)
        channel_weights = np.mean(time_weights, axis=1)
        channel_weights /= np.sum(channel_weights)
        
        return channel_weights, time_weights
    
    # =========================================================================
    # MSSA Decomposition
    # =========================================================================
    
    def _trajectory_matrix(self, x: np.ndarray, p: int) -> np.ndarray:
        """Construct Hankel trajectory matrix H ∈ R^{p×K}, K = T-p+1."""
        return np.lib.stride_tricks.sliding_window_view(x, p).T
    
    def _diagonal_averaging(self, X: np.ndarray, T: int) -> np.ndarray:
        """Hankelization: reconstruct series via anti-diagonal averaging."""
        p, K = X.shape
        result = np.zeros(T)
        counts = np.zeros(T)
        
        for i in range(p):
            for j in range(K):
                t = i + j
                if t < T:
                    result[t] += X[i, j]
                    counts[t] += 1
        
        return result / np.maximum(counts, 1)
    
    def _determine_seasonal_groups(self, S: np.ndarray, n_trend: int) -> int:
        """Auto-determine seasonal component groups via multi-criteria fusion."""
        S_rem = S[n_trend:]
        if len(S_rem) < 4:
            return self.config.min_seasonal_groups
        
        total_var = np.sum(S ** 2)
        var_ratios = S_rem ** 2 / total_var
        
        # Method 1: Elbow on log singular values
        log_S = np.log(S_rem + self.config.epsilon)
        elbow = np.argmax(np.abs(np.diff(np.diff(log_S)))) + 2
        
        # Method 2: Significant variance count
        sig_count = np.sum(var_ratios > self.config.seasonal_threshold)
        
        # Method 3: Ratio drop detection
        ratios = S_rem[1:] / (S_rem[:-1] + self.config.epsilon)
        drops = np.where(ratios < 0.5)[0]
        ratio_elbow = drops[0] + 1 if len(drops) > 0 else len(ratios) + 1
        
        # Fusion: median of candidates
        candidates = [
            min(elbow // 2, self.config.max_seasonal_groups),
            min(sig_count // 2, self.config.max_seasonal_groups),
            min(ratio_elbow, self.config.max_seasonal_groups)
        ]
        
        return int(np.clip(
            np.median(candidates),
            self.config.min_seasonal_groups,
            self.config.max_seasonal_groups
        ))
    
    def mssa_decompose(
        self, Xi: np.ndarray, 
        weights: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, int, np.ndarray]:
        """
        MSSA with entropy-weighted trajectory matrices.
        
        Steps:
            1. H̃^i = W_i·H^i where W_i = diag(w_i(1),...,w_i(K))
            2. H̃ = [H̃^1, ..., H̃^N]
            3. SVD: H̃ = UΣV^T
            4. Group and reconstruct trend/seasonal/residual
        
        Returns:
            (trend, seasonal_list, residual, n_seasonal, singular_values)
        """
        N, T = Xi.shape
        p = self.config.trajectory_window
        K = T - p + 1
        eps = self.config.epsilon
        
        # Get weights
        if weights is None:
            _, time_weights = self.compute_entropy_weights(Xi, target_length=K)
        else:
            _, time_weights = weights
            if time_weights.shape[1] != K:
                time_weights = np.array([
                    np.interp(np.linspace(0, 1, K),
                             np.linspace(0, 1, time_weights.shape[1]),
                             time_weights[i])
                    for i in range(N)
                ])
        
        # Construct weighted concatenated trajectory matrix
        H_list = []
        for i in range(N):
            H_i = self._trajectory_matrix(Xi[i], p)  # [p, K]
            H_tilde_i = H_i * time_weights[i][np.newaxis, :]  # Column scaling
            H_list.append(H_tilde_i)
        
        H_tilde = np.hstack(H_list)  # [p, N*K]
        
        # SVD
        U, S, Vt = svd(H_tilde, full_matrices=False)
        
        # Determine component groups
        n_trend = min(self.config.n_trend_components, len(S))
        n_seasonal = (self._determine_seasonal_groups(S, n_trend) 
                     if self.config.auto_seasonal else self.config.max_seasonal_groups)
        
        remaining = len(S) - n_trend
        comp_per_seasonal = max(1, remaining // (n_seasonal + 1))
        
        # Initialize outputs
        trend = np.zeros((N, T))
        seasonal = [np.zeros((N, T)) for _ in range(n_seasonal)]
        residual = np.zeros((N, T))
        
        # Reconstruct per channel
        for i in range(N):
            V_i = Vt[:, i*K:(i+1)*K]
            W_inv = 1.0 / (time_weights[i] + eps)
            
            # Trend
            M_trend = sum(S[j] * np.outer(U[:, j], V_i[j]) for j in range(n_trend))
            trend[i] = self._diagonal_averaging(M_trend * W_inv, T)
            
            # Seasonal components
            idx = n_trend
            for k in range(n_seasonal):
                M_k = sum(S[idx+j] * np.outer(U[:, idx+j], V_i[idx+j])
                         for j in range(comp_per_seasonal) if idx+j < len(S))
                seasonal[k][i] = self._diagonal_averaging(M_k * W_inv, T)
                idx += comp_per_seasonal
            
            # Residual
            M_res = sum(S[j] * np.outer(U[:, j], V_i[j]) for j in range(idx, len(S)))
            residual[i] = self._diagonal_averaging(M_res * W_inv, T)
        
        return trend, seasonal, residual, n_seasonal, S
    
    def decompose(self, Xi: np.ndarray) -> Dict:
        """
        Execute full MSD pipeline.
        
        Args:
            Xi: Multivariate series [N_channels, T_timesteps]
        
        Returns:
            Dict with keys: trend, seasonal, residual, channel_weights, 
                           time_weights, n_seasonal, singular_values
        """
        p = self.config.trajectory_window
        K = Xi.shape[1] - p + 1
        
        channel_weights, time_weights = self.compute_entropy_weights(Xi, K)
        trend, seasonal, residual, n_seasonal, S = self.mssa_decompose(
            Xi, weights=(channel_weights, time_weights)
        )
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'channel_weights': channel_weights,
            'time_weights': time_weights,
            'n_seasonal': n_seasonal,
            'singular_values': S
        }


# =============================================================================
# Utility Functions
# =============================================================================

def variance_decomposition(result: Dict, Xi: np.ndarray) -> Dict:
    """Compute variance contribution ratios for each component."""
    total_var = np.var(Xi) + 1e-10
    seasonal_vars = [np.var(s) for s in result['seasonal']]
    
    return {
        'trend': np.var(result['trend']) / total_var,
        'seasonal': [v / total_var for v in seasonal_vars],
        'residual': np.var(result['residual']) / total_var
    }


def reconstruction_mse(result: Dict, Xi: np.ndarray) -> float:
    """Compute reconstruction MSE."""
    Xi_hat = result['trend'] + sum(result['seasonal']) + result['residual']
    return float(np.mean((Xi - Xi_hat) ** 2))

