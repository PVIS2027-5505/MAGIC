from __future__ import annotations
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as Fn

def _resolve_F_per_level(n_features: Union[int, List[int]], K: int) -> List[int]:
    if isinstance(n_features, int):
        return [n_features] * K
    per = [int(x) for x in n_features]
    if len(per) != K:
        raise ValueError(
            f"n_features list has {len(per)} entries but resolutions "
            f"has {K}; lengths must match")
    return per

def _ctx_lo(l: int, window) -> int:
    if window is None or window <= 0:
        return 0
    return max(0, l - window)

def _norm_coords(R: int, device) -> torch.Tensor:
    if R == 1:
        v = torch.zeros(1, device=device)
    else:
        v = torch.linspace(-1, 1, R, device=device)
    g = torch.stack(torch.meshgrid(v, v, v, indexing="ij"), dim=-1)
    return g

def _sample_levels_at(grids: List[torch.Tensor],
                       target_R: int,
                       device) -> List[torch.Tensor]:
    g = _norm_coords(target_R, device).view(1, target_R, target_R, target_R, 3)
    out = []
    for grid in grids:

        sampled = Fn.grid_sample(grid, g, mode="bilinear", align_corners=True)
        out.append(sampled)
    return out

def _make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown context-MLP activation '{name}' "
                     f"(expected 'relu' or 'gelu')")

class _LevelContextMLP(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 hidden: int = 32, init_log_sigma: float = math.log(4.0),
                 activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 2 * out_features)
        self.act = _make_activation(activation)

        nn.init.zeros_(self.fc2.weight)
        self.fc2.bias.data[:out_features].zero_()
        self.fc2.bias.data[out_features:].fill_(init_log_sigma)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(ctx)))

class MultiResLevelContextEntropyModel(nn.Module):

    def __init__(self, resolutions: List[int],
                 n_features: Union[int, List[int]],
                 hidden: int = 16,
                 distribution: str = "gaussian",
                 init_scale: float = 64.0, init_sigma: float = 4.0,
                 context_window: int = 0,
                 sample_num: int = 0,
                 context_start_level: int = 1,
                 activation: str = "relu",
                 spatial_context: bool = False,
                 likelihood_clamp_min: float = 1e-12):
        super().__init__()
        assert distribution in ("gaussian", "laplace")
        self.distribution = distribution
        self.activation = activation

        self.spatial_context = bool(spatial_context)

        self.likelihood_clamp_min = float(likelihood_clamp_min)
        self.resolutions = list(resolutions)
        self.K = len(resolutions)
        self.n_features_per_level = _resolve_F_per_level(n_features, self.K)
        self.n_features = max(self.n_features_per_level)

        self.context_window = context_window if context_window and context_window > 0 else None

        self.context_start_level = max(1, int(context_start_level))

        F_list = self.n_features_per_level

        self.log_scale = nn.ParameterList([
            nn.Parameter(torch.full((F_l,), math.log(init_scale)))
            for F_l in F_list
        ])
        self.anchor_mu = nn.ParameterList([
            nn.Parameter(torch.zeros(F_l)) for F_l in F_list
        ])
        self.anchor_log_sigma = nn.ParameterList([
            nn.Parameter(torch.full((F_l,), math.log(init_sigma)))
            for F_l in F_list
        ])

        self.context_mlps = nn.ModuleList()
        for l in range(self.K):
            if l < self.context_start_level:
                self.context_mlps.append(nn.Identity())
            else:
                in_dim = sum(F_list[_ctx_lo(l, self.context_window):l])
                self.context_mlps.append(_LevelContextMLP(
                    in_features=in_dim, out_features=F_list[l],
                    hidden=hidden,
                    init_log_sigma=math.log(init_sigma),
                    activation=activation,
                ))

        self.spatial_mlps = nn.ModuleList()
        if self.spatial_context:
            for l in range(self.K):
                ctx_dim = (sum(F_list[_ctx_lo(l, self.context_window):l])
                           if l >= self.context_start_level else 0)
                in_dim = ctx_dim + 6 * F_list[l]
                self.spatial_mlps.append(_LevelContextMLP(
                    in_features=in_dim, out_features=F_list[l],
                    hidden=hidden,
                    init_log_sigma=math.log(init_sigma),
                    activation=activation,
                ))

        self.register_buffer("resolutions_buf",
                              torch.tensor(self.resolutions, dtype=torch.long))
        self.register_buffer("n_features_per_level_buf",
                              torch.tensor(self.n_features_per_level,
                                            dtype=torch.long))

        self.register_buffer("context_window_buf",
                              torch.tensor([self.context_window or 0],
                                            dtype=torch.long))

        self.register_buffer("context_start_level_buf",
                              torch.tensor([self.context_start_level],
                                            dtype=torch.long))

        self.register_buffer("spatial_context_buf",
                              torch.tensor([1 if self.spatial_context else 0],
                                            dtype=torch.long))

        self.sample_num = int(sample_num) if sample_num and sample_num > 0 else 0
        per = [0] * self.K
        if self.sample_num > 0:
            P = [F_list[l] * self.resolutions[l] ** 3 for l in range(self.K)]

            lo = 0 if self.spatial_context else self.context_start_level
            P_tail = sum(P[lo:]) or 1
            for l in range(lo, self.K):
                m = int(round(self.sample_num * P[l] / P_tail))
                per[l] = max(1, min(self.resolutions[l] ** 3, m))
        self._sample_num_per_level = per

    def _device(self):
        return self.log_scale[0].device

    def _level_scale(self, l: int) -> torch.Tensor:
        return self.log_scale[l].exp().view(1, -1, 1, 1, 1)

    def _level_anchor_priors(self, l: int, target_shape):
        F_l = self.n_features_per_level[l]
        mu = self.anchor_mu[l].view(1, F_l, 1, 1, 1).expand(target_shape)
        sigma = (self.anchor_log_sigma[l].exp().clamp_min(1e-3)
                  .view(1, F_l, 1, 1, 1).expand(target_shape))
        return mu.contiguous(), sigma.contiguous()

    def _level_context_priors(self, l: int,
                                prev_decoded_levels: List[torch.Tensor]):
        assert l >= 1
        R_l = self.resolutions[l]
        device = self._device()
        ctx_src = prev_decoded_levels[_ctx_lo(l, self.context_window):]
        ctx_levels = _sample_levels_at(ctx_src, R_l, device)
        ctx = torch.cat(ctx_levels, dim=1)

        out = self.context_mlps[l](ctx.permute(0, 2, 3, 4, 1))
        out = out.permute(0, 4, 1, 2, 3)
        F_l = self.n_features_per_level[l]
        mu = out[:, :F_l]
        sigma = out[:, F_l:].exp().clamp_min(1e-3)
        return mu.contiguous(), sigma.contiguous()

    def _parity_grid(self, R: int, device) -> torch.Tensor:
        idx = torch.arange(R, device=device)
        i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
        return ((i + j + k) % 2 == 1)

    @staticmethod
    def _gather6_dense(v: torch.Tensor) -> torch.Tensor:
        R = v.shape[-1]
        p = Fn.pad(v, (1, 1, 1, 1, 1, 1))
        nb = [
            p[:, :, 0:R,     1:R + 1, 1:R + 1],
            p[:, :, 2:R + 2, 1:R + 1, 1:R + 1],
            p[:, :, 1:R + 1, 0:R,     1:R + 1],
            p[:, :, 1:R + 1, 2:R + 2, 1:R + 1],
            p[:, :, 1:R + 1, 1:R + 1, 0:R],
            p[:, :, 1:R + 1, 1:R + 1, 2:R + 2],
        ]
        return torch.cat(nb, dim=1)

    def _ctx_dense(self, l: int, prev_decoded_levels: List[torch.Tensor]):
        if l < self.context_start_level:
            return None
        R_l = self.resolutions[l]
        ctx_src = prev_decoded_levels[_ctx_lo(l, self.context_window):]
        ctx_levels = _sample_levels_at(ctx_src, R_l, self._device())
        return torch.cat(ctx_levels, dim=1)

    def _level_spatial_priors(self, l: int,
                               prev_decoded_levels: List[torch.Tensor],
                               level_vals: torch.Tensor):
        nb = self._gather6_dense(level_vals)
        ctx = self._ctx_dense(l, prev_decoded_levels)
        spin = nb if ctx is None else torch.cat([ctx, nb], dim=1)
        out = self.spatial_mlps[l](spin.permute(0, 2, 3, 4, 1))
        out = out.permute(0, 4, 1, 2, 3)
        F_l = self.n_features_per_level[l]
        mu = out[:, :F_l]
        sigma = out[:, F_l:].exp().clamp_min(1e-3)
        return mu.contiguous(), sigma.contiguous()

    def _level_priors_checkerboard(self, l: int,
                                    levels_x_scaled: List[torch.Tensor]):
        x_l = levels_x_scaled[l]
        R_l = self.resolutions[l]
        prev = [levels_x_scaled[i] / self._level_scale(i) for i in range(l)]
        if l < self.context_start_level:
            muA, sigA = self._level_anchor_priors(l, x_l.shape)
        else:
            muA, sigA = self._level_context_priors(l, prev)
        level_vals = x_l / self._level_scale(l)
        muB, sigB = self._level_spatial_priors(l, prev, level_vals)
        par = self._parity_grid(R_l, x_l.device).view(1, 1, R_l, R_l, R_l)
        mu = torch.where(par, muB, muA)
        sigma = torch.where(par, sigB, sigA)
        return mu, sigma

    def quantize_train_level(self, x_l: torch.Tensor, l: int):
        s = self._level_scale(l)
        x_scaled = x_l * s
        noise = torch.empty_like(x_scaled).uniform_(-0.5, 0.5)
        return (x_scaled + noise) / s, x_scaled + noise

    def quantize_eval_level(self, x_l: torch.Tensor, l: int):
        s = self._level_scale(l)
        x_scaled = x_l * s
        x_int = torch.round(x_scaled)
        return x_int / s, x_int

    def _dist(self, mu, sigma):
        if self.distribution == "laplace":
            return torch.distributions.Laplace(mu, sigma)
        return torch.distributions.Normal(mu, sigma)

    def rate_bits_all_levels(self, levels_x_scaled: List[torch.Tensor]):
        total_bits = []
        for l in range(self.K):
            x_l = levels_x_scaled[l]
            if self.spatial_context:
                mu, sigma = self._level_priors_checkerboard(l, levels_x_scaled)
            elif l < self.context_start_level:
                mu, sigma = self._level_anchor_priors(l, x_l.shape)
            else:

                prev = [levels_x_scaled[i] / self._level_scale(i)
                        for i in range(l)]
                mu, sigma = self._level_context_priors(l, prev)
            d = self._dist(mu, sigma)
            prob = (d.cdf(x_l + 0.5) - d.cdf(x_l - 0.5)).clamp_min(self.likelihood_clamp_min)
            total_bits.append((-torch.log2(prob)).sum())
        return torch.stack(total_bits).sum(), total_bits

    def _rate_bits_level_sampled(self, l: int,
                                  levels_x_scaled: List[torch.Tensor]):
        R_l = self.resolutions[l]
        F_l = self.n_features_per_level[l]
        device = self._device()
        x_scaled_l = levels_x_scaled[l]
        M = self._sample_num_per_level[l]

        di = torch.randint(0, R_l, (M,), device=device)
        hi = torch.randint(0, R_l, (M,), device=device)
        wi = torch.randint(0, R_l, (M,), device=device)
        vals = x_scaled_l[0][:, di, hi, wi].t()

        lin = (torch.linspace(-1, 1, R_l, device=device) if R_l > 1
               else torch.zeros(1, device=device))
        coords = torch.stack([lin[di], lin[hi], lin[wi]], dim=-1)
        grid = coords.view(1, M, 1, 1, 3)

        lo = _ctx_lo(l, self.context_window)
        ctx_parts = []
        for i in range(lo, l):
            grid_i = levels_x_scaled[i] / self._level_scale(i)
            sampled = Fn.grid_sample(grid_i, grid, mode="bilinear",
                                      align_corners=True)
            ctx_parts.append(
                sampled.view(self.n_features_per_level[i], M).t())
        ctx = torch.cat(ctx_parts, dim=-1)
        out = self.context_mlps[l](ctx)
        mu = out[:, :F_l]
        sigma = out[:, F_l:].exp().clamp_min(1e-3)
        d = self._dist(mu, sigma)
        prob = (d.cdf(vals + 0.5) - d.cdf(vals - 0.5)).clamp_min(self.likelihood_clamp_min)
        sampled_bits = (-torch.log2(prob)).sum()
        return sampled_bits * (float(R_l) ** 3 / M)

    def _neighbours6_at_points(self, level_vals: torch.Tensor,
                                di, hi, wi) -> torch.Tensor:
        R = level_vals.shape[-1]
        offs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        parts = []
        for dd, dh, dw in offs:
            d2, h2, w2 = di + dd, hi + dh, wi + dw
            valid = ((d2 >= 0) & (d2 < R) & (h2 >= 0) & (h2 < R)
                     & (w2 >= 0) & (w2 < R)).unsqueeze(-1)
            nbv = level_vals[:, d2.clamp(0, R - 1), h2.clamp(0, R - 1),
                             w2.clamp(0, R - 1)].t()
            parts.append(nbv * valid)
        return torch.cat(parts, dim=-1)

    def _rate_bits_level_sampled_spatial(self, l: int,
                                          levels_x_scaled: List[torch.Tensor]):
        R_l = self.resolutions[l]
        F_l = self.n_features_per_level[l]
        device = self._device()
        x_scaled_l = levels_x_scaled[l]
        M = self._sample_num_per_level[l]

        di = torch.randint(0, R_l, (M,), device=device)
        hi = torch.randint(0, R_l, (M,), device=device)
        wi = torch.randint(0, R_l, (M,), device=device)
        vals = x_scaled_l[0][:, di, hi, wi].t()
        par = ((di + hi + wi) % 2 == 1).unsqueeze(-1)

        lo = _ctx_lo(l, self.context_window)
        ctx_parts = []
        if l >= self.context_start_level:
            lin = (torch.linspace(-1, 1, R_l, device=device) if R_l > 1
                   else torch.zeros(1, device=device))
            grid = torch.stack([lin[di], lin[hi], lin[wi]],
                               dim=-1).view(1, M, 1, 1, 3)
            for i in range(lo, l):
                grid_i = levels_x_scaled[i] / self._level_scale(i)
                s = Fn.grid_sample(grid_i, grid, mode="bilinear",
                                   align_corners=True)
                ctx_parts.append(s.view(self.n_features_per_level[i], M).t())
        ctx = (torch.cat(ctx_parts, dim=-1) if ctx_parts
               else vals.new_zeros((M, 0)))

        if l < self.context_start_level:
            muA = self.anchor_mu[l].view(1, F_l).expand(M, F_l)
            sigA = (self.anchor_log_sigma[l].exp().clamp_min(1e-3)
                    .view(1, F_l).expand(M, F_l))
        else:
            outA = self.context_mlps[l](ctx)
            muA = outA[:, :F_l]
            sigA = outA[:, F_l:].exp().clamp_min(1e-3)

        level_vals = (x_scaled_l / self._level_scale(l))[0]
        nb = self._neighbours6_at_points(level_vals, di, hi, wi)
        spin = nb if ctx.shape[-1] == 0 else torch.cat([ctx, nb], dim=-1)
        outB = self.spatial_mlps[l](spin)
        muB = outB[:, :F_l]
        sigB = outB[:, F_l:].exp().clamp_min(1e-3)

        mu = torch.where(par, muB, muA)
        sigma = torch.where(par, sigB, sigA)
        d = self._dist(mu, sigma)
        prob = (d.cdf(vals + 0.5) - d.cdf(vals - 0.5)).clamp_min(self.likelihood_clamp_min)
        return (-torch.log2(prob)).sum() * (float(R_l) ** 3 / M)

    def rate_bits_all_levels_sampled(self, levels_x_scaled: List[torch.Tensor]):
        total_bits = []
        for l in range(self.K):
            if self.spatial_context:
                if self._sample_num_per_level[l] > 0:
                    total_bits.append(
                        self._rate_bits_level_sampled_spatial(l, levels_x_scaled))
                else:
                    x_l = levels_x_scaled[l]
                    mu, sigma = self._level_priors_checkerboard(l, levels_x_scaled)
                    d = self._dist(mu, sigma)
                    prob = (d.cdf(x_l + 0.5) - d.cdf(x_l - 0.5)).clamp_min(self.likelihood_clamp_min)
                    total_bits.append((-torch.log2(prob)).sum())
            elif l < self.context_start_level:
                x_l = levels_x_scaled[l]
                mu, sigma = self._level_anchor_priors(l, x_l.shape)
                d = self._dist(mu, sigma)
                prob = (d.cdf(x_l + 0.5) - d.cdf(x_l - 0.5)).clamp_min(self.likelihood_clamp_min)
                total_bits.append((-torch.log2(prob)).sum())
            else:
                total_bits.append(
                    self._rate_bits_level_sampled(l, levels_x_scaled))
        return torch.stack(total_bits).sum(), total_bits

    def quantize_and_rate_all(self, levels_x: List[torch.Tensor]):
        x_q_list = []
        x_scaled_list = []
        for l, x_l in enumerate(levels_x):
            x_q, x_scaled = self.quantize_train_level(x_l, l)
            x_q_list.append(x_q)
            x_scaled_list.append(x_scaled)
        if self.sample_num > 0:
            rate_total, rate_per_level = \
                self.rate_bits_all_levels_sampled(x_scaled_list)
        else:
            rate_total, rate_per_level = \
                self.rate_bits_all_levels(x_scaled_list)
        return x_q_list, rate_total, rate_per_level

    def quantize_eval_all(self, levels_x: List[torch.Tensor]):
        x_q_list = []
        x_int_list = []
        for l, x_l in enumerate(levels_x):
            x_q, x_int = self.quantize_eval_level(x_l, l)
            x_q_list.append(x_q)
            x_int_list.append(x_int)
        return x_q_list, x_int_list
