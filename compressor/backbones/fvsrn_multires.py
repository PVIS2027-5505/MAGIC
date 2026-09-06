from __future__ import annotations
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn

def _parse_resolutions(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]

def _resolve_features_per_level(opt: dict, K: int) -> List[int]:
    raw = opt.get("n_features_per_level", None)
    if raw is None or raw == "":
        return [int(opt["n_features"])] * K
    if isinstance(raw, str):
        per = [int(x) for x in raw.split(",")]
    else:
        per = [int(x) for x in raw]
    if len(per) != K:
        raise ValueError(
            f"n_features_per_level has {len(per)} entries but resolutions "
            f"has {K}; lengths must match")
    return per

class _ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.relu(self.linear(x))

class _PositionalEncoding(nn.Module):
    def __init__(self, num_terms: int, n_dims: int = 3):
        super().__init__()
        self.n_dims = n_dims
        self.L = num_terms
        L_terms = (torch.arange(num_terms, dtype=torch.float32)
                   .repeat_interleave(2 * n_dims))
        L_terms = torch.pow(2.0, L_terms) * math.pi
        self.register_buffer("L_terms", L_terms, persistent=False)

    def forward(self, x):
        rep = list(x.shape)
        rep[-1] = self.L * 2
        out = x.repeat(1, self.L * 2)
        out = out * self.L_terms
        out[..., 0::6] = torch.sin(out[..., 0::6])
        out[..., 1::6] = torch.sin(out[..., 1::6])
        out[..., 2::6] = torch.sin(out[..., 2::6])
        out[..., 3::6] = torch.cos(out[..., 3::6])
        out[..., 4::6] = torch.cos(out[..., 4::6])
        out[..., 5::6] = torch.cos(out[..., 5::6])
        return out

class FVSRNMultires(nn.Module):

    def __init__(self, opt: dict):
        super().__init__()

        resolutions = _parse_resolutions(opt["feature_grid_shape"])
        self.resolutions = list(resolutions)
        self.K = len(resolutions)

        self.n_features_per_level = _resolve_features_per_level(opt, self.K)

        self.n_features = max(self.n_features_per_level)
        self.full_shape = opt["full_shape"]
        self.num_pe = opt.get("num_positional_encoding_terms", 6)

        self.feature_grid_levels = nn.ParameterList()
        for R, F_l in zip(resolutions, self.n_features_per_level):
            p = nn.Parameter(
                torch.empty(1, F_l, R, R, R, dtype=torch.float32).uniform_(0.0, 1e-2),
                requires_grad=True,
            )
            self.feature_grid_levels.append(p)

        self.pe = _PositionalEncoding(self.num_pe, n_dims=3)
        in_dim = sum(self.n_features_per_level) + self.num_pe * 3 * 2
        layers = [_ReLULayer(in_dim, opt["nodes_per_layer"])]
        for _ in range(opt["n_layers"] - 1):
            layers.append(_ReLULayer(opt["nodes_per_layer"],
                                       opt["nodes_per_layer"]))
        layers.append(nn.Linear(opt["nodes_per_layer"], opt["n_outputs"],
                                  bias=False))
        self.decoder = nn.Sequential(*layers)

        self.register_buffer(
            "volume_min",
            torch.tensor([opt["data_min"]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([opt["data_max"]], dtype=torch.float32),
            persistent=False,
        )

    @property
    def feature_grids(self):
        R_max = max(self.resolutions)
        F_max = max(self.n_features_per_level)
        out = torch.zeros(self.K, F_max, R_max, R_max, R_max,
                           device=self.feature_grid_levels[0].device,
                           dtype=self.feature_grid_levels[0].dtype)
        for l, (R, F_l) in enumerate(zip(self.resolutions,
                                          self.n_features_per_level)):
            out[l, :F_l, :R, :R, :R] = self.feature_grid_levels[l].squeeze(0)
        return out

    @property
    def feature_grid_levels_view(self):
        return [p for p in self.feature_grid_levels]

    def get_model_parameters(self):
        return [{"params": list(self.feature_grid_levels.parameters())},
                 {"params": list(self.decoder.parameters())}]

    def get_transform_parameters(self):
        return [{"params": []}]

    def feature_density(self, x):
        return torch.ones(x.shape[0], device=x.device)

    def forward(self, x: torch.Tensor,
                feature_grids_override: torch.Tensor = None,
                feature_grid_levels_override: list = None) -> torch.Tensor:
        if feature_grid_levels_override is not None:
            grids = feature_grid_levels_override
        elif feature_grids_override is not None:

            grids = []
            for l, (R, F_l) in enumerate(zip(self.resolutions,
                                              self.n_features_per_level)):
                grids.append(feature_grids_override[l:l+1, :F_l, :R, :R, :R]
                              .contiguous())
        else:
            grids = list(self.feature_grid_levels)

        coords = x.view(1, 1, 1, x.shape[0], 3)
        feats = []
        for g in grids:
            sampled = Fn.grid_sample(g, coords, mode="bilinear",
                                       align_corners=True)
            feats.append(sampled.flatten(0, -2).permute(1, 0))
        spatial = torch.cat(feats, dim=1)
        pe = self.pe(x)
        out = self.decoder(torch.cat([pe, spatial], dim=1)).float()
        return out * (self.volume_max - self.volume_min) + self.volume_min
