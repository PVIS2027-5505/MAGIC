from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressor import (
    build_backbone,
    decode_multires_level_ac,
    load_volume, make_coord_grid, eval_full_volume_psnr,
)

_LAM_SUFFIX_RE = re.compile(r"^features_l([0-9eE+\-.]+)\.b$")

from decompress import _build_entropy_model

def _discover_bitstreams(save_dir: Path) -> dict:
    out: dict = {}
    for p in sorted(save_dir.glob("features*.b")):
        if p.name == "features.b":
            out[None] = p
            continue
        m = _LAM_SUFFIX_RE.match(p.name)
        if not m:
            continue
        try:
            out[float(m.group(1))] = p
        except ValueError:
            continue
    return out

class INRModel(nn.Module):

    def __init__(self,
                 backbone: nn.Module,
                 fg_levels: list,
                 full_shape: tuple,
                 data_min: float,
                 data_max: float,
                 data_device: str,
                 mode: str = "mlp"):
        super().__init__()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._fg_levels = list(fg_levels)
        self.full_shape = tuple(int(s) for s in full_shape)
        self.shape = self.full_shape

        self._min = torch.tensor([float(data_min)])
        self._max = torch.tensor([float(data_max)])
        self.data_device = data_device
        self._mode = "mlp"
        self._decoded = None
        self._grad = None
        if mode == "decoded":
            self.set_query_mode("decoded")

    def min(self) -> torch.Tensor:
        return self._min

    def max(self) -> torch.Tensor:
        return self._max

    def get_volume_extents(self) -> tuple:
        return self.full_shape

    def set_fg_levels(self, new_levels: list) -> None:
        self._fg_levels = list(new_levels)
        if self._decoded is not None:
            self._decoded = None
            if self._mode == "decoded":
                self.set_query_mode("decoded")

    def set_minmax(self, lo: float, hi: float) -> None:
        self._min = torch.tensor([float(lo)])
        self._max = torch.tensor([float(hi)])

    @torch.no_grad()
    def set_query_mode(self, mode: str) -> None:
        assert mode in ("mlp", "decoded")
        if mode == "decoded" and self._decoded is None:
            self._decoded = self._decode_full_volume()

            self._min = self._decoded.flatten().min().detach().cpu().reshape(1)
            self._max = self._decoded.flatten().max().detach().cpu().reshape(1)
        self._mode = mode

    @torch.no_grad()
    def _decode_full_volume(self, batch: int = 2 ** 18) -> torch.Tensor:
        D, H, W = self.full_shape
        coords = make_coord_grid((D, H, W), self.data_device)
        n = coords.shape[0]
        pred = torch.empty(n, 1, device=self.data_device)
        for i in range(0, n, batch):
            pred[i:i + batch] = self.backbone(
                coords[i:i + batch],
                feature_grid_levels_override=self._fg_levels)
        vol = pred.view(D, H, W).unsqueeze(0).unsqueeze(0)
        return vol

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_device = x.device
        x = x.to(self.data_device)
        if self._mode == "decoded":

            x_flip = x.flip(-1)
            y = F.grid_sample(self._decoded,
                              x_flip.reshape(([1] * x_flip.shape[-1])
                                              + list(x_flip.shape)),
                              mode="bilinear",
                              align_corners=True).squeeze().unsqueeze(1)
        else:
            y = self.backbone(x, feature_grid_levels_override=self._fg_levels)
        return y.to(x_device)

    @torch.no_grad()
    def _ensure_grad(self):
        if self._grad is not None:
            return
        assert self._mode == "decoded", (
            "_ensure_grad() is the decoded-mode gradient. It used to silently "
            "flip the model into decoded mode when called from mlp mode, which "
            "meant --inr_mode mlp was a no-op the moment shading was on (the "
            "first gradient() call downgraded it) -- so 'network-level' timings "
            "were really measuring decode-once + grid_sample, the same code "
            "path as the GT render. mlp mode now finite-differences through the "
            "backbone instead; see gradient().")
        vol = self._decoded

        pad = (1, 1, 1, 1, 1, 1)
        vp = F.pad(vol, pad, mode="replicate")
        g0 = (vp[:, :, 2:, 1:-1, 1:-1] - vp[:, :, :-2, 1:-1, 1:-1]) * 0.5
        g1 = (vp[:, :, 1:-1, 2:, 1:-1] - vp[:, :, 1:-1, :-2, 1:-1]) * 0.5
        g2 = (vp[:, :, 1:-1, 1:-1, 2:] - vp[:, :, 1:-1, 1:-1, :-2]) * 0.5

        self._grad = torch.cat([g0, g1, g2], dim=1)

    @torch.no_grad()
    def _gradient_fd(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x = x.to(self.data_device)
        N = x.shape[0]
        offs = torch.zeros(6, 3, device=self.data_device, dtype=x.dtype)
        offs[0, 0], offs[1, 0] = eps, -eps
        offs[2, 1], offs[3, 1] = eps, -eps
        offs[4, 2], offs[5, 2] = eps, -eps
        pts = (x[None] + offs[:, None, :]).clamp_(-1.0, 1.0).reshape(-1, 3)
        v = self.backbone(
            pts, feature_grid_levels_override=self._fg_levels).view(6, N)
        return torch.stack([(v[0] - v[1]) / (2 * eps),
                            (v[2] - v[3]) / (2 * eps),
                            (v[4] - v[5]) / (2 * eps)], dim=-1)

    @torch.no_grad()
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode != "decoded":
            return self._gradient_fd(x).to(x.device)
        self._ensure_grad()
        x_device = x.device
        x = x.to(self.data_device)
        x_flip = x.flip(-1)
        g = F.grid_sample(self._grad,
                           x_flip.reshape(([1] * x_flip.shape[-1])
                                           + list(x_flip.shape)),
                           mode="bilinear",
                           align_corners=True)
        g = g.squeeze(-2).squeeze(-2).squeeze(0).T

        return g.to(x_device)

def _split_rest(rest: dict, device: str) -> tuple[dict, dict]:
    em_state, model_state = {}, {}
    for k, v in rest.items():
        if k.startswith("_em."):
            em_state[k[len("_em."):]] = v
        else:

            model_state[k] = v.float() if v.dtype == torch.float16 else v
    return em_state, model_state

def _validate_load_msg(msg, allow_missing_prefix: str = "feature_grid_levels"):
    bad_missing = [k for k in msg.missing_keys
                   if not k.startswith(allow_missing_prefix)]
    if bad_missing or msg.unexpected_keys:
        raise RuntimeError(
            "load_state_dict mismatch:\n"
            f"  unexpected = {msg.unexpected_keys}\n"
            f"  bad_missing = {bad_missing}\n"
            "(missing keys other than feature_grid_levels.* indicate a "
            "checkpoint/backbone version mismatch.)")

def load_inr(save_dir,
             device: str = "cuda:0",
             data_device: Optional[str] = None,
             mode: str = "mlp") -> INRModel:
    save_dir = Path(save_dir)
    if data_device is None:
        data_device = device

    meta_path = save_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {save_dir}")
    metadata = json.loads(meta_path.read_text())
    args = metadata["args"]
    opt = metadata["opt"]

    bitstreams = _discover_bitstreams(save_dir)
    if not bitstreams:
        raise FileNotFoundError(f"No features*.b in {save_dir}")
    chosen_path = bitstreams[None]

    legacy = list(save_dir.glob("rest_l*.pt"))
    if legacy and not (save_dir / "rest.pt").exists():
        raise NotImplementedError(
            f"This save_dir uses legacy per-λ rest files ({[p.name for p in legacy]}). "
            "The renderer only supports the unified `rest.pt` format.")

    backbone = build_backbone("fvsrn_multires", opt).to(device)

    rest_path = save_dir / "rest.pt"
    if not rest_path.exists():
        raise FileNotFoundError(f"rest.pt not found in {save_dir}")
    rest = torch.load(rest_path, map_location=device, weights_only=True)
    em_state, model_state = _split_rest(rest, device)

    entropy_model = _build_entropy_model(args, device, em_state)
    entropy_model.load_state_dict(em_state, strict=False)

    msg = backbone.load_state_dict(model_state, strict=False)
    _validate_load_msg(msg, allow_missing_prefix="feature_grid_levels")

    level_shapes = [(1, F_l, R, R, R)
                    for R, F_l in zip(entropy_model.resolutions,
                                       entropy_model.n_features_per_level)]
    x_int_list = decode_multires_level_ac(
        str(chosen_path), entropy_model, level_shapes,
        distribution=getattr(entropy_model, "distribution", "gaussian"),
    )
    fg_levels = []
    for l, x_l in enumerate(x_int_list):
        fg_levels.append((x_l / entropy_model._level_scale(l)).to(device))

    model = INRModel(
        backbone=backbone,
        fg_levels=fg_levels,
        full_shape=tuple(opt["full_shape"]),
        data_min=float(opt["data_min"]),
        data_max=float(opt["data_max"]),
        data_device=data_device,
        mode=mode,
    )

    print(f"[inr_loader] loaded {save_dir.name}: "
          f"full_shape={model.full_shape}  "
          f"data_range=[{model._min.item():.4f},{model._max.item():.4f}]  "
          f"bitstream={chosen_path.name}")
    return model
