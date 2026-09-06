from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

def load_volume(path: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    p = Path(path)
    if p.suffix == ".nc":
        import netCDF4 as nc
        f = nc.Dataset(str(p))
        v = list(f.variables)[0]
        arr = np.array(f[v]).astype(np.float32)
        f.close()
    elif p.suffix == ".raw":

        parts = p.stem.split("_")
        dims = None
        for tok in parts:
            if "x" in tok and all(s.isdigit() for s in tok.split("x")):
                dims = tuple(int(s) for s in tok.split("x"))
                break
        if dims is None:
            raise ValueError(f"Cannot infer shape from {p.name}; "
                              "filename must contain DxHxW")

        dims = tuple(reversed(dims))
        arr = np.fromfile(p, dtype=np.float32).reshape(dims)
    else:
        raise ValueError(f"Unsupported file type: {p}")

    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max > a_min:
        arr = (2.0 * (arr - a_min) / (a_max - a_min) - 1.0).astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t, arr.shape

def make_coord_grid(shape: tuple[int, ...], device,
                     align_corners: bool = True) -> torch.Tensor:
    if align_corners:
        ranges = [torch.linspace(-1, 1, s, device=device) for s in shape]
    else:
        ranges = [torch.linspace(-1 + 1.0 / s, 1 - 1.0 / s, s, device=device)
                  for s in shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    coords = torch.stack(grids, dim=-1).reshape(-1, len(shape))
    return coords

@torch.no_grad()
def eval_full_volume_psnr(model, gt_volume: torch.Tensor,
                           device: str = "cuda:0",
                           batch: int = 2 ** 18) -> tuple[float, float]:
    full_shape = list(gt_volume.shape[2:])
    model.eval()
    coords = make_coord_grid(full_shape, device)
    n = coords.shape[0]
    pred = torch.empty(n, 1, device=device)
    for i in range(0, n, batch):
        pred[i:i + batch] = model(coords[i:i + batch])
    gt = gt_volume.to(device)
    pred_vol = pred.view(*full_shape).unsqueeze(0).unsqueeze(0)
    mse = F.mse_loss(pred_vol, gt).item()
    data_range = (gt.max() - gt.min()).item()
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-30)
    return psnr, mse
