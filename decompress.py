from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compressor.config import (
    CONTEXT_HIDDEN, CONTEXT_ACTIVATION, AC_DISTRIBUTION,
    INIT_SCALE, INIT_SIGMA,
)
from compressor import (
    build_backbone,
    MultiResLevelContextEntropyModel,
    decode_multires_level_ac,
    load_volume, make_coord_grid,
)

def _build_entropy_model(args, device, em_state):
    resolutions = em_state["resolutions_buf"].tolist()
    n_features = em_state["n_features_per_level_buf"].tolist()

    hidden = CONTEXT_HIDDEN
    context_window = int(em_state["context_window_buf"].view(-1)[0])
    context_start_level = int(em_state["context_start_level_buf"].view(-1)[0])

    em = MultiResLevelContextEntropyModel(
        resolutions=resolutions, n_features=n_features,
        hidden=hidden, distribution=AC_DISTRIBUTION,
        init_scale=INIT_SCALE, init_sigma=INIT_SIGMA,
        context_window=context_window,
        context_start_level=context_start_level,
        activation=CONTEXT_ACTIVATION,
        spatial_context=False,
    )
    return em.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="Path to ORIGINAL volume for PSNR ground-truth.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args_cli = parser.parse_args()

    save_dir = Path(args_cli.save_dir)
    metadata = json.loads((save_dir / "metadata.json").read_text())
    args = metadata["args"]
    opt = metadata["opt"]
    raw_size = int(metadata["raw_size_bytes"])

    gt_volume, full_shape = load_volume(args_cli.data)
    print(f"[decompress] gt={args_cli.data} shape={full_shape} "
          f"raw_size={raw_size/1024/1024:.1f} MB")

    model = build_backbone("fvsrn_multires", opt).to(args_cli.device)

    feat_path = save_dir / "features.b"
    if not feat_path.exists():
        raise FileNotFoundError(f"No features.b in {save_dir}")

    rest_path = save_dir / "rest.pt"
    rest = torch.load(rest_path, map_location=args_cli.device,
                       weights_only=True)
    em_state = {k[len("_em."):]: v for k, v in rest.items()
                if k.startswith("_em.")}

    model_state = {k: (v.float() if v.dtype == torch.float16 else v)
                   for k, v in rest.items() if not k.startswith("_em.")}

    entropy_model = _build_entropy_model(args, args_cli.device, em_state)
    entropy_model.load_state_dict(em_state, strict=False)

    msg = model.load_state_dict(model_state, strict=False)
    print(f"[decompress] loaded state_dict (missing={len(msg.missing_keys)})")

    level_shapes = [(1, F_l, R, R, R)
                    for R, F_l in zip(entropy_model.resolutions,
                                        entropy_model.n_features_per_level)]
    x_int_list = decode_multires_level_ac(
        str(feat_path), entropy_model, level_shapes,
        distribution=getattr(entropy_model, "distribution", "gaussian"),
    )
    fg_levels_recon = []
    for l, x_l in enumerate(x_int_list):
        s = entropy_model._level_scale(l)
        fg_levels_recon.append((x_l / s).to(args_cli.device))

    coords = make_coord_grid(full_shape, args_cli.device)
    model.eval()
    n = coords.shape[0]
    pred = torch.empty(n, 1, device=args_cli.device)
    batch = 2 ** 18
    with torch.no_grad():
        for i in range(0, n, batch):
            pred[i:i + batch] = model(
                coords[i:i + batch],
                feature_grid_levels_override=fg_levels_recon)
    gt = gt_volume.to(args_cli.device)
    pred_vol = pred.view(*full_shape).unsqueeze(0).unsqueeze(0)
    mse = F.mse_loss(pred_vol, gt).item()
    data_range = (gt.max() - gt.min()).item()
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-30)

    feat_bytes = feat_path.stat().st_size
    cr = raw_size / feat_bytes
    print(f"  features.b={feat_bytes/1024:.1f} KB  "
          f"CR(features-only)={cr:.1f}x  PSNR={psnr:.3f} dB")
    row = {
        "bitstream": "features.b",
        "lambda": args.get("entropy_lambda"),
        "features_bytes": feat_bytes,
        "cr_features_only": cr,
        "psnr_db": psnr,
        "mse": mse,
    }

    out_path = save_dir / "decompress_eval.json"
    with open(out_path, "w") as f:
        json.dump({"rows": [row], "raw_size_bytes": raw_size}, f, indent=2)
    print(f"\n[decompress] eval results saved to {out_path}")

if __name__ == "__main__":
    main()
