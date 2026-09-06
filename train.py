from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from compressor.config import (
    N_LAYERS, NODES_PER_LAYER, CONTEXT_WINDOW, CONTEXT_START_LEVEL,
    CONTEXT_HIDDEN, CONTEXT_ACTIVATION, AC_DISTRIBUTION, INIT_SCALE,
    INIT_SIGMA, LIKELIHOOD_CLAMP_MIN, RATE_SAMPLE_NUM, BLOCK_SIZE,
    N_BINS, EPS_H, WARMUP_ITERS, SCHEDULE_K, COVERAGE_PORTION,
    CANDIDATE_MULTIPLIER, ALPHA,
)
from compressor import (
    build_backbone,
    MultiResLevelContextEntropyModel,
    encode_multires_level_ac,
    load_volume, make_coord_grid,
)

BACKBONE = "fvsrn_multires"

def build_options(args, full_shape, data_min, data_max) -> dict:
    return {
        "n_features": None,

        "n_features_per_level": args.n_features_per_level,
        "feature_grid_shape": args.feature_grid_shape,
        "n_layers": N_LAYERS,
        "nodes_per_layer": NODES_PER_LAYER,
        "n_outputs": 1,
        "num_positional_encoding_terms": 0,

        "full_shape": tuple(full_shape),
        "data_min": float(data_min),
        "data_max": float(data_max),
    }

def patched_forward(model, entropy_model, x, training, bypass=False):
    if bypass:
        return model(x), x.new_zeros(())
    levels_x = list(model.feature_grid_levels_view)
    if training:
        x_q_list, bits, _ = entropy_model.quantize_and_rate_all(levels_x)
    else:
        x_q_list, x_int_list = entropy_model.quantize_eval_all(levels_x)
        bits, _ = entropy_model.rate_bits_all_levels(x_int_list)
    y = model(x, feature_grid_levels_override=x_q_list)
    return y, bits

def flat_indices_to_coords(idx: torch.Tensor, shape: tuple[int, ...],
                           device) -> torch.Tensor:
    rem = idx.long()
    unravel = []
    for s in reversed(shape):
        unravel.append(rem % int(s))
        rem = torch.div(rem, int(s), rounding_mode="floor")
    coords = []
    for pos, s in zip(reversed(unravel), shape):
        if int(s) == 1:
            coords.append(torch.zeros_like(pos, dtype=torch.float32))
        else:
            coords.append(pos.to(torch.float32) * (2.0 / (int(s) - 1)) - 1.0)
    return torch.stack(coords, dim=-1).to(device)

def adaptive_scale_at(it: int, total_iters: int, k: float = 10.0) -> float:
    import math
    if total_iters <= 0 or k <= 0:
        return 1.0
    t = it / total_iters
    return 1.0 / (1.0 + math.exp(-k * (t - 0.5)))

@torch.no_grad()
def kd_block_entropy(gt_flat: torch.Tensor, full_shape: tuple[int, ...],
                      block_size, n_bins: int = 16
                      ) -> tuple[torch.Tensor, tuple[int, int, int],
                                  tuple[int, int, int]]:
    D, H, W = (int(s) for s in full_shape)
    if isinstance(block_size, (tuple, list)):
        Bd_edge, Bh_edge, Bw_edge = (int(b) for b in block_size)
    else:
        Bd_edge = Bh_edge = Bw_edge = int(block_size)
    assert D % Bd_edge == 0 and H % Bh_edge == 0 and W % Bw_edge == 0, \
        f"block_size={block_size} must divide {full_shape}"
    Bd, Bh, Bw = D // Bd_edge, H // Bh_edge, W // Bw_edge

    V = gt_flat.view(D, H, W)
    blocks = (V.view(Bd, Bd_edge, Bh, Bh_edge, Bw, Bw_edge)
                .permute(0, 2, 4, 1, 3, 5).contiguous()
                .view(-1, Bd_edge * Bh_edge * Bw_edge))
    N = blocks.shape[0]

    entropy = torch.empty(N, device=blocks.device)
    vox_per_block = blocks.shape[1]

    chunk = max(1, 32_000_000 // max(1, vox_per_block))
    for i in range(0, N, chunk):
        j = min(N, i + chunk)
        b = blocks[i:j]
        vmin = b.min(dim=1, keepdim=True).values
        vmax = b.max(dim=1, keepdim=True).values
        rng = (vmax - vmin).clamp_min(1e-12)

        bin_idx = ((b - vmin) / rng * n_bins).clamp(0, n_bins - 1).to(torch.long)
        hist = torch.zeros(b.shape[0], n_bins, device=b.device)
        hist.scatter_add_(1, bin_idx, torch.ones_like(b))
        del bin_idx
        p = hist / hist.sum(dim=1, keepdim=True).clamp_min(1)
        log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
        e = -(p * log_p).sum(dim=1)
        const_mask = (vmax.squeeze(-1) - vmin.squeeze(-1)) < 1e-12
        e = torch.where(const_mask, torch.zeros_like(e), e)
        entropy[i:j] = e
        del hist, p, log_p, e
    return entropy, (Bd, Bh, Bw), (Bd_edge, Bh_edge, Bw_edge)

class ActiveVoxelSampler:

    def __init__(self, full_shape, gt_volume, args):
        self.shape = tuple(int(s) for s in full_shape)
        self.n_voxels = int(np.prod(self.shape))
        self.flat_gt = gt_volume.view(-1)
        self.args = args
        self.device = gt_volume.device
        self.pool_idx = None
        self.pool_scores = None
        self.last_refresh = -1

        self.kd_prob = None
        self.kd_block_dims = None
        self.kd_block_edges = None
        self.kd_block_E_flat = None
        self._init_kd_entropy()

    def _init_kd_entropy(self):
        block_size = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        E, dims, edges = kd_block_entropy(
            self.flat_gt, self.shape,
            block_size=block_size,
            n_bins=N_BINS,
        )
        self.kd_block_dims = dims
        self.kd_block_edges = edges

        self.kd_block_E_flat = E.detach()

        self.kd_block_size = int(round((edges[0]*edges[1]*edges[2])**(1/3)))
        prob = E / E.sum().clamp_min(1e-12)
        eps = EPS_H / E.numel()
        prob = prob + eps
        prob = prob / prob.sum()
        self.kd_prob = prob

        cdf = torch.cumsum(prob, dim=0)
        cdf[-1] = 1.0
        self.kd_cdf = cdf
        print(f"[kd_entropy] {E.numel()} blocks of {edges[0]}x{edges[1]}x{edges[2]}  "
              f"E_max={E.max().item():.3f}  E_mean={E.mean().item():.3f}  "
              f"frac_const={(E == 0).float().mean().item():.3f}")

    @torch.no_grad()
    def _l4_bitrate_proxy(self, model, entropy_model) -> torch.Tensor:
        levels = list(model.feature_grid_levels_view)
        x_l = levels[-1]
        l = len(levels) - 1
        scale = entropy_model._level_scale(l)
        x_int_abs = (x_l * scale).round().abs()
        return x_int_abs.sum(dim=1).squeeze(0)

    def _blended_topk(self, residuals: torch.Tensor,
                       voxel_idx: torch.Tensor, n_res: int,
                       model=None, entropy_model=None) -> torch.Tensor:
        alpha = ALPHA
        if alpha <= 0 or model is None or entropy_model is None:
            return torch.topk(residuals, n_res).indices

        proxy = self._l4_bitrate_proxy(model, entropy_model)
        R = proxy.shape[0]
        D, H, W = self.shape
        z = voxel_idx % W
        y = (voxel_idx // W) % H
        x = voxel_idx // (W * H)
        gi = ((x * R) // D).clamp_max(R - 1)
        gj = ((y * R) // H).clamp_max(R - 1)
        gk = ((z * R) // W).clamp_max(R - 1)
        rate = proxy[gi, gj, gk]

        res_n = residuals / residuals.mean().clamp_min(1e-8)
        rate_n = (rate / rate.mean().clamp_min(1e-8)).clamp_min(1e-6)
        score = res_n / rate_n.pow(alpha)
        return torch.topk(score, n_res).indices

    def _kd_sample_indices(self, n: int, it: int = None):
        Bd, Bh, Bw = self.kd_block_dims
        Bd_edge, Bh_edge, Bw_edge = self.kd_block_edges
        D, H, W = self.shape
        K = self.kd_cdf.numel()

        mix = getattr(self.args, "entropy_mix", None)
        k = SCHEDULE_K
        if mix is not None and mix >= 0:

            s = float(mix)
        elif it is None or k <= 0:

            s = 1.0
        else:
            s = adaptive_scale_at(it, int(self.args.iterations), k)
        n_adapt = int(round(n * s))
        n_unif = n - n_adapt

        block_parts = []
        if n_adapt > 0:
            u = torch.rand(n_adapt, device=self.device)
            block_parts.append(
                torch.searchsorted(self.kd_cdf, u).clamp_max(K - 1))
        if n_unif > 0:
            block_parts.append(torch.randint(0, K, (n_unif,),
                                              device=self.device))
        block_idx = (block_parts[0] if len(block_parts) == 1
                     else torch.cat(block_parts, dim=0))

        bz = block_idx % Bw
        by = (block_idx // Bw) % Bh
        bx = block_idx // (Bw * Bh)
        lx = torch.randint(0, Bd_edge, (n,), device=self.device)
        ly = torch.randint(0, Bh_edge, (n,), device=self.device)
        lz = torch.randint(0, Bw_edge, (n,), device=self.device)
        x = bx * Bd_edge + lx
        y = by * Bh_edge + ly
        z = bz * Bw_edge + lz
        return x * (H * W) + y * W + z

    def _kd_nint_sample(self, it: int, model, entropy_model=None):
        batch = int(self.args.points_per_iteration)
        if it < WARMUP_ITERS:
            return (torch.randint(0, self.n_voxels, (batch,),
                                   device=self.device), None)

        random_portion = COVERAGE_PORTION
        n_rnd = int(batch * random_portion)
        n_res = batch - n_rnd
        parts = []

        if n_res > 0:
            mult = CANDIDATE_MULTIPLIER
            res_pool_n = min(self.n_voxels,
                             max(n_res, int(round(n_res * mult))))
            res_pool_idx = self._kd_sample_indices(res_pool_n, it=it)
            with torch.no_grad():
                x_res = flat_indices_to_coords(res_pool_idx, self.shape,
                                                self.device)
                y_res_gt = self.flat_gt[res_pool_idx].unsqueeze(-1)
                y_res_pred = model(x_res)
                residuals = (y_res_pred - y_res_gt).abs().squeeze(-1)
            top_pos = self._blended_topk(residuals, res_pool_idx, n_res, model, entropy_model)
            parts.append(res_pool_idx[top_pos])

        if n_rnd > 0:
            parts.append(self._kd_sample_indices(n_rnd, it=it))

        return torch.cat(parts, dim=0), None

    def sample(self, it: int, model, entropy_model):
        return self._kd_nint_sample(it, model, entropy_model)

def train(model, entropy_model, gt_volume, opt, args):
    device = args.device
    model.train(True)

    em_param_ids = set(id(p) for p in entropy_model.parameters())
    main_groups = [
        {"params": [p for g in model.get_model_parameters() for p in g["params"]
                    if id(p) not in em_param_ids]},
        {"params": list(entropy_model.parameters())},
    ]
    optimizer = optim.Adam(main_groups, lr=args.lr,
                           betas=(0.9, 0.99), eps=1e-15)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations)

    full_shape = opt["full_shape"]

    n_grid_params = sum(p.numel() for p in model.feature_grid_levels)

    print(f"[train] backbone={BACKBONE}  iters={args.iterations}  "
          f"fg_params={n_grid_params}  em_params="
          f"{sum(p.numel() for p in entropy_model.parameters())}")

    gt_volume = gt_volume.to(device)
    sampler = ActiveVoxelSampler(full_shape, gt_volume, args)
    print(f"[active] block={BLOCK_SIZE}^3  bins={N_BINS}  "
          f"coverage={COVERAGE_PORTION:.2f}  focused={1.0-COVERAGE_PORTION:.2f}  "
          f"pool={CANDIDATE_MULTIPLIER:.0f}x  alpha={ALPHA}  "
          f"schedule_k={SCHEDULE_K}  warmup={WARMUP_ITERS}", flush=True)

    bypass_em = False
    cur_lam = args.entropy_lambda
    start = time.time()
    for it in range(args.iterations):
        idx, rec_weights = sampler.sample(it, model, entropy_model)
        x = flat_indices_to_coords(idx, full_shape, device)
        y_gt = gt_volume.view(-1)[idx].unsqueeze(-1)

        optimizer.zero_grad()
        y_pred, rate_bits = patched_forward(model, entropy_model,
                                              x, training=True,
                                              bypass=bypass_em)
        rec_error = F.mse_loss(y_pred, y_gt, reduction="none")
        if rec_weights is not None:
            rec_loss = (rec_error * rec_weights).mean()
        else:
            rec_loss = rec_error.mean()
        if cur_lam > 0:
            loss = rec_loss + cur_lam * rate_bits / n_grid_params
        else:
            loss = rec_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.log_every and it % args.log_every == 0:
            print(f"  it={it:5d}  mse={rec_loss.item():.4e}  "
                  f"rate={rate_bits.item():.2e}  "
                  f"bpp_param={(rate_bits.item()/n_grid_params):.3f}  "
                  f"loss={loss.item():.4e}", flush=True)

    elapsed = time.time() - start
    print(f"[train] done in {elapsed:.1f}s")
    return elapsed

def save_compressed(model, entropy_model, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    feat_path = save_dir / "features.b"
    levels_x = list(model.feature_grid_levels_view)
    _, x_int_list = entropy_model.quantize_eval_all(levels_x)
    feat_bytes = encode_multires_level_ac(
        x_int_list, entropy_model,
        distribution=getattr(entropy_model, "distribution", "gaussian"),
        output_path=str(feat_path),
    )

    rest_path = save_dir / "rest.pt"
    rest = {}
    for k, v in model.state_dict().items():
        if "feature_grid_levels" in k:
            continue
        rest[k] = v.to(torch.float16) if v.dtype == torch.float32 else v
    em_state = {f"_em.{k}": v
                for k, v in entropy_model.state_dict().items()}
    rest.update(em_state)
    torch.save(rest, rest_path)

    return {
        "features_bytes": feat_bytes,
        "rest_pt_bytes": rest_path.stat().st_size,
        "level_shapes": [list(p.shape) for p in levels_x],
    }

def main():
    parser = argparse.ArgumentParser(
        description="Train and compress a volume with MAGIC.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to a .raw (name must contain XxYxZ) or .nc volume.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory for features.b, rest.pt and metadata.json.")
    parser.add_argument("--entropy_lambda", type=float, default=0.03,
                        help="Rate-distortion trade-off. Larger = smaller file, "
                             "lower PSNR. This is the only knob that moves the "
                             "operating point.")
    parser.add_argument("--feature_grid_shape", type=str,
                        default="16,32,64,128,256",
                        help="Per-level grid resolutions, coarse to fine.")
    parser.add_argument("--n_features_per_level", type=str,
                        default="16,8,8,4,4",
                        help="Per-level channel counts, aligned with "
                             "--feature_grid_shape.")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--points_per_iteration", type=int, default=2 ** 16)
    parser.add_argument("--log_every", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)

    gt_volume, full_shape = load_volume(args.data)
    data_min, data_max = float(gt_volume.min()), float(gt_volume.max())
    print(f"[data] {args.data} shape={full_shape} "
          f"min={data_min:.4f} max={data_max:.4f}")

    opt = build_options(args, full_shape, data_min, data_max)

    model = build_backbone(BACKBONE, opt).to(args.device)
    F_per_level = list(model.n_features_per_level)
    print(f"[backbone] {BACKBONE}  per-level features = {F_per_level}  "
          f"(resolutions = {model.resolutions})")

    entropy_model = MultiResLevelContextEntropyModel(
        resolutions=model.resolutions, n_features=F_per_level,
        hidden=CONTEXT_HIDDEN,
        distribution=AC_DISTRIBUTION,
        init_scale=INIT_SCALE, init_sigma=INIT_SIGMA,
        context_window=CONTEXT_WINDOW,
        sample_num=RATE_SAMPLE_NUM,
        context_start_level=CONTEXT_START_LEVEL,
        activation=CONTEXT_ACTIVATION,
        spatial_context=False,
        likelihood_clamp_min=LIKELIHOOD_CLAMP_MIN,
    ).to(args.device)

    elapsed = train(model, entropy_model, gt_volume, opt, args)

    raw_size = int(np.prod(full_shape)) * 4

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "args": vars(args),
        "opt": opt,
        "training_time_s": elapsed,
        "raw_size_bytes": raw_size,
        "evaluations": [],
    }

    psnr, mse = _eval(model, entropy_model, gt_volume, args.device)
    print(f"\n[eval] PSNR={psnr:.3f} dB  mse={mse:.4e}  "
          f"train_time={elapsed:.1f}s", flush=True)
    metadata["evaluations"].append({

        "lambda": args.entropy_lambda,
        "psnr_db": psnr, "mse": mse,
    })
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    try:
        _t_enc = time.time()
        sizes = save_compressed(model, entropy_model, save_dir)
        encode_s = time.time() - _t_enc
        total_bytes = sizes["features_bytes"] + sizes["rest_pt_bytes"]
        cr = raw_size / total_bytes
        print(f"features.b={sizes['features_bytes']/1024:.1f} KB  "
              f"rest.pt={sizes['rest_pt_bytes']/1024:.1f} KB")
        print(f"CR(total)={cr:.1f}x  PSNR={psnr:.3f} dB  "
              f"train_time={elapsed:.1f}s  encode_time={encode_s:.1f}s  "
              f"compress_time={elapsed + encode_s:.1f}s")

        metadata["encode_time_s"] = encode_s
        metadata["compress_time_s"] = elapsed + encode_s
        metadata["evaluations"][-1].update({
            "features_bytes": sizes["features_bytes"],
            "rest_pt_bytes": sizes["rest_pt_bytes"],
            "cr_total": cr,
            "encode_time_s": encode_s,
            "compress_time_s": elapsed + encode_s,
        })
        if "bits_per_param" in sizes:
            metadata["evaluations"][-1].update({
                "bits_per_param": sizes["bits_per_param"],
                "n_grid_params": sizes["n_grid_params"],
            })
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        print(f"[save] save_compressed failed: {type(e).__name__}: {e}",
              flush=True)

@torch.no_grad()
def _eval(model, entropy_model, gt_volume, device,
          sample_num: int = 0, seed: int = 1234, bypass: bool = False):
    full_shape = list(gt_volume.shape[2:])
    model.eval()
    gt = gt_volume.to(device)
    data_range = (gt.max() - gt.min()).item()

    if sample_num and sample_num > 0:
        n = int(sample_num)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        n_voxels = int(np.prod(full_shape))
        idx = torch.randint(0, n_voxels, (n,), device=device,
                            generator=gen)
        fg_levels_q = None
        if not bypass:
            levels_x = list(model.feature_grid_levels_view)
            fg_levels_q, _ = entropy_model.quantize_eval_all(levels_x)

        se_sum = torch.zeros((), device=device)
        batch = 2 ** 18
        flat_gt = gt.view(-1)
        for i in range(0, n, batch):
            cur_idx = idx[i:i + batch]
            x = flat_indices_to_coords(cur_idx, tuple(full_shape), device)
            pred = model(x, feature_grid_levels_override=fg_levels_q)
            y = flat_gt[cur_idx].unsqueeze(-1)
            se_sum = se_sum + F.mse_loss(pred, y, reduction="sum")
        mse = (se_sum / n).item()
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-30)
        return psnr, mse

    coords = make_coord_grid(full_shape, device)
    n = coords.shape[0]
    pred = torch.empty(n, 1, device=device)
    batch = 2 ** 18

    fg_levels_q = None
    if not bypass:
        levels_x = list(model.feature_grid_levels_view)
        fg_levels_q, _ = entropy_model.quantize_eval_all(levels_x)
    for i in range(0, n, batch):
        pred[i:i + batch] = model(
            coords[i:i + batch],
            feature_grid_levels_override=fg_levels_q)
    pred_vol = pred.view(*full_shape).unsqueeze(0).unsqueeze(0)
    mse = F.mse_loss(pred_vol, gt).item()
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-30)
    return psnr, mse

if __name__ == "__main__":
    main()
