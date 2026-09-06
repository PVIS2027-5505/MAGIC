from __future__ import annotations
import math
import struct
import torch

import torchac

def _gaussian_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def _laplace_cdf(z: torch.Tensor) -> torch.Tensor:
    z = z.clamp(-30.0, 30.0)
    return 0.5 + 0.5 * torch.sign(z) * (1.0 - torch.exp(-z.abs()))

MAX_CDF_BYTES = 256 * 1024 * 1024

def _chunk_size(K: int) -> int:
    Lp = 2 * K + 2
    return max(1, MAX_CDF_BYTES // (Lp * 4))

def encode_per_position_ac_to_bytes(x_int: torch.Tensor,
                                      mu: torch.Tensor,
                                      scale: torch.Tensor,
                                      distribution: str = "gaussian") -> bytes:
    assert x_int.dim() == 5, "expected [G, C, D, H, W]"
    G, C, D, H, W = x_int.shape
    cdf_fn = _laplace_cdf if distribution == "laplace" else _gaussian_cdf

    n_records = G * C
    payload = struct.pack("I", n_records)
    for g in range(G):
        for c in range(C):
            x_gc = x_int[g, c]
            mu_gc = mu[g, c].detach().cpu().reshape(-1)
            sc_gc = scale[g, c].detach().cpu().reshape(-1)
            K = int(x_gc.abs().max().item())
            if K == 0:
                K = 1
            sym_flat = (x_gc + K).to(torch.int16).cpu().reshape(-1)
            N = sym_flat.numel()
            chunk = _chunk_size(K)
            n_chunks = (N + chunk - 1) // chunk

            l_vals = (torch.arange(2 * K + 2, dtype=torch.float32)
                      - K - 0.5).view(1, -1)
            payload += struct.pack("II", K, n_chunks)
            for ci in range(n_chunks):
                lo = ci * chunk
                hi = min(N, lo + chunk)
                mu_chunk = mu_gc[lo:hi].unsqueeze(-1)
                sc_chunk = sc_gc[lo:hi].unsqueeze(-1)
                z = (l_vals - mu_chunk) / sc_chunk
                cdf = cdf_fn(z)
                sym_chunk = sym_flat[lo:hi]
                stream = torchac.encode_float_cdf(
                    cdf.contiguous(), sym_chunk,
                    needs_normalization=True, check_input_bounds=False,
                )
                payload += struct.pack("I", len(stream))
                payload += stream
                del z, cdf, sym_chunk
    return payload

def decode_per_position_ac_from_bytes(payload: bytes,
                                        mu: torch.Tensor,
                                        scale: torch.Tensor,
                                        distribution: str = "gaussian"
                                        ) -> torch.Tensor:
    assert mu.dim() == 5
    G, C, D, H, W = mu.shape
    N_per = D * H * W
    cdf_fn = _laplace_cdf if distribution == "laplace" else _gaussian_cdf

    out = torch.zeros(G * C, N_per, dtype=torch.float32)
    mu_cpu = mu.view(G * C, N_per).detach().cpu()
    scale_cpu = scale.view(G * C, N_per).detach().cpu()
    pos = 0
    n_records = struct.unpack_from("I", payload, pos)[0]; pos += 4
    assert n_records == G * C, f"records {n_records} != G*C={G*C}"
    for idx in range(n_records):
        K, n_chunks = struct.unpack_from("II", payload, pos); pos += 8
        chunk = _chunk_size(K)
        l_vals = (torch.arange(2 * K + 2, dtype=torch.float32)
                  - K - 0.5).view(1, -1)
        for ci in range(n_chunks):
            length = struct.unpack_from("I", payload, pos)[0]; pos += 4
            stream = payload[pos:pos + length]; pos += length
            lo = ci * chunk
            hi = min(N_per, lo + chunk)
            mu_chunk = mu_cpu[idx, lo:hi].unsqueeze(-1)
            sc_chunk = scale_cpu[idx, lo:hi].unsqueeze(-1)
            z = (l_vals - mu_chunk) / sc_chunk
            cdf = cdf_fn(z)
            sym = torchac.decode_float_cdf(
                cdf.contiguous(), stream,
                needs_normalization=True,
            )
            out[idx, lo:hi] = sym.float() - K
            del z, cdf, sym
    return out.view(G, C, D, H, W)

def _encode_subset(x_int, mu, sigma, sel, distribution):
    F = x_int.shape[1]
    xf = x_int.reshape(1, F, -1)[..., sel].unsqueeze(-1).unsqueeze(-1)
    mf = mu.reshape(1, F, -1)[..., sel].unsqueeze(-1).unsqueeze(-1)
    sf = sigma.reshape(1, F, -1)[..., sel].unsqueeze(-1).unsqueeze(-1)
    return encode_per_position_ac_to_bytes(
        xf.contiguous(), mf.contiguous(), sf.contiguous(),
        distribution=distribution)

def _decode_subset(payload, mu, sigma, sel, distribution):
    F = mu.shape[1]
    mf = mu.reshape(1, F, -1)[..., sel].unsqueeze(-1).unsqueeze(-1)
    sf = sigma.reshape(1, F, -1)[..., sel].unsqueeze(-1).unsqueeze(-1)
    out = decode_per_position_ac_from_bytes(
        payload, mf.contiguous(), sf.contiguous(), distribution=distribution)
    return out.reshape(F, -1)

def _encode_level_checkerboard(x_l, l, x_int_list, entropy_model, distribution):
    device = x_l.device
    R = entropy_model.resolutions[l]
    csl = getattr(entropy_model, "context_start_level", 1)
    prev = [x_int_list[i] / entropy_model._level_scale(i) for i in range(l)]
    if l < csl:
        muA, sigA = entropy_model._level_anchor_priors(l, x_l.shape)
    else:
        muA, sigA = entropy_model._level_context_priors(l, prev)
    par = entropy_model._parity_grid(R, device).reshape(-1)
    idx0 = (~par).nonzero(as_tuple=False).squeeze(-1)
    idx1 = par.nonzero(as_tuple=False).squeeze(-1)
    streamA = _encode_subset(x_l, muA, sigA, idx0, distribution)
    level_vals = x_l / entropy_model._level_scale(l)
    muB, sigB = entropy_model._level_spatial_priors(l, prev, level_vals)
    streamB = _encode_subset(x_l, muB, sigB, idx1, distribution)
    return struct.pack("II", len(streamA), len(streamB)) + streamA + streamB

def encode_multires_level_ac(x_int_list,
                              entropy_model,
                              distribution: str = "gaussian",
                              output_path: str = None) -> int:
    K = len(x_int_list)
    csl = getattr(entropy_model, "context_start_level", 1)
    spatial = getattr(entropy_model, "spatial_context", False)
    bytes_per_level = []
    for l in range(K):
        x_l = x_int_list[l]
        if spatial:
            bytes_l = _encode_level_checkerboard(
                x_l, l, x_int_list, entropy_model, distribution)
            bytes_per_level.append(bytes_l)
            continue
        if l < csl:
            mu, sigma = entropy_model._level_anchor_priors(l, x_l.shape)
        else:
            prev_decoded = [x_int_list[i] / entropy_model._level_scale(i)
                            for i in range(l)]
            mu, sigma = entropy_model._level_context_priors(l, prev_decoded)
        bytes_l = encode_per_position_ac_to_bytes(
            x_l, mu, sigma, distribution=distribution)
        bytes_per_level.append(bytes_l)

    payload = struct.pack("I", K)
    for b in bytes_per_level:
        payload += struct.pack("I", len(b))
    for b in bytes_per_level:
        payload += b

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(payload)
    return len(payload)

def decode_multires_level_ac(payload_path: str,
                              entropy_model,
                              level_shapes,
                              distribution: str = "gaussian"):
    K = len(level_shapes)
    with open(payload_path, "rb") as f:
        payload = f.read()
    pos = 0
    K_read = struct.unpack_from("I", payload, pos)[0]; pos += 4
    assert K_read == K, f"bitstream has {K_read} levels, expected {K}"
    lengths = []
    for _ in range(K):
        lengths.append(struct.unpack_from("I", payload, pos)[0]); pos += 4
    streams = []
    for l in lengths:
        streams.append(payload[pos:pos + l]); pos += l

    x_int_list = []

    device = entropy_model.log_scale[0].device
    csl = getattr(entropy_model, "context_start_level", 1)
    spatial = getattr(entropy_model, "spatial_context", False)
    for l in range(K):
        target_shape = tuple(level_shapes[l])
        if spatial:
            x_l = _decode_level_checkerboard(
                streams[l], l, target_shape, x_int_list,
                entropy_model, distribution, device)
            x_int_list.append(x_l.to(device))
            continue
        if l < csl:
            mu, sigma = entropy_model._level_anchor_priors(l, target_shape)
        else:
            prev_decoded = [x_int_list[i] / entropy_model._level_scale(i)
                            for i in range(l)]
            mu, sigma = entropy_model._level_context_priors(l, prev_decoded)
        x_l = decode_per_position_ac_from_bytes(
            streams[l], mu, sigma, distribution=distribution)
        x_int_list.append(x_l.to(device))
    return x_int_list

def _decode_level_checkerboard(level_bytes, l, target_shape, x_int_list,
                                entropy_model, distribution, device):
    lenA, lenB = struct.unpack_from("II", level_bytes, 0)
    off = 8
    streamA = level_bytes[off:off + lenA]; off += lenA
    streamB = level_bytes[off:off + lenB]
    _, F, R, _, _ = target_shape
    csl = getattr(entropy_model, "context_start_level", 1)
    prev = [x_int_list[i] / entropy_model._level_scale(i) for i in range(l)]

    par = entropy_model._parity_grid(R, device).reshape(-1)
    idx0 = (~par).nonzero(as_tuple=False).squeeze(-1)
    idx1 = par.nonzero(as_tuple=False).squeeze(-1)

    if l < csl:
        muA, sigA = entropy_model._level_anchor_priors(l, target_shape)
    else:
        muA, sigA = entropy_model._level_context_priors(l, prev)
    valsA = _decode_subset(streamA, muA, sigA, idx0, distribution).to(device)

    x_flat = torch.zeros(F, R * R * R, device=device)
    x_flat[:, idx0] = valsA
    x_int = x_flat.view(1, F, R, R, R)

    level_vals = x_int / entropy_model._level_scale(l)
    muB, sigB = entropy_model._level_spatial_priors(l, prev, level_vals)
    valsB = _decode_subset(streamB, muB, sigB, idx1, distribution).to(device)
    x_flat[:, idx1] = valsB
    return x_flat.view(1, F, R, R, R)
