from __future__ import annotations

import json
from pathlib import Path

def load_state(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    d = data[0] if isinstance(data, list) else data
    if not isinstance(d, dict) or "RGBPoints" not in d:
        raise ValueError(f"{path}: not a render-state / colormap JSON "
                          "(no 'RGBPoints')")
    return d

def apply_tf_and_camera(state: dict, tf, camera, verbose: bool = True):
    tf._load_color_data(state)
    if "MappingMinMax" in state:
        lo, hi = [float(v) for v in state["MappingMinMax"]]
        tf.set_mapping_minmax(lo, hi)
    if "Camera" in state:
        camera.set_from_state(state["Camera"])
    if verbose:
        n_rgb = len(state["RGBPoints"]) // 4
        n_op = len(state.get("Points", [])) // 4
        print(f"[state] TF: {n_rgb} colour CPs, {n_op} opacity CPs, "
              f"data range=[{tf.min_value:.4g}, {tf.max_value:.4g}], "
              f"mapping={[round(float(v), 3) for v in tf.mapping_minmax.tolist()]}")
        if "Camera" in state:
            c = state["Camera"]
            print(f"[state] camera: coi={[round(float(v), 1) for v in c['coi']]} "
                  f"dist={float(c['dist']):.1f} fov={float(c['fov']):.1f}")

def shading_from_state(state: dict, force_shading: bool = False,
                        defaults: dict | None = None,
                        overrides: dict | None = None,
                        verbose: bool = True) -> dict:
    base = {"enabled": False, "ambient": 0.6, "diffuse": 0.7,
            "specular": 4.0, "shininess": 128.0, "shade_mode": "abs",
            "grad_threshold": 1e-4}
    if defaults:
        base.update(defaults)
    base.update({k: v for k, v in state.get("Shading", {}).items()})
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    if force_shading:
        base["enabled"] = True
    if verbose:
        print(f"[state] shading: {base}")
    return base

def spp_from_state(state: dict, fallback: int) -> int:
    return max(1, int(state.get("SamplesPerRay", fallback)))

def batch_from_state(state: dict, fallback: int) -> int:
    return max(1, int(state.get("BatchSize", fallback)))

def state_name(path: str) -> str:
    return Path(path).stem
