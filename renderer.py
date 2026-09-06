from __future__ import annotations

import argparse
import os
import sys
import time
from math import ceil
import pathlib
from pathlib import Path
from typing import List, Tuple

import nerfacc
import numpy as np
import torch
import torch.nn.functional as F

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class RawData(torch.nn.Module):

    def __init__(self, volume_path: str, device: str):
        super().__init__()
        self.device = device
        from compressor import load_volume

        vol, shape = load_volume(volume_path)
        self.data = vol.to(device)
        self.full_shape = tuple(int(s) for s in shape)
        self.shape = self.full_shape

    def min(self):
        return self.data.flatten().min()

    def max(self):
        return self.data.flatten().max()

    def get_volume_extents(self):
        return self.full_shape

    def forward(self, x):
        x_device = x.device
        x = x.to(self.device)

        x_flip = x.flip(-1)
        y = F.grid_sample(self.data,
                          x_flip.reshape(([1] * x_flip.shape[-1])
                                          + list(x_flip.shape)),
                          mode="bilinear",
                          align_corners=True).squeeze().unsqueeze(1)
        return y.to(x_device)

    _GRAD_VOLUME_MAX_VOXELS = 2_000_000_000

    def _sample(self, x, padding_mode="zeros"):
        xf = x.flip(-1)
        y = F.grid_sample(self.data,
                          xf.reshape(([1] * xf.shape[-1]) + list(xf.shape)),
                          mode="bilinear", padding_mode=padding_mode,
                          align_corners=True)
        return y.squeeze().unsqueeze(1)

    @torch.no_grad()
    def gradient(self, x):
        x_device = x.device
        if self.data.numel() > self._GRAD_VOLUME_MAX_VOXELS:

            xq = x.to(self.device)
            cols = []
            for a in range(3):
                h = 2.0 / max(1, int(self.full_shape[a]) - 1)
                off = torch.zeros_like(xq)
                off[..., a] = h
                fp = self._sample(xq + off, padding_mode="border")
                fm = self._sample(xq - off, padding_mode="border")
                cols.append((fp - fm) * 0.5)
            return torch.cat(cols, dim=-1).to(x_device)

        if not hasattr(self, "_grad") or self._grad is None:
            vp = F.pad(self.data, (1, 1, 1, 1, 1, 1), mode="replicate")
            g0 = (vp[:, :, 2:, 1:-1, 1:-1] - vp[:, :, :-2, 1:-1, 1:-1]) * 0.5
            g1 = (vp[:, :, 1:-1, 2:, 1:-1] - vp[:, :, 1:-1, :-2, 1:-1]) * 0.5
            g2 = (vp[:, :, 1:-1, 1:-1, 2:] - vp[:, :, 1:-1, 1:-1, :-2]) * 0.5
            self._grad = torch.cat([g0, g1, g2], dim=1)
        x = x.to(self.device).flip(-1)
        g = F.grid_sample(self._grad,
                          x.reshape(([1] * x.shape[-1]) + list(x.shape)),
                          mode="bilinear",
                          align_corners=True)
        return g.squeeze(-2).squeeze(-2).squeeze(0).T.to(x_device)

_COLORMAPS_DIR = Path(__file__).resolve().parent.parent / "Colormaps"

_SHADE_DEFAULTS = {"ambient": 0.2, "diffuse": 0.7,
                    "specular": 0.8, "shininess": 64.0}

class TransferFunction():
    def __init__(self, device,
                 min_value: float = 0.0, max_value: float = 1.0,
                 colormap=None):
        self.device = device
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.mapping_minmax = torch.tensor([0.0, 1.0], device=self.device)
        self.num_dict_entries = 4096

        self.preintegration = False
        self.preint_K = 1024
        self.preint_color = None
        self.preint_alpha = None
        if colormap is None:
            self.coolwarm()
        else:
            self.loadColormap(colormap)

    def loadColormap(self, colormapname):
        import json
        file_location = _COLORMAPS_DIR / colormapname
        if file_location.exists():
            with open(file_location) as f:
                color_data = json.load(f)[0]
        else:
            print(f"Colormap '{colormapname}' not found in {_COLORMAPS_DIR}; "
                  "reverting to coolwarm")
            self.coolwarm()
            return
        self._load_color_data(color_data)

    def _load_color_data(self, color_data: dict):
        rgb_data = color_data['RGBPoints']
        self.color_control_points = torch.tensor(
            rgb_data[0::4], dtype=torch.float32, device=self.device)
        self.color_control_points = (self.color_control_points
                                     - self.color_control_points[0])
        self.color_control_points = (self.color_control_points
                                     / self.color_control_points[-1])
        r = torch.tensor(rgb_data[1::4], dtype=torch.float32, device=self.device)
        g = torch.tensor(rgb_data[2::4], dtype=torch.float32, device=self.device)
        b = torch.tensor(rgb_data[3::4], dtype=torch.float32, device=self.device)
        self.color_values = torch.stack([r, g, b], dim=1)

        if "Points" in color_data:
            a_data = color_data['Points']
            self.opacity_control_points = torch.tensor(
                a_data[0::4], dtype=torch.float32, device=self.device)
            self.opacity_control_points = (self.opacity_control_points
                                           - self.opacity_control_points[0])
            self.opacity_control_points = (self.opacity_control_points
                                           / self.opacity_control_points[-1])
            self.opacity_values = torch.tensor(
                a_data[1::4], dtype=torch.float32, device=self.device)
        else:
            self.opacity_control_points = torch.tensor(
                [0.0, 1.0], dtype=torch.float32, device=self.device)
            self.opacity_values = torch.tensor(
                [0.0, 1.0], dtype=torch.float32, device=self.device)
        self.precompute_maps()

    def loadFromPVSM(self, tf_spec: dict):
        rgb = []
        for (v, r, g, b) in tf_spec["rgb_points"]:
            rgb.extend([v, r, g, b])

        pts = []
        for (v, op) in tf_spec["opacity_points"]:
            pts.extend([v, op, 0.5, 0.0])
        color_data = {"RGBPoints": rgb, "Points": pts}

        rgb_values = [v for (v, _, _, _) in tf_spec["rgb_points"]]
        if rgb_values:
            self.set_minmax(min(rgb_values), max(rgb_values))
        self._load_color_data(color_data)

    def coolwarm(self):
        self.color_control_points = torch.tensor(
            [0.0, 0.5, 1.0], dtype=torch.float32, device=self.device)
        self.opacity_control_points = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device)
        self.color_values = torch.tensor(
            [[59 / 255., 76 / 255., 192 / 255.],
             [221 / 255., 221 / 255., 221 / 255.],
             [180 / 255., 4 / 255., 38 / 255.]],
            dtype=torch.float32, device=self.device)
        self.opacity_values = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device)
        self.precompute_maps()

    def precompute_maps(self):
        self.precompute_color_map()
        self.precompute_opacity_map()
        self._maybe_rebake_preintegration()

    def _maybe_rebake_preintegration(self):
        if self.preintegration:
            self.precompute_preintegration()

    def precompute_preintegration(self):
        K = self.preint_K
        r = self.num_dict_entries // K
        a = self.precomputed_opacity_map[:, 0].reshape(K, r).mean(dim=1)
        c = self.precomputed_color_map.reshape(K, r, 3).mean(dim=1)

        A = torch.cumsum(a, dim=0)
        CA = torch.cumsum(c * a.unsqueeze(-1), dim=0)
        C = torch.cumsum(c, dim=0)

        idx = torch.arange(K, device=self.device)
        ii = torch.minimum(idx[:, None], idx[None, :])
        jj = torch.maximum(idx[:, None], idx[None, :])
        seg = (jj - ii).clamp(min=1).to(torch.float32)

        dA = A[jj] - A[ii]
        alpha_bar = dA / seg
        c_assoc = (CA[jj] - CA[ii]) / (dA.unsqueeze(-1) + 1e-8)
        c_plain = (C[jj] - C[ii]) / seg.unsqueeze(-1)
        alpha_mass = (dA > 1e-8).unsqueeze(-1)
        color_bar = torch.where(alpha_mass, c_assoc, c_plain)

        alpha_bar[idx, idx] = a
        color_bar[idx, idx] = c
        self.preint_alpha = alpha_bar.contiguous()
        self.preint_color = color_bar.contiguous()

    def _preint_index(self, value: torch.Tensor):
        v = (value - self.min_value) / (self.max_value - self.min_value)
        v = self.remap_value(v)
        return (v * self.preint_K).long().clamp_(0, self.preint_K - 1)

    def preint_color_opacity(self, v_front: torch.Tensor,
                             v_back: torch.Tensor):
        value_device = v_front.device
        i = self._preint_index(v_front.to(self.device))
        j = self._preint_index(v_back.to(self.device))
        flat = i * self.preint_K + j
        rgbs = torch.index_select(
            self.preint_color.view(-1, 3), dim=0, index=flat)
        alphas = torch.index_select(
            self.preint_alpha.view(-1, 1), dim=0, index=flat)
        return rgbs.to(value_device), alphas.to(value_device)

    def precompute_color_map(self):
        self.precomputed_color_map = torch.zeros(
            [self.num_dict_entries, 3], dtype=torch.float32, device=self.device)
        for ind in range(self.color_control_points.shape[0] - 1):
            color_a = self.color_values[ind]
            color_b = self.color_values[ind + 1]
            start_ind = int(self.num_dict_entries
                            * self.color_control_points[ind])
            end_ind = int(self.num_dict_entries
                          * self.color_control_points[ind + 1])
            n = end_ind - start_ind
            if n <= 0:
                continue
            color_a = color_a.unsqueeze(0).repeat(n, 1)
            color_b = color_b.unsqueeze(0).repeat(n, 1)
            t = torch.arange(0.0, 1.0, step=(1 / n),
                             dtype=torch.float32,
                             device=self.device).unsqueeze(1)[0:n].repeat(1, 3)
            self.precomputed_color_map[start_ind:end_ind] = (
                color_a * (1 - t) + color_b * t)

    def precompute_opacity_map(self):
        self.precomputed_opacity_map = torch.zeros(
            [self.num_dict_entries, 1], dtype=torch.float32, device=self.device)
        for ind in range(self.opacity_control_points.shape[0] - 1):
            opacity_a = self.opacity_values[ind]
            opacity_b = self.opacity_values[ind + 1]
            start_ind = int(self.num_dict_entries
                            * self.opacity_control_points[ind])
            end_ind = int(self.num_dict_entries
                          * self.opacity_control_points[ind + 1])
            n = end_ind - start_ind
            if n <= 0:
                continue
            opacity_a = opacity_a.unsqueeze(0).repeat(n, 1)
            opacity_b = opacity_b.unsqueeze(0).repeat(n, 1)
            t = torch.arange(0.0, 1.0, step=(1 / n),
                             dtype=torch.float32,
                             device=self.device).unsqueeze(1)[0:n]
            self.precomputed_opacity_map[start_ind:end_ind] = (
                opacity_a * (1 - t) + opacity_b * t)

    def set_minmax(self, lo, hi):
        self.min_value = float(lo)
        self.max_value = float(hi)

    def smooth_opacity_lut(self, sigma_entries: float):
        if sigma_entries <= 0 or not hasattr(self, "precomputed_opacity_map"):
            return
        N = self.num_dict_entries
        radius = max(1, int(round(3.0 * sigma_entries)))
        k = torch.arange(-radius, radius + 1, dtype=torch.float32,
                         device=self.device)
        w = torch.exp(-(k ** 2) / (2.0 * sigma_entries ** 2))
        w = w / w.sum()

        a = self.precomputed_opacity_map.view(1, 1, N)
        a = F.pad(a, [radius, radius], mode="reflect")
        a = F.conv1d(a, w.view(1, 1, -1)).view(N, 1)
        self.precomputed_opacity_map = a
        self._maybe_rebake_preintegration()

    def apply_alpha_floor(self, v_floor: float):
        if not hasattr(self, "precomputed_opacity_map"):
            return
        if self.max_value == self.min_value:
            return
        frac = (v_floor - self.min_value) / (self.max_value - self.min_value)
        idx_cut = int(max(0, min(self.num_dict_entries,
                                   round(frac * self.num_dict_entries))))
        self.precomputed_opacity_map[:idx_cut] = 0.0
        self._maybe_rebake_preintegration()

    def set_mapping_minmax(self, lo, hi):
        self.mapping_minmax = torch.tensor([lo, hi], device=self.device)

    def get_color_points(self):
        return (self.color_control_points.cpu().numpy().copy(),
                self.color_values.cpu().numpy().copy())

    def set_color_points(self, xs, rgb):
        self.color_control_points = torch.tensor(
            np.asarray(xs), dtype=torch.float32, device=self.device)
        self.color_values = torch.tensor(
            np.asarray(rgb), dtype=torch.float32, device=self.device)
        self.precompute_color_map()
        self._maybe_rebake_preintegration()

    def to_paraview_json(self, name: str) -> list:
        rgb_points, points = [], []
        for x, (r, g, b) in zip(self.color_control_points.cpu().tolist(),
                                self.color_values.cpu().tolist()):
            rgb_points += [x, r, g, b]
        for x, a in zip(self.opacity_control_points.cpu().tolist(),
                        self.opacity_values.cpu().tolist()):
            points += [x, a, 0.5, 0.0]
        return [{"ColorSpace": "RGB", "Name": name,
                 "Points": points, "RGBPoints": rgb_points}]

    def remap_value(self, values):
        new_min = -(self.mapping_minmax[0])
        new_max = 2 - self.mapping_minmax[1]
        values = values * (new_max - new_min)
        values += new_min
        return values

    def remap_value_inplace(self, values):
        new_min = -(self.mapping_minmax[0])
        new_max = 2 - self.mapping_minmax[1]
        values *= (new_max - new_min)
        values += new_min

    def update_opacities(self, opacity_control_points, opacity_values):
        self.opacity_control_points = torch.tensor(
            opacity_control_points, dtype=torch.float32, device=self.device)
        self.opacity_values = torch.tensor(
            opacity_values, dtype=torch.float32, device=self.device)
        self.precompute_opacity_map()
        self._maybe_rebake_preintegration()

    def color_at_value(self, value: torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        idx = ((value[:, 0] - self.min_value)
               / (self.max_value - self.min_value)
               * (self.mapping_minmax[1] - self.mapping_minmax[0])
               + self.mapping_minmax[0])
        idx *= self.num_dict_entries
        idx = idx.long().clamp_(0, self.num_dict_entries - 1)
        return torch.index_select(
            self.precomputed_color_map, dim=0, index=idx).to(value_device)

    def opacity_at_value(self, value: torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        idx = ((value[:, 0] - self.min_value)
               / (self.max_value - self.min_value)
               * (self.mapping_minmax[1] - self.mapping_minmax[0])
               + self.mapping_minmax[0])
        idx *= self.num_dict_entries
        idx = idx.long().clamp_(0, self.num_dict_entries - 1)
        return torch.index_select(
            self.precomputed_opacity_map, dim=0, index=idx).to(value_device)

    def color_opacity_at_value(self, value: torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        value = value - self.min_value
        value = value / (self.max_value - self.min_value)
        value = self.remap_value(value)
        idx = (value * self.num_dict_entries).long().clamp_(
            0, self.num_dict_entries - 1)
        return (torch.index_select(
                    self.precomputed_color_map, dim=0, index=idx).to(value_device),
                torch.index_select(
                    self.precomputed_opacity_map, dim=0, index=idx).to(value_device))

    def color_opacity_at_value_inplace(self, value, rgbs, alphas, start_ind):
        value_device = value.device
        value = value.to(self.device)
        idx = self.remap_value(
            (value[:, 0] - self.min_value)
            / (self.max_value - self.min_value))
        idx = (idx * self.num_dict_entries).long().clamp_(
            0, self.num_dict_entries - 1)
        rgbs[start_ind:start_ind + value.shape[0]] = torch.index_select(
            self.precomputed_color_map, dim=0, index=idx).to(value_device)
        alphas[start_ind:start_ind + value.shape[0]] = torch.index_select(
            self.precomputed_opacity_map, dim=0, index=idx).to(value_device)

class Camera():
    def __init__(self, device,
                 scene_aabb: torch.Tensor,
                 coi: torch.Tensor = torch.Tensor([0., 0., 0.]),
                 azi_deg: float = 0., polar_deg: float = 90.,
                 dist: float = 200.):
        self.device = device
        self.fov = torch.tensor([60.0], device=self.device)
        self.azi = torch.zeros(1, device=coi.device)
        self.polar = torch.zeros(1, device=coi.device)
        self.dist = torch.zeros(1, device=coi.device)
        self.coi = torch.zeros(3, device=coi.device)
        self._up_override = None
        self._vmat_override = None
        self._st_rot = self._st_trans = self._st_coi_t = None
        self.set_azi(azi_deg, device=coi.device)
        self.set_polar(polar_deg, device=coi.device)
        self.set_dist(dist, device=coi.device)
        self.set_coi(coi)
        self.set_eye(self.calc_eye())
        self.vMat = self.get_view()

    def position(self):
        return self.eye

    def set_azi(self, azi_deg, device="cuda:0"):
        self.azi = torch.deg2rad(torch.tensor(azi_deg, device=device))
        self.set_eye(self.calc_eye())

    def set_polar(self, polar_deg, device="cuda:0"):
        self.polar = torch.deg2rad(torch.tensor(polar_deg, device=device))
        self.set_eye(self.calc_eye())

    def set_dist(self, dist, device="cuda:0"):
        self.dist = torch.tensor(dist, device=device)
        self.set_eye(self.calc_eye())

    def set_coi(self, coi):
        self.coi = coi
        self.set_eye(self.calc_eye())

    def set_eye(self, eye):
        self.eye = eye

    def calc_eye(self):
        y = self.dist * torch.cos(self.polar)
        dxz = self.dist * torch.sin(self.polar)
        x = torch.sin(self.azi) * dxz
        z = torch.cos(self.azi) * dxz
        eye_origin = torch.stack([x, y, z]).to(self.coi)
        return self.coi + eye_origin

    def get_c2w(self):
        return torch.linalg.inv(self.get_view())

    def get_view(self):
        if self._vmat_override is not None:
            return self._vmat_override
        normalize = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        zaxis = normalize(self.eye - self.coi)
        if self._up_override is not None:
            up = self._up_override.to(zaxis)
        else:
            up = torch.tensor([0., 1., 0.]).to(zaxis)
        xaxis = torch.cross(normalize(up), zaxis)
        xaxis = normalize(xaxis) if xaxis.sum() != 0. else xaxis
        yaxis = torch.cross(zaxis, xaxis)
        vMat = torch.tensor([
            [xaxis[0], xaxis[1], xaxis[2], -torch.dot(self.eye, xaxis)],
            [yaxis[0], yaxis[1], yaxis[2], -torch.dot(self.eye, yaxis)],
            [zaxis[0], zaxis[1], zaxis[2], -torch.dot(self.eye, zaxis)],
            [0., 0., 0., 1.]
        ], device=self.device)
        return vMat

    def set_from_pvsm(self, position, focal_point, view_up,
                       view_angle_deg: float, axis_order=(2, 1, 0)):
        def _reorder(v):
            return [float(v[i]) for i in axis_order]
        pos = torch.tensor(_reorder(position),
                            dtype=torch.float32, device=self.device)
        fp = torch.tensor(_reorder(focal_point),
                           dtype=torch.float32, device=self.device)
        up = torch.tensor(_reorder(view_up),
                           dtype=torch.float32, device=self.device)
        up = up / torch.linalg.norm(up)
        self.coi = fp
        self.eye = pos
        self._up_override = up
        self.fov = torch.tensor([float(view_angle_deg)], device=self.device)

        delta = pos - fp
        self.dist = torch.linalg.norm(delta)

        if float(self.dist.item()) > 0:
            dxz = (delta[0] ** 2 + delta[2] ** 2) ** 0.5
            self.polar = torch.atan2(dxz, delta[1])
            self.azi = torch.atan2(delta[0], delta[2])

    def set_from_state(self, state: dict):
        coi = torch.tensor([float(v) for v in state["coi"]],
                            dtype=torch.float32, device=self.device)
        dist = float(state["dist"])
        rot = torch.tensor(state["rotation"], dtype=torch.float32,
                            device=self.device)
        if rot.shape == (3, 3):
            r4 = torch.eye(4, dtype=torch.float32, device=self.device)
            r4[:3, :3] = rot
            rot = r4
        coi_t = torch.eye(4, dtype=torch.float32, device=self.device)
        coi_t[:3, 3] = -coi
        trans = torch.eye(4, dtype=torch.float32, device=self.device)
        trans[2, 3] = -dist

        self._st_rot, self._st_trans, self._st_coi_t = rot, trans, coi_t
        self._apply_vmat(trans @ rot @ coi_t)
        self.dist = torch.tensor(dist, device=self.device)
        self.coi = coi
        self.fov = torch.tensor([float(state["fov"])], device=self.device)

    def _apply_vmat(self, vmat):
        self._vmat_override = vmat
        self.eye = torch.linalg.inv(vmat)[:3, 3].contiguous()

    def orbit(self, deg: float, pivot=None):
        if self._st_rot is None:
            raise RuntimeError("orbit() needs a --state camera")
        t = torch.deg2rad(torch.tensor(float(deg), device=self.device))
        c, s = torch.cos(t), torch.sin(t)
        ry = torch.eye(4, dtype=torch.float32, device=self.device)
        ry[0, 0], ry[0, 2] = c, s
        ry[2, 0], ry[2, 2] = -s, c
        v0 = self._st_trans @ self._st_rot @ self._st_coi_t
        if pivot is None:
            self._apply_vmat(self._st_trans @ self._st_rot @ ry @ self._st_coi_t)
            return
        p = torch.as_tensor([float(v) for v in pivot], dtype=torch.float32,
                            device=self.device)
        tp = torch.eye(4, dtype=torch.float32, device=self.device)
        tp[:3, 3] = p
        tm = torch.eye(4, dtype=torch.float32, device=self.device)
        tm[:3, 3] = -p
        self._apply_vmat(v0 @ tp @ ry @ tm)

    def generate_dirs(self, width, height):
        x, y = torch.meshgrid(
            torch.arange(width), torch.arange(height), indexing="xy")
        x = x.flatten().to(self.device)
        y = y.flatten().to(self.device)
        x = (2 * (x + 0.5) / width - 1) * torch.tan(
            torch.deg2rad(self.fov / 2)) * (width / height)
        y = (1 - 2 * (y + 0.5) / height) * torch.tan(
            torch.deg2rad(self.fov / 2))
        z = -torch.ones(x.shape).to(self.device)
        camera_dirs = torch.stack([x, y, z], -1)
        c2w = self.get_c2w()
        directions = (c2w[:3, :3] @ camera_dirs.T).T
        directions = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True)
        return directions.reshape(height, width, 3)

class Scene(torch.nn.Module):

    def __init__(self, model, camera, full_shape,
                 image_resolution: Tuple[int],
                 batch_size: int, spp: int,
                 transfer_function: TransferFunction,
                 device="cuda:0", data_device="cuda:0",
                 shading: dict = None):
        super().__init__()
        self.model = model
        self.device = device
        self.data_device = data_device
        self.scene_aabb = torch.tensor(
            [0.0, 0.0, 0.0,
             full_shape[0] - 1, full_shape[1] - 1, full_shape[2] - 1],
            device=self.device)
        self.image_resolution = image_resolution
        self.batch_size = batch_size
        self.spp = spp
        self.estimator = nerfacc.OccGridEstimator(
            self.scene_aabb, resolution=1, levels=1).to(self.device)
        self.estimator.binaries = torch.ones_like(self.estimator.binaries)
        self.transfer_function = transfer_function
        self.amount_empty = 0.0
        self.camera = camera

        self.shading = shading or {"enabled": False}

        self.jitter = False

        self.preintegrate = False
        self.on_setting_change()

    def get_mem_use(self):
        return (torch.cuda.max_memory_allocated(device=self.device)
                / (1024 ** 3)) if "cuda" in str(self.device) else 0.0

    def set_aabb(self, full_shape):
        self.scene_aabb = torch.tensor(
            [0.0, 0.0, 0.0,
             full_shape[0] - 1, full_shape[1] - 1, full_shape[2] - 1],
            device=self.device)
        self.estimator = nerfacc.OccGridEstimator(
            self.scene_aabb, resolution=1, levels=1).to(self.device)
        self.estimator.binaries = torch.ones_like(self.estimator.binaries)

    def generate_viewpoint_rays(self, camera: Camera):
        height, width = self.image_resolution[:2]
        n = height * width
        self.rays_d = camera.generate_dirs(width, height).view(-1, 3)
        self.rays_o = camera.position().unsqueeze(0).expand(n, 3)
        max_view_dist = (self.scene_aabb[3] ** 2
                          + self.scene_aabb[4] ** 2
                          + self.scene_aabb[5] ** 2) ** 0.5
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            self.rays_o, self.rays_d,
            render_step_size=max_view_dist / self.spp,
            stratified=self.jitter)
        return ray_indices, t_starts, t_ends

    def _normalize_locs(self, locs):
        locs = locs / self.scene_aabb[3:]
        locs *= 2
        locs -= 1
        return locs

    def rgb_alpha_fn(self, t_starts, t_ends, ray_indices):
        o = self.rays_o[ray_indices]
        d = self.rays_d[ray_indices]

        sample_locs = self._normalize_locs(
            o + d * (t_starts + t_ends)[:, None] / 2.0)
        sample_locs_d = sample_locs.to(self.data_device)
        if self.preintegrate:

            v_front = self.model(self._normalize_locs(
                o + d * t_starts[:, None]).clamp_(-1.0, 1.0)
                .to(self.data_device)).to(self.device)
            v_back = self.model(self._normalize_locs(
                o + d * t_ends[:, None]).clamp_(-1.0, 1.0)
                .to(self.data_device)).to(self.device)
            rgbs, alphas = self.transfer_function.preint_color_opacity(
                v_front[:, 0], v_back[:, 0])
        else:
            densities = self.model(sample_locs_d).to(self.device)
            rgbs, alphas = self.transfer_function.color_opacity_at_value(
                densities[:, 0])

        if self.shading.get("enabled", False) and hasattr(self.model, "gradient"):

            g = self.model.gradient(sample_locs_d).to(self.device)
            g_norm = torch.linalg.norm(g, dim=-1)
            valid = (g_norm > self.shading.get("grad_threshold", 1e-4)).float()

            n = -g / (g_norm.unsqueeze(-1) + 1e-8)

            d = self.rays_d[ray_indices]
            ndotd = (n * d).sum(dim=-1)
            amb = float(self.shading.get("ambient", 0.2))
            dif = float(self.shading.get("diffuse", 0.7))
            spc = float(self.shading.get("specular", 0.8))
            shin = float(self.shading.get("shininess", 64.0))
            mode = self.shading.get("shade_mode", "abs")

            if mode == "flip":

                flip = (ndotd > 0).float() * 2.0 - 1.0

                n_eff = n * (-flip).unsqueeze(-1)
                diff_amnt = (-(n_eff * d).sum(dim=-1)).clamp(min=0.0)
                spec_amnt = diff_amnt ** shin
            elif mode == "separate":

                front = (-ndotd).clamp(min=0.0)
                back = ndotd.clamp(min=0.0)
                diff_amnt = front + back
                spec_amnt = front ** shin + back ** shin
            else:
                diff_amnt = ndotd.abs().clamp(0.0, 1.0)

                spec_amnt = diff_amnt ** shin

            shaded = rgbs * (amb + dif * diff_amnt).unsqueeze(-1) \
                + spc * spec_amnt.unsqueeze(-1)

            valid_e = valid.unsqueeze(-1)
            rgbs = (shaded * valid_e + rgbs * (1.0 - valid_e)).clamp(0.0, 1.0)

        alphas += 1
        alphas.log_()
        return rgbs, alphas[:, 0]

    def rgb_alpha_fn_batch(self, t_starts, t_ends, ray_indices):
        rgbs = torch.empty([t_starts.shape[0], 3],
                           device=self.device, dtype=torch.float32)
        alphas = torch.empty([t_starts.shape[0], 1],
                             device=self.device, dtype=torch.float32)
        for start in range(0, t_starts.shape[0], self.batch_size):
            end = min(start + self.batch_size, t_starts.shape[0])
            o = self.rays_o[ray_indices[start:end]]
            d = self.rays_d[ray_indices[start:end]]
            if self.preintegrate:

                v_front = self.model(self._normalize_locs(
                    o + d * t_starts[start:end][:, None]
                ).clamp_(-1.0, 1.0).to(self.data_device)).to(self.device)
                v_back = self.model(self._normalize_locs(
                    o + d * t_ends[start:end][:, None]
                ).clamp_(-1.0, 1.0).to(self.data_device)).to(self.device)
                rgbs[start:end], alphas[start:end] = \
                    self.transfer_function.preint_color_opacity(
                        v_front[:, 0], v_back[:, 0])
            else:
                sample_locs = self._normalize_locs(
                    o + d * (t_starts[start:end]
                             + t_ends[start:end])[:, None] / 2.0)
                densities = self.model(sample_locs.to(self.data_device)).to(
                    self.device)
                self.transfer_function.color_opacity_at_value_inplace(
                    densities, rgbs, alphas, start)
        alphas += 1
        alphas.log_()
        return rgbs, alphas[:, 0]

    def render_rays(self, t_starts, t_ends, ray_indices, n_rays):
        colors, _, _, _ = nerfacc.rendering(
            t_starts, t_ends, ray_indices, n_rays,
            rgb_alpha_fn=self.rgb_alpha_fn,
            render_bkgd=torch.tensor(
                [1.0, 1.0, 1.0], dtype=torch.float32, device=self.device))
        colors.clip_(0.0, 1.0)
        return colors

    def render(self, camera):
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.generate_viewpoint_rays(camera)
            colors = self.render_rays(
                t_starts, t_ends, ray_indices,
                self.image_resolution[0] * self.image_resolution[1])
        return colors.reshape(
            self.image_resolution[0], self.image_resolution[1], 3)

    def generate_checkerboard_render_order(self):
        class Rect:
            def __init__(self, x, y, w, h):
                self.x, self.y, self.w, self.h = x, y, w, h
                if w > 1 and h > 1:
                    self.queue = [(x + w // 2, y + h // 2),
                                  (x + w // 2, y), (x, y + h // 2)]
                elif w > 1:
                    self.queue = [(x + w // 2, y)]
                elif h > 1:
                    self.queue = [(x, y + h // 2)]
                else:
                    self.queue = []

            def subdivide(self):
                if self.w > 1 and self.h > 1:
                    return [
                        Rect(self.x, self.y, self.w // 2, self.h // 2),
                        Rect(self.x + self.w // 2, self.y + self.h // 2,
                             self.w - self.w // 2, self.h - self.h // 2),
                        Rect(self.x + self.w // 2, self.y,
                             self.w - self.w // 2, self.h // 2),
                        Rect(self.x, self.y + self.h // 2,
                             self.w // 2, self.h - self.h // 2)]
                elif self.w > 1:
                    return [Rect(self.x, self.y, self.w // 2, self.h),
                            Rect(self.x + self.w // 2, self.y,
                                 self.w - self.w // 2, self.h)]
                elif self.h > 1:
                    return [Rect(self.x, self.y, self.w, self.h // 2),
                            Rect(self.x, self.y + self.h // 2,
                                 self.w, self.h - self.h // 2)]
                return []

            def get_next(self):
                return self.queue.pop(0) if self.queue else None

            def needs_subdivide(self):
                return len(self.queue) == 0

        def checkerboard_render_order(w, h):
            rects = [Rect(0, 0, w, h)]
            order = [(0, 0)]
            rects_to_add = []
            while rects:
                idx_remove = []
                for i, r in enumerate(rects):
                    spot = r.get_next()
                    if spot is not None:
                        order.append(spot)
                    if r.needs_subdivide():
                        idx_remove.append(i)
                        rects_to_add += r.subdivide()
                for i in range(len(idx_remove)):
                    rects.pop(idx_remove[len(idx_remove) - i - 1])
                if not rects:
                    while rects_to_add:
                        rects.append(rects_to_add.pop(0))
            return order

        return checkerboard_render_order(self.strides, self.strides)

    def generate_normal_render_order(self):
        return [(y, x) for x in range(self.strides)
                for y in range(self.strides)]

    def on_setting_change(self):
        self.max_view_dist = (self.scene_aabb[3] ** 2
                               + self.scene_aabb[4] ** 2
                               + self.scene_aabb[5] ** 2) ** 0.5
        self.height, self.width = self.image_resolution[:2]
        n_point_evals = (self.width * self.height * self.spp
                          * (1 - self.amount_empty))
        n_passes = n_point_evals / self.batch_size
        self.strides = int(n_passes ** 0.5) + 1
        self.image = torch.empty(
            [self.height, self.width, 3], device=self.device,
            dtype=torch.float32)
        self.mip = torch.zeros(
            [ceil(self.height / self.strides),
             ceil(self.width / self.strides), 3],
            device=self.device, dtype=torch.float32)
        self.mask = torch.zeros_like(self.image, dtype=torch.bool)
        self.temp_image = torch.empty_like(self.image)
        self.render_order = self.generate_checkerboard_render_order()
        self.current_order_spot = 0
        self.all_rays = self._cam_rays()
        self.cam_origin = self._cam_origin()
        self.y_leftover = self.height % self.strides
        self.x_leftover = self.width % self.strides
        self.passes = 0
        self.mip_level = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cam_rays(self):
        r = self.camera.generate_dirs(self.width, self.height)
        return torch.as_tensor(r, dtype=torch.float32, device=self.device)

    def _cam_origin(self):
        p = self.camera.position()
        t = torch.as_tensor(p, dtype=torch.float32, device=self.device)
        return t.unsqueeze(0)

    def on_tf_change(self):
        self.image.zero_()
        self.mask.zero_()
        self.temp_image.zero_()
        self.mip = torch.zeros(
            [ceil(self.height / self.strides),
             ceil(self.width / self.strides), 3],
            device=self.device, dtype=torch.float32)
        self.current_order_spot = 0
        self.passes = 0
        self.mip_level = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_rotate_zoom_pan(self):
        self.image.zero_()
        self.mask.zero_()
        self.temp_image.zero_()
        self.mip = torch.zeros(
            [ceil(self.height / self.strides),
             ceil(self.width / self.strides), 3],
            device=self.device, dtype=torch.float32)
        self.current_order_spot = 0
        self.all_rays = self._cam_rays()
        self.cam_origin = self._cam_origin()
        self.passes = 0
        self.mip_level = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_resize(self):
        self.height, self.width = self.image_resolution[:2]
        n_point_evals = (self.width * self.height * self.spp
                          * (1 - self.amount_empty))
        n_passes = n_point_evals / self.batch_size
        self.strides = int(n_passes ** 0.5) + 1
        self.image = torch.empty(
            [self.height, self.width, 3], device=self.device,
            dtype=torch.float32)
        self.mip = torch.zeros(
            [ceil(self.height / self.strides),
             ceil(self.width / self.strides), 3],
            device=self.device, dtype=torch.float32)
        self.mask = torch.zeros_like(self.image, dtype=torch.bool)
        self.temp_image = torch.empty_like(self.image)
        self.render_order = self.generate_checkerboard_render_order()
        self.current_order_spot = 0
        self.all_rays = self._cam_rays()
        self.cam_origin = self._cam_origin()
        self.y_leftover = self.height % self.strides
        self.x_leftover = self.width % self.strides
        self.passes = 0
        self.mip_level = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def one_step_update(self):
        if self.current_order_spot == len(self.render_order):
            return
        with torch.no_grad():
            x, y = self.render_order[self.current_order_spot]
            mip_stride = min(self.strides, int(2 ** self.mip_level))
            mip_x = round(mip_stride * (x / self.strides))
            mip_y = round(mip_stride * (y / self.strides))
            y_extra = 1 if y < self.y_leftover else 0
            x_extra = 1 if x < self.x_leftover else 0
            rays_this_iter = self.all_rays[y::self.strides,
                                            x::self.strides].clone().view(-1, 3)
            self.rays_d = rays_this_iter
            num_rays = rays_this_iter.shape[0]
            self.rays_o = self.cam_origin.expand(num_rays, 3)
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                self.rays_o, self.rays_d,
                render_step_size=self.max_view_dist / self.spp,
                stratified=self.jitter)
            new_colors = self.render_rays(
                t_starts, t_ends, ray_indices, num_rays).view(
                    self.height // self.strides + y_extra,
                    self.width // self.strides + x_extra, 3)
            self.image[y::self.strides, x::self.strides, :] = new_colors
            self.mask[y::self.strides, x::self.strides, :] = 1
            self.mip[mip_y:new_colors.shape[0] * mip_stride:mip_stride,
                     mip_x:new_colors.shape[1] * mip_stride:mip_stride,
                     :] = new_colors
            self.temp_image = (self.image * self.mask
                                + F.interpolate(
                                    self.mip.permute(2, 0, 1).unsqueeze(0),
                                    size=[self.height, self.width],
                                    mode='bilinear')[0].permute(1, 2, 0)
                                * ~self.mask)
            self.passes += 1
            if self.passes == int(4 ** self.mip_level):
                self.mip_level += 1
                if (self.mip.shape[0] * 2 > self.height
                        or self.mip.shape[1] * 2 > self.width):
                    upscale_shape = [self.height, self.width]
                else:
                    eff = self.strides / (2 ** self.mip_level)
                    upscale_shape = [ceil(self.height / eff),
                                     ceil(self.width / eff)]
                self.mip = F.interpolate(
                    self.mip.permute(2, 0, 1).unsqueeze(0),
                    size=upscale_shape, mode='nearest')[0].permute(1, 2, 0)
        self.current_order_spot += 1

    def render_checkerboard(self):
        while self.current_order_spot < len(self.render_order):
            self.one_step_update()
        return self.image, []

def _build_argparser():
    p = argparse.ArgumentParser(
        description="Volume render an INR (or raw .nc) to a PNG.")
    src = p.add_argument_group("source")
    src.add_argument("--save_dir", type=str, default=None,
                     help="Path to an INR save dir (containing metadata.json + "
                          "rest.pt + features*.b). Required unless --raw_data.")
    src.add_argument("--raw_data", action="store_true",
                     help="Render a raw .nc volume instead (path via --data).")
    src.add_argument("--data", type=str, default=None,
                     help="Path to a .nc / .raw volume (with --raw_data, or "
                          "with --verify_psnr).")

    cam = p.add_argument_group("camera")
    cam.add_argument("--azi", type=float, default=0.,
                     help="Azimuth around y-axis (deg). 0 aligns +z.")
    cam.add_argument("--polar", type=float, default=90.,
                     help="Polar from +y (deg). 90 is equator.")
    cam.add_argument("--dist", type=float, default=None,
                     help="Distance from COI; defaults to AABB diagonal.")

    rend = p.add_argument_group("rendering")
    rend.add_argument("--spp", type=int, default=None,
                      help="Samples per ray (default 256; with --state the "
                           "preset's SamplesPerRay wins unless you pass this).")
    rend.add_argument("--hw", type=lambda s: [int(x) for x in s.split(",")],
                      default=[512, 512], help="height,width  (e.g. 512,512).")
    rend.add_argument("--batch_size", type=int, default=None,
                      help="Forward batch size; smaller = less VRAM. Default "
                           "2^23; with --state the preset's BatchSize wins "
                           "unless you pass this. MUST be pinned by hand when "
                           "timing methods against each other -- batch size "
                           "changes throughput, and letting the preset decide "
                           "silently gave GT/GridComp 2^25 while the baselines "
                           "(which OOM there) ran at 2^20.")
    rend.add_argument("--colormap", type=str, default=None,
                      help="Paraview colormap JSON in Render/Colormaps/.")
    rend.add_argument("--shading", action="store_true",
                      help="Enable Blinn-Phong headlight shading "
                           "(L = V = -ray_dir).  Requires the model to "
                           "expose a `.gradient(x)` method — INRModel "
                           "auto-builds a gradient field on first call, "
                           "RawData has it built-in.")

    rend.add_argument("--ambient", type=float, default=None)
    rend.add_argument("--diffuse", type=float, default=None)
    rend.add_argument("--specular", type=float, default=None)
    rend.add_argument("--shininess", type=float, default=None)
    rend.add_argument("--shade_mode", type=str, default="abs",
                      choices=["abs", "flip", "separate"],
                      help="Two-sided Blinn-Phong back-face handling. "
                           "'abs' (default) matches NVIDIA IndeX's "
                           "kernel_depth_enhancement.cu — |N·L| and |N·H| "
                           "with an additive specular composite. "
                           "'flip' flips N to face the camera when N·V<0 "
                           "then does single-sided Blinn-Phong. "
                           "'separate' adds two pow(max(0, ±N·H), shin) "
                           "specular terms (one per face).")
    rend.add_argument("--preintegrate", action="store_true",
                      help="(default; kept for backwards compat) Pre-integrated "
                           "transfer function.")
    rend.add_argument("--no_preintegrate", action="store_true",
                      help="Disable the pre-integrated transfer function "
                           "(Engel-style 2D table over segment endpoint "
                           "values). Pre-integration is ON by default: it "
                           "kills the TF-aliasing wood-grain rings that sharp "
                           "opacity peaks produce under point sampling, for a "
                           "2nd model eval per segment (~10% slower). Pass "
                           "this to reproduce the old point-sampled renders "
                           "bit-for-bit.")
    rend.add_argument("--orbit_views", type=int, default=0,
                      help="Time the render over N viewpoints uniformly spaced "
                           "around the volume (orbit about the up axis through "
                           "the COI) and report the mean. A single frame is not "
                           "a fair cost estimate -- how long a frame takes "
                           "depends on how much volume its rays traverse. "
                           "Requires --state.")
    rend.add_argument("--orbit_center", type=str, default="state",
                      help="Pivot for --orbit_views: 'state' (the preset's "
                           "COI, the historical behaviour), 'volume' (the "
                           "centre of the volume's AABB), or explicit world "
                           "coordinates 'x,y,z'. Use 'volume' whenever the "
                           "preset's COI is off-centre, or the volume leaves "
                           "the frame partway round the orbit. View 0 is the "
                           "preset camera either way.")
    rend.add_argument("--jitter", action="store_true",
                      help="Stratified (jittered) ray sampling — breaks the "
                           "coherent wood-grain rings of uniform lattice "
                           "sampling at the cost of fine stochastic noise. "
                           "Default OFF (deterministic CLI renders).")
    rend.add_argument("--state", type=str, default=None,
                      help="Path to a GUI render-state preset "
                           "(RenderStates/*.json): TF colour+opacity control "
                           "points, arcball camera, shading params, data-range "
                           "remap, samples-per-ray, batch size. Overrides "
                           "--colormap / --azi/--polar/--dist / --shading* / "
                           "--spp. Same file works for renderer.py and "
                           "render_baseline.py, so every method renders the "
                           "identical frame.")
    rend.add_argument("--tf_range", type=str, default=None,
                      help="Force the TransferFunction's (min_value, "
                           "max_value) to LO,HI in data units (e.g. "
                           "'-1,1') AFTER --colormap / --pvsm loading. "
                           "Use this to unify TF mapping across INR vs GT "
                           "renders: by default each picks model.min()/"
                           "max() (decoded INR is [-1, 0.94], GT is "
                           "[-1, 1]) so the same data value lands at "
                           "slightly different LUT indices.")
    rend.add_argument("--alpha_floor", type=float, default=None,
                      help="Zero the TF opacity for data values below this "
                           "threshold (in data units, e.g. -0.5 for the "
                           "act_gelu run).  Kills background speckle from "
                           "INR sub-voxel noise without otherwise touching "
                           "the colour map.  Apply *after* loading the TF "
                           "(after --pvsm / --colormap).")
    rend.add_argument("--tf_opacity_smooth", type=float, default=0.0,
                      help="Gaussian σ (in LUT entries; LUT has 4096 bins) "
                           "applied to the opacity LUT after loading. "
                           "Widens sharp opacity peaks, suppresses TF "
                           "aliasing on noisy INRs. Try 40 (= ~2%% of "
                           "data range) as a starting point.")
    rend.add_argument("--img_name", type=str, default="render.png",
                      help="Output file under Render/Output/.")

    misc = p.add_argument_group("misc")
    misc.add_argument("--device", type=str, default="cuda:0")
    misc.add_argument("--data_device", type=str, default=None)
    misc.add_argument("--inr_mode", choices=["mlp", "decoded"], default="decoded",
                      help="INR query mode. 'mlp' = continuous backbone (high "
                           "freq noise between voxels). 'decoded' (default) = "
                           "decode once to (D,H,W) volume then bilinearly "
                           "interpolate — matches GT semantics so the only "
                           "remaining gap is the 47-dB volume error.")
    return p

def _resolve_orbit_center(spec, scene):
    if spec in (None, "", "state"):
        return None
    if spec == "volume":

        c = (scene.scene_aabb[3:] / 2.0).tolist()
        print(f"[orbit] pivot = volume centre {[round(v, 1) for v in c]}")
        return c
    c = [float(v) for v in spec.split(",")]
    assert len(c) == 3, f"--orbit_center needs 3 comma-separated values, got {spec!r}"
    print(f"[orbit] pivot = {c}")
    return c

def _save_view(img, img_name, k, out_root, n_views=100):
    from imageio import imsave
    p = pathlib.PurePath(img_name)
    vdir = out_root / p.parent / "views"
    vdir.mkdir(parents=True, exist_ok=True)

    w = max(2, len(str(max(0, int(n_views) - 1))))
    out = vdir / f"{p.stem}_v{k:0{w}d}.png"
    imsave(out, (img.clamp(0, 1) * 255).cpu().numpy().astype("uint8"))
    return out

def _output_dir() -> Path:
    out = Path(__file__).resolve().parent / "renders"
    out.mkdir(parents=True, exist_ok=True)
    return out

def main():
    args = _build_argparser().parse_args()

    if args.raw_data == bool(args.save_dir):

        if args.raw_data and args.save_dir:
            print("error: pass either --save_dir or --raw_data, not both",
                  file=sys.stderr)
            sys.exit(2)
        if not args.raw_data and not args.save_dir:
            print("error: pass --save_dir <INR>  OR  --raw_data --data <.nc>",
                  file=sys.stderr)
            sys.exit(2)

    if args.raw_data and not args.data:
        print("error: --raw_data requires --data <.nc|.raw>", file=sys.stderr)
        sys.exit(2)

    device = args.device
    data_device = args.data_device or device
    if "cuda" in device and not torch.cuda.is_available():
        print(f"warning: {device} requested but CUDA unavailable; falling back to cpu")
        device = "cpu"
        data_device = "cpu"

    if "cuda" in device:

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    if args.raw_data:
        model = RawData(args.data, data_device)
        full_shape = model.full_shape
    else:
        from inr_loader import load_inr
        model = load_inr(args.save_dir, device=data_device,
                         data_device=data_device, mode=args.inr_mode)
        full_shape = model.full_shape
    model = model.to(data_device)
    model.eval()

    aabb = torch.tensor(
        [0.0, 0.0, 0.0,
         full_shape[0] - 1, full_shape[1] - 1, full_shape[2] - 1])
    if args.dist is None:
        args.dist = float(
            (aabb[3] ** 2 + aabb[4] ** 2 + aabb[5] ** 2) ** 0.5)
    coi = aabb.reshape(2, 3).mean(dim=0)
    camera = Camera(device, scene_aabb=aabb, coi=coi,
                    azi_deg=args.azi, polar_deg=args.polar, dist=args.dist)

    tf = TransferFunction(device, float(model.min().item()),
                          float(model.max().item()), args.colormap)

    if args.tf_range is not None:
        lo, hi = [float(s) for s in args.tf_range.split(",")]
        tf.set_minmax(lo, hi)
        print(f"[tf] range forced to [{lo}, {hi}] "
              f"(overrides model.min()/max())")
    if args.alpha_floor is not None:
        tf.apply_alpha_floor(args.alpha_floor)
        print(f"[tf] alpha_floor applied at v={args.alpha_floor}")
    if args.tf_opacity_smooth > 0:
        tf.smooth_opacity_lut(args.tf_opacity_smooth)
        print(f"[tf] opacity LUT smoothed with σ={args.tf_opacity_smooth} "
              f"entries (~{args.tf_opacity_smooth/tf.num_dict_entries*100:.2f}%"
              f" of data range)")

    cli_shading = {"ambient": args.ambient, "diffuse": args.diffuse,
                    "specular": args.specular, "shininess": args.shininess}
    shading_cfg = {
        "enabled": bool(args.shading),
        "shade_mode": args.shade_mode,
        "grad_threshold": 1e-4,
    }
    shading_cfg.update(_SHADE_DEFAULTS)
    shading_cfg.update({k: v for k, v in cli_shading.items() if v is not None})

    spp = args.spp if args.spp is not None else 256
    batch_size = args.batch_size if args.batch_size is not None else 2 ** 23
    if args.state:
        import render_state as rs
        st = rs.load_state(args.state)
        rs.apply_tf_and_camera(st, tf, camera)
        shading_cfg = rs.shading_from_state(
            st, force_shading=False,
            defaults=dict(_SHADE_DEFAULTS, shade_mode=args.shade_mode),
            overrides=cli_shading)

        spp = args.spp if args.spp is not None else rs.spp_from_state(st, 512)
        batch_size = (args.batch_size if args.batch_size is not None
                      else rs.batch_from_state(st, 2 ** 23))
        print(f"[state] spp={spp} batch_size={batch_size} "
              f"({Path(args.state).name})")

    scene = Scene(model, camera, full_shape, tuple(args.hw),
                  batch_size, spp, tf, device, data_device,
                  shading=shading_cfg)
    scene.jitter = bool(args.jitter)

    if not args.no_preintegrate:

        tf.preintegration = True
        tf.precompute_preintegration()
        scene.preintegrate = True
        print(f"[tf] pre-integrated TF table built (K={tf.preint_K})")

    if args.orbit_views > 0:

        n = args.orbit_views
        pivot = _resolve_orbit_center(args.orbit_center, scene)
        scene.current_order_spot = 0
        camera.orbit(0.0, pivot)
        scene.on_setting_change()
        _ = scene.render_checkerboard()
        times = np.zeros(n)
        for k in range(n):
            camera.orbit(k * 360.0 / n, pivot)
            scene.on_setting_change()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scene.current_order_spot = 0
            t0 = sync_time()
            img, _ = scene.render_checkerboard()
            t1 = sync_time()
            times[k] = t1 - t0
            _save_view(img, args.img_name, k, _output_dir(), n)
        print(f"[ORBIT] views={n} hw={args.hw[1]}x{args.hw[0]} spp={spp} "
              f"batch={batch_size}: mean={times.mean()*1000:.1f}ms "
              f"std={times.std()*1000:.1f}ms min={times.min()*1000:.1f}ms "
              f"max={times.max()*1000:.1f}ms FPS={1/times.mean():.3f}",
              flush=True)
        camera.orbit(0.0, pivot)
        scene.on_setting_change()
        scene.current_order_spot = 0
        img, _ = scene.render_checkerboard()
    else:
        t0 = sync_time()
        img, _ = scene.render_checkerboard()
        t1 = sync_time()
        print(f"[render] {args.hw[1]}x{args.hw[0]} spp={spp}: "
              f"{t1 - t0:.3f}s")

    from imageio import imsave
    out_path = _output_dir() / args.img_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imsave(out_path,
           (img * 255).cpu().numpy().astype(np.uint8))
    print(f"[render] wrote {out_path}")

if __name__ == "__main__":
    main()
