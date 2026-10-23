"""Global terrain heightmap + differentiable analytic height scan (replaces the ray-cast ``height_scanner``).

The heightmap is built once by ray-casting straight down against the static terrain geoms of the env's MuJoCo model
(robot geoms are excluded), and cached on disk. ``height_scan`` reproduces the pattern and output convention of
``mjlab.envs.mdp.height_scan`` for the task's ``height_scanner`` (GridPattern 1.6 x 1.0 m @ 0.1 m, yaw aligned, attached
to ``torso_link``): ``min(z_root - ground_z, max_distance)`` for every ray, rays ordered y-major (meshgrid "xy").

Known limitation: a top-down heightmap cannot represent overhangs (the real ray starts inside the geometry).
"""

from __future__ import annotations

import copy
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from .rotations import yaw_mat

HEIGHTMAP_CACHE_VERSION = 2

MISS_HEIGHT = -100.0
"""Ground height stored for cells without terrain (scan saturates at ``max_distance``)."""


def build_heightmap(mj_model: mujoco.MjModel, resolution: float, terrain_body: str = "terrain") -> tuple[np.ndarray, float, float]:
    """Return ``(heights[ny, nx], x0, y0)``; vertex (j, i) is at ``(x0 + i * resolution, y0 + j * resolution)``."""
    m = copy.deepcopy(mj_model)
    terrain_id = m.body(terrain_body).id
    is_terrain = m.geom_bodyid == terrain_id
    m.geom_group[~is_terrain] = 5  # only the terrain geoms are visible to the rays
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xpos, rbound = d.geom_xpos[is_terrain], m.geom_rbound[is_terrain]
    lo, hi = (xpos - rbound[:, None]).min(0), (xpos + rbound[:, None]).max(0)
    x0, y0 = np.floor(lo[:2] / resolution) * resolution
    nx = int(np.ceil((hi[0] - x0) / resolution)) + 1
    ny = int(np.ceil((hi[1] - y0) / resolution)) + 1
    z_top = float(hi[2]) + 1.0
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[int(m.geom_group[is_terrain][0])] = 1
    heights = np.full((ny, nx), MISS_HEIGHT, dtype=np.float32)
    vec = np.array([0.0, 0.0, -1.0])

    def cast_row(j: int) -> None:
        geomid = np.zeros(1, dtype=np.int32)
        for i in range(nx):
            pnt = np.array([x0 + i * resolution, y0 + j * resolution, z_top])
            dist = mujoco.mj_ray(m, d, pnt, vec, geomgroup, 1, -1, geomid)
            if dist >= 0:
                heights[j, i] = z_top - dist

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        list(pool.map(cast_row, range(ny)))
    return heights, float(x0), float(y0)


def load_or_build_heightmap(
    mj_model: mujoco.MjModel, metadata_yaml: str, resolution: float, cache_dir: str
) -> tuple[np.ndarray, float, float]:
    """Cache key: terrain geoms of the compiled model + the metadata yaml + resolution."""
    terrain_id = mj_model.body("terrain").id
    is_terrain = mj_model.geom_bodyid == terrain_id
    key = hashlib.sha1()
    key.update(f"v{HEIGHTMAP_CACHE_VERSION}".encode())
    with open(os.path.expanduser(metadata_yaml), "rb") as f:
        key.update(f.read())
    key.update(str(resolution).encode())
    for arr in (mj_model.geom_pos, mj_model.geom_quat, mj_model.geom_size, mj_model.geom_type, mj_model.geom_dataid,
                mj_model.geom_group, mj_model.geom_contype, mj_model.geom_conaffinity):  # fmt: skip
        key.update(np.ascontiguousarray(arr[is_terrain]).tobytes())
    # terrain body world transform (geom positions above are body-relative) and mesh geometry + topology
    key.update(np.ascontiguousarray(mj_model.body_pos[terrain_id]).tobytes())
    key.update(np.ascontiguousarray(mj_model.body_quat[terrain_id]).tobytes())
    for arr in (mj_model.mesh_vert, mj_model.mesh_face, mj_model.mesh_vertadr, mj_model.mesh_faceadr, mj_model.mesh_vertnum, mj_model.mesh_facenum):
        key.update(np.ascontiguousarray(arr).tobytes())
    cache_dir = os.path.expanduser(cache_dir)
    path = os.path.join(cache_dir, key.hexdigest() + ".npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["heights"], float(data["x0"]), float(data["y0"])
    heights, x0, y0 = build_heightmap(mj_model, resolution)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, heights=heights, x0=x0, y0=y0)
    return heights, x0, y0


class HeightScanner:
    """Bilinear heightmap lookup on the task's yaw-aligned ``height_scanner`` grid."""

    def __init__(
        self,
        heights: np.ndarray,
        x0: float,
        y0: float,
        resolution: float,
        size: tuple[float, float],
        pattern_resolution: float,
        max_distance: float,
        device: str,
    ):
        self.heights = torch.tensor(heights, device=device)[None, None]  # (1, 1, ny, nx)
        self.x0, self.y0, self.res = x0, y0, resolution
        self.max_distance = max_distance
        # same construction as mjlab ``GridPatternCfg.generate_rays``
        sx, sy = size
        x = torch.arange(-sx / 2, sx / 2 + pattern_resolution * 0.5, pattern_resolution, device=device)
        y = torch.arange(-sy / 2, sy / 2 + pattern_resolution * 0.5, pattern_resolution, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        self.offsets = torch.stack((grid_x.flatten(), grid_y.flatten()), -1)  # (K, 2)
        self.num_rays = self.offsets.shape[0]

    def ground_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear ground height at world points ``xy`` (B, K, 2) -> (B, K)."""
        ny, nx = self.heights.shape[-2:]
        gx = (xy[..., 0] - self.x0) / (self.res * (nx - 1)) * 2 - 1
        gy = (xy[..., 1] - self.y0) / (self.res * (ny - 1)) * 2 - 1
        grid = torch.stack((gx, gy), -1).unsqueeze(0)  # (1, B, K, 2)
        out = F.grid_sample(self.heights, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out[0, 0]

    def height_scan(self, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Height of the root above the ground at every ray: (B, 187), differentiable w.r.t. ``p`` and ``R``."""
        H = yaw_mat(R)[..., :2, :2]  # (B, 2, 2)
        xy = p[:, None, :2] + self.offsets @ H.transpose(-1, -2)  # (B, K, 2)
        return torch.clamp(p[:, 2:3] - self.ground_height(xy), max=self.max_distance)
