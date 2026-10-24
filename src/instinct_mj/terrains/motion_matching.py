"""Query the realized motion-matched tiles without choosing motion or reset state."""
from pathlib import Path
import math

import torch
import yaml

from instinct_mj.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg


def motion_matched_origins(scene, *, metadata_yaml):
    """Return actual world origins keyed by declared terrain IDs, not tile ordinals.

    Mesh generation maps difficulty to an ordered metadata row. The row's
    terrain_id is a label (not necessarily contiguous); other tile types and
    tiles built from another metadata source are not matches.
    """
    path = Path(metadata_yaml).resolve()
    metadata = yaml.safe_load(path.read_text())
    records = metadata['terrains']
    labels = [row['terrain_id'] for row in records]
    if (not labels or any(type(label) is not int for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('motion-matched terrain metadata requires unique integer terrain IDs')
    terrain = scene.terrain
    origins = terrain.terrain_origins
    configs = terrain.subterrain_specific_cfgs
    if (origins.ndim != 3 or origins.shape[-1] != 3
            or len(configs) != origins.shape[0] * origins.shape[1]):
        raise ValueError('realized terrain tiles and origin grid differ')
    selected = {label: [] for label in labels}
    for cfg, origin in zip(configs, origins.reshape(-1, 3), strict=True):
        if not isinstance(cfg, MotionMatchedTerrainCfg) or Path(cfg.metadata_yaml).resolve() != path:
            continue
        difficulty = float(cfg.difficulty)
        if not math.isfinite(difficulty) or not bool(torch.isfinite(origin).all()):
            raise ValueError('motion-matched tile difficulty and world origin must be finite')
        ordinal = min(max(int(difficulty * len(records)), 0), len(records)-1)
        selected[labels[ordinal]].append(origin)
    return {label: (torch.stack(rows).clone() if rows else origins.new_empty((0, 3)))
            for label, rows in selected.items()}
