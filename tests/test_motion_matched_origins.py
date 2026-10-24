"""Realized tile labels, not difficulty ordinals, determine motion placement."""
from types import SimpleNamespace

import pytest
import torch
import yaml

from instinct_mj.terrains.motion_matching import motion_matched_origins
from instinct_mj.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg


def test_realized_tiles_keep_noncontiguous_ids_and_exclude_unmatched_tiles(tmp_path):
    metadata = tmp_path/'terrain.yaml'
    metadata.write_text(yaml.safe_dump({'terrains': [{'terrain_id': 10}, {'terrain_id': 20}]}))
    configs = []
    for difficulty in (.0, .99):
        cfg = MotionMatchedTerrainCfg(path=str(tmp_path), metadata_yaml=str(metadata))
        cfg.difficulty = difficulty
        configs.append(cfg)
    # A non-matched tile's difficulty is not evidence of a motion-terrain match.
    configs.append(SimpleNamespace(difficulty=.5))
    other = MotionMatchedTerrainCfg(path=str(tmp_path), metadata_yaml=str(tmp_path/'other.yaml'))
    other.difficulty = 0.
    configs.append(other)
    origins = torch.tensor([[[1., 2., 3.], [10., 20., 30.], [99., 0., 0.], [98., 0., 0.]]])
    scene = SimpleNamespace(terrain=SimpleNamespace(terrain_origins=origins, subterrain_specific_cfgs=configs))
    rng = torch.get_rng_state().clone()
    value = motion_matched_origins(scene, metadata_yaml=metadata)
    torch.testing.assert_close(value[10], origins[0, :1])
    torch.testing.assert_close(value[20], origins[0, 1:2])
    assert torch.equal(rng, torch.get_rng_state())
    origins.fill_(9.)
    assert value[10][0, 0] == 1.  # query result owns its snapshot
    configs[0].difficulty = float('nan')
    with pytest.raises(ValueError, match='finite'):
        motion_matched_origins(scene, metadata_yaml=metadata)


def test_duplicate_terrain_labels_reject_before_scene_access(tmp_path):
    metadata = tmp_path/'terrain.yaml'
    metadata.write_text(yaml.safe_dump({'terrains': [{'terrain_id': 10}, {'terrain_id': 10}]}))
    with pytest.raises(ValueError, match='unique integer'):
        motion_matched_origins(None, metadata_yaml=metadata)
