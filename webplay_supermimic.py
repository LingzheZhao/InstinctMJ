"""Launch the viser web viewer for the SuperMimic-trained perceptive shadowing policy.

Usage: python webplay_supermimic.py [checkpoint]   (default: newest model_*.pt of the overnight run)
"""
import glob
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

import instinct_mj.scripts.instinct_rl.play as playmod
import instinct_mj.tasks  # noqa: F401  (register tasks)
from instinct_mj.scripts.instinct_rl.play import PlayConfig, run_play
from instinct_mj.scripts.supermimic.common import apply_motion_folder

MOTION_FOLDER = os.path.expanduser("~/Xyk/Datasets/20251116_50cm_kneeClimbStep1")
RUN_DIR = "logs/instinct_rl/g1_perceptive_shadowing/2026-10-22_13-02-06_supermimic"

_real_load = playmod.load_env_cfg


def _patched_load(task_id, play=False):
    cfg = _real_load(task_id, play=play)
    apply_motion_folder(cfg, MOTION_FOLDER, restore_baseline_sampling=False)
    # one terrain row per terrain of the dataset, so every motion has its terrain in the scene
    with open(os.path.join(MOTION_FOLDER, "metadata.yaml"), encoding="utf-8") as f:
        num_terrains = len(yaml.safe_load(f)["terrains"])
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum, generator.num_rows, generator.num_cols = True, num_terrains, 1
    cfg.episode_length_s = 30.0  # play each motion to its end
    # no reference visualization (reference robot and link markers)
    cfg.scene.entities.pop("robot_reference", None)
    motion_reference_cfg = next(s for s in cfg.scene.sensors if s.name == "motion_reference")
    motion_reference_cfg.debug_vis = False
    motion_reference_cfg.reference_entity_name = None
    return cfg


playmod.load_env_cfg = _patched_load

if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    CKPT = max(glob.glob(f"{RUN_DIR}/model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
print(f"[webplay] checkpoint: {CKPT}", flush=True)
cfg = PlayConfig(viewer="viser", num_envs=1, checkpoint_file=CKPT)
run_play("Instinct-Perceptive-Shadowing-G1-v0", cfg)
