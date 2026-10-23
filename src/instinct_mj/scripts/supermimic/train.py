"""Train the task's depth policy with SuperMimic (single stage: world model with a learned depth head + BPTT policy).

Example:
    instinct-supermimic-train --motion-folder ~/Xyk/Datasets/20251116_50cm_kneeClimbStep1 --num-envs 1024

The simulator is the unchanged ``Instinct-Perceptive-Shadowing-G1-v0`` env (DR, resets, adaptive sampling, depth camera,
terminations) and is only used to collect data. The policy is the task's own actor, trained ONLY by back-propagation
through the world model; checkpoints are written in the stock ``OnPolicyRunner`` format, so they load with
``instinct-play`` on the Play task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tyro
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.config import SuperMimicRunnerCfg
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_vecenv_cls
from instinct_mj.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import (
    TerrainMotionCfg,
    _maybe_disable_single_motion_binning,
)
from instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg import make_curriculum, make_events
from instinct_mj.utils.dict import class_to_dict
from instinct_mj.utils.noise import (
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

TASK_ID = "Instinct-Perceptive-Shadowing-G1-v0"


@dataclass
class TrainArgs:
    task: str = TASK_ID
    num_envs: int = 1024
    motion_folder: str | None = None
    """Dataset root with ``metadata.yaml`` (default: the task's ``MOTION_FOLDER``)."""
    device: str = "cuda:0"
    depth_noise: bool = True
    """Enable the camera noise models (``--depth-noise`` / ``--no-depth-noise``), see ``_enable_depth_noise``."""
    resume_init_new_heads: bool = False
    """Allow resuming from a sidecar written before the failure head existed: the head is created fresh (explicit opt-in)."""
    resume: str | None = None
    """Checkpoint (``model_<it>.pt``) to resume from (its ``supermimic_state_<it>.pt`` sidecar must be next to it)."""
    log_root: str = "logs/instinct_rl"
    agent: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)


def _enable_depth_noise(cfg) -> None:
    """Turn on the camera noise models that are commented out in ``make_perceptive_scene_sensors`` (same params), in front
    of ``normalize`` / ``crop_and_resize``. Left out: ``reflection_artifact_noise`` / ``sky_artifact_noise`` (the camera
    range is clipped to 2 m, so sky artifacts never show up and reflections are covered by the generic artifacts)."""
    camera_cfg = next(s for s in cfg.scene.sensors if s.name == "camera")
    tail = {k: camera_cfg.noise_pipeline[k] for k in ("normalize", "crop_and_resize")}
    camera_cfg.noise_pipeline = {
        "depth_contour_noise": DepthContourNoiseCfg(contour_threshold=1.8, maxpool_kernel_size=1),
        "depth_artifact_noise": DepthArtifactNoiseCfg(),
        "stereo_noise": RangeBasedGaussianNoiseCfg(max_value=1.2, min_value=0.12, noise_std=0.02),
        "gaussian_blur_noise": GaussianBlurNoiseCfg(kernel_size=3, sigma=0.5),
        "stereo_too_close_noise": StereoTooCloseNoiseCfg(),
        **tail,
    }


def apply_motion_folder(cfg, motion_folder: str | None, restore_baseline_sampling: bool = True) -> None:
    """Override the dataset of the task's env cfg (shared by training and evaluation).

    ``restore_baseline_sampling``: re-enable the training bin sampling / curriculum (the Play cfg must keep its own)."""
    if motion_folder is None:
        return
    motion_reference_cfg = next(sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "motion_reference")
    motion_buffer = next(iter(motion_reference_cfg.motion_buffers.values()))
    # same as ``G1PerceptiveShadowingEnvCfg.__post_init__``
    motion_buffer.path = os.path.expanduser(motion_folder)
    motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
    terrain_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path = motion_buffer.path
    terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
    if not restore_baseline_sampling:
        return
    # ``load_env_cfg`` already ran ``__post_init__`` for the default dataset, which may have disabled the bin
    # sampling (single-motion datasets). Restore the baseline sampling before re-evaluating for the new dataset.
    baseline = TerrainMotionCfg()
    motion_buffer.motion_start_from_middle_range = baseline.motion_start_from_middle_range
    motion_buffer.motion_bin_length_s = baseline.motion_bin_length_s
    motion_buffer.env_starting_stub_sampling_strategy = baseline.env_starting_stub_sampling_strategy
    cfg.curriculum["beyond_adaptive_sampling"] = make_curriculum()["beyond_adaptive_sampling"]
    cfg.events["bin_fail_counter_smoothing"] = make_events()["bin_fail_counter_smoothing"]
    if _maybe_disable_single_motion_binning(motion_buffer):
        cfg.curriculum["beyond_adaptive_sampling"] = None
        cfg.events["bin_fail_counter_smoothing"] = None


def build_env_cfg(task: str, num_envs: int, motion_folder: str | None, depth_noise: bool = True):
    """The task's env cfg with only the motion folder overridden (and ``auto_reset`` off, see the runner)."""
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    # The runner needs the pre-reset data of finished envs and resets them explicitly (see ``SuperMimicRunner.collect``).
    cfg.auto_reset = False
    apply_motion_folder(cfg, motion_folder)
    if depth_noise:
        _enable_depth_noise(cfg)
    # NOTE: the task's ``height_scanner`` / critic ``height_scan`` stay in the env although the observation is constant
    # (the sensor only includes geom group 0, which is empty; a task bug, not fixed here): removing the critic term
    # would change the critic input size of the stock ``EncoderActorCritic`` and break ``runner.load`` in Play. The world
    # model / critic use the analytic heightmap scan instead.
    return cfg


def make_vec_env(task: str, cfg, device: str):
    env = load_env_cls(task)(cfg=cfg, device=device, render_mode=None)
    return load_vecenv_cls(task)(env, policy_group="policy", critic_group="critic")


def main() -> None:
    args = tyro.cli(TrainArgs)
    from instinct_mj.rl.supermimic.runner import SuperMimicRunner

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    cfg = build_env_cfg(args.task, args.num_envs, args.motion_folder, args.depth_noise)
    cfg.seed = args.agent.seed
    stock_cfg = load_instinct_rl_cfg(args.task)  # the task's own policy architecture / normalizer
    stock_cfg.device, stock_cfg.seed = args.device, args.agent.seed
    log_dir = Path(args.log_root) / stock_cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_supermimic")
    log_dir = log_dir.expanduser().resolve()
    print(f"[INFO] Task={args.task}, device={args.device}, num_envs={args.num_envs}, log_dir={log_dir}")
    vec_env = make_vec_env(args.task, cfg, args.device)
    runner = SuperMimicRunner(vec_env, {**args.agent.to_dict(), "stock": stock_cfg.to_dict()}, log_dir=str(log_dir), device=args.device)
    if args.resume is not None:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        runner.load(args.resume, init_new_heads=args.resume_init_new_heads)
    dump_yaml(log_dir / "params" / "env.yaml", class_to_dict(cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", stock_cfg.to_dict())
    dump_yaml(log_dir / "params" / "supermimic.yaml", args.agent.to_dict())
    try:
        runner.learn(num_learning_iterations=args.agent.max_iterations, init_at_random_ep_len=False)
    except KeyboardInterrupt:
        print("[WARN] Training interrupted.")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
