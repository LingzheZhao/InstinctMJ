"""Explicit runtime inputs for the existing native flat/perceptive task registry.

Configuration imports never inspect a dataset, select a motion, or write a
metadata file. MotionReferenceManager remains the only reference-clock owner.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from collections.abc import Mapping


def make_shadowing_env_cfg(
    task_id: str, *, play: bool = False, motion_buffers: Mapping | None = None,
    motion_folder: str | None = None, robot=None, robot_model_path: str | None = None,
    num_envs: int | None = None, physics_dt: float | None = None, decimation: int | None = None,
    terrain=None, terrain_metadata: str | None = None, sampling: Mapping | None = None,
    reference_frame_interval_s: float | None = None,
):
    """Build the registered task with explicit source, asset, clock and terrain inputs.

    Supply either native ``MotionBufferCfg`` objects (including downstream
    ``class_type`` subclasses), or a native motion folder. The factory does not
    own another registry, motion clock, sampler or training loop. ``sampling``
    holds explicitly selected native buffer settings; no dataset-size heuristic
    enables/disables sampling. With native buffers, their metadata is not read.
    """
    from instinct_mj.tasks.registry import load_env_cfg
    cfg = load_env_cfg(task_id, play=play)
    return configure_shadowing_env_cfg(cfg, motion_buffers=motion_buffers, motion_folder=motion_folder,
        robot=robot, robot_model_path=robot_model_path, num_envs=num_envs, physics_dt=physics_dt,
        decimation=decimation, terrain=terrain, terrain_metadata=terrain_metadata,
        sampling=sampling, reference_frame_interval_s=reference_frame_interval_s)


def configure_shadowing_env_cfg(
    cfg, *, motion_buffers=None, motion_folder=None, robot=None, robot_model_path=None,
    num_envs=None, physics_dt=None, decimation=None, terrain=None, terrain_metadata=None,
    sampling=None, reference_frame_interval_s=None,
):
    """Apply explicit inputs to a fresh native task config; perform no file I/O.

    This also permits native applications that already own a resolved registered
    config to use the same input boundary, without loading a task twice.
    """
    if (motion_buffers is None) == (motion_folder is None):
        raise ValueError("Supply exactly one of native motion_buffers or motion_folder")
    reference = next((sensor for sensor in cfg.scene.sensors if sensor.name == "motion_reference"), None)
    if reference is None:
        raise ValueError("The selected task has no native motion_reference sensor")
    if motion_buffers is not None:
        if not isinstance(motion_buffers, Mapping) or not motion_buffers:
            raise ValueError("motion_buffers must contain named native MotionBufferCfg inputs")
        reference.motion_buffers = deepcopy(dict(motion_buffers))
    else:
        if not isinstance(motion_folder, str) or not motion_folder.strip():
            raise ValueError("motion_folder must be an explicit nonempty path")
        for buffer in reference.motion_buffers.values():
            buffer.path = str(Path(motion_folder).expanduser())
            if hasattr(buffer, "metadata_yaml"):
                buffer.metadata_yaml = (str(Path(terrain_metadata).expanduser()) if terrain_metadata is not None
                                        else str(Path(buffer.path) / "metadata.yaml"))
    reference.mp_split_method = "None"
    if sampling is not None and not isinstance(sampling, Mapping):
        raise ValueError("sampling must hold explicit native motion-buffer settings")
    for buffer in reference.motion_buffers.values():
        if not isinstance(buffer.class_type, type):
            raise ValueError("Every native motion buffer requires its class_type")
        for key, value in (sampling or {}).items():
            if not hasattr(buffer, key):
                raise ValueError(f"Unknown native motion sampling setting: {key}")
            setattr(buffer, key, deepcopy(value))
    # Sampling ownership is explicit. Downstream samplers do not inherit donor
    # bin-weight updates or smoothing merely because the template contained them.
    if "beyond_adaptive_sampling" in cfg.curriculum:
        cfg.curriculum["beyond_adaptive_sampling"] = None
    if "bin_fail_counter_smoothing" in cfg.events:
        cfg.events["bin_fail_counter_smoothing"] = None
    if robot is not None:
        if not robot_model_path:
            raise ValueError("A supplied robot asset requires its matching robot_model_path")
        cfg.scene.entities["robot"] = deepcopy(robot)
        if "robot_reference" in cfg.scene.entities:
            cfg.scene.entities["robot_reference"] = deepcopy(robot)
    if robot_model_path is not None:
        reference.robot_model_path = str(Path(robot_model_path).expanduser())
    if num_envs is not None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        cfg.scene.num_envs = num_envs
    if physics_dt is not None:
        if type(physics_dt) not in (int, float) or not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive")
        cfg.sim.mujoco.timestep = physics_dt
    if decimation is not None:
        if type(decimation) is not int or decimation < 1:
            raise ValueError("decimation must be a positive integer")
        cfg.decimation = decimation
    reference.update_period = float(cfg.sim.mujoco.timestep) * cfg.decimation
    if reference_frame_interval_s is not None:
        if (type(reference_frame_interval_s) not in (int, float)
                or not math.isfinite(reference_frame_interval_s) or reference_frame_interval_s <= 0):
            raise ValueError("reference_frame_interval_s must be finite and positive")
        reference.frame_interval_s = reference_frame_interval_s
    if terrain is not None:
        cfg.scene.terrain = deepcopy(terrain)
    generator = getattr(cfg.scene.terrain, "terrain_generator", None)
    matched = None if generator is None else generator.sub_terrains.get("motion_matched")
    if matched is not None:
        if terrain_metadata is None:
            metadata = {getattr(buffer, "metadata_yaml", None) for buffer in reference.motion_buffers.values()}
            if len(metadata) != 1 or None in metadata or "" in metadata:
                raise ValueError("Motion-matched terrain requires explicit terrain_metadata")
            terrain_metadata = next(iter(metadata))
        matched.metadata_yaml = str(Path(terrain_metadata).expanduser())
        matched.path = str(Path(motion_folder).expanduser() if motion_folder is not None
                           else Path(matched.metadata_yaml).parent)
    elif terrain_metadata is not None:
        raise ValueError("terrain_metadata requires a motion-matched native terrain")
    return cfg
