"""Reads the SuperMimic state / context of every env from the live mjlab perceptive shadowing env."""

from __future__ import annotations

import torch

from .mdp import MDP, XI_COM, XI_DQ, XI_FRICTION_START, XI_KD, XI_KP, XI_MASS
from .rotations import quat_to_mat


class SimReader:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        idx = self.robot.indexing
        self.free_q = idx.free_joint_q_adr.long()
        self.free_v = idx.free_joint_v_adr.long()
        self.joint_q = idx.joint_q_adr.long()
        self.joint_v = idx.joint_v_adr.long()
        self.contact_sensor = env.scene["contact_forces"]
        self.contact_body_ids = list(env.reward_manager.get_term_cfg("undesired_contacts").params["asset_cfg"].body_ids)
        self.contact_threshold = env.reward_manager.get_term_cfg("undesired_contacts").params["threshold"]

    def state(self):
        """Exact simulator state (from ``qpos`` / ``qvel``, no derived-quantity staleness)."""
        data = self.env.sim.data
        qpos, qvel = data.qpos, data.qvel
        p = qpos[:, self.free_q[:3]].clone()
        R = quat_to_mat(qpos[:, self.free_q[3:7]])
        v = qvel[:, self.free_v[:3]].clone()  # linear velocity of the body-frame origin, world frame
        w = (R @ qvel[:, self.free_v[3:6]].unsqueeze(-1)).squeeze(-1)  # free-joint angular velocity is in the body frame
        return p, R, v, w, qpos[:, self.joint_q].clone(), qvel[:, self.joint_v].clone()

    def last_action(self) -> torch.Tensor:
        return self.env.action_manager.action.clone()

    def contact(self) -> torch.Tensor:
        """Per-body indicator used by the ``undesired_contacts`` reward: max force over the sensor history > threshold."""
        force_norm = torch.norm(self.contact_sensor.data.force_history, dim=-1).max(-1).values
        return (force_norm[:, self.contact_body_ids] > self.contact_threshold).float()

    def context(self, mdp: MDP) -> torch.Tensor:
        """Per-env episode-constant context xi (what the startup domain randomization changed), see ``mdp.XI_*``."""
        env, robot = self.env, self.robot
        model = env.sim.model
        mj = env.sim.mj_model
        xi = torch.zeros(env.num_envs, mdp.xi_dim, device=env.device)
        xi[:, XI_DQ] = robot.data.default_joint_pos - robot.data.default_joint_pos_nominal
        torso = int(robot.indexing.body_ids[robot.body_names.index("torso_link")])
        xi[:, XI_COM] = model.body_ipos[:, torso] - torch.as_tensor(mj.body_ipos[torso], device=env.device)
        for act in robot.actuators:
            m_ = len(act.target_ids)
            ctrl = act.global_ctrl_ids.long()
            joints = act.target_ids.long()
            xi[:, XI_KP.start + joints] = model.actuator_gainprm[:, ctrl[:m_], 0] / mdp.kp_nom[joints]
            xi[:, XI_KD.start + joints] = model.actuator_gainprm[:, ctrl[m_:], 0] / mdp.kd_nom[joints]
        mass_ids, _ = robot.find_bodies(["torso_link", "left_ankle.*", "right_ankle.*", "left_wrist.*", "right_wrist.*"])
        body_ids = robot.indexing.body_ids[mass_ids].long()
        xi[:, XI_MASS] = model.body_mass[:, body_ids] / torch.as_tensor(mj.body_mass[body_ids.cpu().numpy()], device=env.device, dtype=torch.float32)
        xi[:, XI_FRICTION_START:] = model.geom_friction[:, mdp.collision_geom_ids].flatten(1)
        return xi


# The task's ``height_scanner`` (``make_perceptive_scene_sensors``): GridPattern 1.6 x 1.0 m @ 0.1 m, max_distance 5 m.
HEIGHT_SCANNER_SIZE = (1.6, 1.0)
HEIGHT_SCANNER_RESOLUTION = 0.1
HEIGHT_SCANNER_MAX_DISTANCE = 5.0


def build_mdp(env, heightmap_resolution: float, heightmap_cache_dir: str, device: str):
    """Instantiate FK, motion library, heightmap scanner and analytic MDP for the live (unwrapped) env."""
    from .kinematics import G1Kinematics
    from .motion_lib import MotionLib
    from .terrain import HeightScanner, load_or_build_heightmap

    manager = env.scene["motion_reference"]
    motion = MotionLib(manager, device)
    kin = G1Kinematics(device, list(env.scene["robot"].joint_names), list(manager.cfg.link_of_interests))
    metadata_yaml = next(iter(manager.motion_buffers.values())).cfg.metadata_yaml
    heights, x0, y0 = load_or_build_heightmap(env.sim.mj_model, metadata_yaml, heightmap_resolution, heightmap_cache_dir)
    scanner = HeightScanner(
        heights, x0, y0, heightmap_resolution, HEIGHT_SCANNER_SIZE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_MAX_DISTANCE, device
    )
    return motion, kin, scanner, MDP(kin, motion, scanner, env, env.step_dt, device)
