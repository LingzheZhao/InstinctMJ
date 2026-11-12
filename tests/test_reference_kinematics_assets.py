"""The motion reference's kinematics on each named G1's model file agree with MuJoCo's (CPU, no scene)."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

LINKS = ('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link',
         'right_knee_link', 'right_ankle_roll_link', 'torso_link', 'left_shoulder_roll_link', 'left_elbow_link',
         'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link')


@pytest.mark.parametrize('name', ['instinct_torso_popsicle', 'mjlab_pelvis'])
def test_reference_link_poses_equal_mujoco_kinematics_to_float32_rounding(name):
    from instinct_mj.assets.g1_assets import g1_asset
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager
    path = g1_asset(name).model_path
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    hinges = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    (free,) = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    manager = MotionReferenceManager.__new__(MotionReferenceManager)
    manager.cfg, manager.device = SimpleNamespace(robot_model_path=path, link_of_interests=list(LINKS)), 'cpu'
    manager._num_joints, manager.sim_joint_names = len(hinges), list(hinges)
    manager._initialize_robot_kinematics()
    rng = np.random.default_rng(0)
    for _ in range(4):
        q = np.array([0.8 * rng.uniform(*model.jnt_range[model.joint(j).id]) for j in hinges])
        data.qpos[:] = model.qpos0
        start = int(model.jnt_qposadr[free])
        data.qpos[start:start + 7] = [0, 0, 0, 1, 0, 0, 0]
        for joint, value in zip(hinges, q):
            data.qpos[int(model.jnt_qposadr[model.joint(joint).id])] = value
        mujoco.mj_kinematics(model, data)
        poses = manager.target_link_pose_forward_kinematics(torch.tensor(q, dtype=torch.float32)[None])[0].numpy()
        for k, link in enumerate(LINKS):
            body = model.body(link).id
            np.testing.assert_allclose(poses[k, :3], data.xpos[body], rtol=0, atol=1e-6, err_msg=link)
            quaternion = poses[k, 3:] / np.linalg.norm(poses[k, 3:])
            assert 1 - abs(float(np.dot(quaternion, data.xquat[body]))) < 1e-6, link
