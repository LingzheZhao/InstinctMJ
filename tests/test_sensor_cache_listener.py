"""Sensor reads after a state write see the written state.

mjlab_research's Simulation notifies its data listeners after every step, forward and reset, and InstinctRlEnv
registers the scene's sensor-cache invalidation as one. The first test runs a real InstinctRlEnv (MuJoCo Warp on the
CPU unless CUDA is available); the second reads the motion reference manager's own frames on the CPU.
"""
from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import Sensor, SensorCfg
from mjlab.sim.sim import SimulationCfg

from instinct_mj.envs import InstinctRlEnv, InstinctRlEnvCfg

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_ENVS = 2
BOX_XML = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
SENSOR = 'box_height'


@dataclass(kw_only=True)
class HeightSensorCfg(SensorCfg):
    def build(self):
        return HeightSensor(self)


class HeightSensor(Sensor[torch.Tensor]):
    """The box's world height, copied out of the simulation data as mjlab's contact sensor copies its data.

    mjlab's builtin sensors return a view of sensordata, which a stale cache would still follow.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def edit_spec(self, scene_spec, entities):
        del scene_spec, entities

    def initialize(self, mj_model, model, data, device):
        del model, device
        self._sim_data = data
        self._body = mj_model.body('box/box').id

    def _compute_data(self):
        return self._sim_data.xpos[:, self._body, 2:3].clone()


def sensed_height(env):
    return env.scene[SENSOR].data.clone()


def lift_box(env, env_ids):
    """A step event that writes the box 1 m higher, at rest, after the reward has read the sensor."""
    del env_ids
    box = env.scene['box']
    pose = box.data.root_link_pose_w.clone()
    pose[:, 2] += 1.
    box.write_root_link_pose_to_sim(pose)
    box.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def test_observation_after_a_step_event_write_sees_the_written_state():
    cfg = InstinctRlEnvCfg(
        scene=SceneCfg(num_envs=NUM_ENVS, env_spacing=3.,
                       entities={'box': EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BOX_XML))},
                       sensors=(HeightSensorCfg(name=SENSOR),)),
        observations={'obs': ObservationGroupCfg(terms={'height': ObservationTermCfg(func=sensed_height)})},
        rewards={'height': RewardTermCfg(func=lambda env: sensed_height(env).sum(1), weight=1.)},
        events={'lift': EventTermCfg(func=lift_box, mode='step')},
        sim=SimulationCfg(njmax=10), decimation=2, episode_length_s=10., scale_rewards_by_dt=False)
    env = InstinctRlEnv(cfg=cfg, device=DEVICE)
    env.reset()
    for _ in range(3):
        obs, reward, *_ = env.step(torch.zeros(NUM_ENVS, 0, device=DEVICE))
        written = env.scene['box'].data.root_link_pos_w[:, 2:3]
        # The reward read the sensor before the lift; the observation reads it after the lift and the forward.
        torch.testing.assert_close(reward[:, None], written - 1., rtol=0, atol=1e-4)
        torch.testing.assert_close(obs['obs'], written, rtol=0, atol=1e-6)
    env.close()


def test_reference_frames_at_the_robot_root_follow_a_state_write_at_the_same_reference_time():
    from types import SimpleNamespace

    from test_reference_clock import manager
    value, _ = manager()
    root = SimpleNamespace(root_link_pos_w=torch.tensor([[1., 2., 0.7], [3., 4., 0.8]]),
                           root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
    value._entity = SimpleNamespace(data=root)
    first = value.reference_relative_base_pos.clone()
    torch.testing.assert_close(first[:, :2], root.root_link_pos_w[:, :2])
    # Without a data change the frame stays as it was.
    value._invalidate_cache()
    torch.testing.assert_close(value.reference_relative_base_pos, first, rtol=0, atol=0)
    # A written root at the same reference time is read after the scene's invalidation.
    root.root_link_pos_w = root.root_link_pos_w + torch.tensor([.5, -.5, 0.])
    assert value.elapsed_time.tolist() == [0., 0.]
    value._invalidate_cache()
    torch.testing.assert_close(value.reference_relative_base_pos[:, :2], root.root_link_pos_w[:, :2])
