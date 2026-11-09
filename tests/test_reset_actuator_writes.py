"""Post-reset control writes through InstinctRlEnv on a real mjlab entity with delayed actuators.

The lifecycle is the CPU stand-in of test_runtime_lifecycle; only the scene holds a real mjlab
Entity and Simulation (MuJoCo Warp on the CPU unless CUDA is available). No training runs.
The entity set-up follows mjlab's tests/test_delayed_actuator.py (Apache-2.0, LICENSES/).
"""
import mujoco
import pytest
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

from test_runtime_lifecycle import CpuLifecycleEnv

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
NUM_ENVS = 3
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def make_entity(actuator_cfg):
    cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
                    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg(
                        target_names_expr=('joint.*',), effort_limit=100., stiffness=80., damping=10.,
                        delay_min_lag=2, delay_max_lag=2),)))
    entity = Entity(cfg)
    model = entity.compile()
    sim = Simulation(num_envs=NUM_ENVS, cfg=SimulationCfg(), model=model, device=DEVICE)
    entity.initialize(model, sim.model, sim.data, DEVICE)
    return entity, sim


class EntityScene:
    """The lifecycle's scene events, with control writes and resets passed to a real entity."""
    num_envs = NUM_ENVS

    def __init__(self, env, entity):
        self.env, self.entity = env, entity

    def reset(self, ids):
        self.env.events.append(('scene.reset', ids.tolist()))
        self.entity.reset(ids.to(DEVICE))

    def update(self, dt):
        self.env.events.append(('scene.update', dt))

    def write_data_to_sim(self, env_ids=None):
        self.env.events.append('write' if env_ids is None else ('write', env_ids.tolist()))
        self.entity.write_data_to_sim(None if env_ids is None else env_ids.to(DEVICE))


@pytest.mark.parametrize('actuator_cfg', [BuiltinPositionActuatorCfg, IdealPdActuatorCfg])
def test_partial_reset_writes_only_reset_worlds_and_holds_their_pose(actuator_cfg):
    entity, sim = make_entity(actuator_cfg)
    env = CpuLifecycleEnv()
    env.scene = EntityScene(env, entity)
    entity.write_joint_state_to_sim(torch.zeros(NUM_ENVS, 2, device=DEVICE), torch.zeros(NUM_ENVS, 2, device=DEVICE))
    for step in range(5):
        entity.set_joint_position_target(torch.full((NUM_ENVS, 2), step / 10., device=DEVICE))
        entity.write_data_to_sim()
    delay = entity.actuators[0]._delay_buffer
    keep = torch.tensor([1, 2], device=DEVICE)
    history, ctrl = delay._buffer.buffer[keep].clone(), sim.data.ctrl[keep].clone()
    lags, steps = delay._current_lags[keep].clone(), delay._step_count[keep].clone()

    reset = torch.tensor([0])
    pose = torch.tensor([[0.3, -0.2]], device=DEVICE)
    real_reset = env.scene.reset
    def reset_to_pose(ids):
        real_reset(ids)
        entity.write_joint_state_to_sim(pose, torch.zeros_like(pose), env_ids=ids.to(DEVICE))
    env.scene.reset = reset_to_pose
    env.reset(env_ids=reset)

    assert env.events.count(('write', [0])) == 1 and 'write' not in env.events
    torch.testing.assert_close(delay._buffer.buffer[keep], history, rtol=0, atol=0)
    torch.testing.assert_close(sim.data.ctrl[keep], ctrl, rtol=0, atol=0)
    torch.testing.assert_close(delay._current_lags[keep], lags, rtol=0, atol=0)
    torch.testing.assert_close(delay._step_count[keep], steps, rtol=0, atol=0)
    # The reset world holds its post-reset pose: the builtin actuator writes the pose as its target,
    # the ideal PD actuator writes kp * (pose - q) = 0.
    expected = pose[0] if actuator_cfg is BuiltinPositionActuatorCfg else torch.zeros(2, device=DEVICE)
    torch.testing.assert_close(sim.data.ctrl[0], expected)
