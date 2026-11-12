"""Model-field events on the research mjlab: the derived constants and the physics they feed.

The event manager of LingzheZhao/mjlab_research derives the mujoco_warp.set_const groups from the model fields an
event declares. Each InstinctMJ event that writes model fields must leave the derived constants equal to a full
recompute, in every world. Runs a real InstinctRlEnv (MuJoCo Warp on the CPU unless CUDA is available); no training.
"""
import mujoco
import pytest
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from mjlab.managers.event_manager import _DERIVED_FIELDS, recompute_groups
from mjlab.scene import SceneCfg
from mjlab.sim.sim import SimulationCfg
from mujoco_warp import SetConstGroup

from instinct_mj.envs import InstinctRlEnv, InstinctRlEnvCfg
from instinct_mj.envs.mdp.events.motion_reference import virtualize_articulation
from instinct_mj.envs.mdp.events.randomization import randomize_rigid_body_com
from instinct_mj.tasks.locomotion.mdp.events import randomize_rigid_body_mass

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_ENVS = 3

# A floating base with two hinged links, no ground: nothing touches anything.
ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0.3 0 0">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="0.4"/>
      </body>
      <body name="link2" pos="-0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="link2_geom" type="capsule" fromto="0 0 0 -0.3 0 0" size="0.05" mass="0.4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
DERIVED = tuple(dict.fromkeys(field for fields in _DERIVED_FIELDS.values() for field in fields))


def make_env(events):
    robot = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
                      articulation=EntityArticulationInfoCfg(actuators=(BuiltinPositionActuatorCfg(
                          target_names_expr=('joint.*',), effort_limit=100., stiffness=80., damping=10.),)))
    cfg = InstinctRlEnvCfg(
        scene=SceneCfg(num_envs=NUM_ENVS, env_spacing=3., entities={'robot': robot}),
        observations={'obs': ObservationGroupCfg(terms={'qpos': ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:, :3])})},
        events=events, sim=SimulationCfg(njmax=20), decimation=2, episode_length_s=10.)
    return InstinctRlEnv(cfg=cfg, device=DEVICE)


def derived_constants(env):
    # Read through the Warp arrays: some derived fields (eq_data) have vector types with no Torch view.
    return {field: torch.from_numpy(getattr(env.sim.wp_model, field).numpy().copy()) for field in DERIVED}


def assert_equal_to_full_recompute(env):
    after_event = derived_constants(env)
    env.sim.recompute_constants(SetConstGroup.ALL)
    fresh = derived_constants(env)
    for field in DERIVED:
        torch.testing.assert_close(after_event[field], fresh[field], rtol=1e-5, atol=1e-6, msg=field)
    return after_event


BASE = SceneEntityCfg('robot', body_names=('base',))
CASES = {
    'randomize_rigid_body_com': (randomize_rigid_body_com, 'body_ipos', ('body_ipos', 'body_iquat'),
                                 {'asset_cfg': BASE, 'com_range': {'x': (-.05, .05), 'y': (-.05, .05),
                                                                   'z': (-.05, .05)}}),
    'randomize_rigid_body_mass': (randomize_rigid_body_mass, 'body_mass', ('body_mass', 'body_ipos',
                                                                           'body_inertia', 'body_iquat'),
                                  {'asset_cfg': BASE, 'mass_distribution_params': (-.5, .5), 'operation': 'add'}),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_inertial_events_leave_derived_constants_equal_to_a_full_recompute_in_every_world(name):
    func, written, fields, params = CASES[name]
    assert func.model_fields == fields
    torch.manual_seed(3)
    reference = make_env({})
    default = derived_constants(reference)
    default_written = getattr(reference.sim.model, written).clone()
    reference.close()
    env = make_env({name: EventTermCfg(func=func, mode='startup', params=params)})
    base = env.scene['robot'].indexing.body_ids[0]
    # The event wrote a different value in each world ...
    value = getattr(env.sim.model, written)[:, base]
    assert value.shape[0] == NUM_ENVS and not torch.allclose(value[0], value[1])
    assert not torch.allclose(value, default_written[:, base].expand_as(value))
    # ... the event manager recomputed the inertial constants of each world ...
    after = assert_equal_to_full_recompute(env)
    groups = recompute_groups(fields, env.sim.mj_model)
    assert groups & SetConstGroup.INERT0
    for field in ('dof_invweight0', 'body_invweight0', 'actuator_acc0'):
        assert after[field].shape[0] == NUM_ENVS, field
        assert not torch.allclose(after[field][0], after[field][1]), field
        assert not torch.allclose(after[field], default[field].expand_as(after[field])), field
    env.close()


def test_gravity_compensation_needs_no_recompute_and_holds_only_the_written_worlds():
    env = make_env({'virtualize': EventTermCfg(func=virtualize_articulation, mode='reset',
                                               params={'asset_cfg': SceneEntityCfg('robot')})})
    assert virtualize_articulation.model_fields == ('body_gravcomp',)
    assert recompute_groups(virtualize_articulation.model_fields, env.sim.mj_model) == SetConstGroup.NONE
    before = derived_constants(env)
    env.reset(env_ids=torch.tensor([0], device=DEVICE))
    bodies = env.scene['robot'].indexing.body_ids
    gravcomp = env.sim.model.body_gravcomp
    assert gravcomp.shape[0] == NUM_ENVS
    assert bool((gravcomp[0, bodies] == 1.).all()) and not bool(gravcomp[1:, bodies].any())
    after = assert_equal_to_full_recompute(env)
    for field in DERIVED:
        torch.testing.assert_close(after[field], before[field], rtol=0, atol=0, msg=field)
    # The next step's accelerations, with no recompute and after a full one: the compensated world floats,
    # the others fall at g.
    env.sim.forward()
    qacc = env.sim.data.qacc.clone()
    env.sim.recompute_constants(SetConstGroup.ALL)
    env.sim.forward()
    torch.testing.assert_close(env.sim.data.qacc, qacc, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(qacc[0, :3], torch.zeros(3, device=DEVICE), rtol=0, atol=1e-5)
    gravity = float(env.sim.mj_model.opt.gravity[2])
    torch.testing.assert_close(qacc[1:, 2], torch.full((NUM_ENVS - 1,), gravity, device=DEVICE), rtol=1e-5,
                               atol=1e-5)
    env.close()
