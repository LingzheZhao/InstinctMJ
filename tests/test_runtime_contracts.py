"""CPU checks of explicit task inputs and native lifecycle boundaries; no simulation steps."""
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_task_config_imports_do_not_write_selections_or_inspect_datasets():
    code = '''
import mjlab.envs, mujoco, instinct_rl.env
import builtins
original = builtins.open
def readonly(path, mode='r', *args, **kwargs):
    if any(flag in mode for flag in 'wax+'):
        raise AssertionError(f'config import wrote {path}')
    return original(path, mode, *args, **kwargs)
builtins.open = readonly
import instinct_mj.tasks.registry
from instinct_mj.tasks.shadowing.beyondmimic.config.g1 import beyondmimic_plane_cfg
from instinct_mj.tasks.shadowing.whole_body.config.g1 import plane_shadowing_cfg
from instinct_mj.tasks.shadowing.perceptive.config.g1 import perceptive_shadowing_cfg
beyondmimic_plane_cfg.g1_beyondmimic_plane_env_cfg()
plane_shadowing_cfg.g1_plane_shadowing_env_cfg()
perceptive_shadowing_cfg.G1PerceptiveShadowingEnvCfg()
perceptive_shadowing_cfg.G1PerceptiveShadowingOneMotionEnvCfg()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('task', ['Instinct-Shadowing-WholeBody-Plane-G1-v0',
                                 'Instinct-Perceptive-Shadowing-G1-v0'])
def test_factory_injects_native_buffer_and_clock_without_io(task, tmp_path):
    import instinct_mj.tasks
    from instinct_mj.motion_reference.motion_reference_cfg import MotionBufferCfg
    from instinct_mj.tasks.shadowing.factory import make_shadowing_env_cfg
    class NativeBuffer:
        pass
    source = MotionBufferCfg(class_type=NativeBuffer)
    kwargs = {} if 'WholeBody' in task else {'terrain_metadata':str(tmp_path/'metadata.yaml')}
    cfg = make_shadowing_env_cfg(task, motion_buffers={'current':source}, num_envs=3,
        physics_dt=.005, decimation=4, reference_frame_interval_s=.02, **kwargs)
    reference = next(sensor for sensor in cfg.scene.sensors if sensor.name == 'motion_reference')
    assert reference.motion_buffers['current'].class_type is NativeBuffer
    assert reference.motion_buffers['current'] is not source
    assert reference.mp_split_method == 'None'
    assert reference.update_period == .02 and reference.frame_interval_s == .02
    assert cfg.scene.num_envs == 3 and cfg.curriculum['beyond_adaptive_sampling'] is None
    assert cfg.events['bin_fail_counter_smoothing'] is None
    assert not (tmp_path/'metadata.yaml').exists()
    import mujoco
    spec = mujoco.MjSpec.from_file(reference.robot_model_path)
    assert list(spec.worldbody.bodies)[0].name == 'torso_link'


class CachedEnv:
    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.device, self.max_episode_length = 2, 'cpu', 10
        self.action_manager = SimpleNamespace(total_action_dim=3)
        self.observation_manager = SimpleNamespace(active_terms={'policy':['proprio']},
            group_obs_term_dim={'policy':[(3,)]}, compute=self.forbidden_compute)
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.obs_buf, self.resets, self.steps = {}, 0, 0
    def forbidden_compute(self):
        raise AssertionError('a cache read must never recompute noise/history')
    def reset(self, **kwargs):
        self.resets += 1
        self.obs_buf = {'policy':torch.full((2,3),float(self.resets))}
        return self.obs_buf, {}
    def step(self, actions):
        self.steps += 1
        self.action = actions.clone()
        self.obs_buf = {'policy':actions.clone()}
        return self.obs_buf, torch.zeros(2), torch.zeros(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool), {}


def test_wrapper_has_one_startup_reset_owner_and_cache_only_reads():
    from instinct_mj.rl.vecenv_wrapper import InstinctRlVecEnvWrapper
    native = CachedEnv()
    wrapper = InstinctRlVecEnvWrapper(native, clip_actions=.5)
    assert native.resets == 1
    state = torch.get_rng_state().clone()
    for _ in range(3):
        assert wrapper.get_observations()[0].tolist() == [[1.]*3]*2
    assert torch.equal(state,torch.get_rng_state()) and native.resets == 1
    wrapper.step(torch.ones(2,3))
    assert native.action.max() == .5
    assert wrapper.get_observations()[0].max() == .5
    native = CachedEnv()
    native.reset()
    wrapper = InstinctRlVecEnvWrapper(native, reset_on_init=False)
    assert native.resets == 1 and wrapper.get_observations()[0].max() == 1


def test_wrapper_depth_reads_owned_camera_outputs_and_cached_clean_pipeline():
    from instinct_mj.rl.vecenv_wrapper import InstinctRlVecEnvWrapper
    native = CachedEnv()
    image = torch.arange(8.).reshape(2,2,2,1)
    native.scene = {'camera':SimpleNamespace(data=SimpleNamespace(output={'distance_to_image_plane_noised':image}))}
    wrapper = InstinctRlVecEnvWrapper(native)
    wrapper._depth_pipeline = SimpleNamespace(clean_from_sensor=lambda:torch.ones(2,1,2,2))
    torch.testing.assert_close(wrapper.processed_depth(),image.permute(0,3,1,2))
    noisy = wrapper.processed_depth()
    noisy.zero_()
    assert image.max() == 7 and wrapper.processed_depth(noisy=False).sum() == 8


def test_hooks_preserve_real_tick_count_and_endpoint_order_without_extra_sense():
    from instinct_mj.envs.manager_based_rl_env import InstinctRlEnv
    from instinct_mj.envs.runtime_hooks import RuntimeCallbacks
    events = []
    native = SimpleNamespace(runtime_callbacks=RuntimeCallbacks(), _inside_control_step=True,
        observation_manager=SimpleNamespace(compute=lambda **kwargs: {}),
        scene=SimpleNamespace(update=lambda dt:events.append(('scene',dt))),
        sim=SimpleNamespace(forward=lambda:events.append('forward')),
        termination_manager=SimpleNamespace(compute=lambda:events.append('termination'),
            terminated=torch.tensor([False]),time_outs=torch.tensor([True])))
    unhook = native.runtime_callbacks.register(native.runtime_callbacks.physics,
        lambda env,dt:events.append(('physics',dt)))
    native.runtime_callbacks.register(native.runtime_callbacks.terminal,
        lambda env,term,trunc:events.append(('terminal',bool(trunc[0]))))
    InstinctRlEnv._install_runtime_hooks(native)
    native.scene.update(.005)
    native.termination_manager.compute()
    assert events == [('scene',.005),('physics',.005),'forward',('scene',0.),'termination',('terminal',True)]
    unhook(); unhook()
    events.clear(); native.scene.update(.005)
    assert events == [('scene',.005)]


def test_reference_clock_read_is_nonadvancing_and_owned():
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager
    manager = object.__new__(MotionReferenceManager)
    manager._timestamp = torch.tensor([.02,.1])
    clock = manager.elapsed_time
    clock.zero_()
    torch.testing.assert_close(manager.elapsed_time,torch.tensor([.02,.1]))


def test_passive_terminal_observer_cannot_change_physical_results_or_sampling_counts():
    from instinct_mj.envs.manager_based_rl_env import InstinctRlEnv
    from instinct_mj.envs.runtime_hooks import RuntimeCallbacks
    def run(observed):
        torch.manual_seed(19)
        counts = dict(physics=0, forward=0, sense=0, observation=0, history=0)
        state = SimpleNamespace(integrated=0., reached=0., reference_time=0.)
        terminal = []
        def forward():
            counts['forward'] += 1
            state.reached = state.integrated
        def scene(dt):
            state.reference_time += dt
        def compute(**kwargs):
            counts['observation'] += 1
            counts['history'] += int(kwargs['update_history'])
            return {'policy': torch.tensor([[state.reached]]) + torch.rand(1, 1)}
        native = SimpleNamespace(runtime_callbacks=RuntimeCallbacks(), _inside_control_step=True,
            observation_manager=SimpleNamespace(compute=compute), scene=SimpleNamespace(update=scene),
            sim=SimpleNamespace(forward=forward), obs_buf={},
            termination_manager=SimpleNamespace(compute=lambda: torch.tensor([state.reached >= 8.]),
                terminated=torch.tensor([False]), time_outs=torch.tensor([False])))
        native.runtime_callbacks.register(native.runtime_callbacks.physics,
            lambda env,dt:counts.__setitem__('physics',counts['physics']+1))
        if observed:
            native.runtime_callbacks.register(native.runtime_callbacks.terminal,
                lambda env,done,truncated:terminal.append(state.reached))
        InstinctRlEnv._install_runtime_hooks(native)
        results = []
        for _ in range(3):
            for _ in range(4):
                # A split-step solver has integrated its qpos while the last
                # forward-derived body reading is still stale until forward().
                state.integrated += 1.
                native.scene.update(.005)
            done = native.termination_manager.compute()
            reward = state.reached * .1
            counts['sense'] += 1
            native.obs_buf = native.observation_manager.compute(update_history=True)
            results.append((done, reward, native.obs_buf['policy'].clone()))
        return results, counts, state.reference_time, torch.get_rng_state(), terminal
    plain, seen = run(False), run(True)
    for left, right in zip(plain[0], seen[0], strict=True):
        assert torch.equal(left[0], right[0]) and left[1] == right[1] and torch.equal(left[2], right[2])
    assert [value[1] for value in plain[0]] == pytest.approx([.4, .8, 1.2])
    assert plain[1] == seen[1] == dict(physics=12, forward=3, sense=3, observation=3, history=3)
    assert plain[2] == seen[2] == pytest.approx(.06)
    assert torch.equal(plain[3], seen[3])
    assert seen[4] == [4., 8., 12.]


def test_terrain_resampling_changes_start_time_with_motion_and_origin():
    from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
    buffer = object.__new__(TerrainMotion)
    buffer.buffer_device, buffer.output_device = 'cpu', 'cpu'
    buffer._all_motion_num_selectable_origins = torch.tensor([0.,1.])
    buffer._assigned_env_motion_selection = torch.tensor([0,1])
    buffer._motion_buffer_start_time_s = torch.tensor([99.,.25])
    buffer._motion_weights = torch.ones(2)
    buffer.cfg = SimpleNamespace(motion_start_from_middle_range=[0.,0.])
    buffer._all_motion_origins_in_scene = torch.tensor([[[0.,0.,0.]],[[4.,1.,0.]]])
    buffer._assigned_env_origins = torch.zeros(2,3)
    buffer._safe_motion_resampling(torch.tensor([0,1]))
    buffer._sample_assigned_env_origins(torch.tensor([0,1]))
    assert buffer._assigned_env_motion_selection.tolist() == [1,1]
    assert buffer._motion_buffer_start_time_s.tolist() == [0.,.25]
    torch.testing.assert_close(buffer._assigned_env_origins,torch.tensor([[4.,1.,0.],[4.,1.,0.]]))


def test_height_scanner_sees_collision_terrain_not_robot_groups():
    import mujoco
    from instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg import _edit_perceptive_scene_spec
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="terrain">
        <geom name="floor" type="box" size="1 1 .1" group="2"/>
        <geom name="visual" type="box" size="1 1 .1" group="2" contype="0" conaffinity="0"/>
        </body></worldbody></mujoco>''')
    _edit_perceptive_scene_spec(spec)
    assert spec.geom('floor').group == 0 and spec.geom('visual').group == 2


def test_physical_state_refresh_does_not_reset_reference_or_advance_any_clock():
    from instinct_mj.envs.manager_based_rl_env import InstinctRlEnv
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager
    reference = object.__new__(MotionReferenceManager)
    reference._timestamp = torch.tensor([.02,.04,.06])
    events = []
    native = object.__new__(InstinctRlEnv)
    native.scene = SimpleNamespace(num_envs=3, sensors={'reference':reference,
        'camera':SimpleNamespace(reset=lambda ids:events.append(('sensor',ids.tolist())))})
    native.sim = SimpleNamespace(device='cpu',forward=lambda:events.append('forward'),
        sense=lambda:events.append('sense'))
    def compute(*, update_history, env_ids):
        events.append(('observations',update_history,env_ids.tolist()))
        return {'policy':torch.ones(3,4)}
    native.observation_manager = SimpleNamespace(reset=lambda ids:events.append(('history_reset',ids.tolist())),
                                                 compute=compute)
    actual = native.refresh_observations_after_state_write(env_ids=torch.tensor([1]))
    assert events == [('sensor',[1]),'forward','sense',('history_reset',[1]),('observations',True,[1])]
    torch.testing.assert_close(reference.elapsed_time,torch.tensor([.02,.04,.06]))
    assert actual['policy'].shape == (3,4)
    with pytest.raises(TypeError):
        actual['policy'] = torch.zeros(3,4)


@pytest.mark.parametrize('concatenate', [False, True])
def test_partial_reset_preserves_issued_noisy_observations_and_native_history(concatenate):
    from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg
    from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
    from instinct_mj.envs.runtime_hooks import scope_reset_observation_cache
    calls = []
    def value(env):
        calls.append(True)
        return env.value
    env = SimpleNamespace(num_envs=3, device='cpu', value=torch.zeros(3,2), obs_buf={})
    group = ObservationGroupCfg(terms={
        'plain':ObservationTermCfg(func=value, noise=UniformNoiseCfg(n_min=-1., n_max=1.)),
        'history':ObservationTermCfg(func=value, history_length=3),
    }, enable_corruption=True, concatenate_terms=concatenate)
    env.observation_manager = ObservationManager({'policy':group}, env)
    scope_reset_observation_cache(env)
    env.obs_buf = env.observation_manager.compute(update_history=True)
    def clone(tree):
        return {k:clone(v) for k,v in tree.items()} if isinstance(tree,dict) else tree.clone()
    previous = clone(env.obs_buf)
    ids, keep = torch.tensor([1]), torch.tensor([0,2])
    env.value[1] = 7.
    env.observation_manager.reset(ids)
    calls.clear()
    env.obs_buf = env.observation_manager.compute(update_history=True, env_ids=ids)
    assert len(calls) == 2  # exactly one native compute, one call per term
    if concatenate:
        torch.testing.assert_close(env.obs_buf['policy'][keep],previous['policy'][keep],rtol=0,atol=0)
        assert bool((env.obs_buf['policy'][1,2:] == 7.).all())
    else:
        for name in ('plain','history'):
            torch.testing.assert_close(env.obs_buf['policy'][name][keep],previous['policy'][name][keep],rtol=0,atol=0)
        assert bool((env.obs_buf['policy']['history'][1] == 7.).all())
    state = torch.get_rng_state()
    assert env.observation_manager.compute() is env.obs_buf
    assert env.observation_manager.compute(update_history=True,env_ids=torch.tensor([],dtype=torch.long)) is env.obs_buf
    assert torch.equal(state,torch.get_rng_state()) and len(calls) == 2
