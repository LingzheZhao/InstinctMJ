"""Pinned SDK lifecycle tests: real observation manager, CPU-only fake physics.

The fake solver deliberately leaves derived state stale until forward. No
MuJoCo/Warp physics execution or training is performed by these tests.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
from instinct_mj.envs.manager_based_rl_env import InstinctRlEnv
from instinct_mj.envs.runtime_hooks import RuntimeCallbacks


class Manager:
    def __init__(self, env, name):
        self.env, self.name = env, name
    def reset(self, ids, **kwargs):
        self.env.events.append((self.name + '.reset', ids.tolist()))
        return {}
    def compute(self, **kwargs):
        self.env.events.append((self.name + '.compute', kwargs))
    def compute_substep(self):
        self.env.events.append('metrics.substep')
    def update(self, **kwargs):
        self.env.events.append('monitor.update')
        return {'monitor': 1.}
    def process_action(self, action):
        self.env.events.append('action.process')
    def apply_action(self):
        self.env.events.append('action.apply')


class Solver:
    device = 'cpu'
    def __init__(self, env):
        self.env = env
    def reset(self, ids):
        self.env.events.append(('sim.reset', ids.tolist()))
        self.env.integrated[ids] = 0.
    def step(self):
        self.env.events.append('sim.step')
        self.env.integrated += 1.
    def forward(self):
        self.env.events.append('forward')
        self.env.reached.copy_(self.env.integrated)
    def sense(self):
        self.env.events.append('sense')


class Scene:
    num_envs = 3
    def __init__(self, env):
        self.env = env
    def reset(self, ids):
        self.env.events.append(('scene.reset', ids.tolist()))
        self.env.reference_time[ids] = 0.
    def update(self, dt):
        self.env.events.append(('scene.update', dt))
        self.env.reference_time += dt
    def write_data_to_sim(self):
        self.env.events.append('write')


class Terminations(Manager):
    def compute(self):
        self.env.events.append('termination.compute')
        self.terminated = self.env.reached[:, 0] >= torch.tensor([4., 8., 100.])
        self.time_outs = torch.zeros(3, dtype=torch.bool)
        return self.terminated | self.time_outs


class Rewards(Manager):
    def compute(self, dt):
        self.env.events.append(('reward.compute', dt))
        return self.env.reached[:, 0].clone() * .1


class Events(Manager):
    available_modes = ('reset', 'step', 'interval')
    def apply(self, mode, **kwargs):
        self.env.events.append(('event.' + mode, kwargs))
        if mode == 'step' and self.env.event_push:
            self.env.integrated += 10.


class Recorder:
    def __init__(self, env):
        self.env = env
    def record_pre_reset(self, ids):
        self.env.events.append(('recorder.pre_reset', ids.tolist()))
    def record_post_reset(self, ids):
        self.env.events.append(('recorder.post_reset', ids.tolist()))
    def record_post_step(self):
        self.env.events.append('recorder.post_step')


class CpuLifecycleEnv(InstinctRlEnv):
    """Exercise the real class methods/managers without constructing a simulator."""
    def __init__(self, *, auto_reset=True, delayed=False):
        self.cfg = SimpleNamespace(auto_reset=auto_reset, decimation=4,
            sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=.005)))
        self.events, self.runtime_callbacks, self.obs_buf = [], RuntimeCallbacks(), {}
        self.integrated, self.reached = torch.zeros(3, 2), torch.zeros(3, 2)
        self.reference_time = torch.zeros(3)
        self.event_push = False
        self.sim, self.scene = Solver(self), Scene(self)
        self.extras, self._sim_step_counter, self.common_step_counter = {}, 0, 0
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self._manual_reset_pending = torch.zeros(3, dtype=torch.bool)
        self._command_dt = torch.zeros(3)
        for name in ('action', 'command', 'metrics', 'curriculum', 'monitor'):
            setattr(self, name + '_manager', Manager(self, name))
        self.reward_manager = Rewards(self, 'reward')
        self.termination_manager = Terminations(self, 'termination')
        self.event_manager = Events(self, 'event')
        self.recorder_manager = Recorder(self)
        def values(env):
            env.events.append('observation.term')
            return env.reached.clone()
        terms = {'plain': ObservationTermCfg(func=values, noise=UniformNoiseCfg(n_min=-1., n_max=1.)),
                 'history': ObservationTermCfg(func=values, history_length=3)}
        if delayed:
            terms['delayed'] = ObservationTermCfg(func=values, delay_min_lag=1, delay_max_lag=2)
        self.observation_manager = ObservationManager({'policy': ObservationGroupCfg(
            terms=terms, enable_corruption=True, concatenate_terms=False)}, self)
        self.events.clear()


def clone(value):
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value.clone() if isinstance(value, torch.Tensor) else value


def test_explicit_step_orders_fresh_endpoint_terminal_events_reset_and_publication():
    env = CpuLifecycleEnv()
    env.reset()
    env.events.clear()
    env.add_physics_step_callback(lambda native, dt: native.events.append(('physics', dt)))
    terminal = []
    def capture(native, terminated, truncated):
        terminal.append((native.reached.clone(), terminated.clone(), native.reference_time.clone()))
        native.events.append('terminal')
    remove = env.add_terminal_callback(capture)
    publications = []
    env.add_observation_callback(lambda native, ids: publications.append(clone(dict(native.cached_observations))))
    obs, reward, done, truncated, extras = env.step(torch.zeros(3, 2))
    expected_tick = ['action.apply', 'write', 'sim.step', ('scene.update', .005), ('physics', .005), 'metrics.substep']
    assert env.events[:1] == ['action.process']
    assert env.events[1:25] == expected_tick * 4
    assert env.events[25:32] == ['forward', ('scene.update', 0.), 'termination.compute', 'terminal',
                                ('reward.compute', .02), ('metrics.compute', {}), ('event.step', {'dt': .02})]
    assert env.events[32] == ('event.interval', {'dt': .02})
    assert env.events[33] == ('recorder.pre_reset', [0])
    assert env.events.index('sense') > env.events.index(('termination.reset', [0]))
    assert env.events[-2:] == ['recorder.post_step', 'monitor.update']
    assert env.events.count('forward') == 2 and env.events.count('sense') == 1
    assert env.events.count('observation.term') == 2
    assert done.tolist() == [True, False, False] and not bool(truncated.any())
    torch.testing.assert_close(reward, torch.full((3,), .4))
    torch.testing.assert_close(terminal[0][0], torch.full((3, 2), 4.))
    torch.testing.assert_close(terminal[0][2], torch.full((3,), .02))
    assert env.reference_time.tolist() == pytest.approx([0., .02, .02])
    command = next(value[1]['dt'] for value in env.events if isinstance(value, tuple) and value[0] == 'command.compute')
    torch.testing.assert_close(command, torch.tensor([0., .02, .02]))
    assert extras['step'] == {'monitor': 1.}
    torch.testing.assert_close(publications[0]['policy']['plain'], obs['policy']['plain'])
    assert env.episode_length_buf.tolist() == [0, 1, 1]
    remove(); remove()


def test_passive_observer_does_not_change_results_clocks_rng_or_computation_counts():
    def run(observed):
        torch.manual_seed(19)
        env = CpuLifecycleEnv(delayed=True)
        terminal, physics = [], []
        env.add_physics_step_callback(lambda native, dt: physics.append(dt))
        if observed:
            env.add_terminal_callback(lambda native, done, truncated: terminal.append(native.reached.clone()))
        env.reset()
        env.events.clear()
        results = []
        for _ in range(8):
            obs, reward, done, truncated, _ = env.step(torch.zeros(3, 2))
            results.append((clone(obs), reward.clone(), done.clone(), truncated.clone()))
        return env, results, torch.get_rng_state(), terminal, physics
    plain, seen = run(False), run(True)
    for left, right in zip(plain[1], seen[1], strict=True):
        for name in ('plain', 'history', 'delayed'):
            torch.testing.assert_close(left[0]['policy'][name], right[0]['policy'][name], rtol=0, atol=0)
        for a, b in zip(left[1:], right[1:], strict=True):
            assert torch.equal(a, b)
    assert torch.equal(plain[2], seen[2])
    assert torch.equal(plain[0].reference_time, seen[0].reference_time)
    assert len(seen[3]) == 8 and plain[4] == seen[4] == [.005] * 32
    for env in (plain[0], seen[0]):
        assert env.events.count('forward') == 16 and env.events.count('sense') == 8
        assert env.events.count('observation.term') == 24
        assert env._sim_step_counter == 32 and env.common_step_counter == 8


def test_post_event_forward_and_manual_reset_preserve_terminal_state_and_scope():
    env = CpuLifecycleEnv(auto_reset=False)
    env.reset()
    env.event_push = True
    obs, reward, done, *_ = env.step(torch.zeros(3, 2))
    torch.testing.assert_close(reward, torch.full((3,), .4))
    assert done.tolist() == [True, False, False]
    assert bool((obs['policy']['history'][:, -2:] == 14.).all())
    before = len(env.events)
    with pytest.raises(RuntimeError, match='must be reset'):
        env.step(torch.zeros(3, 2))
    assert len(env.events) == before
    previous, clock = clone(obs), env.reference_time.clone()
    env.reset(env_ids=torch.tensor([0]))
    assert not bool(env._manual_reset_pending.any())
    for name in ('plain', 'history'):
        torch.testing.assert_close(env.obs_buf['policy'][name][1:], previous['policy'][name][1:], rtol=0, atol=0)
    torch.testing.assert_close(env.reference_time[1:], clock[1:])
    env.step(torch.zeros(3, 2))


def test_reset_uses_sdk_manager_order_once_without_replacing_native_methods():
    env = CpuLifecycleEnv()
    assert not env.obs_buf and env.events == []  # no constructor reset
    env.reset(env_ids=torch.tensor([1]))
    assert env.events[:5] == [('monitor.reset', [1]), ('curriculum.compute', {'env_ids': torch.tensor([1])}),
                             ('sim.reset', [1]), ('scene.reset', [1]),
                             ('event.reset', {'env_ids': torch.tensor([1]), 'global_env_step_count': 0})]
    ordered = [x[0] for x in env.events if isinstance(x, tuple) and x[0].endswith('.reset')]
    assert ordered == ['monitor.reset', 'sim.reset', 'scene.reset', 'event.reset', 'action.reset',
                       'reward.reset', 'metrics.reset', 'curriculum.reset', 'command.reset', 'event.reset', 'termination.reset']
    assert env.events.count('forward') == env.events.count('sense') == 1
    assert env.events.count('observation.term') == 2
    assert 'compute' not in vars(env.observation_manager)
    assert 'compute' not in vars(env.termination_manager)
    assert 'update' not in vars(env.scene)


def test_lifecycle_explicitly_targets_the_installed_pinned_sdk():
    from importlib.metadata import version
    assert version('mjlab') == '1.6.0'
    assert InstinctRlEnv.reset is not ManagerBasedRlEnv.reset
    assert InstinctRlEnv.step is not ManagerBasedRlEnv.step
    module = Path(inspect.getfile(InstinctRlEnv)).read_text()
    tree = ast.parse(module)
    forbidden = {('scene', 'update'), ('termination_manager', 'compute'), ('observation_manager', 'compute')}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
                    assert (target.value.attr, target.attr) not in forbidden
    assert '_install_runtime_hooks' not in module and 'scope_reset_observation_cache' not in module


def test_partial_reset_keeps_unselected_real_delay_history_and_lag_schedule():
    torch.manual_seed(37)
    env = CpuLifecycleEnv(auto_reset=False, delayed=True)
    env.reset()
    env.step(torch.zeros(3, 2))
    delay = env.observation_manager._group_obs_term_delay_buffer['policy']['delayed']
    history = env.observation_manager._group_obs_term_history_buffer['policy']['history']
    keep = torch.tensor([1, 2])
    before = {'delay': delay._buffer.buffer[keep].clone(), 'lags': delay._current_lags[keep].clone(),
              'steps': delay._step_count[keep].clone(), 'phases': delay._phase_offsets[keep].clone(),
              'history': history.buffer[keep].clone(), 'issued': clone(env.obs_buf)}
    env.events.clear()
    env.reset(env_ids=torch.tensor([0]))
    assert env.events.count('observation.term') == 3  # one compute, one call per native term
    torch.testing.assert_close(delay._buffer.buffer[keep], before['delay'], rtol=0, atol=0)
    torch.testing.assert_close(delay._current_lags[keep], before['lags'])
    torch.testing.assert_close(delay._step_count[keep], before['steps'])
    torch.testing.assert_close(delay._phase_offsets[keep], before['phases'])
    torch.testing.assert_close(history.buffer[keep], before['history'], rtol=0, atol=0)
    for name in ('plain', 'history', 'delayed'):
        torch.testing.assert_close(env.obs_buf['policy'][name][keep], before['issued']['policy'][name][keep], rtol=0, atol=0)
    state, events = torch.get_rng_state(), len(env.events)
    assert env._compute_observations(update_history=True, env_ids=torch.tensor([], dtype=torch.long)) is env.obs_buf
    assert torch.equal(state, torch.get_rng_state()) and len(env.events) == events
