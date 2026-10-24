"""Small callback collection for native physical-step and terminal observations.

Callbacks read the owned native environment; no simulation, reference clocks,
reset decisions or learning algorithms are implemented here.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch


def scope_reset_observation_cache(env):
    """Publish new observations only for explicitly reset worlds.

    mjlab scopes history/delay backfills to ``env_ids``, but computes stateless
    noisy terms for the full batch. Keep those unreset worlds' already-issued
    observations too. The native manager still owns the single computation and
    its noise draws; a cache read never calls it again. No claim is made that
    mjlab's batched noise generator draws only the selected rows.
    """
    compute = env.observation_manager.compute

    def snapshot(value, keep):
        if isinstance(value, Mapping):
            return {name: snapshot(item, keep) for name, item in value.items()}
        return value[keep].clone()

    def retain(current, previous, keep):
        if isinstance(previous, Mapping):
            if current.keys() != previous.keys():
                raise RuntimeError("Observation schema changed during partial reset")
            for name, value in previous.items():
                retain(current[name], value, keep)
        else:
            current[keep] = previous

    def scoped_compute(update_history=False, env_ids=None):
        previous, keep = None, None
        if env_ids is not None and env.obs_buf:
            keep = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            keep[env_ids] = False
            if bool(keep.all()):
                return env.obs_buf
            if bool(keep.any()):
                previous = snapshot(env.obs_buf, keep)
        current = compute(update_history=update_history, env_ids=env_ids)
        if previous is not None:
            # In-place publication also updates the native manager's cache.
            retain(current, previous, keep)
        return current

    env.observation_manager.compute = scoped_compute


class RuntimeCallbacks:
    def __init__(self):
        self.physics = []
        self.terminal = []

    @staticmethod
    def register(collection, callback):
        if not callable(callback):
            raise TypeError("A runtime callback must be callable")
        collection.append(callback)
        def remove():
            if callback in collection:
                collection.remove(callback)
        return remove

    def physics_step(self, env, dt):
        for callback in tuple(self.physics):
            callback(env, dt)

    def terminal_step(self, env, terminated, truncated):
        for callback in tuple(self.terminal):
            callback(env, terminated, truncated)
