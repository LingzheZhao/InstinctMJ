"""Small callback collection for native physical-step and terminal observations.

Callbacks read the owned native environment; no simulation, reference clocks,
reset decisions or learning algorithms are implemented here.
"""
from __future__ import annotations

from collections.abc import Mapping



def snapshot_observation_rows(value, keep):
    """Copy selected rows of the nested native observation schema."""
    if isinstance(value, Mapping):
        return {name: snapshot_observation_rows(item, keep) for name, item in value.items()}
    return value[keep].clone()


def retain_observation_rows(current, previous, keep):
    """Retain issued rows in the native cache after a scoped computation."""
    if isinstance(previous, Mapping):
        if current.keys() != previous.keys():
            raise RuntimeError("Observation schema changed during partial reset")
        for name, value in previous.items():
            retain_observation_rows(current[name], value, keep)
    else:
        current[keep] = previous


class RuntimeCallbacks:
    def __init__(self):
        self.physics = []
        self.terminal = []
        self.observations = []

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

    def observation_step(self, env, env_ids):
        for callback in tuple(self.observations):
            callback(env, env_ids)
