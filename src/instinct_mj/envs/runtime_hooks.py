"""Small callback collection for native physical-step and terminal observations.

Callbacks read the owned native environment; no simulation, reference clocks,
reset decisions or learning algorithms are implemented here.
"""
from __future__ import annotations


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
