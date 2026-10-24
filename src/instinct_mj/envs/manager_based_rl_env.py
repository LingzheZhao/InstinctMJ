from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from instinct_mj.managers import MultiRewardCfg


class InstinctRlEnv(ManagerBasedRlEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def __init__(
        self,
        cfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        del kwargs  # Unused.
        from instinct_mj.envs.runtime_hooks import RuntimeCallbacks
        self.runtime_callbacks = RuntimeCallbacks()
        self._inside_control_step = False
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self._instinct_body_lin_acc_cache: dict[str, dict[str, object]] = {}
        self.extras = {}
        self.obs_buf = {}
        # Initialize the manual-reset state here because InstinctRlEnv
        # customizes scene construction instead of calling ManagerBasedRlEnv.__init__.
        self._manual_reset_pending = torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=device)
        # Scratch buffer for per-env command dt; see ManagerBasedRlEnv.step().
        self._command_dt = torch.zeros(self.cfg.scene.num_envs, device=device)

        # Use InstinctScene so terrain cfg.class_type is honored (e.g. hacked_generator importer).
        self.scene = self.cfg.scene_class_type(self.cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            device=device,
            spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info(),
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=device, dtype=torch.long)
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(
                model=self.sim.mj_model,
                cfg=self.cfg.viewer,
                scene=self.scene,
                sim_model=self.sim.model,
                expanded_fields=self.sim.expanded_fields,
            )
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt

        self.load_managers()
        self.setup_manager_visualizers()
        self._install_runtime_hooks()

    def load_managers(self) -> None:
        """Extend mjlab manager loading with InstinctLab multi-reward routing."""
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = {}

        super().load_managers()

        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = self.cfg.multi_reward_manager_class_type(
                self.cfg.rewards,
                self,
                scale_by_dt=self.cfg.scale_rewards_by_dt,
            )
            print_info(f"[INFO] {self.reward_manager}")

        self.monitor_manager = self.cfg.monitor_manager_class_type(self.cfg.monitors, self)
        print_info(f"[INFO] Monitor Manager: {self.monitor_manager}")

    def setup_manager_visualizers(self) -> None:
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = self.monitor_manager

    def step(self, action: torch.Tensor):
        self._inside_control_step = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._inside_control_step = False
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        extras.setdefault("step", {})
        extras["step"].update(monitor_infos)
        return obs, reward, terminated, truncated, extras

    @property
    def cached_observations(self):
        """Observations last produced by reset/step; callers must not mutate tensors."""
        if not self.obs_buf:
            raise RuntimeError("Reset the environment before reading cached observations")
        return MappingProxyType(self.obs_buf)

    def get_observations(self):
        """Read cached observations without another noise sample/history advance."""
        return self.cached_observations

    def refresh_observations_after_state_write(self, *, env_ids=None):
        """Refresh an explicitly written physical state without reference reset/advance.

        Physical sensors and observation histories restart only for selected
        worlds. MotionReferenceManager's assignments, timestamps and sampler are
        deliberately untouched. This owns the forward/sense and observation
        reset/compute; callers must not repeat those operations around it.
        """
        from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager
        ids = (torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None
               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1))
        for sensor in self.scene.sensors.values():
            if not isinstance(sensor, MotionReferenceManager):
                sensor.reset(ids)
        self.sim.forward()
        self.sim.sense()
        self.observation_manager.reset(ids)
        self.obs_buf = self.observation_manager.compute(update_history=True, env_ids=ids)
        return self.cached_observations

    @property
    def terrain_placement(self):
        """Actual scene origins in world meters, copied from the owned native terrain."""
        terrain = self.scene.terrain
        if terrain is None:
            raise ValueError("This environment has no terrain")
        return {name:None if getattr(terrain,name) is None else getattr(terrain,name).clone()
                for name in ("env_origins","terrain_origins")}

    def terrain_heightmap(self, *, resolution, bounds=None):
        """Raycast the compiled physical terrain into an owned world-meter height grid.

        Output ``heights[y,x]`` and ``origin=[world_x0,world_y0]`` describe cells
        spaced by ``resolution`` meters; ``valid`` marks rays that found terrain.
        Infinite planes require explicit finite world-XY bounds. This top-down
        representation cannot describe overhangs. No content hash/disk cache,
        second physical terrain, reference advance or simulation step is used.
        """
        from instinct_mj.rl.supermimic.terrain import build_heightmap, MISS_HEIGHT
        heights,x0,y0 = build_heightmap(self.sim.mj_model,resolution,bounds=bounds)
        value = torch.as_tensor(heights,device=self.device,dtype=torch.float32)
        return {"heights":value,"origin":value.new_tensor([x0,y0]),"resolution":float(resolution),
                "valid":value != MISS_HEIGHT,"axes":"y,x","units":"m"}

    def add_physics_step_callback(self, callback):
        """Register ``callback(env, physics_dt)`` after each actual scene update.

        Native solver contacts/capacity counters and lazy sensor measurements can
        be read here. No control/reference step or noise computation is added.
        Returns an idempotent unregister callable.
        """
        return self.runtime_callbacks.register(self.runtime_callbacks.physics, callback)

    def add_terminal_callback(self, callback):
        """Register ``callback(env, terminated, truncated)`` before reward/reset.

        Runs every control after refreshed endpoint kinematics and termination
        evaluation, before rewards, interval events and automatic resets. Camera
        sensing remains at its native post-step phase: for terminal depth use
        ``auto_reset=False``, read step's cache, then explicitly reset done worlds.
        Callback consumers must clone any state they retain across a reset.
        """
        return self.runtime_callbacks.register(self.runtime_callbacks.terminal, callback)

    @property
    def control_endpoint(self) -> str:
        """Physical kinematics seen by every termination/reward, with or without observers."""
        return "fresh_forward_before_termination_and_reward"

    def _install_runtime_hooks(self):
        from instinct_mj.envs.runtime_hooks import scope_reset_observation_cache
        scope_reset_observation_cache(self)
        update_scene = self.scene.update
        def scene_update(dt):
            result = update_scene(dt)
            if self._inside_control_step and dt > 0:
                self.runtime_callbacks.physics_step(self, dt)
            return result
        self.scene.update = scene_update
        compute_terminations = self.termination_manager.compute
        def terminal_compute():
            # mj_step integrates after its forward pass. The environment owns
            # this endpoint, even when no passive measurement is registered:
            # termination and reward must see the same reached physical state.
            # dt=0 does not advance reference/camera noise or policy history.
            self.sim.forward()
            self.scene.update(dt=0.0)
            result = compute_terminations()
            self.runtime_callbacks.terminal_step(self, self.termination_manager.terminated,
                                                  self.termination_manager.time_outs)
            return result
        self.termination_manager.compute = terminal_compute

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        terrain = self.scene.terrain
        if terrain is not None:
            terrain.debug_vis(visualizer)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if isinstance(env_ids, Sequence):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)

        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)

        super()._reset_idx(env_ids)

        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)

    """
  Properties.
  """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)

    @property
    def body_lin_acc_cache(self) -> dict[str, dict[str, object]]:
        """Get the per-entity body linear acceleration cache."""
        return self._instinct_body_lin_acc_cache
