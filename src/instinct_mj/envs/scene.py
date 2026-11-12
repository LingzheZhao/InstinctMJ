from __future__ import annotations

from mjlab.scene import Scene
from mjlab.utils.spec import attach_unprefixed


class InstinctScene(Scene):
    """Scene variant that builds the terrain through ``cfg.class_type``.

  This mirrors IsaacLab's ``InteractiveScene``, which constructs every terrain
  as ``cfg.class_type(cfg)`` with no special-casing, so both the plain terrain
  entity and the Instinct terrain importer (e.g. the ``hacked_generator``
  variant) stay overridable from config.
  """

    def _add_terrain(self) -> None:
        if self._cfg.terrain is None:
            return
        terrain_cfg = self._cfg.terrain
        terrain_cfg.num_envs = self.num_envs
        terrain_cfg.env_spacing = self.env_spacing
        terrain = terrain_cfg.class_type(terrain_cfg, device=self.device)
        self._terrain = terrain
        self.entities["terrain"] = terrain
        frame = self.spec.worldbody.add_frame()
        attach_unprefixed(self.spec, terrain.spec, frame)
