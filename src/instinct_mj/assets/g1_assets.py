"""Named Unitree G1 assets that a task can select.

Two assets are named. ``instinct_torso_popsicle`` is InstinctMJ's torso-rooted G1 and refers to the objects of
``unitree_g1``. ``mjlab_pelvis`` is the pelvis-rooted G1 of the installed mjlab and refers to the objects of
``mjlab.asset_zoo.robots.unitree_g1.g1_constants``; nothing of it is copied here.

The two are two physical conditions, not two spellings of one robot: actuator class, passive joint damping and
frictionloss, collision shapes and contact parameters, and how the torso's mass is split into bodies differ. Selecting
an asset selects all of them together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import EntityCfg

from instinct_mj.assets import unitree_g1

FOOT_BODIES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
"""The two sole-bearing bodies of either G1."""


@dataclass
class G1Asset:
    """One selectable G1: its entity, its model file and the selectors that depend on its tree."""

    name: str
    root_body: str
    """The body that carries the free joint."""
    model_path: str
    """The MJCF file of the same tree, read by the motion reference's kinematics."""
    action_scale: Mapping[str, float]
    """Joint-position action scale, keyed by joint-name regex."""
    spec_fn: Callable[[], mujoco.MjSpec]
    entity_cfg_fn: Callable[[], EntityCfg]
    foot_collision_geom_names_fn: Callable[[], tuple[str, ...]]

    def entity_cfg(self) -> EntityCfg:
        """A fresh entity configuration; no object of the asset module is shared with the caller."""
        return deepcopy(self.entity_cfg_fn())

    def foot_collision_geom_names(self) -> tuple[str, ...]:
        """Names of the collision shapes of the two ankle-roll bodies."""
        return self.foot_collision_geom_names_fn()


def collision_geom_names_of(spec: mujoco.MjSpec, bodies: tuple[str, ...]) -> tuple[str, ...]:
    """Collision shapes of the named bodies, by the rule of ``unitree_g1.foot_collision_geom_names``."""
    names = tuple(geom.name for name in bodies for geom in spec.body(name).geoms if geom.contype or geom.conaffinity)
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError(f"the asset must declare distinct named collision shapes on {bodies}")
    return names


def _mjlab_pelvis_foot_collision_geom_names() -> tuple[str, ...]:
    return collision_geom_names_of(g1_constants.get_spec(), FOOT_BODIES)


def _instinct_torso_popsicle_entity_cfg() -> EntityCfg:
    return unitree_g1.G1_29DOF_TORSOBASE_POPSICLE_CFG


DEFAULT_G1_ASSET = "instinct_torso_popsicle"
"""The asset the registered G1 shadowing tasks build when no asset is named."""

G1_ASSETS: dict[str, G1Asset] = {
    "instinct_torso_popsicle": G1Asset(
        name="instinct_torso_popsicle",
        root_body="torso_link",
        model_path=unitree_g1.G1_MJCF_PATH,
        action_scale=unitree_g1.beyondmimic_action_scale,
        spec_fn=unitree_g1.get_g1_spec,
        entity_cfg_fn=_instinct_torso_popsicle_entity_cfg,
        foot_collision_geom_names_fn=unitree_g1.foot_collision_geom_names,
    ),
    "mjlab_pelvis": G1Asset(
        name="mjlab_pelvis",
        root_body="pelvis",
        model_path=str(g1_constants.G1_XML),
        action_scale=g1_constants.G1_ACTION_SCALE,
        spec_fn=g1_constants.get_spec,
        entity_cfg_fn=g1_constants.get_g1_robot_cfg,
        foot_collision_geom_names_fn=_mjlab_pelvis_foot_collision_geom_names,
    ),
}


def g1_asset(name: str) -> G1Asset:
    """The named G1 asset."""
    try:
        return G1_ASSETS[name]
    except KeyError:
        raise ValueError(f"Unknown G1 asset {name!r}; known: {sorted(G1_ASSETS)}") from None


__all__ = ["DEFAULT_G1_ASSET", "FOOT_BODIES", "G1Asset", "G1_ASSETS", "collision_geom_names_of", "g1_asset"]
