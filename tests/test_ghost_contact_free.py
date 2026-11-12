"""A supplied robot's play ghost takes no part in contact, in the compiled scene (CPU, no simulation step)."""
import pytest


def compiled_play_scene(task, **robot):
    import instinct_mj.tasks
    from instinct_mj.envs.scene import InstinctScene
    from instinct_mj.motion_reference.motion_reference_cfg import MotionBufferCfg
    from instinct_mj.tasks.shadowing.factory import make_shadowing_env_cfg

    class StandInBuffer:
        pass
    cfg = make_shadowing_env_cfg(task, play=True, motion_buffers={'current': MotionBufferCfg(class_type=StandInBuffer)},
                                 num_envs=1, **robot)
    return InstinctScene(cfg.scene, device='cpu').compile()


def colliding_geoms(model, prefix):
    bodies = [model.body(int(model.geom_bodyid[g])).name for g in range(model.ngeom)]
    ours = [g for g in range(model.ngeom) if bodies[g].startswith(prefix)]
    return len(ours), sum(1 for g in ours if model.geom_contype[g] or model.geom_conaffinity[g])


def pelvis_robot():
    from instinct_mj.assets.g1_assets import g1_asset
    asset = g1_asset('mjlab_pelvis')
    return {'robot': asset.entity_cfg(), 'robot_model_path': asset.model_path}


@pytest.mark.parametrize('task', ['Instinct-BeyondMimic-Plane-G1-v0', 'Instinct-Shadowing-WholeBody-Plane-G1-v0'])
@pytest.mark.parametrize('robot', ['instinct_torso_popsicle', 'mjlab_pelvis', 'robot='])
def test_supplied_robot_ghost_has_no_colliding_geom_and_the_robot_keeps_its_own(task, robot):
    model = compiled_play_scene(task, **(pelvis_robot() if robot == 'robot=' else {'asset': robot}))
    ghost_geoms, ghost_colliding = colliding_geoms(model, 'robot_reference/')
    robot_geoms, robot_colliding = colliding_geoms(model, 'robot/')
    assert ghost_geoms == robot_geoms > 0
    assert ghost_colliding == 0
    assert robot_colliding > 0
