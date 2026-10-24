"""Compiled shared-asset identities select physical feet, never unnamed/visual shapes."""
import numpy as np


def test_actual_g1_geom_names_and_foot_selector_preserve_collision_physics():
    import mujoco
    from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH,get_g1_spec,foot_collision_geom_names
    original=mujoco.MjSpec.from_file(G1_MJCF_PATH).compile()
    current=get_g1_spec().compile()
    names=[current.geom(i).name for i in range(current.ngeom)]
    assert all(names) and len(set(names))==current.ngeom
    selected=foot_collision_geom_names()
    assert len(selected)==14
    assert len(foot_collision_geom_names())==len(selected)
    for name in selected:
        geom=current.geom(name);body=current.body(int(geom.bodyid.item()))
        assert body.name in ('left_ankle_roll_link','right_ankle_roll_link')
        assert int(geom.contype.item()) or int(geom.conaffinity.item())
    for field in ('geom_bodyid','geom_type','geom_size','geom_pos','geom_quat','geom_friction',
                  'geom_priority','geom_solref','geom_solimp','geom_contype','geom_conaffinity'):
        np.testing.assert_array_equal(getattr(current,field),getattr(original,field))
