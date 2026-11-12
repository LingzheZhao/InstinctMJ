"""The two named G1 assets: their objects, roots, joints, gains and the physical conditions that set them apart."""
import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import Entity

from instinct_mj.assets import unitree_g1
from instinct_mj.assets.g1_assets import FOOT_BODIES, G1_ASSETS, g1_asset

TORSO, PELVIS = 'instinct_torso_popsicle', 'mjlab_pelvis'


def compiled(name):
    return Entity(g1_asset(name).entity_cfg()).compile()


def joint_terms(model):
    """Per hinge name: stiffness, damping gain, effort limit, armature, range, passive damping, frictionloss."""
    terms = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        elements = [a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == j]
        dof = int(model.jnt_dofadr[j])
        if model.jnt_actfrclimited[j]:
            effort = float(model.jnt_actfrcrange[j, 1])
        else:
            (element,) = elements
            effort = float(model.actuator_forcerange[element, 1])
        terms[model.joint(j).name] = dict(
            stiffness=-sum(float(model.actuator_biasprm[a, 1]) for a in elements),
            damping_gain=-sum(float(model.actuator_biasprm[a, 2]) for a in elements),
            effort=effort, armature=float(model.dof_armature[dof]), range=tuple(model.jnt_range[j]),
            passive_damping=float(model.dof_damping[dof]), frictionloss=float(model.dof_frictionloss[dof]),
            elements=len(elements))
    return terms


def test_torso_asset_refers_to_the_unitree_g1_objects():
    asset = g1_asset(TORSO)
    assert asset.entity_cfg_fn() is unitree_g1.G1_29DOF_TORSOBASE_POPSICLE_CFG
    assert asset.entity_cfg() is not unitree_g1.G1_29DOF_TORSOBASE_POPSICLE_CFG
    assert asset.model_path == unitree_g1.G1_MJCF_PATH
    assert asset.action_scale is unitree_g1.beyondmimic_action_scale
    assert asset.spec_fn is unitree_g1.get_g1_spec
    assert asset.foot_collision_geom_names_fn is unitree_g1.foot_collision_geom_names


def test_pelvis_asset_refers_to_the_installed_mjlab_objects():
    asset = g1_asset(PELVIS)
    assert asset.entity_cfg_fn is g1_constants.get_g1_robot_cfg
    assert asset.model_path == str(g1_constants.G1_XML)
    assert asset.action_scale is g1_constants.G1_ACTION_SCALE
    assert asset.spec_fn is g1_constants.get_spec
    first, second = asset.entity_cfg(), asset.entity_cfg()
    assert first.articulation is not second.articulation and first.init_state is not second.init_state


def test_an_unknown_asset_name_is_an_argument_error():
    with pytest.raises(ValueError, match='Unknown G1 asset'):
        g1_asset('g1')


@pytest.mark.parametrize('name, root', [(TORSO, 'torso_link'), (PELVIS, 'pelvis')])
def test_root_body_carries_the_free_joint_in_the_entity_and_in_the_model_file(name, root):
    model = compiled(name)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1 and model.body(int(model.jnt_bodyid[free[0]])).name == root
    assert g1_asset(name).root_body == root
    assert list(mujoco.MjSpec.from_file(g1_asset(name).model_path).worldbody.bodies)[0].name == root


def test_joint_names_gains_effort_armature_and_ranges_agree_across_the_two_assets():
    torso, pelvis = joint_terms(compiled(TORSO)), joint_terms(compiled(PELVIS))
    assert len(torso) == 29 and set(torso) == set(pelvis)
    for joint in torso:
        a, b = torso[joint], pelvis[joint]
        for key in ('stiffness', 'damping_gain', 'armature'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-8, err_msg=f'{joint} {key}')
        assert a['effort'] == b['effort'] and a['range'] == b['range'], joint
        assert a['stiffness'] > 0 and a['damping_gain'] > 0
        assert (a['elements'], b['elements']) == (2, 1)


def test_passive_joint_terms_are_two_physical_conditions():
    torso, pelvis = joint_terms(compiled(TORSO)), joint_terms(compiled(PELVIS))
    assert {(t['passive_damping'], t['frictionloss']) for t in torso.values()} == {(0.5, 0.1)}
    assert {(t['passive_damping'], t['frictionloss']) for t in pelvis.values()} == {(0.0, 0.0)}


@pytest.mark.parametrize('name', [TORSO, PELVIS])
def test_action_scale_is_a_quarter_effort_over_stiffness_for_every_joint(name):
    import re
    terms, scale = joint_terms(compiled(name)), g1_asset(name).action_scale
    for joint, term in terms.items():
        (value,) = [v for pattern, v in scale.items() if re.fullmatch(pattern, joint)]
        np.testing.assert_allclose(value, 0.25 * term['effort'] / term['stiffness'], rtol=1e-8, err_msg=joint)


@pytest.mark.parametrize('name', [TORSO, PELVIS])
def test_foot_selector_names_seven_colliding_shapes_on_each_ankle_roll_body(name):
    model = compiled(name)
    selected = g1_asset(name).foot_collision_geom_names()
    assert len(selected) == 14 and len(set(selected)) == 14
    for body in FOOT_BODIES:
        assert sum(model.body(int(model.geom(n).bodyid.item())).name == body for n in selected) == 7
    for geom_name in selected:
        geom = model.geom(geom_name)
        assert int(geom.contype.item()) or int(geom.conaffinity.item())


def test_the_pelvis_foot_selector_names_the_mjlab_foot_capsules():
    expected = tuple(f'{side}_foot{i}_collision' for side in ('left', 'right') for i in range(1, 8))
    assert g1_asset(PELVIS).foot_collision_geom_names() == expected


def test_every_named_asset_is_listed_under_its_own_name():
    assert all(asset.name == name for name, asset in G1_ASSETS.items())
