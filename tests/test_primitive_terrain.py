"""Actual CPU-compiled native geometry; no dynamics or GPU environment."""
import numpy as np
import pytest
import mujoco
import yaml

from instinct_mj.terrains.trimesh.mesh_terrains import motion_matched_terrain
from instinct_mj.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg
from instinct_mj.rl.supermimic.terrain import build_heightmap


def compile_scene(tmp_path,mutate=None):
    primitives=[dict(name='floor',type='plane',size=[15.,8.,.1],pos=[0.,0.,0.]),
                dict(name='box',type='box',size=[.2,.2,.075],pos=[2.,1.,.075])]
    if mutate:mutate(primitives)
    path=tmp_path/'terrain.yaml';path.write_text(yaml.safe_dump(dict(terrains=[dict(terrain_id=10,collision_primitives=primitives)])))
    cfg=MotionMatchedTerrainCfg(metadata_yaml=str(path),path=str(tmp_path),size=(30.,16.),use_input_origin_frame=True)
    spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
    output=motion_matched_terrain(cfg,.5,spec,np.random.default_rng(7))
    # The existing generator applies this exact tile-center translation.
    for item in output.geometries:item.geom.pos=np.asarray(item.geom.pos)-np.array([15.,8.,0.])
    robot=spec.worldbody.add_body(name='robot')
    robot.add_geom(name='nonterrain_occluder',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.4,.4,.4],pos=[0.,0.,1.])
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data,output


def test_actual_plane_box_rays_and_physical_heightmap_exclude_robot(tmp_path):
    model,data,output=compile_scene(tmp_path)
    np.testing.assert_array_equal(output.origin,[15.,8.,0.])
    assert [int(model.geom_type[i]) for i in range(2)]==[mujoco.mjtGeom.mjGEOM_PLANE,mujoco.mjtGeom.mjGEOM_BOX]
    assert (model.geom_group[:2]==2).all() and (model.geom_contype[:2]==1).all()
    group=np.zeros(6,np.uint8);group[2]=1
    for xy,height in [((0.,0.),0.),((2.,1.),.15),((2.4,1.),0.)]:
        distance=mujoco.mj_ray(model,data,np.array([*xy,3.]),np.array([0.,0.,-1.]),group,1,-1,None)
        assert 3.-distance==pytest.approx(height,abs=1e-12)
    with pytest.raises(ValueError,match='explicit finite heightmap bounds'):build_heightmap(model,.1)
    heights,x0,y0=build_heightmap(model,.1,bounds=(-.5,.5,-.5,.5))
    np.testing.assert_array_equal(heights,0.)
    assert (x0,y0)==(-.5,-.5)
    assert not list(tmp_path.glob('.coacd_cache/*'))


def test_public_native_generator_preserves_the_input_origin_and_exact_surfaces(tmp_path):
    from instinct_mj.terrains.terrain_generator import FiledTerrainGenerator
    from instinct_mj.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
    path=tmp_path/'terrain.yaml'
    path.write_text(yaml.safe_dump(dict(terrains=[dict(terrain_id=10,collision_primitives=[
        dict(name='floor',type='plane',size=[15.,8.,.1],pos=[0.,0.,0.]),
        dict(name='box',type='box',size=[.2,.2,.075],pos=[2.,1.,.075])])])) )
    sub=MotionMatchedTerrainCfg(metadata_yaml=str(path),path=str(tmp_path),use_input_origin_frame=True)
    cfg=FiledTerrainGeneratorCfg(size=(30.,16.),num_rows=1,num_cols=1,border_width=0.,seed=7,sub_terrains={'motion_matched':sub})
    generator=FiledTerrainGenerator(cfg,device='cpu');spec=mujoco.MjSpec();generator.compile(spec)
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(generator.terrain_origins,[[[0.,0.,0.]]])
    assert model.nmesh==0 and model.ngeom==2
    np.testing.assert_array_equal(data.geom_xpos,[[0.,0.,0.],[2.,1.,.075]])
    assert not list(tmp_path.glob('.coacd_cache/*'))


def test_repeated_box_only_tiles_keep_unique_native_geometry_names(tmp_path):
    from instinct_mj.terrains.terrain_generator import FiledTerrainGenerator
    from instinct_mj.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
    path=tmp_path/'terrain.yaml'
    path.write_text(yaml.safe_dump(dict(terrains=[dict(terrain_id=10,collision_primitives=[
        dict(name='box',type='box',size=[.2,.2,.075],pos=[0.,0.,.075])])])) )
    sub=MotionMatchedTerrainCfg(metadata_yaml=str(path),path=str(tmp_path),use_input_origin_frame=True)
    cfg=FiledTerrainGeneratorCfg(size=(4.,4.),num_rows=1,num_cols=2,border_width=0.,seed=7,sub_terrains={'motion_matched':sub})
    generator=FiledTerrainGenerator(cfg,device='cpu');spec=mujoco.MjSpec();generator.compile(spec)
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    names=[model.geom(i).name for i in range(model.ngeom)]
    assert len(names)==len(set(names))==2
    np.testing.assert_allclose(data.geom_xpos,[[0.,-2.,.075],[0.,2.,.075]],rtol=0,atol=1e-12)


@pytest.mark.parametrize('mutate',[
    lambda rows:rows[0].update(size=[0.,8.,.1]),
    lambda rows:rows[0].update(pos=[0.,float('nan'),0.]),
    lambda rows:rows[1].update(name='floor'),
    lambda rows:rows[0].update(type='mesh'),
    lambda rows:rows.append(dict(name='second-plane',type='plane',size=[15.,8.,.1],pos=[0.,0.,0.])),
])
def test_invalid_primitive_geometry_is_refused_before_compilation(tmp_path,mutate):
    with pytest.raises(ValueError,match='primitive geometry'):compile_scene(tmp_path,mutate)
