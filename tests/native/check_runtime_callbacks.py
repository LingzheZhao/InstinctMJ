"""Provisioned-runtime diagnostic; not collected by CPU pytest and never an auto-launch.

Use a committed/pinned environment and explicitly supplied native motion inputs.
This checks physical callback timing, cached sensor reads and reset clocks; it
makes no claim about tracking fidelity or learned-policy retention.
"""
import argparse
import json
from pathlib import Path
import torch


def write_default_pose_motion(path):
    """Fresh native-schema input from the selected asset, only for clock diagnostics."""
    import numpy as np
    import mujoco
    from mjlab.utils.string import resolve_expr
    from instinct_mj.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG
    asset = G1_29DOF_TORSOBASE_POPSICLE_CFG
    model = asset.spec_fn().compile()
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1 and model.body(model.jnt_bodyid[free[0]]).name == 'torso_link'
    names = [model.joint(i).name for i in range(model.njnt) if i not in free]
    pose = resolve_expr(asset.init_state.joint_pos, names, 0.)
    path.mkdir(parents=True, exist_ok=False)
    np.savez(path/'default_pose_retargetted.npz', framerate=50., joint_names=np.array(names),
        joint_pos=np.tile(pose, (101, 1)), base_pos_w=np.tile(asset.init_state.pos, (101, 1)),
        base_quat_w=np.tile(asset.init_state.rot, (101, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--task', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--motion-folder')
    source.add_argument('--synthetic-default-pose', action='store_true',
                        help='fresh asset-default input; lifecycle diagnostic, never skill evidence')
    parser.add_argument('--terrain-metadata')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True, help='fresh output root')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('the output root must not exist')
    args.out.mkdir(parents=True)
    if args.synthetic_default_pose:
        args.motion_folder = str(args.out/'inputs')
        write_default_pose_motion(Path(args.motion_folder))
    from instinct_mj.tasks.shadowing.factory import make_shadowing_env_cfg
    from instinct_mj.tasks.registry import load_env_cls
    from instinct_mj.rl.vecenv_wrapper import InstinctRlVecEnvWrapper
    cfg = make_shadowing_env_cfg(args.task, motion_folder=args.motion_folder,
        terrain_metadata=args.terrain_metadata, num_envs=4, physics_dt=.005, decimation=4)
    cfg.seed = 42
    cfg.episode_length_s = .06  # ensure real automatic resets inside the eight controls
    result = {'task':args.task, 'motion_folder':args.motion_folder, 'terrain_metadata':args.terrain_metadata,
        'synthetic_default_pose':args.synthetic_default_pose,
        'num_envs':4, 'physics_dt':.005, 'decimation':4, 'controls':8,
        'episode_length_s':.06, 'seed':42, 'device':args.device,
        'physics_callbacks':0, 'terminal_callbacks':0, 'callback_failures':[], 'skipped':[]}
    env, wrapper = None, None
    try:
        env = load_env_cls(args.task)(cfg=cfg,device=args.device,render_mode=None)
        def physical(native, dt):
            result['physics_callbacks'] += 1
            if dt != .005:
                result['callback_failures'].append('wrong physical callback period')
        def terminal(native, terminated, truncated):
            result['terminal_callbacks'] += 1
            if not bool(torch.isfinite(native.scene['robot'].data.root_link_pos_w).all()):
                result['callback_failures'].append('nonfinite endpoint position')
        env.add_physics_step_callback(physical)
        env.add_terminal_callback(terminal)
        wrapper = InstinctRlVecEnvWrapper(env)
        reference = env.scene['motion_reference']
        rng_state = (lambda:torch.cuda.get_rng_state(args.device)) if args.device.startswith('cuda') else torch.get_rng_state
        for _ in range(result['controls']):
            observation, _ = wrapper.get_observations()
            before, rng = reference.elapsed_time, rng_state()
            repeated, _ = wrapper.get_observations()
            assert torch.equal(observation,repeated) and torch.equal(before,reference.elapsed_time)
            assert torch.equal(rng,rng_state())
            wrapper.step(torch.zeros(env.num_envs,wrapper.num_actions,device=args.device))
        before = reference.elapsed_time
        cached, _ = wrapper.get_observations()
        cached = cached.clone()
        wrapper.reset(env_ids=torch.tensor([1],device=args.device))
        following, _ = wrapper.get_observations()
        keep = torch.tensor([0,2,3],device=args.device)
        assert torch.equal(before[keep],reference.elapsed_time[keep])
        assert torch.equal(cached[keep],following[keep])
        assert result['physics_callbacks'] == result['controls']*cfg.decimation
        assert result['terminal_callbacks'] == result['controls']
        assert not result['callback_failures']
        result['pass'] = True
    except BaseException as error:
        result.update({'pass':False,'error':repr(error)})
        raise
    finally:
        if wrapper is not None:
            wrapper.close()
        elif env is not None:
            env.close()
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
