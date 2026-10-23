"""Correctness gate of the SuperMimic analytic heads: compares them against the live env managers and FAILS (exit 1)
when a tolerance is violated or a comparison is empty.

Runs the real perceptive shadowing env (``auto_reset=False``, as in the runner, so terminal states stay readable) with
zero / random actions and checks: motion-library frame indexing (incl. fractional phases), forward kinematics, reward
terms (incl. terminal transitions), terminations (natural and forced on both sides of every threshold by writing
perturbed states), goal / proprio observations, observation noise ranges and history ordering, applied torque, the
terrain height scan, and that pre-reset terminal states differ from the reset state.

    .venv/bin/python tests/supermimic/check_analytic_heads.py --motion-folder ~/Xyk/Datasets/20251116_50cm_kneeClimbStep1
"""

from __future__ import annotations

import argparse
import copy
import sys

import mujoco
import numpy as np
import torch

from instinct_mj.rl.supermimic.mdp import CAMERA_CTX_DIM, DEPTH_SHAPE, GOAL_POS_NOISE, GOAL_ROT_NOISE, PROPRIO_NOISE
from instinct_mj.rl.supermimic.depth import DepthPipeline
from instinct_mj.rl.supermimic.rotations import quat_to_mat, rot_angle_sq, so3_exp, yaw_mat
from instinct_mj.rl.supermimic.sim import SimReader, build_mdp, camera_context
from instinct_mj.scripts.supermimic.train import TASK_ID, build_env_cfg, make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument("--motion-folder", default="~/Xyk/Datasets/20251116_50cm_kneeClimbStep1")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

dev = "cuda:0"
cfg = build_env_cfg(TASK_ID, args.num_envs, args.motion_folder)
assert cfg.auto_reset is False
cfg.observations["policy"].enable_corruption = False  # noise-free policy group: comparable with the analytic assembly
cfg.seed = args.seed
vec = make_vec_env(TASK_ID, cfg, dev)
env = vec.unwrapped
motion, kin, scanner, mdp = build_mdp(env, 0.05, "~/.cache/instinct_mj/supermimic_heightmaps", dev)
rd = SimReader(env)
robot = env.scene["robot"]
mgr = env.scene["motion_reference"]
xi = rd.context(mdp)
mdp.set_xi_stats(xi)
depth_pipe = DepthPipeline(env)
real_depth, imag_depth, clean_depth = [], [], []
link_ids = robot.find_bodies(mgr.cfg.link_of_interests, preserve_order=True)[0]

stats: dict[str, list[torch.Tensor]] = {}
tol: dict[str, tuple[float, float | None]] = {}  # name -> (max abs err tolerance, mean abs err tolerance)
failures: list[str] = []


def log(name: str, err: torch.Tensor, max_tol: float, mean_tol: float | None = None) -> None:
    tol[name] = (max_tol, mean_tol)
    stats.setdefault(name, []).append(err.detach().abs().flatten().float().cpu())


def fail_if(cond: bool, msg: str) -> None:
    if cond:
        failures.append(msg)


# terrain-only CPU model for direct ray casting (the "env semantics" reference of the scan: rays start at the root)
cpu = copy.deepcopy(env.sim.mj_model)
terrain_id = cpu.body("terrain").id
is_terrain = cpu.geom_bodyid == terrain_id
cpu.geom_group[~is_terrain] = 5
cpu_data = mujoco.MjData(cpu)
mujoco.mj_forward(cpu, cpu_data)
geomgroup = np.zeros(6, dtype=np.uint8)
geomgroup[int(cpu.geom_group[is_terrain][0])] = 1


def direct_scan(p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    H = yaw_mat(R)[..., :2, :2]
    xy = p[:, None, :2] + scanner.offsets @ H.transpose(-1, -2)
    out = torch.full(xy.shape[:2], scanner.max_distance)
    pc, xyc = p.cpu().numpy(), xy.cpu().numpy()
    geomid = np.zeros(1, dtype=np.int32)
    for b in range(xyc.shape[0]):
        for k in range(xyc.shape[1]):
            dist = mujoco.mj_ray(cpu, cpu_data, np.array([xyc[b, k, 0], xyc[b, k, 1], pc[b, 2]]), np.array([0.0, 0.0, -1.0]), geomgroup, 1, -1, geomid)
            if 0 <= dist <= scanner.max_distance:
                out[b, k] = dist
    return out.to(p.device)


# ---- static checks: pattern, noise ranges ----
from mjlab.sensor import GridPatternCfg

ref_off, ref_dir = GridPatternCfg(resolution=0.1, size=(1.6, 1.0)).generate_rays(None, dev)
log("scan pattern offsets vs mjlab GridPatternCfg", ref_off[:, :2] - scanner.offsets, 0.0, 0.0)
fail_if(not (scanner.num_rays == ref_off.shape[0] == 187), "height scan is not 187 rays")
terms_cfg = env.cfg.observations["policy"].terms
for name, idx, mine in zip(("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action"), range(5), PROPRIO_NOISE):
    n = terms_cfg[name].noise
    env_range = 0.0 if n is None else float(n.n_max)
    fail_if(n is not None and float(n.n_min) != -float(n.n_max), f"asymmetric noise in {name}")
    log(f"noise range: {name}", torch.tensor(env_range - mine), 0.0, 0.0)
log("noise range: position_ref", torch.tensor(float(terms_cfg["position_ref"].noise.n_max) - GOAL_POS_NOISE), 0.0, 0.0)
log("noise range: rotation_ref", torch.tensor(float(terms_cfg["rotation_ref"].noise.n_max) - GOAL_ROT_NOISE), 0.0, 0.0)

# ---- camera mount context: the fitted rotation vector must reproduce the randomized ray directions ----
cam = env.scene["camera"]
ctx = camera_context(env)
d_nom = cam._compute_pinhole_rays_from_intrinsics(cam._camera_data.intrinsic_matrices)[1]
d_rec = (so3_exp(ctx[:, 3:])[:, None] @ d_nom.unsqueeze(-1)).squeeze(-1)
log("camera ctx: rotated nominal rays vs actual rays", d_rec - cam.ray_directions, 3e-3, 1e-3)
log("camera ctx: position offset vs ray starts", cam.ray_starts - ctx[:, None, :3], 1e-6, 1e-7)
fail_if(ctx.shape[-1] != CAMERA_CTX_DIM or float(ctx.abs().max()) == 0.0, "camera context is empty")
print(f"camera ctx: |pos| max {ctx[:, :3].abs().max().item():.4f} m, |rotvec| max {ctx[:, 3:].abs().max().item():.4f} rad")

# ---- stepping ----
env.reset()
rng = torch.Generator(device=dev).manual_seed(args.seed)
obs_hist: list[torch.Tensor] = []  # noise-free proprio observations, newest last
hist_len = env.cfg.observations["critic"].terms["joint_pos"].history_length
steps_since_reset = torch.zeros(args.num_envs, dtype=torch.long, device=dev)
n_done_total = 0
phases = []
for step in range(args.steps):
    phase_ = step * 3 // args.steps
    n_act = env.action_manager.total_action_dim
    if phase_ == 0:
        action = torch.zeros(args.num_envs, n_act, device=dev)
    elif phase_ == 1:
        action = 0.3 * torch.randn(args.num_envs, n_act, device=dev, generator=rng)
    else:
        action = 1.0 * torch.randn(args.num_envs, n_act, device=dev, generator=rng)
    a_prev = rd.last_action()
    obs, rew, terminated, truncated, extras = env.step(action)
    rew_total = rew["rewards"].clone()  # the manager reuses this buffer in compute()
    a = env.action_manager.action.clone()
    done = terminated | truncated
    steps_since_reset += 1

    S = rd.state()
    f, phase, origin = motion.current_frame()
    ref = mgr.reference_frame
    valid = ref.validity[:, 0].bool()  # envs whose reference clip is not exhausted
    phases.append(phase.abs().cpu())
    fail_if(not bool(valid.any()), "no valid frame in a step")
    log("frame: base_pos_w", (motion.root_p[f] + origin - ref.base_pos_w[:, 0])[valid], 1e-5, 1e-6)
    log("frame: base_quat_w (R)", (motion.root_R[f] - quat_to_mat(ref.base_quat_w[:, 0]))[valid], 1e-5, 1e-6)
    log("frame: link_pos_w", (motion.link_p[f] + origin[:, None] - ref.link_pos_w[:, 0])[valid], 1e-5, 1e-6)
    log("frame: link_lin_vel_w", (motion.link_v[f] - ref.link_lin_vel_w[:, 0])[valid], 1e-5, 1e-6)
    log("frame: link_ang_vel_w", (motion.link_w[f] - ref.link_ang_vel_w[:, 0])[valid], 1e-5, 1e-6)
    log("frame: joint_pos", (motion.q[f] - ref.joint_pos[:, 0])[valid], 1e-5, 1e-6)
    log("frame: joint_vel", (motion.qd[f] - ref.joint_vel[:, 0])[valid], 1e-5, 1e-6)
    d = robot.data
    log("state: root_link_pos_w", S[0] - d.root_link_pos_w, 1e-4, 1e-5)
    log("state: root_link_quat -> R", S[1] - quat_to_mat(d.root_link_quat_w), 1e-5, 1e-6)
    log("state: root_link_lin_vel_w", S[2] - d.root_link_lin_vel_w, 1e-4, 1e-5)
    log("state: root_link_ang_vel_w", S[3] - d.root_link_ang_vel_w, 1e-4, 1e-5)

    # ---- forward kinematics vs sim ----
    bp, bR, bv, bw = kin.forward(*S)
    log("fk: link pos", bp - d.body_link_pos_w[:, link_ids], 1e-4, 1e-5)
    log("fk: link rot (matrix)", bR - quat_to_mat(d.body_link_quat_w[:, link_ids]), 1e-4, 1e-5)
    log("fk: link lin vel", bv - d.body_link_lin_vel_w[:, link_ids], 1e-3, 1e-4)
    log("fk: link ang vel", bw - d.body_link_ang_vel_w[:, link_ids], 1e-3, 1e-4)

    # ---- rewards (ALL envs incl. terminal transitions; reference-exhausted envs excluded) ----
    stale = {k: v.clone() for k, v in env.reward_manager._termwise_reward_buf["rewards"].items()}
    env.reward_manager.compute(env.step_dt)
    fresh = env.reward_manager._termwise_reward_buf["rewards"]
    contact = rd.contact()
    r, terms = mdp.reward(S, f, origin, a, a_prev, xi, contact, terms=True)
    for k, v in terms.items():
        if k == "base_position_imitation_gauss":  # the env multiplies by the reference validity
            v = v * ref.validity[:, 0]
        # link_pos / link_rot: the env caches ``reference_relative_*`` per timestamp, i.e. they stay partly stale
        t = (5e-2, 3e-3) if k in ("link_pos_imitation_gauss", "link_rot_imitation_gauss") else (1e-4, 1e-5)
        log(f"reward[{k}] vs env (fresh state)", (v - fresh[k])[valid], *t)
        # stale: what the env actually used (one-substep-old xpos/cvel); only a loose sanity bound
        log(f"reward[{k}] vs env (stale, as used)", (v - stale[k])[valid], 3.0, 0.1)
    log("reward total vs env (stale, as used)", (r - rew_total)[valid], 5e-2, 2e-3)  # terminal transitions (falls) have the largest one-substep lag
    n_done_total += int(done.sum())

    # ---- terminations, natural stepping (the env evaluates them on the one-substep-stale state) ----
    tmargin = mdp.termination(S, f, origin)
    tm = env.termination_manager
    for k, v in tmargin.items():
        flag_env = tm.get_term(k)
        mismatch = ((v > 0) != flag_env) & valid
        log(f"termination[{k}] natural: |margin| where flags differ", torch.where(mismatch, v.abs(), torch.zeros_like(v)), 2e-2, None)
        log(f"termination[{k}] natural: flag mismatches", mismatch.float(), 1.0, 0.02)

    # ---- torque ----
    tau = mdp._torque(S[4], S[5], a, xi)
    qfrc = env.sim.data.qfrc_actuator[:, rd.joint_v]
    log("torque: analytic PD vs qfrc_actuator (all joints)", tau - qfrc, 1e-3, 1e-5)
    log("torque: analytic PD vs qfrc_actuator (ankle/wrist)", (tau - qfrc)[:, mdp.torque_joint_ids], 1e-3, 1e-5)

    # ---- observations / goal (noise-free) ----
    o = mdp.obs(S, a, xi)
    log("obs: projected_gravity", o[:, 0:3] - d.projected_gravity_b, 1e-5, 1e-6)
    log("obs: base_ang_vel (body)", o[:, 3:6] - d.root_link_ang_vel_b, 1e-4, 1e-5)
    log("obs: joint_pos_rel", o[:, 6:35] - (d.joint_pos - d.default_joint_pos), 1e-5, 1e-6)
    log("obs: joint_vel_rel", o[:, 35:64] - (d.joint_vel - d.default_joint_vel), 1e-5, 1e-6)
    log("obs: last_action", o[:, 64:93] - a, 1e-6, 1e-7)
    goal = mdp.goal(S, f, phase, noisy=False)
    cm = env.command_manager
    g = mdp.motion.goal_frames
    jp_cmd = cm.get_term("joint_pos_ref_command").command
    jv_cmd = cm.get_term("joint_vel_ref_command").command
    pos_cmd = cm.get_term("position_b_ref_command").command
    rot_cmd = cm.get_term("rotation_ref_command").command
    log("goal: joint_pos_ref vs command", (goal[:, : 29 * g] - jp_cmd.flatten(1))[valid], 1e-5, 1e-6)
    log("goal: joint_vel_ref vs command", (goal[:, 29 * g : 58 * g] - jv_cmd.flatten(1))[valid], 1e-5, 1e-6)
    log("goal: position_ref (b) vs command", (goal[:, 58 * g : 61 * g] - pos_cmd.flatten(1))[valid], 1e-4, 1e-5)
    log("goal: rotation_ref vs command", (goal[:, 61 * g :] - rot_cmd.flatten(1))[valid], 1e-4, 1e-5)

    # ---- observation history ordering (env critic group is noise-free) ----
    obs_hist.append(o.clone())
    obs_hist = obs_hist[-hist_len:]
    ok_hist = (steps_since_reset >= hist_len) & ~done
    if len(obs_hist) == hist_len and bool(ok_hist.any()):
        mine = torch.stack(obs_hist, 1)  # (N, hist, 93), newest last
        c = {k: v.reshape(args.num_envs, hist_len, -1) for k, v in obs["critic"].items() if k in ("joint_pos", "joint_vel", "base_ang_vel", "last_action")}  # flattened (time-major) history
        log("history: joint_pos (oldest first)", (mine[:, :, 6:35] - c["joint_pos"])[ok_hist], 1e-5, 1e-6)
        log("history: joint_vel (oldest first)", (mine[:, :, 35:64] - c["joint_vel"])[ok_hist], 1e-5, 1e-6)
        log("history: base_ang_vel (oldest first)", (mine[:, :, 3:6] - c["base_ang_vel"])[ok_hist], 1e-4, 1e-5)
        log("history: last_action (oldest first)", (mine[:, :, 64:93] - c["last_action"])[ok_hist], 1e-6, 1e-7)

    # ---- the analytic policy observation vs the env's policy group (all terms except depth) ----
    pol = obs["policy"]
    depth_env = pol["depth_image"]
    log("depth_image: shape", torch.tensor(float(tuple(depth_env.shape[1:]) != DEPTH_SHAPE)), 0.0, 0.0)
    log("depth_image: outside [0, 1]", torch.relu(-depth_env) + torch.relu(depth_env - 1), 0.0, 0.0)
    for name, sl in (("joint_pos_ref", slice(0, 29 * g)), ("joint_vel_ref", slice(29 * g, 58 * g)), ("position_ref", slice(58 * g, 61 * g)), ("rotation_ref", slice(61 * g, None))):
        log(f"policy obs: {name} vs env policy group", (goal[:, sl] - pol[name].flatten(1))[valid], 1e-4, 1e-5)
    if len(obs_hist) == hist_len and bool(ok_hist.any()):
        order = env.observation_manager.active_terms["policy"]
        env_flat = torch.cat([pol[n].flatten(1) for n in order], -1)
        mine_flat = mdp.policy_obs(torch.stack(obs_hist, 1), goal, depth_env)
        keep = torch.ones(env_flat.shape[-1], dtype=torch.bool, device=dev)
        d0 = sum(pol[n].flatten(1).shape[-1] for n in order[: order.index("depth_image")])
        keep[d0 : d0 + depth_env[0].numel()] = False  # depth is compared through the depth head, not here
        log("policy obs: assembled flat vector vs env (all but depth)", (mine_flat - env_flat)[ok_hist][:, keep], 1e-4, 1e-5)
        fail_if(order != ["joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref", "depth_image", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action"], f"unexpected policy term order {order}")

    # ---- depth noise: real noised images vs imagination-noised images of the clean sensor measurement ----
    clean = depth_pipe.clean_from_sensor()
    real_depth.append(depth_env.clone()), clean_depth.append(clean), imag_depth.append(depth_pipe.noise_predicted(clean))

    # ---- height scan (vs direct terrain ray casts from the root, the env sensor semantics) ----
    if step % 10 == 0:
        scan = scanner.height_scan(S[0], S[1])
        ref_scan = direct_scan(S[0], S[1])
        log("height_scan: heightmap lookup vs direct rays (mean)", scan - ref_scan, 5.0, 1e-2)
        log("height_scan: fraction of rays |err|>0.05 (overhangs/edges)", ((scan - ref_scan).abs() > 0.05).float(), 1.0, 0.05)

    # ---- explicit reset of the finished envs; pre-reset terminal state must differ from the reset state ----
    if done.any():
        ids = done.nonzero().squeeze(-1)
        env.reset(env_ids=ids)
        S_after = rd.state()
        diff = ((S_after[0][ids] - S[0][ids]).abs().amax(-1) + (S_after[4][ids] - S[4][ids]).abs().amax(-1))
        log("terminal: pre-reset s1 differs from reset state (1 - min(1, diff/0.01))", (1 - (diff / 0.01).clamp(max=1.0)), 0.0, None)
        steps_since_reset[ids] = 0
        obs_hist = []  # envs reset: restart the history comparison

# ---- forced termination checks: write perturbed states so both sides of every threshold are hit ----
env.reset()
env.step(torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=dev))
S0 = rd.state()
for case, offsets in (("height", (0.15, 0.35)), ("tilt", (0.5, 1.2))):
    for sign_i in range(2):
        value = torch.full((args.num_envs,), offsets[sign_i], device=dev)
        pose = torch.cat((S0[0], torch.nn.functional.normalize(robot.data.root_link_quat_w.clone(), dim=-1)), -1)
        if case == "height":
            pose[:, 2] += value
        else:
            half = value / 2
            q_tilt = torch.stack((half.cos(), half.sin(), torch.zeros_like(half), torch.zeros_like(half)), -1)
            w1, x1, y1, z1 = q_tilt.unbind(-1)
            w2, x2, y2, z2 = pose[:, 3:7].unbind(-1)
            pose[:, 3:7] = torch.stack((w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2), -1)
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        env.termination_manager.compute()
        S_f = rd.state()
        f_f, _, origin_f = motion.current_frame()
        margins = mdp.termination(S_f, f_f, origin_f)
        for k, v in margins.items():
            flag_env = env.termination_manager.get_term(k)
            mismatch = ((v > 0) != flag_env).float()
            log(f"termination[{k}] forced: flag mismatches", mismatch, 0.0, 0.0)
            stats.setdefault(f"__forced_true[{k}]", []).append(flag_env.float().cpu())
        env.reset(env_ids=torch.arange(args.num_envs, device=dev))
        env.step(torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=dev))
        S0 = rd.state()

# ---- report + gates ----
print(f"{'quantity':84s} {'max abs err':>12s} {'mean abs err':>13s}  n")
for name, errs in stats.items():
    e = torch.cat(errs)
    if name.startswith("__forced_true"):
        k = name[len("__forced_true["):-1]
        n_true, n_false = int(e.sum()), int((1 - e).sum())
        print(f"forced {k}: env flag true {n_true}, false {n_false}")
        fail_if(n_true == 0 or n_false == 0, f"forced termination check for {k} did not hit both sides")
        continue
    mx, mean = e.max().item(), e.mean().item()
    print(f"{name:84s} {mx:12.3e} {mean:13.3e}  {len(e)}")
    fail_if(len(e) == 0, f"empty comparison: {name}")
    max_tol, mean_tol = tol[name]
    fail_if(mx > max_tol, f"{name}: max err {mx:.3e} > {max_tol:.3e}")
    if mean_tol is not None:
        fail_if(mean > mean_tol, f"{name}: mean err {mean:.3e} > {mean_tol:.3e}")
real_d, imag_d, clean_d = (torch.cat(x) for x in (real_depth, imag_depth, clean_depth))
stat = lambda x: (x.mean().item(), x.std().item(), (x >= 1.0).float().mean().item(), (x <= 0.0).float().mean().item())  # noqa: E731
print("depth image statistics (mean, std, frac==max, frac==0):")
for name, x in (("clean", clean_d), ("real noised (env obs)", real_d), ("imagination noised (from clean)", imag_d)):
    print(f"  {name:34s} " + " ".join(f"{v:.4f}" for v in stat(x)))
for i, (name, tol_) in enumerate((("mean", 0.03), ("std", 0.03), ("frac==max", 0.05), ("frac==0", 0.02))):
    fail_if(abs(stat(real_d)[i] - stat(imag_d)[i]) > tol_, f"depth noise statistics differ ({name}): real {stat(real_d)[i]:.4f} vs imagination {stat(imag_d)[i]:.4f}")
fail_if((real_d - clean_d).abs().mean().item() < 1e-3, "the env's depth observation is not noised")
ph = torch.cat(phases)
print(f"reference phases |x - round(x)|: nonzero fraction {(ph > 1e-3).float().mean().item():.2f}, max {ph.max().item():.3f}")
fail_if(not bool((ph > 1e-3).any()), "no env had a nonzero fractional reference phase")
print(f"terminal (done) transitions seen: {n_done_total}")
fail_if(n_done_total == 0, "no terminal transition was exercised")
if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
