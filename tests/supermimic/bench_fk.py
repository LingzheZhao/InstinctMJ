"""Parity + speed of the level-batched forward kinematics vs the original per-body loop (kept here as the reference).

    .venv/bin/python tests/supermimic/bench_fk.py [--batch 8192]
Uses the G1 kinematic tree from the MJCF only (no env, tiny GPU use)."""

from __future__ import annotations

import argparse
import time

import torch

from instinct_mj.rl.supermimic.kinematics import G1Kinematics
from instinct_mj.rl.supermimic.rotations import axis_angle_to_mat, quat_to_mat

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=8192)
args = parser.parse_args()
dev = "cuda:0"
JOINTS = [
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
TRACKED = ["pelvis", "torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link",
           "left_wrist_yaw_link", "right_wrist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "left_knee_link",
           "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"]  # fmt: skip
kin = G1Kinematics(dev, JOINTS, TRACKED)


def reference_forward(self, p, R, v, w, q, qd):
    """The original per-body loop (identical to the code before the level-batched version)."""
    P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
    for i in range(self.num_bodies):
        par = self.parent[i]
        if par < 0:
            P[i], Rs[i], V[i], W[i] = p, R, v, w
            continue
        j = self.joint_idx[i]
        Rp = Rs[par]
        pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
        R_off = Rp @ self.off_R[i]
        Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
        axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
        wi = W[par] + axis_w * qd[..., j : j + 1]
        vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
        P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
    idx = self.tracked_idx
    return (torch.stack([P[i] for i in idx], -2), torch.stack([Rs[i] for i in idx], -3), torch.stack([V[i] for i in idx], -2), torch.stack([W[i] for i in idx], -2))


def make_inputs(B, seed=0, grad=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
    p, v, w = r(B, 3), r(B, 3), r(B, 3)
    R = quat_to_mat(r(B, 4))
    q, qd = 1.2 * r(B, 29), 3 * r(B, 29)
    out = [p, R, v, w, q, qd]
    return [x.clone().requires_grad_(grad) for x in out]


# ---- parity: values and gradients ----
ins_a, ins_b = make_inputs(args.batch, grad=True), make_inputs(args.batch, grad=True)
out_ref = reference_forward(kin, *ins_a)
out_new = kin.forward(*ins_b)
names = ("pos", "rot", "lin vel", "ang vel")
for n, a, b in zip(names, out_ref, out_new):
    print(f"forward {n:8s} max abs diff {float((a - b).abs().max()):.3e}   (value scale {float(a.abs().max()):.2f})")
weights = [torch.randn_like(x) for x in out_ref]
sum((o * wgt).sum() for o, wgt in zip(out_ref, weights)).backward()
sum((o * wgt).sum() for o, wgt in zip(out_new, weights)).backward()
for n, a, b in zip(("p", "R", "v", "w", "q", "qd"), ins_a, ins_b):
    print(f"grad wrt {n:3s} max abs diff {float((a.grad - b.grad).abs().max()):.3e}   (grad scale {float(a.grad.abs().max()):.2f})")
assert all(float((a - b).abs().max()) < 1e-5 for a, b in zip(out_ref, out_new)), "forward parity failed"


# ---- speed (forward + backward), GPU is shared: interleaved repeats, median ----
def timed(fn, reps=7):
    ts = []
    for _ in range(reps):
        ins = make_inputs(args.batch, grad=True)
        torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*ins)
        sum(o.sum() for o in out).backward()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


timed(lambda *a: reference_forward(kin, *a), 2), timed(kin.forward, 2)
ref_ms = [timed(lambda *a: reference_forward(kin, *a), 3) for _ in range(3)]
new_ms = [timed(kin.forward, 3) for _ in range(3)]
print(f"fwd+bwd at B={args.batch}: per-body loop {sorted(ref_ms)[1]:.1f} ms, level-batched {sorted(new_ms)[1]:.1f} ms")
torch.cuda.reset_peak_memory_stats()
timed(lambda *a: reference_forward(kin, *a), 1)
m_ref = torch.cuda.max_memory_allocated() / 2**20
torch.cuda.reset_peak_memory_stats()
timed(kin.forward, 1)
print(f"peak memory fwd+bwd: per-body loop {m_ref:.0f} MiB, level-batched {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
