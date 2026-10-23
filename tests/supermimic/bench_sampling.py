"""Parity (statistical) + speed of the sync-free failure-weighted window sampling vs the original nonzero-based version.

    .venv/bin/python tests/supermimic/bench_sampling.py"""

from __future__ import annotations

import time

import torch

from instinct_mj.rl.supermimic.buffer import ReplayBuffer

dev = "cuda:0"
torch.manual_seed(0)
T, N, K, k, B, H, q = 300, 128, 36, 8, 2048, 16, 0.25
buf = ReplayBuffer(T, N, K, dev, holdout_envs=0.1)
buf.size, buf.ptr = T, 0
buf.data["terminated"] = torch.rand(T, N, device=dev) < 0.004  # sparse real terminations


def old_select(B, k, lo, hi, failure_frac, horizon):
    """The original selection (host syncs: int(sum), nonzero)."""
    size = buf.size
    n_win = size - k + 1
    n_fail = int(B * failure_frac)
    rows_all = buf._row(torch.arange(size, device=dev))
    term = buf.data["terminated"][rows_all][:, lo:hi].int()
    cum = torch.nn.functional.pad(term.cumsum(0), (0, 0, 1, 0))
    t0_all = torch.arange(n_win, device=dev)
    t1_all = (t0_all + horizon).clamp(max=size)
    fail_set = (cum[t1_all] - cum[t0_all]) > 0
    n_failure_windows = int(fail_set.sum())
    t0 = torch.randint(0, n_win, (B,), device=dev)
    env = torch.randint(lo, hi, (B,), device=dev)
    idx = fail_set.nonzero(as_tuple=False)
    pick = idx[torch.randint(0, len(idx), (n_fail,), device=dev)]
    t0[:n_fail], env[:n_fail] = pick[:, 0], pick[:, 1] + lo
    perm = torch.randperm(B, device=dev)
    t0, env = t0[perm], env[perm]
    qq = n_fail / B
    in_fail = fail_set[t0, env - lo]
    w = torch.where(in_fail, 1.0 / ((1 - qq) + qq * fail_set.numel() / n_failure_windows), torch.full((B,), 1.0 / (1 - qq), device=dev))
    return t0, env, w, fail_set


lo, hi = buf.train_envs
stats = {"old": [], "new": []}
cells_old = torch.zeros(T - k + 1, hi - lo, device=dev)
cells_new = torch.zeros_like(cells_old)
for _ in range(60):
    t0, env, w, fail_set = old_select(B, k, lo, hi, q, H)
    out = buf.sample_windows(B, k, buf.train_envs, q, H)
    in_old, in_new = fail_set[t0, env - lo], fail_set[out["age0"], out["env"] - lo]
    stats["old"].append((float(in_old.float().mean()), float(w.mean())))
    stats["new"].append((float(in_new.float().mean()), float(out["weight"].mean())))
    cells_old[t0[in_old], env[in_old] - lo] += 1
    cells_new[out["age0"][in_new], out["env"][in_new] - lo] += 1
for name in ("old", "new"):
    a = torch.tensor(stats[name])
    print(f"{name}: share of windows inside the failure set {a[:, 0].mean():.4f}, mean importance weight {a[:, 1].mean():.4f} (must be ~1)")
fs = fail_set
F = int(fs.sum())
exp = 1.0 / F
for name, cells in (("old", cells_old), ("new", cells_new)):
    sel = cells[fs]
    print(f"{name}: failure windows hit {int((sel > 0).sum())}/{F} cells; mean count {float(sel.mean()):.2f}, std/mean {float(sel.std() / sel.mean()):.3f} (Poisson ~ {float(1 / sel.mean().sqrt()):.3f})")
a_old, a_new = torch.tensor(stats["old"]), torch.tensor(stats["new"])
assert abs(float(a_old[:, 0].mean() - a_new[:, 0].mean())) < 0.01 and abs(float(a_new[:, 1].mean()) - 1.0) < 0.02


def timed(fn, reps=15):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


print(f"selection only: original {timed(lambda: old_select(B, k, lo, hi, q, H)):.2f} ms; new (full sample_windows incl. gathers) {timed(lambda: buf.sample_windows(B, k, buf.train_envs, q, H)):.2f} ms")
