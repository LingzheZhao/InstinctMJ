"""Component timings / peak memory of the SuperMimic runner on a real (small) env: sim step (collect), wm_update, policy_update.

    .venv/bin/python tests/supermimic/bench_runner.py --num-envs 64 --reps 5 [--profile] [--set key=value ...]
The GPU may be shared: numbers are medians of interleaved repeats and only meaningful relative to each other.
Config overrides (``--set pi_batch=256`` ...) toggle runner options (e.g. ``--set fk_compile=false``)."""

from __future__ import annotations

import argparse
import os
import time

import torch

from instinct_mj.rl.supermimic.runner import SuperMimicRunner
from instinct_mj.scripts.supermimic.common import TASK_ID, build_env_cfg, make_vec_env
from instinct_mj.tasks.registry import load_instinct_rl_cfg

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--reps", type=int, default=5)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--only", default="")
parser.add_argument("--metrics", action="store_true")
parser.add_argument("--cudnn-benchmark", action="store_true")
parser.add_argument("--set", nargs="*", default=[])
args = parser.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")
torch.backends.cudnn.benchmark = args.cudnn_benchmark
cfg = load_instinct_rl_cfg(TASK_ID)
cfg.device = "cuda:0"
defaults = dict(buffer_steps=150, warmup_steps=64, wm_batch=512, depth_windows=128, depth_pairs=128, fail_windows=128, pi_batch=256,
                wm_only_iters=0, compile=True)  # fmt: skip
for k, v in defaults.items():
    setattr(cfg, k, v)
for kv in args.set:
    k, v = kv.split("=")
    old = getattr(cfg, k)
    setattr(cfg, k, (v.lower() == "true") if isinstance(old, bool) else type(old)(v))
env_cfg = build_env_cfg(TASK_ID, args.num_envs, "~/Xyk/Datasets/20251116_50cm_kneeClimbStep1")
env_cfg.seed = 42  # as the training script (a random terrain grid can lack a terrain, see the note in the report)
vec = make_vec_env(TASK_ID, env_cfg, "cuda:0")
runner = SuperMimicRunner(vec, cfg.to_dict(), log_dir=None, device="cuda:0")
runner.setup_training()
stats = {"r": [], "r_gap": [], "ep_len": [], "term": [], "done": []}
runner.collect(cfg.warmup_steps, stats)
runner.it = 10**6  # the failure head is "warmed up": exercise its imagination path


def timed(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        out.append((time.time() - t0) * 1e3)
    return min(out), sorted(out)[len(out) // 2]


for name, fn in (("collect (1 sim step)", lambda: runner.collect(1, stats)), ("wm_update", runner.wm_update), ("policy_update", runner.policy_update)):
    if args.only and args.only not in name:
        continue
    fn(), fn()  # warm-up / compile
    torch.cuda.reset_peak_memory_stats()
    lo, med = timed(fn, args.reps)
    print(f"{name:22s} min {lo:8.1f} ms  median {med:8.1f} ms   peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
def proxies(fn):
    """Contention-robust proxies from the profiler: summed GPU kernel time, CPU time without the host waits (launch /
    Python overhead), number of kernel launches and number of host syncs."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    cuda_ms = sum(e.self_device_time_total for e in ka) / 1e3
    sync = [e for e in ka if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize")]
    cpu_ms = (sum(e.self_cpu_time_total for e in ka) - sum(e.self_cpu_time_total for e in sync)) / 1e3
    launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    return cuda_ms, cpu_ms, launches, sum(e.count for e in sync)


if args.metrics:
    for name, fn in (("collect", lambda: runner.collect(1, stats)), ("wm_update", runner.wm_update), ("policy_update", runner.policy_update)):
        if args.only and args.only not in name:
            continue
        c_ms, p_ms, n_launch, n_sync = proxies(fn)
        print(f"METRICS {name:14s} GPU kernel time {c_ms:8.1f} ms | CPU time w/o host waits {p_ms:8.1f} ms | kernel launches {n_launch:6d} | host syncs {n_sync:4d}")
if args.profile:
    from torch.profiler import ProfilerActivity, profile

    for name, fn in (("policy_update", runner.policy_update), ("wm_update", runner.wm_update), ("collect", lambda: runner.collect(1, stats))):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        print(f"\n=== {name}: top CUDA kernels ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=60))
        print(f"--- {name}: top CPU (self) ---")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14, max_name_column_width=60))
vec.close()
