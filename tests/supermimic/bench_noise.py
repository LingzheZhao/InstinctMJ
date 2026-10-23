"""Parity + speed of the camera noise-model changes (sync-free / early-out versions of ``noise_model``).

  * contour noise: ``data[mask] = 0`` -> ``torch.where`` (bit-identical)
  * artifact masks: ``env_onehot`` via ``F.one_hot`` instead of an index_put (bit-identical for the same RNG state)
  * stereo-too-close: per-image early-out, a single draw of the (only effective) block value / artifact buffer and ``torch.where``
    instead of boolean-index assignments (same per-image distribution, different RNG consumption -> statistical test)

    .venv/bin/python tests/supermimic/bench_noise.py"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from instinct_mj.utils.noise import DepthContourNoiseCfg, StereoTooCloseNoiseCfg, noise_model

dev = "cuda:0"
cfg = StereoTooCloseNoiseCfg()
cfg.device = dev
ccfg = DepthContourNoiseCfg(contour_threshold=1.8, maxpool_kernel_size=1)
ccfg.device = dev
ids = None


# ---------------- reference implementations: the original code ----------------
def ref_contour(data, cfg):
    kernel = torch.zeros((8, 1, 3, 3), dtype=torch.float32, device=cfg.device)
    for k, ((cy, cx), (ny, nx), val) in enumerate([((1, 1), (0, 0), 0.5), ((1, 1), (0, 1), 0.1), ((1, 1), (0, 2), 0.5), ((1, 1), (1, 0), 1.2),
                                                   ((1, 1), (1, 2), 1.2), ((1, 1), (2, 0), 0.5), ((1, 1), (2, 1), 0.1), ((1, 1), (2, 2), 0.5)]):  # fmt: skip
        kernel[k, :, cy, cx], kernel[k, :, ny, nx] = val, -val
    x = data.permute(0, 3, 1, 2)
    mask = F.max_pool2d(torch.abs(F.conv2d(x, kernel, padding=1)).max(dim=1, keepdim=True)[0], kernel_size=cfg.maxpool_kernel_size, stride=1, padding=int(cfg.maxpool_kernel_size / 2)) > cfg.contour_threshold  # fmt: skip
    x = x.clone()
    x[mask] = 0.0
    return x.permute(0, 2, 3, 1)


def ref_stereo_all(data, cfg):
    n, h, w, c = data.shape
    too_close_mask = data < cfg.close_threshold
    vertical = noise_model._recognize_top_down_too_close(too_close_mask)
    full_block_mask, half_block_mask = vertical & too_close_mask, (~vertical) & too_close_mask
    for pixel_value in random.sample(cfg.full_block_values, len(cfg.full_block_values)):
        buf = noise_model._add_depth_artifacts(torch.ones_like(data), cfg.full_block_artifacts_prob, cfg.full_block_height_mean_std, cfg.full_block_width_mean_std, device=data.device)  # fmt: skip
        data[full_block_mask] = ((1 - buf) * pixel_value)[full_block_mask]
    spark = noise_model._torch_rand_float(0.0, 1.0, (n, h * w), device=data.device).view(n, h, w, 1) < cfg.half_block_spark_prob
    data[half_block_mask] = (spark.to(torch.float32) * cfg.half_block_value)[half_block_mask]
    return data


def make(n, frac_close, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = 0.3 + 2.0 * torch.rand(n, 27, 48, 1, device=dev, generator=g)
    close_img = torch.rand(n, device=dev, generator=g) < frac_close
    block = torch.rand(n, device=dev, generator=g) < 0.4  # vertical blocks (>60% rows) vs partial patches
    for i in close_img.nonzero().flatten().tolist():
        if bool(block[i]):
            x[i, :, 10:14] = 0.05
        else:
            x[i, 5:9, 20:30] = 0.05
    return x, close_img


# ---------------- parity ----------------
x, _ = make(512, 0.1, 0)
x[:, :, 20:24] += 3.0  # sharp distance jumps so the contour mask fires
assert torch.equal(noise_model.depth_contour_noise(x.clone(), ccfg, ids), ref_contour(x, ccfg)), "contour noise changed"
print("contour noise: bit-identical to the original (zeros set:", int((ref_contour(x, ccfg) == 0).sum()), "pixels)")

n, a_cnt = 256, 100
torch.manual_seed(1)
ref_mask_idx = torch.randint(0, n, (a_cnt,), device=dev)
old_oh = torch.zeros((a_cnt, n), device=dev)
old_oh[torch.arange(a_cnt), ref_mask_idx] = 1.0
assert torch.equal(old_oh, F.one_hot(ref_mask_idx, n).to(torch.float32))
print("env_onehot: F.one_hot is bit-identical to the index_put construction")

vals = [0.0, 0.25, 0.5, 1.0, 3.0, cfg.half_block_value]
hist_old, hist_new = torch.zeros(len(vals) + 1), torch.zeros(len(vals) + 1)
# NOTE: every call draws ONE order / value of the block values (the last one wins for all affected pixels of the call), so the
# value histogram has to be averaged over many calls.
for seed in range(600):
    x, close_img = make(256, 0.1, seed)
    old = ref_stereo_all(x.clone(), cfg)
    new = noise_model.stereo_too_close_noise(x.clone(), cfg, ids)
    assert torch.equal(new[~close_img], x[~close_img]) and torch.equal(old[~close_img], x[~close_img]), "images without close pixels must be untouched"
    mask = x < cfg.close_threshold
    assert torch.equal(new[~mask], x[~mask]) and torch.equal(old[~mask], x[~mask]), "pixels outside the too-close mask must be unchanged"
    for out, hist in ((old, hist_old), (new, hist_new)):
        v = out[mask]
        for k, val in enumerate(vals):
            hist[k] += float((v == val).sum())
        hist[-1] += float(((v != torch.tensor(vals, device=dev)[:, None]).all(0)).sum())
hist_old, hist_new = hist_old / hist_old.sum(), hist_new / hist_new.sum()
print("stereo-too-close: fraction of too-close pixels set to", vals + ["other(unchanged)"])
print("  original (loop, boolean-index):", [f"{float(h):.4f}" for h in hist_old])
print("  new (single draw, torch.where):", [f"{float(h):.4f}" for h in hist_new])
print(f"  max abs difference of the fractions: {float((hist_old - hist_new).abs().max()):.4f}")
assert float((hist_old - hist_new).abs().max()) < 0.03, "value distribution differs"


# ---------------- speed ----------------
def timed(fn, frac, reps=7):
    ts = []
    for _ in range(reps):
        x, _ = make(1024, frac, 0)
        torch.cuda.synchronize()
        t0 = time.time()
        fn(x)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


for frac in (0.0, 0.03, 0.2):
    timed(lambda x: noise_model.stereo_too_close_noise(x, cfg, ids), frac, 2)
    print(f"stereo-too-close, 1024 images, {frac * 100:.0f}% with close pixels: original {timed(lambda x: ref_stereo_all(x, cfg), frac):.1f} ms, new {timed(lambda x: noise_model.stereo_too_close_noise(x, cfg, ids), frac):.1f} ms")
