"""Parity + speed of ``DepthPipeline.noise_predicted`` (segmented / compiled) vs the plain stage-by-stage loop.

    .venv/bin/python tests/supermimic/bench_depth_noise.py [--batch 1024]
Uses a stub camera (no env): the noise pipeline of the training cfg on synthetic clean depth images."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from instinct_mj.rl.supermimic.depth import FAR_METERS, DepthPipeline
from instinct_mj.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1024)
args = parser.parse_args()
dev = "cuda:0"


def make_pipeline(stages, compile_stages):
    for st in stages:
        st.device = dev
    cam = SimpleNamespace(noise_pipeline=stages, data=SimpleNamespace(output={"distance_to_image_plane": torch.zeros(4, 27, 48, 1, device=dev)}))
    return DepthPipeline(SimpleNamespace(scene={"camera": cam}), compile_stages=compile_stages)


def tail():
    return [DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True), CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32))]


def reference_noise_predicted(self, clean):
    """The plain loop (what ``noise_predicted`` did before the segmentation)."""
    x = self._meters_from_clean(clean)
    ids = torch.arange(x.shape[0], device=x.device)
    for stage in self.noise_stages:
        x = stage.func(x, stage, ids)
    return self._process(x)


def clean_images(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    base = torch.rand(n, 1, 6, 8, device=dev, generator=g)
    img = F.interpolate(base, size=(18, 32), mode="bilinear", align_corners=False)
    img[:, :, :, 24:] = 1.0  # saturated (far) region
    img[: n // 20, :, 8:12, 4:9] = 0.02  # too-close region in 5% of the images
    return img.clamp(0, 1)


# ---- deterministic parity (random stages made deterministic: noise_std = 0) ----
det = lambda: [DepthContourNoiseCfg(contour_threshold=1.8, maxpool_kernel_size=1), RangeBasedGaussianNoiseCfg(max_value=1.2, min_value=0.12, noise_std=0.0),
               GaussianBlurNoiseCfg(kernel_size=3, sigma=0.5), *tail()]  # fmt: skip
x = clean_images(256, 0)
ref = reference_noise_predicted(make_pipeline(det(), False), x)
for compiled in (False, True):
    out = make_pipeline(det(), compiled).noise_predicted(x)
    print(f"deterministic stages (contour, blur, normalize, crop), compile={compiled}: max abs diff {float((out - ref).abs().max()):.2e}")
    assert float((out - ref).abs().max()) < 1e-5

# ---- statistical parity of the full noise pipeline ----
full = lambda: [DepthContourNoiseCfg(contour_threshold=1.8, maxpool_kernel_size=1), DepthArtifactNoiseCfg(),
                RangeBasedGaussianNoiseCfg(max_value=1.2, min_value=0.12, noise_std=0.02), GaussianBlurNoiseCfg(kernel_size=3, sigma=0.5),
                StereoTooCloseNoiseCfg(), *tail()]  # fmt: skip
p_ref, p_new = make_pipeline(full(), False), make_pipeline(full(), True)
stat = lambda t: torch.stack([t.mean(), t.std(), (t >= 1.0).float().mean(), (t <= 0.0).float().mean()])  # noqa: E731
s_ref, s_new = [], []
for seed in range(40):
    x = clean_images(1024, seed)
    s_ref.append(stat(reference_noise_predicted(p_ref, x))), s_new.append(stat(p_new.noise_predicted(x)))
s_ref, s_new = torch.stack(s_ref), torch.stack(s_new)
print("full pipeline over 40 batches of 1024 images (mean, std, frac==1, frac==0):")
print("   plain loop :", [f"{float(v):.4f}" for v in s_ref.mean(0)], " batch-to-batch std", [f"{float(v):.4f}" for v in s_ref.std(0)])
print("   segmented+compiled:", [f"{float(v):.4f}" for v in s_new.mean(0)])
assert float((s_ref.mean(0) - s_new.mean(0)).abs().max()) < 0.01


# ---- speed ----
def timed(fn, reps=9):
    x = clean_images(args.batch, 0)
    for _ in range(3):
        fn(x)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn(x)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1e3)
    return min(ts), sorted(ts)[len(ts) // 2]


p_eager = make_pipeline(full(), False)
for name, fn in (("plain loop", lambda x: reference_noise_predicted(p_eager, x)), ("segmented (eager)", p_eager.noise_predicted), ("segmented + compiled", p_new.noise_predicted)):
    lo, med = timed(fn)
    print(f"noise_predicted, {args.batch} images, {name:22s}: min {lo:6.2f} ms, median {med:6.2f} ms")
