"""Clean / noised processed depth images of the task's camera (DAWN-style training signal for the depth head).

The camera pipeline is ``noise stages (meters, raw resolution) -> normalize -> crop_and_resize``. The policy's real
observation is the noised processed image. The depth head is trained on the CLEAN processed image (the same
deterministic ``normalize`` + ``crop_and_resize`` applied to the sensor's raw ``distance_to_image_plane``), and in
imagination the predicted clean image is pushed through the same noise models before the policy sees it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from instinct_mj.utils.noise import (
    CropAndResizeCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
)

FAR_METERS = 1.0e6  # the camera's ``max_distance`` (misses / far pixels) in the raw image


# noise stages without data-dependent shapes / host syncs: consecutive ones are fused (and optionally ``torch.compile``d)
_PURE_STAGES = (DepthContourNoiseCfg, RangeBasedGaussianNoiseCfg, GaussianBlurNoiseCfg, DepthNormalizationCfg, CropAndResizeCfg)


class DepthPipeline:
    def __init__(self, env, compile_stages: bool = False):
        self.cam = env.scene["camera"]
        pipeline = list(self.cam.noise_pipeline)
        self.normalize_cfg, self.crop_cfg = pipeline[-2], pipeline[-1]
        assert isinstance(self.normalize_cfg, DepthNormalizationCfg) and isinstance(self.crop_cfg, CropAndResizeCfg)
        self.noise_stages = pipeline[:-2]
        self.raw_hw = tuple(self.cam.data.output["distance_to_image_plane"].shape[1:3])
        self.device = self.cam.data.output["distance_to_image_plane"].device
        # ``noise_predicted`` runs [stages..., normalize, crop] as segments: runs of pure stages are one (compiled) function each,
        # the data-dependent stages (artifacts, stereo: ``nonzero``) stay eager
        self._segments, run = [], []
        for stage in [*self.noise_stages, self.normalize_cfg, self.crop_cfg]:
            if isinstance(stage, _PURE_STAGES):
                run.append(stage)
                continue
            if run:
                self._segments.append(self._chain(run, compile_stages))
                run = []
            self._segments.append(self._chain([stage], False))
        self._segments.append(self._chain(run, compile_stages))
        self._to_meters = torch.compile(self._meters_from_clean) if compile_stages else self._meters_from_clean

    @staticmethod
    def _chain(stages, compile_stage: bool):
        def run(x, ids):
            for stage in stages:
                x = stage.func(x, stage, ids)
            return x

        return torch.compile(run) if compile_stage and len(stages) > 0 else run

    def _meters_from_clean(self, clean: torch.Tensor) -> torch.Tensor:
        """Inverse of crop / resize / normalize: (B, 1, 18, 32) processed image -> (B, H, W, 1) raw-resolution meters."""
        up, down, left, right = self.crop_cfg.crop_region
        H, W = self.raw_hw
        d_min, d_max = self.normalize_cfg.depth_range
        o_min, o_max = self.normalize_cfg.output_range
        meters = (clean - o_min) / (o_max - o_min) * (d_max - d_min) + d_min
        meters = F.interpolate(meters, size=(H - up - down, W - left - right), mode="bilinear", align_corners=False)
        # saturated pixels were beyond the range (or ray misses, which the sensor reports as ``max_distance``): restore
        # a far value so the distance-jump based noise models (contour, ...) see the same silhouettes as on raw data
        meters = torch.where(meters >= d_max * (1 - 1e-3), torch.full_like(meters, FAR_METERS), meters)
        return F.pad(meters, (left, right, up, down), mode="replicate").permute(0, 2, 3, 1).contiguous()

    def _process(self, meters: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) meters -> normalized + cropped / resized (B, 1, 18, 32)."""
        ids = torch.arange(meters.shape[0], device=meters.device)
        x = self.normalize_cfg.func(meters, self.normalize_cfg, ids)
        return self.crop_cfg.func(x, self.crop_cfg, ids).permute(0, 3, 1, 2)

    def clean_from_sensor(self) -> torch.Tensor:
        """Clean processed depth image of every env (B, 1, 18, 32) from the sensor's latest raw measurement."""
        return self._process(self.cam.data.output["distance_to_image_plane"].clone())

    def noise_predicted(self, clean: torch.Tensor) -> torch.Tensor:
        """Apply the camera's noise models to a (predicted) clean processed image (B, 1, 18, 32): map it back to a raw
        resolution meters image (inverse of crop / resize / normalize, border replicated), run the noise stages, then
        the same normalize + crop / resize."""
        if len(self.noise_stages) == 0:
            return clean
        x = self._to_meters(clean)
        ids = torch.arange(x.shape[0], device=x.device)
        for segment in self._segments:
            x = segment(x, ids)
        return x.permute(0, 3, 1, 2)
