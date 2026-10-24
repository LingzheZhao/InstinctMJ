"""Clean processed depth images of the task's camera (target of the latent reconstruction loss).

The camera pipeline is ``noise stages (meters, raw resolution) -> normalize -> crop_and_resize``. The policy's real
observation is the noised processed image. The decoder of the visual latent reconstructs the CLEAN processed image: the
same deterministic ``normalize`` + ``crop_and_resize`` applied to the sensor's raw ``distance_to_image_plane``.
"""

from __future__ import annotations

import torch

from instinct_mj.utils.noise import CropAndResizeCfg, DepthNormalizationCfg


class DepthPipeline:
    def __init__(self, env):
        self.cam = env.scene["camera"]
        pipeline = list(self.cam.noise_pipeline)
        self.normalize_cfg, self.crop_cfg = pipeline[-2], pipeline[-1]
        assert isinstance(self.normalize_cfg, DepthNormalizationCfg) and isinstance(self.crop_cfg, CropAndResizeCfg)

    def _process(self, meters: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) meters -> normalized + cropped / resized (B, 1, 18, 32)."""
        ids = torch.arange(meters.shape[0], device=meters.device)
        x = self.normalize_cfg.func(meters, self.normalize_cfg, ids)
        return self.crop_cfg.func(x, self.crop_cfg, ids).permute(0, 3, 1, 2)

    def clean_from_sensor(self) -> torch.Tensor:
        """Clean processed depth image of every env (B, 1, 18, 32) from the sensor's latest raw measurement."""
        return self._process(self.cam.data.output["distance_to_image_plane"].clone())
