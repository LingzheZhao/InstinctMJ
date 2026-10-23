"""Reference motion library for imagination, built from the already-loaded motion buffer of the task.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.

All motions of the ``TerrainMotion`` buffer are concatenated along the frame axis (``clip_start`` / ``clip_end``). Data
are in the motion's local frame; the world frame of an env is ``local + origin`` (``origin`` = the terrain origin the
motion was assigned to, ``TerrainMotion._assigned_env_origins``).
"""

from __future__ import annotations

import torch

from instinct_mj.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager

from .rotations import quat_to_mat, tan_norm_from_mat


class MotionLib:
    def __init__(self, manager: MotionReferenceManager, device: str):
        assert len(manager.motion_buffers) == 1, "SuperMimic supports a single TerrainMotion buffer."
        name, buf = next(iter(manager.motion_buffers.items()))
        assert isinstance(buf, TerrainMotion)
        self.manager, self.buffer, self.device = manager, buf, device
        seq = buf._all_motion_sequences
        cat = lambda a: getattr(seq, a).concatenated_tensor.to(device)  # noqa: E731
        self.root_p, self.root_v, self.root_w = cat("base_pos_w"), cat("base_lin_vel_w"), cat("base_ang_vel_w")
        self.root_R = quat_to_mat(cat("base_quat_w"))
        self.q, self.qd = cat("joint_pos"), cat("joint_vel")
        self.link_p, self.link_v, self.link_w = cat("link_pos_w"), cat("link_lin_vel_w"), cat("link_ang_vel_w")
        self.link_R = quat_to_mat(cat("link_quat_w"))
        self.clip_len = seq.base_pos_w.batch_sizes.to(device)
        self.clip_start = seq.base_pos_w.batch_starts.to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(self.clip_len.sum())
        self.clip_id = torch.repeat_interleave(torch.arange(len(self.clip_len), device=device), self.clip_len)
        framerate = seq.framerate.to(device)
        assert (framerate == framerate[0]).all()
        self.framerate = float(framerate[0])
        self.goal_stride = int(round(float(manager.cfg.frame_interval_s) * self.framerate))
        self.goal_frames = manager.cfg.num_frames
        # static goal parts: reference root frame inverse is applied to the future base positions
        self.root_Rt = self.root_R.transpose(-1, -2)

    # ---------------- frame bookkeeping (identical to ``MotionReferenceManager.reference_frame``) ----------------
    def current_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Global frame index (clamped to the clip), fractional phase and origin (N,3) of the env's reference frame.

        ``phase = (start + t) * fps - round(...)`` is the fractional reference time the manager rounds from; every
        requested future timestamp is rounded independently from it (see :meth:`future_frames`). All outputs are
        clones: auto-reset / resampling overwrites ``_assigned_env_origins`` and the clocks in place.
        """
        buf, mgr = self.buffer, self.manager
        motion_id = buf._assigned_env_motion_selection
        x = (buf._motion_buffer_start_time_s + mgr._timestamp) * buf._all_motion_sequences.framerate[motion_id]
        local = torch.round(x)
        f = self.clip_start[motion_id] + torch.minimum(local.long(), self.clip_len[motion_id] - 1)
        return f, (x - local).clone(), buf._assigned_env_origins.clone()

    def future_frames(self, f: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Frames at the times of the goal (``i * frame_interval`` ahead), each rounded independently from the
        fractional reference time ``f + phase``; clamped to the clip, plus their validity (not past the clip)."""
        fut = f.unsqueeze(-1) + torch.round(phase.unsqueeze(-1) + self.goal_stride * torch.arange(self.goal_frames, device=f.device)).long()
        end = self.clip_end[self.clip_id[f]].unsqueeze(-1)
        return torch.minimum(fut, end), fut <= end

    def goal_static(self, f: torch.Tensor, phase: torch.Tensor, default_q: torch.Tensor) -> torch.Tensor:
        """State independent goal parts: joint_pos_ref, joint_vel_ref, position_ref (``position_b_ref_command``)."""
        fut, valid = self.future_frames(f, phase)
        jp = (self.q[fut] - default_q) * valid.unsqueeze(-1)
        jv = self.qd[fut] * valid.unsqueeze(-1)
        Rt0 = self.root_Rt[f].unsqueeze(1)
        pos_b = (Rt0 @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        return torch.cat((jp.flatten(1), jv.flatten(1), pos_b.flatten(1)), -1)

    def goal_rotation(self, f: torch.Tensor, phase: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """``rotation_ref_command``: tangent-normal of ``R^T R_ref`` for the future frames (zero where invalid)."""
        fut, valid = self.future_frames(f, phase)
        rel = R.transpose(-1, -2).unsqueeze(1) @ self.root_R[fut]
        return (tan_norm_from_mat(rel) * valid.unsqueeze(-1)).flatten(1)
