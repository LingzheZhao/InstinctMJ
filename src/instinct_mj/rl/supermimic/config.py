"""Runner config of the SuperMimic-style trainer (hyperparameters follow Table A.5 of the paper / reference code)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(kw_only=True)
class SuperMimicRunnerCfg:
    """Hyperparameters of :class:`SuperMimicRunner` (the runner receives ``to_dict()``)."""

    seed: int = 42
    max_iterations: int = 320_000
    save_interval: int = 2000
    log_interval: int = 50

    # Algorithm A.1 (collect -> k-step world-model fit -> H-step BPTT policy + GAE critic).
    buffer_steps: int = 1000
    """Replay buffer length in vectorized env steps ((T, N) layout)."""
    utd: float = 0.4
    """Updates per vectorized simulator step."""
    horizon: int = 32
    """Imagination horizon H of the BPTT policy update."""
    goal_frames: int = 10
    """Number of future reference frames in the goal (task: ``MotionReferenceManagerCfg.num_frames``)."""
    hist: int = 8
    """Proprio observation history length (task: ``PROPRIO_HISTORY_LENGTH``)."""
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 1024
    """Imagination starts per policy update (each step renders depth and runs the conv encoders: memory heavy)."""
    depth_windows: int = 512
    """Number of the world-model windows (of ``wm_batch``) that also train the depth heads (memory)."""
    depth_pairs: int = 512
    """(anchor, target) pairs per world-model batch for the teacher-forced depth-head term."""
    depth_loss_weight: float = 5.0
    fail_head: bool = True
    """Use the learned failure ("continue") head in imagination: ``d = 1 - (1 - d_analytic) (1 - p_fail)``. The head is trained
    either way. The gradient through ``p_fail`` is kept (it pushes the policy away from failure states)."""
    fail_head_warmup_iters: int = 300
    """The head only affects the policy this many iterations after it was created (it is trained from the start)."""
    fail_windows: int = 512
    """Windows (of ``wm_batch``) that train the head (the feature computation runs the FK)."""
    fail_clip: float = 0.95
    """Upper clip of ``p_fail`` in imagination."""
    fail_pos_weight: float = 1.0
    """BCE positive weight (default 1: the failure oversampling is importance-weighted to keep the head calibrated)."""
    wm_failure_frac: float = 0.25
    """Fraction of each world-model batch drawn from windows that start within ``wm_failure_horizon`` steps before a real
    termination (dynamics, depth head and failure head see them; the failure head importance-weights them)."""
    wm_failure_horizon: int = 16
    wm_holdout_envs: float = 0.1
    """Fraction of envs never used for world-model / head training, only for held-out calibration diagnostics."""
    depth_ensemble: int = 3
    """Number of depth heads (bootstrap-resampled); each imagined rollout renders with one random member.
    The predicted clean depth is passed through the camera's noise models before the policy sees it."""
    depth_grad: bool = False
    """Back-propagate the policy loss through the rendered depth image into the imagined states (default: stop-gradient
    on the image fed to the policy, as WMP does for world-model features, so the actor cannot exploit depth-head errors)."""
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    lr_final_ratio: float = 0.03
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    """Simulator steps collected before any update."""
    wm_only_iters: int = 500
    """World-model-only iterations at the start (no policy update)."""
    amp: str = "bf16"
    """Autocast dtype of the MLP matmuls ("bf16" or "none")."""
    compile: bool = True
    """Use ``torch.compile`` on the imagination / world-model functions."""
    compile_policy: bool = True
    """``torch.compile`` the imagination step, the reward / termination and the critic-feature functions (level-batched FK)."""
    policy_autocast: bool = False
    """bf16 autocast of the policy network (conv encoder + MLP) in the imagination only (collection / play stay fp32)."""
    checkpoint_imagination: bool = False
    """Gradient checkpointing of every imagination step (recompute in backward): less memory, more compute."""
    heightmap_resolution: float = 0.05
    """Resolution [m] of the global terrain heightmap used by the analytic height scan."""
    heightmap_cache_dir: str = "~/.cache/instinct_mj/supermimic_heightmaps"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
