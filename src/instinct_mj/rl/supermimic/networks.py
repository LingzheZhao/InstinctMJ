"""World model (+ depth head), and critic (paper Table A.4). The policy is the task's own EncoderActorCritic actor.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.
Task-specific changes: the world model also takes the local terrain height scan and predicts per-body contact logits
(used for the ``undesired_contacts`` reward); ``DepthHead`` renders the depth image the task's depth policy needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdp import INV_DIM, NUM_J, invariant_features
from .rotations import so3_exp, so3_log, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps
        self._host_count = 0.0  # host-side mirror of ``count`` (avoids a device sync per update)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._host_count = float(self.count)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self._host_count >= self.max_count:
            return
        n = x.shape[0]
        self._host_count += n
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi, terrain scan) predicting frame-local state increments, plus
    logits of the per-body contact indicators at s_{t+1}."""

    def __init__(self, scan_dim: int, num_contact: int, xi_dim: int, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.in_dim = INV_DIM + NUM_J + xi_dim + scan_dim
        self.net = mlp(self.in_dim, list(hidden), WM_OUT + num_contact, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[WM_OUT:] = -3.0  # contacts are rare
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n, scan):
        return torch.cat((invariant_features(S), a, xi_n, scan), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, scan):
        """Returns the next state tuple and the contact logits at the next state."""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, scan)))
        d = y[..., :WM_OUT] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., WM_OUT:]

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class DepthHead(nn.Module):
    """Learned depth "renderer": the camera's processed depth image (1 x 18 x 32, in [0, 1]) at an imagined state.

    Inputs: the REAL depth image at the anchor state, the root pose at the state relative to the anchor (camera mount
    = root), the joint positions (self-occlusion by arms / knees), the randomized camera mount offset and the local
    terrain height scan. Regions not visible in the anchor image are hallucinated by the decoder.
    """

    def __init__(self, scan_dim: int, joint_dim: int = NUM_J, pose_dim: int = 9, ctx_dim: int = 6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ELU(), nn.Flatten()
        )  # (1, 18, 32) -> (64, 5, 8)
        self.enc_fc = nn.Linear(64 * 5 * 8, 256)
        self.feat = mlp(pose_dim + joint_dim + ctx_dim + scan_dim, [256, 256], 256)
        self.fc = nn.Sequential(nn.Linear(512, 64 * 5 * 8), nn.ELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=(0, 1)), nn.ELU(),  # -> (9, 16)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=(1, 1)), nn.ELU(),  # -> (18, 32)
        )  # fmt: skip
        self.out = nn.Conv2d(32 + 1, 1, 3, 1, 1)  # the anchor image is also fed to the last layer (skip connection)
        self.register_buffer("ctx_scale", torch.tensor([50.0] * 3 + [5.0] * 3), persistent=False)

    def forward(self, anchor, pose, q, ctx, scan):
        """anchor: (B, 1, 18, 32); pose (B, 9); q (B, 29); ctx (B, 6); scan (B, 187) -> (B, 1, 18, 32)."""
        feat_in = torch.cat((pose, q, ctx * self.ctx_scale, scan / 5.0), -1)
        with torch.autocast("cuda", dtype=AMP_DTYPE or torch.float32, enabled=AMP_DTYPE is not None):
            z = torch.cat((F.elu(self.enc_fc(self.enc(anchor))), self.feat(feat_in)), -1)
            h = self.dec(self.fc(z).view(-1, 64, 5, 8))
            return torch.sigmoid(self.out(torch.cat((h, anchor.to(h.dtype)), 1))).float()


class DepthEnsemble(nn.Module):
    """Ensemble of :class:`DepthHead` (different init, trained on bootstrap resamples of the same data)."""

    def __init__(self, num_members: int, scan_dim: int):
        super().__init__()
        self.heads = nn.ModuleList([DepthHead(scan_dim) for _ in range(num_members)])

    def forward_all(self, *inputs) -> torch.Tensor:
        """All members' images: (M, B, 1, 18, 32)."""
        return torch.stack([h(*inputs) for h in self.heads])

    @staticmethod
    def groups(member: torch.Tensor) -> list[torch.Tensor]:
        """Sample indices of every member for a fixed assignment ``member`` (B,): ONE host sync (the group sizes), after which
        ``forward`` runs without any sync (use it when the assignment is constant over many calls, e.g. an imagination)."""
        order = member.argsort()
        counts = torch.bincount(member, minlength=int(member.max()) + 1).tolist()
        return list(order.split(counts))

    def forward(self, member: torch.Tensor | list[torch.Tensor], *inputs) -> torch.Tensor:
        """Image of member ``member[b]`` for sample ``b`` (``member``: (B,) long, or the output of :meth:`groups`)."""
        groups = self.groups(member) if isinstance(member, torch.Tensor) else member
        out = None
        for head, idx in zip(self.heads, groups):
            if len(idx) == 0:
                continue
            img = head(*(x[idx] for x in inputs))
            out = torch.zeros(sum(len(g) for g in groups), *img.shape[1:], device=img.device) if out is None else out
            out = out.index_copy(0, idx, img)
        return out


class FailureHead(nn.Module):
    """Dreamer-style "continue" predictor: P(the transition into this state is a REAL termination), from the privileged critic
    features of the (true or imagined) state (invariant state, robot-vs-reference errors, xi, goal, terrain scan)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = mlp(in_dim, [256, 128], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B,)."""
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
