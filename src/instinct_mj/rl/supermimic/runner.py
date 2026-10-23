"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update, single stage.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich (``train.py``).

The simulator is the unchanged mjlab perceptive shadowing env (including the depth camera) wrapped by
``InstinctRlVecEnvWrapper`` (DR, resets, the task's BeyondMimic adaptive bin sampling, terminations and timeouts stay the
task's; the reference's return-based reset-bin sampler is therefore NOT used). Rewards / terminations in imagination are
the analytic heads of ``mdp.py``. The policy IS the task's own actor (``EncoderActorCritic`` with the Conv2d depth
encoder, built by a stock ``instinct_rl`` ``OnPolicyRunner``) and acts on the task's ``policy`` observation group; in
imagination that observation is assembled analytically (``MDP.policy_obs``) with the depth image rendered by the
world model's ``DepthEnsemble``.
"""

from __future__ import annotations

import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from instinct_rl.runners import OnPolicyRunner
from tensorboardX import SummaryWriter

from . import networks
from .buffer import ReplayBuffer
from .depth import DepthPipeline
from .mdp import DEPTH_DIM, DEPTH_SHAPE, flatten_state, relative_root_pose, unflatten_state
from .networks import Critic, DepthEnsemble, FailureHead, WorldModel
from .sim import SimReader, build_mdp, camera_context

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2)))  # fmt: skip
CONTACT_LOSS_WEIGHT = 0.1
DEPTH_GAP_STEPS = (1, 8, 16, 32)


class SuperMimicRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cuda:0"):
        """``train_cfg`` = ``SuperMimicRunnerCfg.to_dict()`` plus the key ``"stock"``: the task's own
        ``InstinctRlOnPolicyRunnerCfg.to_dict()`` (policy architecture + policy normalizer)."""
        self.cfg, self.env, self.log_dir, self.device = train_cfg, env, log_dir, device
        c = train_cfg
        self.unwrapped = env.unwrapped
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        networks.AMP_DTYPE = torch.bfloat16 if c["amp"] == "bf16" else None

        # ---- the task's policy: stock runner builds EncoderActorCritic + optimizer + policy normalizer ----
        self.stock = OnPolicyRunner(env, copy.deepcopy(c["stock"]), log_dir=log_dir, device=device)
        self.ac = self.stock.alg.actor_critic
        self.opt_pi = self.stock.alg.optimizer
        self.pol_norm = self.stock.normalizers["policy"]
        self.pol_norm.eval()  # statistics are updated explicitly from collected real observations
        for g in self.opt_pi.param_groups:
            g["lr"], g["betas"], g["weight_decay"] = c["lr"], (0.9, 0.95), 0.0
        offset = 0
        for name, shape in env.get_obs_format()["policy"].items():
            if name == "depth_image":
                break
            offset += math.prod(shape)
        self.depth_slice = slice(offset, offset + DEPTH_DIM)
        assert math.prod(env.get_obs_format()["policy"]["depth_image"]) == DEPTH_DIM

        self.N = env.num_envs
        self.motion, self.kin, self.scanner, self.mdp = build_mdp(
            self.unwrapped, c["heightmap_resolution"], c["heightmap_cache_dir"], device
        )
        self.reader = SimReader(self.unwrapped)
        self.xi = self.reader.context(self.mdp)  # (N, xi_dim) episode-constant context of every env
        self.mdp.set_xi_stats(self.xi)
        self.xi_n = self.mdp.normalize_xi(self.xi)
        self.cam_ctx = camera_context(self.unwrapped)  # (N, 6) randomized camera mount offsets
        self.depth_pipe = DepthPipeline(self.unwrapped, compile_stages=c["compile_policy"])
        self.K = len(self.reader.contact_body_ids)
        self.hist, self.horizon, self.goal_dim = c["hist"], c["horizon"], self.mdp.goal_dim
        self.scan_dim = self.scanner.num_rays
        assert c["goal_frames"] == self.motion.goal_frames
        if not bool((self.motion.clip_len > self.horizon).any()):
            raise ValueError(f"No reference clip is longer than the imagination horizon ({self.horizon} frames).")
        assert c["warmup_steps"] >= max(self.horizon, c["wm_window"])

        self.wm = WorldModel(self.scan_dim, self.K, self.mdp.xi_dim).to(device)
        self.depth_head = DepthEnsemble(c["depth_ensemble"], self.scan_dim).to(device)
        self.critic = Critic(self.mdp.critic_dim()).to(device)
        self.opt_wm = torch.optim.Adam([*self.wm.parameters(), *self.depth_head.parameters()], lr=c["lr"])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr"])
        self.fail_head = FailureHead(self.mdp.critic_dim()).to(device)
        self.opt_fail = torch.optim.Adam(self.fail_head.parameters(), lr=c["lr"])
        self.fail_head_start_it = 0  # iteration at which the head was created (a resume may create it later)
        self.err_floor = ERR_FLOOR.to(device)
        self.buffer = ReplayBuffer(c["buffer_steps"], self.N, self.K, device, c["wm_holdout_envs"])
        self.obs, _ = env.get_observations()  # real flat policy observation (post-reset)
        self.it, self.sim_steps, self.wall_offset = 0, 0, 0.0
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self._build_fns()

    # ------------------------------------------------------------------------------------------------
    def _build_fns(self):
        mdp, wm = self.mdp, self.wm
        k = self.cfg["wm_window"]

        def _wm_rollout(s, a, xi_n):
            S = unflatten_state(s[:, 0])
            preds, logits = [], []
            for j in range(k):
                S, lg = wm.step(S, a[:, j], xi_n, mdp.scan(S, noisy=False))
                preds.append(S), logits.append(lg)
            return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1)  # (B, k, ...)

        def _wm_err(S_hat, logits, s1, contact, mask):
            std = torch.maximum(wm.out_norm.std(), self.err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)  # (B, k)
            bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none").sum(-1)
            B = s1.shape[0]
            return ((e + CONTACT_LOSS_WEIGHT * bce) * mask).sum() / B, (e[:, -1], (bce * mask).sum() / B)

        def _reward_term(S, f, origin, a, a_prev, xi, logits):
            r, terms = mdp.reward(S, f, origin, a, a_prev, xi, torch.sigmoid(logits), terms=True)
            return r, mdp.soft_termination(S, f, origin), terms

        def _critic_input(S, f, phase, origin, xi_n):
            return mdp.critic_features(S, f, origin, xi_n, mdp.goal(S, f, phase, noisy=False), mdp.scan(S, noisy=False))

        ac, pol_norm, depth_grad, policy_autocast = self.ac, self.pol_norm, self.cfg["depth_grad"], self.cfg["policy_autocast"]

        def _imag_core(S, o_hist, f, phase, depth, xi, xi_n, scan):
            """One imagination step up to (and excluding) the depth rendering: policy observation in the env's format, actor,
            world-model step, terrain scan and proprio history of the next state."""
            obs = mdp.policy_obs(o_hist, mdp.goal(S, f, phase), depth if depth_grad else depth.detach())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy_autocast):  # training imagination only
                a = ac.act_inference(pol_norm(obs)).float()
            a = torch.clamp(a, -20, 20)
            S1, logits = wm.step(S, a, xi_n, scan)
            o_hist1 = torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.obs(S1, a, xi)).unsqueeze(1)), 1)
            return a, S1, logits, mdp.scan(S1, noisy=False), o_hist1

        if self.cfg["compile"]:
            _wm_rollout, _wm_err = (torch.compile(fn) for fn in (_wm_rollout, _wm_err))
        if self.cfg["compile_policy"]:
            # level-batched FK keeps the graph small enough to compile (the former unrolled 30-body loop did not finish)
            _reward_term, _critic_input = (torch.compile(fn, dynamic=True) for fn in (_reward_term, _critic_input))
            _imag_core = torch.compile(_imag_core)
        self.wm_rollout, self.wm_err = _wm_rollout, _wm_err
        self.reward_term, self.critic_input, self.imag_core = _reward_term, _critic_input, _imag_core

    def _render_inputs(self, anchor, S_anchor, S, ctx):
        return anchor, relative_root_pose(S_anchor, S), S[4], ctx, self.mdp.scan(S, noisy=False)

    def render(self, member, anchor, S_anchor, S, ctx):
        """Depth image at state ``S`` (tuple) rendered by ensemble member ``member`` (B,) from the real ``anchor`` image
        taken at ``S_anchor``."""
        return self.depth_head(member, *self._render_inputs(anchor, S_anchor, S, ctx))

    def render_all(self, anchor, S_anchor, S, ctx):
        """(M, B, 1, 18, 32): every ensemble member."""
        return self.depth_head.forward_all(*self._render_inputs(anchor, S_anchor, S, ctx))

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c["lr_decay_iters"]:
            return c["lr"] * c["lr_final_ratio"]
        return c["lr"] * (c["lr_final_ratio"] + (1 - c["lr_final_ratio"]) * 0.5 * (1 + math.cos(math.pi * it / c["lr_decay_iters"])))

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        """Step the task env with the CURRENT policy on the real policy observation (real depth) + action noise.

        The env runs with ``auto_reset=False`` and finished envs are reset explicitly right after the pre-reset data
        (``s1``, the applied action, contacts, depth image, rewards, flags) has been read. This is the same sequence
        ``ManagerBasedRlEnv.step`` runs for ``auto_reset=True`` (rewards -> terminations -> ``_reset_idx`` ->
        ``write_data_to_sim`` -> ``forward`` -> commands with dt=0 for reset envs -> ``sense`` -> observations of the
        reset envs), because ``reset(env_ids)`` performs exactly those steps for the given envs; the only difference is
        that the finished envs' commands also advance once with ``dt`` before being reset, which the reset overwrites.
        """
        mdp, reader, env = self.mdp, self.reader, self.env
        for _ in range(n_steps):
            S = reader.state()
            f, phase, origin = self.motion.current_frame()
            a_prev = reader.last_action()
            ep_step = self.unwrapped.episode_length_buf.clone()
            scan_c = mdp.scan(S, noisy=False)
            depth = self.obs[:, self.depth_slice].view(self.N, *DEPTH_SHAPE)
            self.pol_norm.update(self.obs)
            a = self.ac.act_inference(self.pol_norm(self.obs))
            a = torch.clamp(a + self.cfg["action_noise"] * torch.randn_like(a), -20, 20)

            obs1, rew, dones, _ = env.step(a)
            # ---- pre-reset data of this transition ----
            a = self.unwrapped.action_manager.action.clone()
            done = dones.bool()
            terminated = self.unwrapped.reset_terminated.clone()
            S1 = reader.state()
            contact = reader.contact()
            r_sim = rew.sum(-1).clone()
            depth1_clean = self.depth_pipe.clean_from_sensor()  # head target: clean image of the same measurement as obs1
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            f1, phase1, origin1 = self.motion.current_frame()
            self.obs = obs1
            if len(done_ids) > 0:
                self.unwrapped.reset(env_ids=done_ids)
                self.obs = torch.where(done[:, None], env.get_observations()[0], obs1)  # post-reset obs of reset envs
            self.buffer.add({
                "s": flatten_state(S), "s1": flatten_state(S1), "a": a, "a_prev": a_prev, "contact": contact,
                "origin": origin, "r": r_sim, "f": f, "phase": phase, "ep_step": ep_step.int(), "done": done,
                "f1": f1, "phase1": phase1, "origin1": origin1, "terminated": terminated,
                "depth": depth.half(), "depth1_clean": depth1_clean.half(),
            })  # fmt: skip
            self.sim_steps += self.N
            self.wm.in_norm.update(self.wm.inputs(S, a, self.xi_n, scan_c))
            self.wm.out_norm.update(self.wm.targets(S, S1))
            r_an = mdp.reward(S1, f1, origin1, a, a_prev, self.xi, contact)
            stats["r_gap"].append((r_an - r_sim).abs().mean() / mdp.dt)
            stats["r"].append(r_sim.mean() / mdp.dt)
            if len(done_ids) > 0:
                stats["ep_len"].append((ep_step[done] + 1).float().mean())
                stats["term"].append(terminated[done].float().sum())
                stats["done"].append(done.float().sum())

    def wm_update(self):
        """One world-model gradient step on a k-step window batch (dynamics, contact, depth and failure losses)."""
        c = self.cfg
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"], self.buffer.train_envs, c["wm_failure_frac"], c["wm_failure_horizon"])
        env_idx = w["env"]
        S_hat, logits = self.wm_rollout(w["s"], w["a"], self.xi_n[env_idx])
        loss, (e_last, bce) = self.wm_err(S_hat, logits, w["s1"], w["contact"], w["mask"])
        k = w["mask"].shape[1]
        Bd = min(c["depth_windows"], w["mask"].shape[0])  # the depth head is trained on the first Bd windows (memory)
        ctx = self.cam_ctx[env_idx[:Bd]]
        # depth head, anchored on the real (noised) image at the window start, target = CLEAN image:
        #  (1) on the WM-PREDICTED states (detached: the head learns to be robust to state errors, the dynamics are
        #      not shaped by the rendering loss), steps 1..k
        S0 = unflatten_state(w["s"][:Bd, 0])
        rep = lambda x: x.unsqueeze(1).expand(Bd, k, *x.shape[1:]).flatten(0, 1)  # noqa: E731
        S_flat = tuple(x[:Bd].detach().flatten(0, 1) for x in S_hat)
        M = len(self.depth_head.heads)
        bootstrap = lambda n: torch.poisson(torch.ones(M, n, device=self.device))  # noqa: E731  (per-member resample weights)
        pred = self.render_all(rep(w["depth"][:Bd, 0].float()), tuple(rep(x) for x in S0), S_flat, rep(ctx))  # (M, Bd*k, 1, H, W)
        l1 = (pred - w["depth1_clean"][:Bd].flatten(0, 1).float()).abs().mean((2, 3, 4)).view(M, Bd, k)
        depth_pred = (l1 * w["mask"][:Bd] * bootstrap(Bd).unsqueeze(-1)).sum() / (Bd * M)
        #  (2) teacher-forced on the TRUE states with random gaps 1..horizon between anchor and target
        pr = self.buffer.sample_pairs(c["depth_pairs"], self.horizon, self.buffer.train_envs)
        S1_true = unflatten_state(pr["s1"])
        pred_tf = self.render_all(pr["anchor_depth"].float(), unflatten_state(pr["s0"]), S1_true, self.cam_ctx[pr["env"]])
        l1_tf = (pred_tf - pr["target_depth"].float()).abs().mean((2, 3, 4))  # (M, Bp)
        depth_tf = (l1_tf * pr["valid"] * bootstrap(l1_tf.shape[1])).sum() / (pr["valid"].sum().clamp(min=1) * M)
        total = loss + c["depth_loss_weight"] * (depth_pred + depth_tf)
        self.opt_wm.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_([*self.wm.parameters(), *self.depth_head.parameters()], 1.0)
        self.opt_wm.step()
        # failure ("continue") head on TRUE and (detached) WM-PREDICTED states of the windows, importance-weighted
        Bf = min(c["fail_windows"], w["mask"].shape[0])
        with torch.no_grad():
            flat_f = lambda x: x[:Bf].flatten(0, 1)  # noqa: E731
            xn = self.xi_n[env_idx[:Bf]].unsqueeze(1).expand(Bf, k, -1).flatten(0, 1)
            args_f = (flat_f(w["f1"]), flat_f(w["phase1"]), flat_f(w["origin1"]), xn)
            X_true = self.critic_input(unflatten_state(flat_f(w["s1"])), *args_f)
            X_pred = self.critic_input(tuple(flat_f(x) for x in S_hat), *args_f)
            self.fail_head.norm.update(X_true)
        label = flat_f(w["terminated"]).float()
        weight = (w["mask"][:Bf] * w["weight"][:Bf, None]).flatten()
        pos_weight = torch.where(label > 0.5, c["fail_pos_weight"], 1.0)
        head_loss = lambda X: (F.binary_cross_entropy_with_logits(self.fail_head(X), label, reduction="none") * pos_weight * weight).sum() / weight.sum().clamp(min=1e-6)  # noqa: E731
        fail_loss = head_loss(X_true) + head_loss(X_pred)
        self.opt_fail.zero_grad(set_to_none=True)
        fail_loss.backward()
        nn.utils.clip_grad_norm_(self.fail_head.parameters(), 1.0)
        self.opt_fail.step()
        m = w["mask"][:, -1]
        return {
            "fail_loss": fail_loss.detach() / 2, "fail_batch_pos": (label * w["mask"][:Bf].flatten()).sum() / w["mask"][:Bf].sum(),
            "wm": loss.detach() / k, "wm8": (e_last.detach() * m).sum() / m.sum().clamp(min=1), "bce": bce.detach(),
            "depth_pred": (l1.detach() * w["mask"][:Bd]).sum() / (w["mask"][:Bd].sum() * M).clamp(min=1),
            "depth_tf": (l1_tf.detach() * pr["valid"]).sum() / (pr["valid"].sum() * M).clamp(min=1),
        }  # fmt: skip

    @torch.no_grad()
    def depth_gap(self, num: int = 256, images: bool = False):
        """Depth error vs open-loop horizon step on real windows of the buffer (anchor = real image at the window start):
        world-model rollout (``pred``, ensemble mean), true states (``tf``), "copy the anchor image" (``copy``) and the
        ensemble disagreement (mean per-pixel std across members) on the rollout states."""
        steps = [n for n in DEPTH_GAP_STEPS if n <= self.horizon]
        w = self.buffer.sample_windows(num, self.horizon, self.buffer.holdout_envs)
        S0 = unflatten_state(w["s"][:, 0])
        anchor, ctx = w["depth"][:, 0].float(), self.cam_ctx[w["env"]]
        S, out, shown = S0, {}, None
        for j in range(self.horizon):
            S, _ = self.wm.step(S, w["a"][:, j], self.xi_n[w["env"]], self.mdp.scan(S, noisy=False))
            if j + 1 in steps:
                tgt, m = w["depth1_clean"][:, j].float(), w["mask"][:, j]
                p_all = self.render_all(anchor, S0, S, ctx)
                p = p_all.mean(0)
                t = self.render_all(anchor, S0, unflatten_state(w["s1"][:, j]), ctx).mean(0)
                masked_mean = lambda x: (x * m).sum() / m.sum().clamp(min=1)  # noqa: E731
                for name, img in (("pred", p), ("tf", t), ("copy", anchor)):
                    out[f"depth_gap/step_{j + 1}_{name}"] = masked_mean((img - tgt).abs().mean((1, 2, 3)))
                out[f"depth_gap/step_{j + 1}_ensemble_disagreement"] = masked_mean(p_all.std(0).mean((1, 2, 3)))
                if images and j + 1 == steps[1 if len(steps) > 1 else 0]:
                    shown = (anchor[:8], tgt[:8], t[:8], p[:8])
        return (out, shown) if images else out

    @torch.no_grad()
    def fail_calibration(self, num: int = 2048) -> dict:
        """Failure head vs actual REAL terminations on held-out windows (envs never used for training), on true states and
        on open-loop WM-predicted states: mean predicted vs actual rate, reliability bins, AUROC, per motion."""
        k = self.cfg["wm_window"]
        w = self.buffer.sample_windows(num, k, self.buffer.holdout_envs)
        env = w["env"]
        xn = self.xi_n[env].unsqueeze(1).expand(num, k, -1).flatten(0, 1)
        args_f = (w["f1"].flatten(), w["phase1"].flatten(), w["origin1"].flatten(0, 1), xn)
        S_hat, _ = self.wm_rollout(w["s"], w["a"], self.xi_n[env])
        mask = w["mask"].flatten() > 0.5
        label = w["terminated"].flatten()[mask].float()
        out = {}
        motion = self.motion.clip_id[w["f1"].flatten()][mask]
        for name, S in (("true_states", unflatten_state(w["s1"].flatten(0, 1))), ("wm_states", tuple(x.flatten(0, 1) for x in S_hat))):
            p = torch.sigmoid(self.fail_head(self.critic_input(S, *args_f)))[mask]
            out[f"fail/{name}/pred_rate"], out[f"fail/{name}/true_rate"] = float(p.mean()), float(label.mean())
            for lo, hi in ((0.0, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 1.01)):
                m = (p >= lo) & (p < hi)
                if m.any():
                    out[f"fail/{name}/bin_{lo}_{hi}/pred"], out[f"fail/{name}/bin_{lo}_{hi}/actual"] = float(p[m].mean()), float(label[m].mean())
            if 0 < label.sum() < len(label):
                order = p.argsort()
                ranks = torch.empty_like(p)
                ranks[order] = torch.arange(1, len(p) + 1, device=self.device, dtype=p.dtype)
                npos = label.sum()
                out[f"fail/{name}/auroc"] = float((ranks[label > 0.5].sum() - npos * (npos + 1) / 2) / (npos * (len(p) - npos)))
            if name == "true_states":
                for m_id in motion.unique().tolist():
                    sel = motion == m_id
                    out[f"fail/motion_{m_id}/pred_rate"], out[f"fail/motion_{m_id}/true_rate"] = float(p[sel].mean()), float(label[sel].mean())
        return out

    @torch.no_grad()
    def termination_check(self, num: int = 256) -> dict:
        """Imagination vs reality from the same held-out real start states: fraction of the starts that terminate within the
        horizon in imagination (analytic hard termination; and 1 - survival with the failure head) vs in the buffer."""
        H, c = self.horizon, self.cfg
        eligible = self.buffer.eligible_starts(
            lambda f: f + H <= self.motion.clip_end[self.motion.clip_id[f]], self.buffer.holdout_envs, max_age=self.buffer.size - H - 1
        )
        if len(eligible) < 8:
            return {}
        pick = eligible[torch.randint(0, len(eligible), (min(num, len(eligible)),), device=self.device)]
        s, h = self.buffer.starts_at(pick[:, 0], pick[:, 1], self.hist)
        im = self._imagine(s, h)
        B = len(pick)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        rep = lambda x: x.unsqueeze(0).expand(H, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in im["Ss"]]) for i in range(6))
        f_all = flat(im["Fs"])
        hard = self.mdp.hard_termination(S_all, f_all, rep(im["origin"])).view(H, B).any(0)
        soft = self.mdp.soft_termination(S_all, f_all, rep(im["origin"])).view(H, B)
        p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_all, f_all, rep(im["phase"]), rep(im["origin"]), rep(im["xi_n"])))).view(H, B).clamp(max=c["fail_clip"])
        survive_with_head = ((1 - soft) * (1 - p_fail)).prod(0)
        # real outcome: a REAL termination within the next H transitions of the same episode
        rows = self.buffer._row(pick[:, 0].unsqueeze(-1) + torch.arange(H, device=self.device))
        env = pick[:, 1].unsqueeze(-1).expand_as(rows)
        done, term = self.buffer.data["done"][rows, env], self.buffer.data["terminated"][rows, env]
        same_episode = torch.cumsum(done.int(), 1) - done.int() == 0
        real = (term & same_episode).any(1)
        out = {"termcheck/imagined_hard": float(hard.float().mean()), "termcheck/imagined_soft_with_head": float((1 - survive_with_head).mean()),
               "termcheck/real": float(real.float().mean())}
        motion = self.motion.clip_id[s["f"]]
        for m_id in motion.unique().tolist():
            sel = motion == m_id
            out[f"termcheck/motion_{m_id}/imagined_hard"], out[f"termcheck/motion_{m_id}/real"] = float(hard[sel].float().mean()), float(real[sel].float().mean())
        return out

    def sample_valid_starts(self, B: int):
        """Uniform over all buffered states whose imagination stays inside the reference clip (``f + H <= clip end``)."""
        eligible = self.buffer.eligible_starts(lambda f: f + self.horizon <= self.motion.clip_end[self.motion.clip_id[f]])
        if len(eligible) == 0:
            raise RuntimeError("No buffered state has a full imagination horizon left inside its reference clip.")
        pick = eligible[torch.randint(0, len(eligible), (B,), device=self.device)]
        return self.buffer.starts_at(pick[:, 0], pick[:, 1], self.hist)

    def _imagine(self, s: dict, h: dict) -> dict:
        """H-step imagination of the policy from buffer start states ``s`` / histories ``h`` (WM + depth ensemble member per
        rollout + camera noise). Gradients flow as in ``policy_update`` (call under ``no_grad`` for diagnostics)."""
        c, mdp, dev = self.cfg, self.mdp, self.device
        B, HZ = s["f"].shape[0], self.horizon
        S = unflatten_state(s["s"])
        f, phase, origin, a_prev, env_idx = s["f"], s["phase"], s["origin"], s["a_prev"], s["env"]
        xi, xi_n, ctx = self.xi[env_idx], self.xi_n[env_idx], self.cam_ctx[env_idx]
        depth0 = s["depth"].float()  # real image at the start state = anchor of the rendered images
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], self.xi[h["env"]].unsqueeze(1))
            # one noise sample per distinct historical observation (backfilled slots duplicate it, as in collection)
            noise = (2 * torch.rand_like(o_hist) - 1) * mdp.proprio_noise
            o_hist = o_hist + noise.gather(1, h["slot"].unsqueeze(-1).expand_as(noise))
        member = torch.randint(0, len(self.depth_head.heads), (B,), device=dev)  # one ensemble member per rollout
        member = self.depth_head.groups(member)  # constant over the horizon: one host sync instead of one per step and member
        S0, f0, depth, scan = S, f, depth0, mdp.scan(S, noisy=False)
        Ss, As, Aps, Fs, Ls = [], [], [], [], []
        for _ in range(HZ):
            Aps.append(a_prev)
            # ---- policy observation in EXACTLY the env's policy-group format; depth = rendered image ----
            # step 0: the REAL buffered (noised) image; later steps: the rendered clean image + the camera's noise models
            step_args = (S, o_hist, f, phase, depth, xi, xi_n, scan)
            if c["checkpoint_imagination"] and torch.is_grad_enabled():
                # recompute the step in the backward pass instead of keeping its activations (same RNG state is replayed)
                a_prev, S, logits, scan, o_hist = torch.utils.checkpoint.checkpoint(self.imag_core, *step_args, use_reentrant=False)
            else:
                a_prev, S, logits, scan, o_hist = self.imag_core(*step_args)
            with torch.set_grad_enabled(c["depth_grad"]):
                depth = self.depth_pipe.noise_predicted(self.render(member, depth0, S0, S, ctx))
            f = f + 1
            Ss.append(S), As.append(a_prev), Fs.append(f), Ls.append(logits)
        return {"S0": S0, "f0": f0, "S": S, "f": f, "phase": phase, "origin": origin, "xi": xi, "xi_n": xi_n, "Ss": Ss, "As": As, "Aps": Aps, "Fs": Fs, "Ls": Ls}

    def policy_update(self):
        """One policy step: H-step imagination from buffer states, BPTT of the discounted return, then GAE critic fit."""
        c, mdp, dev = self.cfg, self.mdp, self.device
        B, HZ = c["pi_batch"], self.horizon
        s, h = self.sample_valid_starts(B)
        frozen = (self.wm, self.depth_head, self.critic, self.fail_head)
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        im = self._imagine(s, h)
        S0, f0, S, f, phase, origin, xi, xi_n = im["S0"], im["f0"], im["S"], im["f"], im["phase"], im["origin"], im["xi"], im["xi_n"]
        Ss, As, Aps, Fs, Ls = im["Ss"], im["As"], im["Aps"], im["Fs"], im["Ls"]
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        rep = lambda x: x.unsqueeze(0).expand(HZ, *x.shape).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        r, d, terms = self.reward_term(S_all, flat(Fs), rep(origin), flat(As), flat(Aps), rep(xi), flat(Ls))
        p_fail_mean = torch.zeros((), device=dev)
        if c["fail_head"] and self.it >= self.fail_head_start_it + c["fail_head_warmup_iters"]:
            # learned failure ("continue") head, combined with the analytic soft termination; the gradient is kept
            p_fail = torch.sigmoid(self.fail_head(self.critic_input(S_all, flat(Fs), rep(phase), rep(origin), rep(xi_n)))).clamp(max=c["fail_clip"])
            d = 1 - (1 - d) * (1 - p_fail)
            p_fail_mean = p_fail.detach().mean()
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        cT = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not c["no_critic"]:
            J = J + (c["gamma"] ** HZ) * cT * self.critic(self.critic_input(S, f, phase, origin, xi_n))
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.ac.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        for m_ in frozen:
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros((), device=dev)
        if not c["no_critic"]:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
                rep1 = lambda x: x.unsqueeze(0).expand(HZ + 1, *x.shape).flatten(0, 1)  # noqa: E731
                X = self.critic_input(S_c, f_c, rep1(phase), rep1(origin), rep1(xi_n)).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c["gamma"] * (1 - dones[t]) * ((1 - c["lam"]) * V[t + 1] + c["lam"] * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c["critic_minibatches"]
            for _ in range(c["critic_epochs"]):
                perm = torch.randperm(n, device=dev)
                for i in range(c["critic_minibatches"]):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        term_means = {k: v.detach().view(HZ, B).mean() for k, v in terms.items()}
        term_means["p_fail_mean"] = p_fail_mean
        return loss.detach(), v_loss.detach(), (r_sum / HZ / mdp.dt).mean(), gn.detach(), cT.mean().detach(), term_means

    # ------------------------------------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Train until iteration ``num_learning_iterations`` (total, so a resumed run continues). The env's random
        initial episode lengths are not used: resets come from the task's reference sampling."""
        c = self.cfg
        stats = {"r": [], "r_gap": [], "ep_len": [], "term": [], "done": []}
        self.collect(c["warmup_steps"], stats)  # also refills the (unsaved) replay buffer after a resume
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        agg: dict[str, list] = {k: [] for k in ("wm", "wm8", "bce", "depth_pred", "depth_tf", "fail_loss", "fail_batch_pos", "pi", "v", "ir", "gn", "surv")}
        terms_acc: dict[str, list] = {}
        while self.it < num_learning_iterations:
            it = self.it
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v, self.opt_fail):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / c["utd"]
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c["wm_batches"]):
                m = self.wm_update()
            for k_ in ("wm", "wm8", "bce", "depth_pred", "depth_tf", "fail_loss", "fail_batch_pos"):
                agg[k_].append(m[k_])
            if it >= c["wm_only_iters"]:
                pl, vl, ir, gn, surv, terms = self.policy_update()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                for k, v in terms.items():
                    terms_acc.setdefault(k, []).append(v)
            self.it += 1
            if self.it % c["log_interval"] == 0:
                mean = lambda x: float(sum(float(v) for v in x) / len(x)) if len(x) else float("nan")  # noqa: E731
                ips = c["log_interval"] / (time.time() - t_log)
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = {
                    "wm/loss": mean(agg["wm"]), "wm/err_8step": mean(agg["wm8"]), "wm/contact_bce": mean(agg["bce"]),
                    "depth/l1_predicted_states": mean(agg["depth_pred"]), "depth/l1_true_states_gap1to32": mean(agg["depth_tf"]),
                    "fail/bce_train": mean(agg["fail_loss"]), "fail/positive_share_in_batch": mean(agg["fail_batch_pos"]),
                    "policy/loss": mean(agg["pi"]), "policy/grad_norm": mean(agg["gn"]), "critic/value_loss": mean(agg["v"]),
                    "imagination/reward": mean(agg["ir"]), "imagination/survival": mean(agg["surv"]),
                    "sim/reward": mean(stats["r"]), "sim/analytic_reward_abs_gap": mean(stats["r_gap"]),
                    "sim/episode_length": mean(stats["ep_len"]), "sim/terminated_fraction": term_frac,
                    "perf/it_per_s": ips, "perf/lr": lr, "perf/sim_steps": float(self.sim_steps), "perf/wall_h": wall_h,
                    "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                }  # fmt: skip
                images = None
                real = self.obs[:, self.depth_slice].view(self.N, *DEPTH_SHAPE)
                imag = self.depth_pipe.noise_predicted(self.depth_pipe.clean_from_sensor())
                for name, x in (("real", real), ("imagination", imag)):
                    row[f"depth_noise/{name}_mean"], row[f"depth_noise/{name}_std"] = float(x.mean()), float(x.std())
                    row[f"depth_noise/{name}_frac_max"], row[f"depth_noise/{name}_frac_zero"] = float((x >= 1.0).float().mean()), float((x <= 0.0).float().mean())
                if self.buffer.size >= self.horizon:
                    row.update(self.fail_calibration())
                    if self.it % (5 * c["log_interval"]) == 0:
                        row.update(self.termination_check())
                    gap = self.depth_gap(images=self.writer is not None and self.it % (10 * c["log_interval"]) == 0)
                    if isinstance(gap, tuple):
                        gap, images = gap
                    row.update({k: float(v) for k, v in gap.items()})
                for k, v in terms_acc.items():
                    row[f"imagination/reward_term/{k}"] = mean(v)
                if self.writer is not None:
                    for k, v in row.items():
                        if not math.isnan(v):
                            self.writer.add_scalar(k, v, self.it)
                    if images is not None:
                        for name, img in zip(("anchor", "real_step8", "true_state_pred_step8", "open_loop_pred_step8"), images):
                            self.writer.add_images(f"depth/{name}", img, self.it)
                    self.writer.flush()
                g = lambda k: row.get(f"depth_gap/step_{k}_pred", float("nan"))  # noqa: E731
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm/loss']:.4f} e8 {row['wm/err_8step']:.4f} bce {row['wm/contact_bce']:.4f} "
                      f"| depth l1 pred {row['depth/l1_predicted_states']:.4f} tf {row['depth/l1_true_states_gap1to32']:.4f} "
                      f"gap@8 {g(8):.4f} @32 {g(32):.4f} | pi {row['policy/loss']:.3f} v {row['critic/value_loss']:.4f} "
                      f"imag_r {row['imagination/reward']:.3f} surv {row['imagination/survival']:.3f} | sim_r {row['sim/reward']:.3f} "
                      f"gap {row['sim/analytic_reward_abs_gap']:.4f} ep_len {row['sim/episode_length']:.1f} term {term_frac:.2f} "
                      f"| {ips:.2f} it/s mem {row['perf/max_mem_gb']:.1f}G", flush=True)  # fmt: skip
                t_log = time.time()
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
                terms_acc = {}
            if self.it % c["save_interval"] == 0 or self.it == num_learning_iterations:
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{self.it}.pt"))

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sidecar(path: str) -> str:
        """``.../model_<it>.pt`` -> ``.../supermimic_state_<it>.pt`` (does not match play's ``model_.*.pt`` lookup)."""
        d, name = os.path.split(path)
        return os.path.join(d, "supermimic_state_" + name[len("model_"):])

    # keys of the sidecar that belong to modules added after the first release (see ``load``)
    NEW_HEAD_KEYS = ("fail_head", "opt_fail", "fail_head_start_it")

    def save(self, path: str) -> None:
        """``path``: stock ``OnPolicyRunner`` checkpoint (actor-critic, policy optimizer, policy normalizer): directly
        playable. Sidecar: world model, depth head, failure head, critic, their optimizers and the training state (resume)."""
        self.stock.current_learning_iteration = self.it
        self.stock.save(path)
        torch.save({
            "wm": self.wm.state_dict(), "depth_head": self.depth_head.state_dict(), "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "opt_v": self.opt_v.state_dict(), "iteration": self.it,
            "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": {k: v for k, v in self.cfg.items() if k != "stock"},
            "xi_mean": self.mdp.xi_mean, "xi_scale": self.mdp.xi_scale,
            "fail_head": self.fail_head.state_dict(), "opt_fail": self.opt_fail.state_dict(), "fail_head_start_it": self.fail_head_start_it,
        }, self._sidecar(path))  # fmt: skip

    def load(self, path: str, init_new_heads: bool = False) -> None:
        """Resume from a stock checkpoint + its sidecar. Existing modules must load strictly (missing keys error). A sidecar
        written before the failure head existed has exactly ``NEW_HEAD_KEYS`` missing: that is only accepted with
        ``init_new_heads=True`` (``--resume-init-new-heads``), in which case the head is initialized fresh (and warmed up for
        ``fail_head_warmup_iters`` iterations from the loaded iteration before it affects the policy)."""
        self.stock.load(path)
        ck = torch.load(self._sidecar(path), map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"]), self.depth_head.load_state_dict(ck["depth_head"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_v.load_state_dict(ck["opt_v"])
        self.it, self.wall_offset, self.sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        assert self.it == self.stock.current_learning_iteration
        self.mdp.xi_mean, self.mdp.xi_scale = ck["xi_mean"].to(self.device), ck["xi_scale"].to(self.device)
        self.xi_n = self.mdp.normalize_xi(self.xi)
        missing = [k for k in self.NEW_HEAD_KEYS if k not in ck]
        if missing:
            if len(missing) != len(self.NEW_HEAD_KEYS):
                raise KeyError(f"sidecar {self._sidecar(path)} has only part of the failure-head keys; missing {missing}")
            if not init_new_heads:
                raise KeyError(f"sidecar {self._sidecar(path)} has no failure head ({list(self.NEW_HEAD_KEYS)}); pass --resume-init-new-heads to create it fresh")
            self.fail_head_start_it = self.it
            print(f"[INFO] sidecar has no failure head: initialized it fresh at iteration {self.it} "
                  f"(warmup {self.cfg['fail_head_warmup_iters']} iterations before it affects the policy).")
        else:
            self.fail_head.load_state_dict(ck["fail_head"]), self.opt_fail.load_state_dict(ck["opt_fail"])
            self.fail_head_start_it = ck["fail_head_start_it"]
