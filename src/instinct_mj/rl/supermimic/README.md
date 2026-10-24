# SuperMimic world-model training for perceptive shadowing parkour (G1)

Trains the depth policy of `Instinct-Perceptive-Shadowing-G1-v0` inside a learned world model, following
[SuperMimic](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) and its open reproduction
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT, Chong Zhang).

- **Single stage, no PPO.** The simulator only collects data. A differentiable world model is fit to it, and the policy is trained by
  back-propagation through 32-step imagined rollouts with analytic task rewards and a critic bootstrap.
- **Vision in the world-model state.** The state is (s, z): the robot state and the latent z of the policy's own depth encoder.
  One MLP predicts both, so no image is rendered in imagination.
- **Unchanged task and policy.** All domain randomization, camera noise and resets of the task are kept. The policy is the task's
  `EncoderActorCritic` (Conv2d depth encoder); checkpoints use the `OnPolicyRunner` format, so `instinct-play` and its ONNX export
  load them unchanged.

![world model](assets/world_model.png)

## Results

One RTX 3090, 1024 environments, dataset `20251116_50cm_kneeClimbStep1` (10 motions on 5 terrains). Play setting (each motion from
its first frame, no pushes, camera noise on, the three tracking terminations on), 12–13 episodes per motion; success = the whole
motion without any termination.

| Iteration | Training time | Success | Completion | Link pos error | Joint pos error |
|---|---|---|---|---|---|
| 2000 | 0.8 h | 10% | 45% | 0.054 m | 0.156 rad |
| 4000 | 1.7 h | 48% | 70% | 0.054 m | 0.173 rad |
| 6000 | 2.5 h | 83% | 91% | 0.053 m | 0.158 rad |
| 8000 | 3.2 h | 97% | 98% | 0.044 m | 0.134 rad |

Per motion at iteration 8000: diveroll4 (4 terrains) 85–100%, rollVault11 100% / 100%, jumpsit2 100% / 100%, kneelClimbStep1
85% / 100% (with / without wall). One seed, flags `--agent.fail-head False --agent.wm-failure-frac 0 --agent.wm-holdout-envs 0`;
the times include evaluations that shared the GPU. The earlier version with a learned depth-image renderer reached 95% at
iteration 8000 after 5.7 h.

## Method

```
for each iteration:
    1. collect   run the current policy (+ action noise) in the task env for ~2.5 steps (UTD 0.4), store transitions
    2. fit WM    10 batches of k=8-step autoregressive windows: dynamics + contact + latent + reconstruction (+ failure) losses
    3. policy    from real buffered states, imagine 32 steps with the frozen world model; analytic reward and soft
                 termination, critic bootstrap after 32 steps; maximize the return by BPTT
    4. critic    regress GAE(λ) targets computed on the imagined transitions
```

The first 500 iterations only fit the world model. Optimizers, learning rate (2e-4, cosine decay), gradient clipping, γ = 0.99,
λ = 0.95 and the update-to-data ratio follow SuperMimic.

**World model** (9.7M parameters, deterministic): MLP 3 × 2048 (LayerNorm, SiLU), `(s', z', contacts) = f(s, z, a, ξ, scan)`.
Inputs: heading- and position-invariant state (70), action (29), episode-constant simulation context ξ (friction, kp/kd, masses,
CoM, joint offsets; 188), terrain height scan from a heightmap (17 × 11), visual latent z (32). Outputs: frame-local state
increments (integrated into s'), per-body contact logits, and the increment of z. Loss over 8-step windows that start from a real
state and z₀ = E(real image): normalized state error, contact BCE, and the latent error against E(real image) at every step.

**Visual latent.** E is the policy's depth encoder (exported as `0-depth_image.onnx`). It is trained only by a training-only
decoder that reconstructs the clean depth image from E(noised image); the policy loss does not train it.

**Failure head** (optional, on by default): a continue predictor on privileged features, combined with the analytic soft
termination in imagination, `d = 1 − (1 − d_analytic)(1 − p_fail)`.

**Imagination.** The policy observation has exactly the format of the task's `policy` group (reference commands with noise, 8 steps
of noisy proprioception); the actor MLP gets z in place of the encoder output, with stop-gradient. Rewards and terminations are
analytic re-implementations of the task's terms (sigmoid-softened terminations); the critic (MLP 256-128) uses privileged features.

**Simulator side.** The env runs with `auto_reset=False` (set by the runner), so terminal transitions stay in the buffer. The camera
noise models (contour, artifacts, stereo, blur, too-close) are enabled in `G1PerceptiveShadowingEnvCfg` and inherited by Play.

## Usage

The dataset is the task's `MOTION_FOLDER` (default `~/Datasets/20251116_50cm_kneeClimbStep1`, download it from the
[release](https://github.com/project-instinct/InstinctMJ/releases/latest)).

```bash
# train / resume (the replay buffer is refilled on resume)
instinct-train Instinct-Perceptive-Shadowing-G1-v0 --num-envs 1024 --agent.max-iterations 8000
instinct-train Instinct-Perceptive-Shadowing-G1-v0 --num-envs 1024 --agent.load-run <run> --agent.load-checkpoint model_<it>.pt
# evaluate (results in <run>/eval/; --run-dir <run> --watch evaluates every new checkpoint, --video-only records videos)
python -m instinct_mj.scripts.supermimic.evaluate --checkpoint <run>/model_<it>.pt --motion-folder <dataset> --num-envs 128
# viser web viewer
python webplay_supermimic.py <run>/model_<it>.pt
```

A run directory holds `model_<it>.pt` (policy, optimizer, normalizers; `OnPolicyRunner` format), `supermimic_state_<it>.pt` (world
model, latent decoder, failure head, critic and their optimizers; read on resume), `params/`, `git/` and TensorBoard events
(world-model losses, latent error vs. open-loop horizon, imagined vs. real terminations, policy / critic, simulator statistics).
Single GPU only.

## Configuration

All fields are in `config.py` (`SuperMimicRunnerCfg`, part of the task's agent cfg) and can be set as `--agent.<field>`
(e.g. `--agent.pi-batch 2048`, `--agent.fail-head False`). The main ones:

| Field | Default | Meaning |
|---|---|---|
| `buffer_steps`, `utd`, `warmup_steps` | 1000, 0.4, 64 | Replay buffer length, updates per simulator step, steps before the first update |
| `wm_window`, `wm_batches`, `wm_batch` | 8, 10, 2048 | Window length k, world-model steps per iteration, windows per batch |
| `wm_only_iters` | 500 | World-model-only iterations at the start |
| `latent_windows`, `latent_weight` | 2048, 1.0 | Windows with the latent prediction loss and its weight |
| `latent_recon_images`, `latent_recon_weight` | 512, 1.0 | Images per batch and weight of the reconstruction loss that trains E |
| `fail_head`, `wm_failure_frac`, `wm_holdout_envs` | True, 0.25, 0.1 | Failure head, failure-window oversampling, held-out envs for diagnostics |
| `horizon`, `pi_batch` | 32, 1024 | Imagination horizon and starts per policy update |
| `gamma`, `lam`, `action_noise` | 0.99, 0.95, 0.1 | Discount, GAE λ, collection action noise |
| `lr`, `lr_decay_iters`, `lr_final_ratio` | 2e-4, 250000, 0.03 | Learning rate and cosine decay |
| `amp`, `compile`, `compile_policy` | `bf16`, True, True | bf16 MLPs, `torch.compile` of the world model and the imagination step |
| `checkpoint_imagination` | False | Gradient checkpointing of the imagination (less memory, more time) |

## Performance

RTX 3090, 1024 environments: 1.2–1.4 s per iteration measured for the results above (15.5 GB peak memory). Since then the small
rotation products of FK, rewards and the state integration are written as elementwise products that `torch.compile` fuses
(policy update about 2x faster in a benchmark than with batched 3 × 3 GEMMs, same results up to fp32 rounding). Most of the
remaining time is the simulator step (sensors, camera noise, managers).

## Files

| File | Content |
|---|---|
| `runner.py` | `SuperMimicRunner`, the task's runner: policy, normalizers, save / load, `export_as_onnx`; collection, world-model fit, imagination / BPTT, critic and logging (built lazily by `setup_training`) |
| `networks.py` | `WorldModel`, `LatentDecoder`, `FailureHead`, `Critic`, `RunningNorm` |
| `mdp.py` | Analytic rewards, terminations, policy observation, critic features |
| `kinematics.py`, `rotations.py` | Differentiable forward kinematics, rotation utilities |
| `terrain.py` | Terrain heightmap (ray-cast once, cached) and height scan |
| `buffer.py`, `sim.py`, `motion_lib.py`, `depth.py` | Replay buffer, env readers, reference motions, clean depth images |
| `scripts/supermimic/evaluate.py` | Evaluation entry point |
| `tests/supermimic/` | Analytic-head checks against the env, compile parity, Play loading, failure diagnosis, benchmarks |

## Known limitations

- The heightmap has a single surface, so overhangs are not represented in the world-model input.
- Latent prediction errors grow with the imagination horizon; what the camera has not seen at the start must be inferred.
- The analytic heads use the post-step state, the env managers body poses one physics substep older (small differences near
  thresholds).
- Issues of the existing task: the critic `height_scan` observation is constant (the scanner only includes geom group 0, the terrain
  is in group 2; kept for checkpoint compatibility), and `TerrainMotion._safe_motion_resampling` can index past the end of a
  replacement motion for some seeds (seed 42 is safe).

## Differences from SuperMimic

| | SuperMimic | This package |
|---|---|---|
| Task | Motion tracking on flat ground | Perceptive motion tracking on terrain |
| World-model input | State, action, context | Adds the terrain height scan and the visual latent; context extended to 188 |
| Outputs | State increments | Adds contact logits, the latent increment and a failure head |
| Policy | MLP encoder / decoder | The task's own depth policy network |
| Reset sampling | Return-based bins | The task's adaptive motion bins |
