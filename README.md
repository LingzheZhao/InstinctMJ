# InstinctMJ · SuperMimic

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/simulator-MuJoCo%20Warp-silver.svg)](https://github.com/google-deepmind/mujoco_warp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

This branch trains the G1 perceptive shadowing parkour policy (`Instinct-Perceptive-Shadowing-G1-*`, depth camera) with
**SuperMimic**: the simulator only collects data, one differentiable world model predicts the robot state and the visual latent of
the policy's depth encoder, and the policy is trained by backpropagation through 32-step imagined rollouts. No PPO. InstinctMJ is
the `mjlab` port of [InstinctLab](https://github.com/project-instinct/InstinctLab) ([Project-Instinct](https://project-instinct.github.io/)).

![world model](src/instinct_mj/rl/supermimic/assets/world_model_simple.png)

One RTX 3090, 1024 environments: **97% success** on the 10 motions of the released dataset (diveroll, knee climb onto a 50 cm
step, jump-sit, roll vault) after 8000 iterations (3.2 h). Method, results and configuration:
[src/instinct_mj/rl/supermimic/README.md](src/instinct_mj/rl/supermimic/README.md).

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync
```

- Python 3.10–3.13, Linux x86_64.
- `mjlab 1.6.0` must be next to this directory as `../mjlab` (installed editable from that path).
- `instinct_rl 1.0.2` is installed from the commit recorded in `uv.lock`.

## Data and weights

Download from the [SuperMimic release](https://github.com/project-instinct/InstinctMJ/releases/latest):

- `20251116_50cm_kneeClimbStep1.zip`: the dataset. Unzip into `~/Datasets/` (the default `MOTION_FOLDER` in
  `src/instinct_mj/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`).
- The trained policy checkpoint, its ONNX export and the world-model state (to resume training).

## Usage

```bash
# train
uv run instinct-train Instinct-Perceptive-Shadowing-G1-v0 --num-envs 1024
# evaluate (success rate per motion)
uv run python -m instinct_mj.scripts.supermimic.evaluate --checkpoint <run>/model_<it>.pt --motion-folder ~/Datasets/20251116_50cm_kneeClimbStep1 --num-envs 128
# web viewer (viser)
uv run python webplay_supermimic.py <run>/model_<it>.pt
# play + ONNX export
uv run instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --checkpoint-file <run>/model_<it>.pt --viewer none --export-onnx True
```

Runs are written to `logs/instinct_rl/g1_perceptive_shadowing/`.

## License

[CC BY-NC 4.0](LICENSE): no commercial use. Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md).

Checkpoints trained outside InstinctMJ are not compatible: the joint order comes from the MJCF and differs from IsaacLab.
