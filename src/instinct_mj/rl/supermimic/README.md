# SuperMimic 风格 model-based 训练器（G1 perceptive shadowing，单阶段深度策略）

参考实现：SuperMimic（MIT，Chong Zhang / ETH Zurich）。`rotations / kinematics / networks / buffer / mdp / runner(train)` 沿用其结构，各文件头部保留署名。
算法（Algorithm A.1）：**collect → k=8 窗口拟合可微世界模型 → H=32 BPTT 策略 + GAE critic**。
**单阶段**：不再区分 teacher/student。世界模型除动力学外还学习“视觉”（深度渲染头 `DepthHead`），策略就是任务自带的深度策略
（`EncoderActorCritic` + Conv2d 深度编码器），**仅通过世界模型想象中的 BPTT 训练**；仿真只用于采集数据，没有 PPO/其它在仿真中的策略训练。
训练的是**已有任务** `Instinct-Perceptive-Shadowing-G1-v0`（没有新建任务，没有改任务文件）。

![world model](assets/world_model_architecture.png)

## 结果（一张 RTX 3090，1024 envs，数据集 20251116_50cm_kneeClimbStep1）

![success curve](assets/success_curve.png)

评估设置与 Play 相同（从动作第 0 帧开始，没有推力，深度噪声开着），保留三项跟踪终止。
每个 env 跑一个完整 episode，整条动作跑完且没有触发任何终止记为成功；每条动作 12–13 个 episode。

| 迭代 | 训练时长 | 成功率 | 完成度 | link 误差 | 关节误差 |
|---|---|---|---|---|---|
| 2000 | 1.2 h | 14% | 49% | 0.057 m | 0.153 rad |
| 4000 | 2.6 h | 72% | 81% | 0.058 m | 0.167 rad |
| 8000 | 5.7 h | 95% | 97% | 0.045 m | 0.140 rad |
| 12000 | 8.7 h | 98% | 98% | 0.042 m | 0.121 rad |

12000 次迭代时：diveroll4（4 种地形）92–100%，rollVault11 100%，jumpsit2 92% / 100%，kneelClimbStep1 92% / 100%。
这次训练用的是加入失败头之前的代码；用当前代码复现，需要加 `--agent.no-fail-head --agent.wm-failure-frac 0 --agent.wm-holdout-envs 0`。
可视化：仓库根目录的 `webplay_supermimic.py [checkpoint]` 启动 viser 网页查看器。

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | `SuperMimicRunnerCfg`（论文 Table A.5 超参 + 深度头选项） |
| `runner.py` | `SuperMimicRunner(env, train_cfg, log_dir, device)`：`learn / save / load` |
| `mdp.py` | 解析 head：任务的 reward 项、终止、**policy 观测组的解析装配 `MDP.policy_obs`**、critic 特征；状态 `S=(p,R,v,w,q,qd)`（root 为 `torso_link`） |
| `depth.py` | `DepthPipeline`：干净/带噪的处理后深度图；对预测深度图施加相机噪声模型 |
| `motion_lib.py` | 由任务已加载的 `TerrainMotion` concat buffer 直接构造的参考库（局部系，世界 = 局部 + origin） |
| `kinematics.py` / `terrain.py` | 可微 FK；全地形高度图（对静态地形 geom 向下 `mj_ray`，磁盘缓存）+ 可微 `height_scan` |
| `sim.py` | 从 mjlab 读取状态（`qpos/qvel`）、接触、per-env 上下文 ξ、**相机安装偏置 `camera_context`**；`build_mdp` |
| `networks.py` / `buffer.py` | WorldModel（3x2048 SiLU+LN，含 contact head）、`DepthHead` / `DepthEnsemble`、Critic、GPU replay buffer |

入口：`src/instinct_mj/scripts/supermimic/train.py`（`instinct-supermimic-train`），校验：`tests/supermimic/check_analytic_heads.py`、`tests/supermimic/check_play_load.py`。

## 运行

`pyproject.toml` 新增了 `instinct-supermimic-train` / `instinct-supermimic-eval` 两个入口，需要重新安装（`uv pip install -e .`）后才能使用；
也可以直接运行 `python -m instinct_mj.scripts.supermimic.train ...` / `python -m instinct_mj.scripts.supermimic.evaluate ...`。

```bash
.venv/bin/instinct-supermimic-train --motion-folder ~/Xyk/Datasets/20251116_50cm_kneeClimbStep1 --num-envs 1024
# 常用: --agent.max-iterations N --agent.wm-only-iters 500 --agent.buffer-steps 1000 --agent.pi-batch 1024
#       --no-depth-noise  --agent.depth-ensemble 3  --agent.depth-grad  --agent.no-compile  --resume logs/instinct_rl/g1_perceptive_shadowing/<run>/model_<it>.pt
```
输出目录与 `instinct-train` 一致：`logs/instinct_rl/g1_perceptive_shadowing/<timestamp>_supermimic/`，其中
- `model_<it>.pt`：**stock `OnPolicyRunner.save` 格式**（actor-critic、policy 优化器、policy normalizer），`instinct-play ... Instinct-Perceptive-Shadowing-G1-Play-v0` 的 load_run/checkpoint 机制直接可用，无需额外参数；策略结构 = 任务的 `rl_cfg.G1PerceptiveShadowingPPORunnerCfg().policy`（actor [512,256,128]、ELU、Conv2d 深度编码器 channels [32,32]、输出 32，原样；没有放大）。
- `supermimic_state_<it>.pt`：边车文件（世界模型、深度头集成、critic、它们的优化器、ξ 归一化统计、训练状态），仅用于 `--resume`（文件名不会被 play 的 `model_.*.pt` 匹配到）。

## 设计

- **仿真器 = 任务本身（含深度相机）**，DR、reset、BeyondMimic 自适应 bin 采样、终止/超时保持原样，不使用参考实现的 return-based bin 采样。
  env 以 `auto_reset=False` 运行：先读 pre-reset 数据（`s1`、实际动作、接触、奖励、flag、干净深度图），再显式 `env.reset(env_ids)`。
  这与 `step(auto_reset=True)` 的顺序等价（reward→termination→`_reset_idx`→`write_data_to_sim`→`forward`→reset env 的 command(dt=0)→`sense`→reset env 的观测），
  仅差别是 done env 的 command 在被 reset 覆盖前多推进了一次 dt。终止那一步的 transition 保留，reference 的 exclusive 边界 mask 只屏蔽其后的 transition。
- **采集时策略作用于真实的 `policy` 观测组**（真实带噪深度 + 带噪本体历史），加动作噪声；policy normalizer 用采集到的真实观测更新，BPTT 期间冻结。
- **想象中的策略观测**严格按 env 的 policy 组装配（`MDP.policy_obs`）：`joint_pos_ref, joint_vel_ref, position_ref, rotation_ref`（解析 goal，噪声同任务）、`depth_image`、
  然后 8 步（oldest first、time-major）的 `projected_gravity, base_ang_vel, joint_pos, joint_vel, last_action`（噪声同任务）。校验脚本对真实状态逐项核对。
- **深度渲染头 `DepthHead`**（世界模型的一部分）：输入 = 想象起点的**真实**深度图（anchor）、root（= 相机安装体 `torso_link`）相对 anchor 的位姿（3+6D）、关节角（自遮挡）、
  per-env 相机安装偏置（`randomize_ray_offsets` 随机化的位置 3 + 旋转向量 3，由 `sim.camera_context` 从传感器的光线读出）、局部高度图 scan；
  小卷积编码器 + MLP + 转置卷积解码器（带 anchor 跳连），sigmoid 输出 [0,1]。
  训练：(1) 在窗口的**世界模型预测状态**上（detach，使头对状态误差鲁棒但不影响动力学），anchor = 窗口首帧真实图，目标 = 步 1..8 的图；
  (2) 在**真实状态**上做 teacher forcing，anchor/目标间隔随机 1..H（=32），覆盖想象中的长间隔。两项均按 episode 边界 mask。
- **DAWN 式的噪声处理**：相机管线 = 噪声模型（米，原始分辨率）→ normalize → crop/resize。策略的真实观测 = 带噪处理图；深度头的 anchor 输入 = 带噪图；深度头目标 = **干净**图
  （对传感器原始 `distance_to_image_plane` 施加同样确定性的 normalize + crop/resize）。想象中把预测的干净图经 `DepthPipeline.noise_predicted` 送入同一套噪声模型
  （逆 crop/resize/normalize 回到原始分辨率米制图，饱和像素还原为远距离值，再跑噪声阶段和 normalize/crop）后才给策略。
- **相机噪声默认开启（`--depth-noise/--no-depth-noise`，只在训练脚本里改 cfg，不改任务文件）**：使用 `make_perceptive_scene_sensors` 中被注释掉的列表及相同参数：
  `DepthContourNoise(1.8,1)`、`DepthArtifactNoise()`、`RangeBasedGaussianNoise(max 1.2, min 0.12, std 0.02)`、`GaussianBlurNoise(3, 0.5)`、`StereoTooCloseNoise()`，位于 normalize/crop 之前。
  未启用 reflection / sky artifact：相机量程被裁到 2 m，天空伪影不会出现，反射类与通用 artifact 重复。
- **鲁棒性措施（均在世界模型训练内、可配置）**：(a) 想象步 0 用 buffer 里的真实图，步 ≥1 用深度头；(b) 深度头集成（默认 3，不同初始化 + Poisson bootstrap 重采样），每条想象轨迹固定随机选一个成员；
  (c) 预测图经相机噪声模型后才给策略；(d) 默认 **stop-gradient**：给策略的预测深度不回传梯度（WMP 的做法，避免策略学着去利用深度头误差），`--agent.depth-grad` 可开启（与带噪声模型的梯度未充分测试）。
- **ξ（episode 常量上下文）**：默认关节偏移、torso COM 偏移、kp/kd 缩放、被随机化 body 质量比，以及每个碰撞 geom 各自的 3 个摩擦分量（见 `sim.SimReader.context`）。
- **接触项**：`undesired_contacts` 用世界模型的 learned contact head（BCE），想象中用 sigmoid 概率；终止的硬版本用于校验，软版本用于想象。
- **Critic**：SuperMimic 的 critic（特权特征 + goal + ξ + 高度图 scan），EncoderActorCritic 自带的 critic 不使用（其输入维度与任务一致，保证 stock `load` 通过）。
- **torch.compile** 只作用于世界模型 rollout/loss；FK 展开的 reward/critic 特征和想象步保持 eager。

## 校验结果

`tests/supermimic/check_analytic_heads.py`（会断言容差，失败以非零码退出；16 envs、120 步、含终止 transition，数据集 20251116_50cm_kneeClimbStep1，**通过**）：

| 量 | max / mean abs err |
|---|---|
| 参考帧索引（base/link/joint，相位均非零）vs `reference_frame` | 0 / 0 |
| 状态读取、FK（link pos/rot/lin vel/ang vel）vs sim | ≤2.4e-6；1.9e-6 / 7.0e-7 / 2.3e-5 / 6.7e-6 |
| reward 各项（同一最新状态；含终止 transition） | ≤2.1e-6，link_pos/link_rot 因 env 缓存 `reference_relative_*` 为 ≤2.7e-2（mean 2.6e-3 / 7e-4） |
| reward 总和 vs env 返回值（env 状态滞后一个 substep） | 2.2e-2 / 7.8e-4 |
| 终止 flag：自然步进 / 写入扰动状态强制命中阈值两侧 | 仅阈值附近不一致（|margin|≤6e-3）/ 0 不一致（三项均两侧命中） |
| PD 力矩 vs `qfrc_actuator`，观测，goal，历史顺序 | ≤1.5e-5 |
| **解析装配的 policy 观测 vs env policy 组（除深度外，关闭噪声）** | 逐项：ref 项 ≤2e-5；整条 flat 向量 1.4e-5 / 3e-8；term 顺序一致 |
| depth_image：形状 (1,18,32)，值域 | 形状一致，全部在 [0,1] |
| 相机偏置：用拟合的旋转向量旋转标称光线 vs 实际光线；位置偏置 | 1.9e-3（mean 2.6e-4）；1.9e-9 |
| 终止前 `s1` 与 reset 后状态 | 全部不同 |
| 高度图 scan vs 直接 `mj_ray` | mean 3.2e-3；1.4% 射线误差 >5 cm |

**想象噪声图 vs 真实带噪图的统计**（同一批干净传感器测量；mean / std / 占比 =1.0 / 占比 =0）：
干净 0.7284 / 0.1979 / 0.2019 / 0.0000；真实带噪（env 观测）0.7008 / 0.2117 / 0.1785 / 0.0010；想象带噪（由干净图经 `noise_predicted`）0.6927 / 0.2023 / 0.1659 / 0.0001。
均值/方差/饱和占比接近；想象图里“零像素”（contour/artifact 抹零）比真实少约 10 倍（0.0001 vs 0.0010）：从 18x32 预测图还原到原始分辨率再加噪，不会复现全部边缘。

## Smoke run（256 envs，300 it，wm_only_iters=100，buffer 200，wm batch 1024，pi batch 256，bf16，相机噪声开启）

- 深度（L1，[0,1] 量纲）：预测状态上 0.035→0.0285，teacher forcing 0.041→0.035；所有 loss 有限。
- 开环深度误差随步数（真实窗口，anchor = 带噪真实图，目标 = 干净图；“copy” = 直接复制 anchor）：
  step1 0.024 vs copy 0.037；step8 0.032 vs 0.086；step16 0.038 vs 0.103；step32 0.061 vs 0.132（teacher-forced 真实状态：0.024/0.030/0.034/0.046）。
  即渲染头优于复制 anchor，但误差随步数增长（世界模型状态误差累积 + anchor 视野外区域靠“想象”）。
- 集成分歧（逐像素 std 均值）约 0.010 且几乎不随步数增长，远小于真实误差：成员高度相关/一致欠拟合，所以分歧不是误差的好估计（诚实记录：集成目前主要提供多样性，作用有限，未做消融）。
- 策略（仅 BPTT）：想象 survival 0.16→0.53，imagined reward 1.3→2.1，仿真 episode length 29→49（仅 300 次迭代，不代表收敛）。
- 保存 / `--resume`（300→310）正常；stock `play.run_play`（Play 任务，headless 300 步 + `export_as_onnx`，导出 `0-depth_image.onnx`、`actor.onnx`、`policy_normalizer.npz`）无需额外参数即可加载。
- 显存 / 速度（默认超参：wm batch 2048、depth_windows 512、depth_pairs 512、pi batch 1024、buffer 1000）：1024 envs 峰值 13.4 GB、约 0.2-0.3 it/s；2048 envs 峰值 15.9 GB、约 0.2-0.3 it/s（每次迭代含 10 个世界模型 batch 与一次 32 步想象，想象含深度渲染与卷积编码器，主导耗时）。默认 `--num-envs 1024`（与任务一致，余量充足）。

## 失败（continue）头与失败加权采样（针对想象过于乐观）

诊断（kneelClimbStep1）显示世界模型在“跪姿起身”阶段过于乐观（想象中 27% 以内终止 vs 真实），因此在世界模型训练内加入：
- **`FailureHead`**（Dreamer 的 continue predictor）：输入 = critic 的特权特征（不变量状态、与参考的误差、ξ、goal、地形 scan）——终止由结果状态决定，所以不额外输入动作；
  输出 P(进入该状态的 transition 是**真实终止**)，标签来自 buffer 的 `terminated`（仅 `reset_terminated`，不含超时/数据集结束）。
  在 k=8 窗口的**真实状态**和**世界模型预测状态（detach）**上训练（`fail_windows` 个窗口，独立的优化器 `opt_fail`）。未采用“m 步内终止”的第二输出（每步一个标签已和解析终止逐步对齐，m 步标签与逐步组合会重复计数）。
- **想象中**：`d = 1 − (1 − d_analytic)(1 − p_fail)`，`p_fail` 上限 `fail_clip=0.95`，梯度保留（`--agent.no-fail-head` 关闭）。头从创建起就训练，创建后 `fail_head_warmup_iters`（默认 300）次迭代内不影响策略。
- **失败加权窗口采样**：每个 WM batch 的 `wm_failure_frac`（默认 0.25）窗口取自“窗口起点后 `wm_failure_horizon`=16 步内有真实终止”的窗口，其余均匀；动力学、深度头、失败头共用该 batch。
  为保持校准，失败头的 BCE 按重要性权重（均匀密度/混合密度）还原采样偏差（`fail_pos_weight` 默认 1）。深度头的 teacher-forcing 间隔采样仍为均匀。
- **留出**：`wm_holdout_envs`（默认 10%）的 env 不参与任何世界模型/头的训练，只用于校准诊断。
- **日志**：`fail/{true_states,wm_states}/{pred_rate,true_rate,bin_*,auroc}`、`fail/motion_<k>/*`（留出窗口上预测 vs 真实终止率），
  `termcheck/*`（每 5 个 log 间隔：同一批留出真实起点上，想象中硬终止 / 含失败头的终止概率 vs buffer 里真实终止的比例，及逐动作）。
- **从旧 checkpoint 续训**：旧的 `supermimic_state_<it>.pt` 没有 `fail_head / opt_fail / fail_head_start_it`。`runner.load` 对已有模块仍严格加载（缺 key 直接报错）；
  只有这三个新 key 恰好全缺且传 `--resume-init-new-heads` 时才新建失败头并打印提示，否则抛 `KeyError`。

## 性能与显存优化（数值等价，均有对拍脚本）

| 项 | 改动 | 对拍 | 效果（64 envs，pi_batch 256，GPU 与训练进程共享，取 min/median） |
|---|---|---|---|
| 相机噪声 `stereo_too_close_noise`（真实相机 + 想象共用） | 只处理有 <close_threshold 像素的图像；原循环里 5 次写同一批像素、只有最后一次有效，改为只抽最后一次；布尔索引赋值→`torch.where`；`env_onehot`→`F.one_hot` | contour / one_hot 逐位相同；stereo 每图分布相同（值直方图差 ≤0.012，RNG 消耗不同）`tests/supermimic/bench_noise.py` | 1024 张图：35–63 ms → 1–2.5 ms |
| 想象中的主机同步 | 深度头集成成员分组一次性（原每步每成员 `nonzero`）；`DepthHead` 每次调用新建常量张量→buffer；`sample_windows` 无同步（`multinomial`） | 集成/常量逐位相同；采样统计等价 `bench_sampling.py` | 每次 `policy_update` 的 host sync：962 → ~100；wall：median 1290–1390 → ~500 ms |
| 想象步 / reward / critic 特征 `torch.compile`（`compile_policy`，默认开） | 层级批处理 FK（`kinematics.py`，按树深度，10 层而非 31 个 body）使图足够小可编译；想象步（obs 装配+actor+WM+scan+proprio 历史）、`reward_term`、`critic_input` 编译 | 固定输入、关闭所有随机噪声：loss 差 0，actor 梯度相对差 1.4e-7（`check_policy_update_parity.py`）；FK 前向 ≤4e-6（数值尺度 ≤25，相对 1.5e-7）、梯度相对 1.4e-7（`bench_fk.py`） | kernel launch 50.7k → 27k；policy_update min 400 → 328 ms；首次编译约 1.5 分钟 |
| 相机噪声分段编译（`DepthPipeline.noise_predicted`） | 连续的纯阶段（contour / range / blur / normalize / crop）融合并编译，含数据依赖形状的阶段（artifact / stereo）保持 eager | 确定性阶段差 ≤3.6e-7；完整管线 40 批统计一致（`bench_depth_noise.py`） | 10.2 → 7.2 ms / 1024 张 |
| 想象步梯度检查点（`checkpoint_imagination`，默认关） | 每个想象步在反向时重算（RNG 状态重放） | 梯度相对差 2e-7（同上脚本） | policy_update 峰值显存 **−80%**（pi_batch 256：2.5 → 0.5 GiB；pi_batch 1024：1.35 GiB，估计 eager ≈10 GiB），耗时 +10% |
| 策略网络 bf16 autocast（`policy_autocast`，默认关） | 仅想象中 | 非精确：actor 梯度相对差 2e-3（loss 差 8e-7）；采集 / play / ONNX 仍为 fp32 | 显存 −20%，速度基本不变（已有 checkpoint 时不建议） |

未采纳（实测无收益）：`channels_last` / 编译深度头（36 ms → 37 / 35 ms）、`cudnn.benchmark`（更慢）、深度头梯度检查点（峰值几乎不变，+25% 时间）、buffer 深度图 uint8（量化误差 ~1e-3 ≈ 深度损失量级的 7%，保持 fp16）。
`wm_update`（每次迭代 10 次，占迭代时间的大头）受限于深度头集成的卷积计算（≈90% GPU 时间，算力受限），无法在不改算法/超参的前提下加速。
`depth_gap` / `fail_calibration` / `termination_check` 每 50 / 250 次迭代一次，约 50–275 ms，摊销 <3 ms/it，未处理。
基准：`tests/supermimic/bench_runner.py`（`--metrics` / `--profile`）、`bench_fk.py`、`bench_noise.py`、`bench_depth_noise.py`、`bench_sampling.py`。

## 评估（`instinct-supermimic-eval`，`scripts/supermimic/evaluate.py`）

```bash
instinct-supermimic-eval --checkpoint <run>/model_5000.pt --motion-folder ~/Xyk/Datasets/20251116_50cm_kneeClimbStep1 [--num-envs 64] [--video]
instinct-supermimic-eval --run-dir <run> --every 5 --watch --motion-folder ...     # 轮询新 checkpoint，结果追加
```
Play 设定（无 DR、无 push、从 t=0 开始、`independent` 采样）+ 同样的 `--motion-folder` 与相机噪声（`--no-depth-noise` 关闭）；用 stock `OnPolicyRunner.load` + `get_inference_policy` 加载。
与 Play 的有意差异：恢复 Play 去掉的三项跟踪终止（否则 success 无意义）；`episode_length_s` 提到 30 s，使 `dataset_exhausted` 成为成功的终止；地形生成器改 curriculum 模式、每个 terrain 一行（保证每条动作都有可放置的地形；训练用的随机 3x3 网格可能缺 terrain）。
每个 env 只跑一个 episode，之后忽略；动作按 `env % K` 显式分配（每条 ≥3 个 episode，`num_envs ≥ 3K`），通过对应 motion 的 one-hot 采样权重 + `env.reset(env_ids)` 走 env 自己的 reset 路径（唯一的内部访问是 buffer 的 `_motion_weights`）。
指标（逐动作和总体）：success rate（`dataset_exhausted` 触发且同步无失败终止）、完成比例、终止原因计数、base pos / 相对世界系 link pos / joint pos 的逐步平均误差、归一化回报（逐步任务回报 / 正权重和×dt）。
输出 `<run_dir>/eval/results.csv`、`eval_<it>.json`，`--video` 时 `<run_dir>/eval/videos/*.mp4`（复用 mjlab `VideoRecorder`，单 env 渲染环境，EGL 无头）。

## 已知限制

- 解析 head 读取最新的 post-step 状态（`qpos/qvel`），env 的 manager 看到的是滞后一个 substep 的 `xpos/cvel`，奖励/终止在边界附近有小差别。
- 不建模参考帧耗尽的 validity：想象起点保证 `f+H` 不越过 clip 末帧。单层高度图无法表示悬空结构。
- 学习的深度渲染误差会在 32 步想象 horizon 内累积（见上表）；anchor 视野之外的区域是被“想象”出来的；相机在 60 Hz 刷新，图像时延被渲染头吸收而不是显式建模。
- 想象噪声图的零像素比例偏低（见上）。
- **任务自身的 bug（未修复）**：`height_scanner` 只包含 geom group 0，而地形 geom 在 group 2，任务 critic 的 `height_scan` 恒为 5.0；仅有 ray-cast 传感器时 warp 的 BVH 对空 group 会 segfault（有相机时不会）。
  训练脚本因此保留传感器和 critic 项（删掉会改变 stock critic 输入维度，导致 Play `load` 失败），世界模型/critic 使用自己的解析高度图 scan。
- 无 provenance 记录（不提供 `add_git_repo_to_log`）。
