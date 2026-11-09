# orion_learning

Soil simulator, training, ONNX export and replay-bag generation for the ORION rover's
adaptive traction controller. Pure Python (NumPy + PyTorch), no ROS needed.
Contract: `docs/SPEC.md` sections 1, 3 and 6, and the ONNX artefact contract below.

## Install and test

```bash
pip install -e "learning[train,bags,test]"     # or: pip install -r learning/requirements.txt
cd learning
python -m pytest -q                              # ~60 s, includes the smoke training run
python -m orion_learning.train --config configs/smoke.yaml   # ~40 s, what CI runs
```

`scripts/verify.sh learning` does both.

## Package map

| module | what it is |
| --- | --- |
| `obs_spec.py` | the only definition of the observation / action layout (unchanged; helpers now also take batches) |
| `soil.py` | vectorised Bekker / Janosi-Hanamoto wheel-soil model, gravity coupling, traction descriptors; scalar reference for tests |
| `estimator.py` | body-speed Kalman filter, confidence, effort model, sinkage proxy, low-pass. Shared with the rover runtime (orion_terrain should import it) |
| `env.py` | N-rover vectorised environment, randomisation, sandbox cells, `clone_state`/`restore_state`/`tile_state` |
| `gym_env.py` | Gymnasium single-rover wrapper (passes `gymnasium.utils.env_checker`) |
| `baselines.py` | B0, stuck rule, escape library E, critic gating. Pure NumPy, reusable by orion_policy |
| `history.py` | the [40, 22] adapter history buffer (same layout as on the rover) |
| `nets.py`, `ppo.py`, `student.py`, `risk.py` | networks, PPO, DAgger student, entrapment-risk critic |
| `train.py` | entry point, all stages, run directories |
| `export_onnx.py` | ONNX artefacts + manifest, parity-checked against PyTorch |
| `evaluate.py` | controllers x sandbox cells x {9.81, 1.62} m/s^2 -> CSV |
| `synth_bag.py` | rosbag2 + truth.csv from a rollout, for the ROS packages' replay tests |
| `scripts/sensitivity_a_s.py` | B0 per cell vs the guessed loose-soil a_s and dig rate (table below) |
| `scripts/dig_rate_belly_out.py` | belly-out time distribution produced by the dig-rate prior (table below) |

## The environment in one paragraph

20 Hz control, 4 physics substeps. Each wheel has Bekker static sinkage, slip sinkage (dug at
a per-episode dig rate LogU(0.03, 0.3) m per metre of slipped rim travel, relaxing at 0.3 m
per metre travelled), Janosi-Hanamoto thrust, compaction resistance, lateral friction, a PI
speed loop limited to 3 N.m, and load transfer. The 7 velocity states per rover are integrated
with a secant-implicit Euler step (explicit Euler is unstable at this step size; the contact
coefficients lag the state by one 12.5 ms substep, and a test checks that the kinetic energy
gained never exceeds the motor work). Randomised per episode: soil (per side with p = 0.5,
plus a loose patch, see below), mass 12-30 kg, g 1.62-9.81 with k_phi_eff = k_phi
(g/9.81)^beta, thrust / resistance multipliers LogU(0.7, 1.4), rear-wheel multi-pass
compaction k_phi x U(1, 1.5), 1-4 steps latency, torque fault, sensor noise, IMU roll / pitch
bias U(-2, 2) deg, sinkage-proxy normaliser I_ref x U(0.7, 1.3), speed-fix dropouts and
accelerometer bias walk. The policy sees only `obs_spec` observations built from noisy sensors
through the same estimator maths the rover uses. Episodes end at the goal line (1.5 m),
belly-out (z >= 8 cm), ENTRAPPED, leaving the 2 x 2 m fence, or 60 s. ENTRAPPED (info key
`hard_stuck`) is the rover's own orion_safety rule, checked step by step against
`SafetyMonitor` in `tests/test_entrapment.py`: FLAG_STUCK after 3 s of |cmd v| > 0.1 m/s with
estimated speed < 0.03 m/s (trusted when the speed fix is < 0.5 s old or the confidence
c = exp(-t_since_fix / 3) * clip(1 - kf_std / 0.1, 0, 1) is > 0.3; `estimator.py`) or
driven-direction side slip > 0.8, then
ENTRAPPED when that stuck episode is 20 s old and the last 20 s moved < 0.1 m. Belly-out stays
a terminal in simulation but is **unobservable on hardware** (the rover does not measure
sinkage); there it shows up later as ENTRAPPED or a timeout.

## Training vs evaluation distribution (council B1)

The SPEC section 6 ranges put an impassable loose patch on every path and make most main soils
impassable too: B0 completes 9 % of those episodes (belly-out 83 %). Trained on that, the
reward-optimal teacher stops. The training distribution (`EnvConfig()`) therefore differs:

* the loose patch exists with p = 0.5, centre y ~ U(-0.9, 0.9), radius U(0.3, 0.6) m, and its
  soil is the episode's main soil degraded: k_phi x LogU(0.4, 0.9), a_s x U(1, 2),
  K x U(1, 1.5);
* mixture rejection sampling: with p = `p_dp_restricted` the episode physics (soil, beta, g,
  mass, multipliers) is redrawn until the rover-level analytic peak steady DP/W at static load
  reaches the threshold; the rest is unrestricted;
* curriculum: the PPO stages anneal the threshold 0.08 -> 0.0 over the first third of their
  iterations (`dp_curriculum_fraction`); student and critic use the end value.

**Pre-registered target: B0 completes 40-60 % of training episodes.** Every PPO iteration
logs `b0_completable` (shadow B0 rollouts on the current training distribution) and
`dp_threshold` in `metrics.csv`; `summary.json` reports the mean and whether it is in the band
(`b0_completable_band` in the configs). Measured B0 success (800 episodes each):

| sampler | B0 success | belly-out |
| --- | --- | --- |
| SPEC ranges (evaluation, `spec_eval_config()`) | 9 % | 83 % |
| new patch, no rejection | 27 % | 62 % |
| `p_dp_restricted` 0.7 (council proposal), threshold 0.0 / 0.03 / 0.08 | 69 / 74 / 78 % | 26 / 21 / 18 % |
| **`p_dp_restricted` 0.4 (default), threshold 0.0 / 0.03 / 0.08** | **46 / 50 / 52 %** | 43 / 40 / 39 % |

The council's 70 / 30 mixture lands above the band in this sim, so the default is 0.4.
Evaluation keeps the SPEC distribution: `evaluate.py` uses `spec_eval_config()` as the base
of the sandbox cells and adds a `spec` cell (the full SPEC randomisation at fixed g).

## Training stages

`python -m orion_learning.train --config configs/full.yaml [--stages ...] [--run-dir runs/X]`

1. **teacher**: PPO; the policy gets `z = mu(privileged)` where `privileged` is the 10-D
   functional traction vector (per side: DP/W at s = 0.3, slip at DP = 0.1 W, z0/r, Rc/W;
   plus m/14 and g/9.81), computed by a steady-state solve of the soil model. Value function
   is asymmetric (sees privileged), training-only.
2. **student**: TCN adapter over the last 40 steps of [obs, previous action] -> z-hat, trained
   DAgger-style: the student acts (teacher policy on z-hat; teacher latent with probability
   beta, decaying 1 -> 0), visited histories are labelled with the teacher's z, MSE. The
   actions go through `baselines.CriticGate` as on the rover (stuck -> escape), driven by a
   surrogate risk (no critic exists yet) that spends ~30 % of the time in the probe, retreat
   and escape bands, so the adapter sees gated histories (council M9; `frac_*` metrics).
3. **critic**: states visited by the gated student (as above) are cloned; from each, K = 4
   branches run escape manoeuvre b at once, then B0, each with its own noise stream and no
   time limit; label = fraction of branches with an entrapment *dated within 10 s* of the
   state. A belly-out is dated when it happens, an ENTRAPPED verdict at the onset of its stuck
   episode (FLAG_STUCK), the same anchor the rover analysis uses (council M2), so a branch
   runs up to 10 + 20 + 15 s. BCE on (obs, z-hat); temperature fitted on one half of held-out
   seeds, ECE / Brier reported on the other half (`summary.json`).
4. **b2**: same PPO without the latent (robust non-adaptive baseline).
5. **export**: ONNX for `student` (adapter + policy + critic) and `B2` (policy).
6. **evaluate**: B0, B2, student, student + critic gating over the 4 cells at g = 9.81 and 1.62.

Each run writes `runs/<timestamp>/`: `config.yaml`, `run_info.json` (seed, git SHA, stages),
`metrics.csv` (stage, iteration, metric, value, wall time), checkpoints, `export/`,
`eval_summary.csv`, `eval_episodes.csv`, `summary.json`.

## Training for real (configs/full.yaml)

Measured on this sandbox (one shared core): the full-size teacher runs at ~8 k env steps/s
including inference and PPO updates (512 envs x 256 steps = 16 s per iteration). The NumPy env
is single-threaded, so throughput scales by running stages as separate processes.

| stage | budget | single process, 1 fast core |
| --- | --- | --- |
| teacher | 1500 it x 131 k = 200 M steps | ~6-7 h |
| b2 | same | ~6-7 h (run in parallel with teacher) |
| student | 20 M steps + adapter fits | ~1 h |
| critic | ~60 k states x 4 branches x ~350 steps (onset-dated labels) = ~85 M branch steps | ~2-3 h |
| evaluate | 5 controllers x 8 cells x 25 trials | ~10 min |

On a 16-vCPU cloud CPU box (e.g. c7i.4xlarge, about USD 0.7/h): run `--stages teacher` and
`--stages b2` as two processes into the same `--run-dir`, then
`--stages student critic export evaluate`. Expect ~9-10 h wall, ~USD 7. No GPU needed (the
networks are small; the env is the bottleneck).

Before trusting a full run, check in `metrics.csv` that `b0_completable` stays in the
pre-registered 40-60 % band (`summary.json`: `b0_completable_teacher.in_band`), that teacher
success rises and belly-out falls, that `onpolicy_latent_mse` of the student keeps falling with beta = 0, and that the
critic's `ece` and `brier` beat `brier_base_rate`. If the student can't beat B2 in sim, the
adaptation claim is dropped before hardware (SPEC F6).

## What the sim can and cannot say about H1 (council M1, W2)

**The sim is uninformative on H1 by construction; the hardware kill test is the evidence.**
Whether B0 slows down or gets trapped on heavy-loose is decided by two guessed parameters of
the sinkage model, not by anything measured. `scripts/sensitivity_a_s.py` (25 trials per
cell, g = 9.81, SPEC evaluation settings):

| dig rate | loose a_s | light-packed | light-loose | heavy-packed | heavy-loose | heavy-loose vs light-packed speed |
| --- | --- | --- | --- | --- | --- | --- |
| 0.03 | 0.3 | 100 / 0 / 0.254 | 100 / 0 / 0.233 | 100 / 0 / 0.252 | 100 / 0 / 0.217 | -14 % |
| 0.03 | 0.5 | 100 / 0 / 0.254 | 100 / 0 / 0.228 | 100 / 0 / 0.252 | 100 / 0 / 0.210 | -17 % |
| 0.03 | 0.75 | 100 / 0 / 0.254 | 100 / 0 / 0.219 | 100 / 0 / 0.252 | 100 / 0 / 0.038 | -85 % |
| 0.03 | 1 | 100 / 0 / 0.254 | 0 / 0 / 0.022 | 100 / 0 / 0.252 | 72 / 0 / 0.028 | -89 % |
| 0.03 | 1.5 | 100 / 0 / 0.254 | 0 / 0 / 0.022 | 100 / 0 / 0.252 | 4 / 0 / 0.024 | -91 % |
| 0.03 | 2 | 100 / 0 / 0.254 | 0 / 0 / 0.022 | 100 / 0 / 0.252 | 0 / 0 / 0.023 | -91 % |
| 0.1 | 0.3 | 100 / 0 / 0.254 | 100 / 0 / 0.233 | 100 / 0 / 0.252 | 100 / 0 / 0.216 | -15 % |
| 0.1 | 0.5 | 100 / 0 / 0.254 | 100 / 0 / 0.228 | 100 / 0 / 0.252 | 100 / 0 / 0.208 | -18 % |
| 0.1 | 0.75 | 100 / 0 / 0.254 | 100 / 0 / 0.216 | 100 / 0 / 0.252 | 0 / 0 / 0.020 | -92 % |
| 0.1 | 1 | 100 / 0 / 0.254 | 0 / 0 / 0.009 | 100 / 0 / 0.252 | 0 / 0 / 0.015 | -94 % |
| 0.1 | 1.5 | 100 / 0 / 0.254 | 0 / 0 / 0.009 | 100 / 0 / 0.252 | 0 / 0 / 0.012 | -95 % |
| 0.1 | 2 | 100 / 0 / 0.254 | 0 / 0 / 0.008 | 100 / 0 / 0.252 | 0 / 0 / 0.011 | -96 % |
| 0.3 | 0.3 | 100 / 0 / 0.254 | 100 / 0 / 0.232 | 100 / 0 / 0.252 | 100 / 0 / 0.216 | -15 % |
| 0.3 | 0.5 | 100 / 0 / 0.254 | 100 / 0 / 0.227 | 100 / 0 / 0.252 | 100 / 0 / 0.207 | -18 % |
| 0.3 | 0.75 | 100 / 0 / 0.254 | 100 / 0 / 0.215 | 100 / 0 / 0.252 | 0 / 0 / 0.018 | -93 % |
| 0.3 | 1 | 100 / 0 / 0.254 | 0 / 0 / 0.006 | 100 / 0 / 0.252 | 0 / 0 / 0.012 | -95 % |
| 0.3 | 1.5 | 100 / 0 / 0.254 | 0 / 0 / 0.005 | 100 / 0 / 0.252 | 0 / 0 / 0.009 | -96 % |
| 0.3 | 2 | 100 / 0 / 0.254 | 0 / 0 / 0.004 | 100 / 0 / 0.252 | 0 / 0 / 0.007 | -97 % |

Cells: B0 success % / entrapped % (belly-out or ENTRAPPED) / mean effective speed (m/s); the remaining trials are 60 s timeouts.

Below loose a_s = 0.75 every cell is traversable and heavy-loose is only 14-18 % slower; from
a_s = 0.75-1.0 the loose cells flip to 0 % success. The dig rate barely matters for the
outcome. Do not quote a sim speed drop as support for H1. Note also the entrapped column: the
stuck B0 rovers are 60 s timeouts, never ENTRAPPED, because with 0.03 m/s speed-fix noise
the estimated speed stays below the 0.03 m/s stuck threshold for only ~73 % of the steps, so
the 3 s speed rule almost never holds, and the backward escapes count as > 0.1 m of progress.
If the rover's estimator is that noisy, the same will happen on hardware.

The dig rate itself (W2) is a model-form guess (SPEC: instant slip sinkage, which buries 84 %
of randomised rovers within 1 s of a full-throttle start). Its prior LogU(0.03, 0.3) gives
these belly-out times on the SPEC evaluation distribution (`scripts/dig_rate_belly_out.py`,
1000 episodes):

**full throttle from rest**

| dig rate | episodes | belly-out <= 1 s | <= 3 s | <= 10 s | ever (60 s) | median time if it happens (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.03-0.05 | 215 | 0 % | 15 % | 80 % | 80 % | 3.8 |
| 0.05-0.08 | 191 | 0 % | 69 % | 89 % | 90 % | 2.6 |
| 0.08-0.13 | 220 | 1 % | 74 % | 80 % | 85 % | 1.8 |
| 0.13-0.2 | 190 | 16 % | 82 % | 89 % | 89 % | 1.2 |
| 0.2-0.3 | 184 | 60 % | 83 % | 86 % | 88 % | 1.0 |
| **prior LogU(0.03, 0.3)** | 1000 | 14 % | 64 % | 85 % | 86 % | 2.0 |

**B0**

| dig rate | episodes | belly-out <= 1 s | <= 3 s | <= 10 s | ever (60 s) | median time if it happens (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.03-0.05 | 215 | 0 % | 0 % | 22 % | 83 % | 12.2 |
| 0.05-0.08 | 191 | 0 % | 1 % | 76 % | 90 % | 6.7 |
| 0.08-0.13 | 220 | 0 % | 23 % | 80 % | 87 % | 4.2 |
| 0.13-0.2 | 190 | 1 % | 71 % | 89 % | 90 % | 2.1 |
| 0.2-0.3 | 184 | 8 % | 78 % | 88 % | 89 % | 1.5 |
| **prior LogU(0.03, 0.3)** | 1000 | 2 % | 33 % | 70 % | 88 % | 4.8 |

## Export and deploy

```bash
python -m orion_learning.export_onnx --run runs/<ts> --controller student   # or B2
```

writes `runs/<ts>/export/<controller>/{adapter,policy,critic}.onnx` + `manifest.json`:

* `adapter.onnx`: `history` float32 [1, 40, 22] -> `latent` [1, 8]
* `policy.onnx`: `obs` [1, 20], `latent` [1, 8] -> `action` [1, 2] in [-1, 1] (deterministic)
* `critic.onnx`: `obs`, `latent` -> `risk` [1, 1], calibrated (temperature inside)
* metadata_props on every file: obs_spec_hash, control_hz, history, latent_dim, obs_dim,
  action_dim, model_kind, controller, git_sha; manifest has sha256 of every file.

Observation normalisation statistics are frozen inside the graphs: feed the raw
`obs_spec.build_observation` output. The export fails unless onnxruntime matches PyTorch to
1e-5. Copy the directory to the rover and point orion_policy at it.

Runtime loop that matches training exactly (see `history.py`, `baselines.CriticGate`):
history starts as zeros at episode/run start; each 20 Hz step push the row
`[obs_t, a_(t-1)]` (a_(t-1) = the action sent on the previous step, [-1, 1], zeros at start),
run the adapter on the full buffer, run the policy on `(obs_t, latent)`, optionally gate with
the critic, send the action. `prev_cmd` inside the observation is the rate-limited twist
published on the previous step.

## Replay bags

```bash
python -m orion_learning.synth_bag --out /tmp/bag_hl --cell heavy-loose --seconds 20 [--storage mcap]
```

rosbag2 (version 9, sqlite3 or mcap; checked with Jazzy `ros2 bag info` and `rosbag2_py`):
`/joint_states` (80 Hz, N.m), `/imu/data` (80 Hz, gravity on +z, orientation),
`/diff_drive_controller/odom` (wheel odometry, not truth), `/orion/ground_truth_pose`
(20 Hz, frame `sandbox`, absent during dropouts), `/cmd_vel/policy` (TwistStamped), and
`truth.csv` with the true pose, body speed, per-wheel and per-side slip and sinkage per step.
