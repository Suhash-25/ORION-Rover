# orion_learning

Soil simulator, training, ONNX export and replay-bag generation for the ORION rover's
adaptive traction controller. Pure Python (NumPy + PyTorch), no ROS needed.
Contract: `docs/SPEC.md` sections 1, 3 and 6, and the ONNX artefact contract below.

## Install and test

```bash
pip install -e "learning[train,bags,test]"     # or: pip install -r learning/requirements.txt
cd learning
python -m pytest -q                              # ~30 s, includes the smoke training run
python -m orion_learning.train --config configs/smoke.yaml   # ~25 s, what CI runs
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

## The environment in one paragraph

20 Hz control, 4 physics substeps. Each wheel has Bekker static sinkage, slip sinkage (dug at
0.1 m per metre of slipped rim travel, relaxing at 0.3 m per metre travelled), Janosi-Hanamoto
thrust, compaction resistance, lateral friction, a PI speed loop limited to 3 N.m, and load
transfer. The 7 velocity states per rover are integrated with a secant-implicit Euler step
(explicit Euler is unstable at this step size). Randomised per episode: soil (per side with
p = 0.5, plus one loose patch), mass 12-30 kg, g 1.62-9.81 with k_phi_eff = k_phi (g/9.81)^beta,
thrust / resistance multipliers LogU(0.7, 1.4), 1-4 steps latency, torque fault, sensor noise,
speed-fix dropouts and accelerometer bias walk. The policy sees only `obs_spec` observations
built from noisy sensors through the same estimator maths the rover uses. Episodes end at the
goal line (1.5 m), belly-out (z >= 8 cm), hard-stuck (10 s), leaving the 2 x 2 m fence, or 60 s.

## Training stages

`python -m orion_learning.train --config configs/full.yaml [--stages ...] [--run-dir runs/X]`

1. **teacher**: PPO; the policy gets `z = mu(privileged)` where `privileged` is the 10-D
   functional traction vector (per side: DP/W at s = 0.3, slip at DP = 0.1 W, z0/r, Rc/W;
   plus m/14 and g/9.81), computed by a steady-state solve of the soil model. Value function
   is asymmetric (sees privileged), training-only.
2. **student**: TCN adapter over the last 40 steps of [obs, previous action] -> z-hat, trained
   DAgger-style: the student acts (teacher policy on z-hat; teacher latent with probability
   beta, decaying 1 -> 0), visited histories are labelled with the teacher's z, MSE.
3. **critic**: states visited by the student are cloned; from each, K = 4 branches run the
   escape library E for 10 s (branch b starts at manoeuvre b, own noise stream); label =
   fraction ending in belly-out or hard-stuck. BCE on (obs, z-hat); temperature fitted on one
   half of held-out seeds, ECE / Brier reported on the other half (`summary.json`).
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
| critic | ~60 k states x 4 x 200 = 50 M branch steps | ~1-1.5 h |
| evaluate | 5 controllers x 8 cells x 25 trials | ~10 min |

On a 16-vCPU cloud CPU box (e.g. c7i.4xlarge, about USD 0.7/h): run `--stages teacher` and
`--stages b2` as two processes into the same `--run-dir`, then
`--stages student critic export evaluate`. Expect ~9-10 h wall, ~USD 7. No GPU needed (the
networks are small; the env is the bottleneck).

Before trusting a full run, check in `metrics.csv` that teacher success rises and belly-out
falls, that `onpolicy_latent_mse` of the student keeps falling with beta = 0, and that the
critic's `ece` and `brier` beat `brier_base_rate`. If the student can't beat B2 in sim, the
adaptation claim is dropped before hardware (SPEC F6).

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
