# FIX-A fix-learning: council findings on the learning side (2026-11-09)

Branch `claude/fix-learning` from `1d5c6ed`. Scope: `learning/` only (env, soil, gym_env,
student / risk collection loops, train, evaluate, configs, tests, README, new `scripts/`).
Closes B1, W1, W2, M1, M2 (env side), M9, m4, m5, m6 of
`docs/engineering-log/council/2026-11-09-integration-review.md`.

## What I built

- **B1 training distribution** (`env.py`). The loose patch is a degraded copy of the episode's
  main soil (k_phi x LogU(0.4, 0.9), a_s x U(1, 2), K x U(1, 1.5)), on with p = 0.5, centre
  y ~ U(-0.9, 0.9). Mixture rejection sampling: with p = `p_dp_restricted` the whole episode
  physics (soil, beta, g, mass, m_fH, m_fR) is redrawn until the rover-level analytic peak
  steady DP/W at static load (`env.peak_dp`, new `soil.steady_drawbar_curve` /
  `peak_drawbar_coefficient`) reaches the threshold. Linear curriculum 0.08 -> 0.0 over the
  first third of each PPO stage (`EnvConfig.dp_threshold`, `dp_curriculum_steps`, set by
  `train.ppo_stage` from `dp_curriculum_fraction`); student and critic use the end value.
  `spec_eval_config()` = SPEC section 6 as written (evaluation distribution); `evaluate.py`
  uses it as the base of the cells and adds a `spec` cell. `train.B0CompletableMonitor` logs
  `dp_threshold` and `b0_completable` (shadow B0 rollouts) every PPO iteration;
  `summary.json` compares the mean with the pre-registered band 40-60 % (in both configs).
- **W2**: dig rate per episode LogU(0.03, 0.3); `scripts/dig_rate_belly_out.py`.
- **W1**: energy test (`tests/test_physics_energy.py`: per substep, kinetic energy gain <=
  motor work at the new wheel speeds) and the one-substep coefficient lag documented in the
  module docstring and at the solve.
- **M1**: `scripts/sensitivity_a_s.py` (CSV + markdown); table in `learning/README.md` with
  "the sim is uninformative on H1 by construction; the hardware kill test is the evidence".
- **M2 (env side)**: the env's terminal "hard_stuck" is now orion_safety's FLAG_STUCK (3 s) +
  ENTRAPPED (20 s window, < 0.1 m net displacement) rule, thresholds copied from
  `orion_safety/config/safety.yaml`, inputs = the command sent, the estimated body speed /
  confidence / low-passed side slip, and the (true) pose. `info['entrapment_time']` dates
  ENTRAPPED at the stuck onset and belly-out when it happens. Critic labels count an
  entrapment only if its date is within T_h = 10 s of the state. Belly-out stays a sim
  terminal, documented as unobservable on hardware.
- **M9**: the DAgger loop (`student.py`) and the critic's state-visiting loop (`risk.py`) send
  actions through `baselines.CriticGate` (stuck -> escape) with `SurrogateRisk` driving the
  probe / retreat / escape bands; mode fractions logged (`frac_*`, `visit_frac_*`).
- **m4** IMU roll / pitch bias U(-2, 2) deg per episode (`measured_tilt`); **m5** rear wheels
  k_phi x U(1, 1.5) per episode (lower z0 follows; privileged descriptors now per axle);
  **m6** sinkage-proxy normaliser I_ref x U(0.7, 1.3) per episode.

## Decisions (and options rejected)

- **`p_dp_restricted` = 0.4, not the council's 0.7.** Measured B0 success (800 episodes):
  0.7 gives 69 / 74 / 78 % at thresholds 0.0 / 0.03 / 0.08, above the pre-registered 40-60 %
  band; 0.4 gives 46 / 50 / 52 %. The band is the pre-registered objective, 0.7 was the
  council's estimate of how to reach it. Recorded in README and the commit.
- **Rover-level peak DP/W** = max over a common slip of the mean of the two side curves at
  static load W = m g / 4, ignoring patch, multi-pass and load transfer. Rejected: min over
  sides (too strict for a skid-steer that can pull on one side).
- **Restricted episodes redraw the whole physics**, not only the soil: a fixed (g, beta, m)
  combination can have no passing soil, which would loop forever. Hard cap 200 rounds, then a
  RuntimeError naming the threshold.
- **ENTRAPPED progress uses the true pose** (camera on the rover); the monitor's odometer
  fallback during camera loss is not modelled.
- **Label branches: escape manoeuvre b, then B0** (was: E back to back). The rover's stuck
  rule only fires while driving, so E-only branches could never be ENTRAPPED. Branches have
  no time limit and stop once the label is decided (cap T_h + 20 s + 15 s slack).
- **SurrogateRisk instead of the critic in the collection loops.** The critic needs z-hat from
  the student, so it cannot gate the student's own training; random piecewise-constant rho
  (30 % of draws in [0.3, 1], hold 0.5-3 s) exposes the adapter and critic to all gate modes.
  Rejected: a second DAgger pass with the trained critic (doubles the student cost; possible
  later).
- **"No event" is -1.0 (`NO_TIME`), not NaN**: gymnasium's env checker compares infos with
  `==`, and NaN != NaN.
- Dig rate, multi-pass, IMU bias and I_ref stay random in the sandbox cells (they are
  model / calibration uncertainty, not cell properties).

## Results

- B0 on the training sampler: 5.5 % success / 86 % belly-out before (256 episodes, old
  defaults) -> 46 % / 43 % at the end of the curriculum (800 episodes). SPEC evaluation
  distribution unchanged: 9 % / 83 %.
- Critic label positive rate on B0-visited training states (K = 4, T_h = 10 s; old package
  vs new, `label_rate.py` in my scratchpad): mean label 0.72 -> 0.42, share of states with
  any positive branch 91 % -> 54 %. Label cost x3.5 (branches run until decided).
- Smoke (`configs/smoke.yaml`, 37 s, was 23 s): teacher b0_completable 0.53 / 0.50 / 0.51
  (in band), b2 0.44 / 0.37 / 0.36 (35 shadow episodes, flagged out of band); belly-out of the
  untrained teacher 90 % -> 50 %, B2 88 % -> 75 %; critic label_mean_train 0.70 -> 0.17
  (n = 32 / 24, too small to mean much).
- Sensitivity (README): heavy-loose flips from 100 % success (a_s <= 0.5, -14 to -18 %
  speed) to 0-4 % success at a_s >= 1.0 (0 % already at 0.75 unless the dig rate is 0.03).
  The dig rate barely changes the outcome. Dig-rate prior: full-throttle belly-out within
  1 s for 14 % of SPEC-distribution rovers (median 2.0 s), B0 2 % (median 4.8 s).
- Golden test: the env's FLAG_STUCK / ENTRAPPED equal `orion_safety.monitor.SafetyMonitor`
  step by step on the same inputs (random training episodes with B0, and stalled rovers).

## Mistakes made and fixed

- First progress-restart test expected the new stuck episode one step late. Root cause: I
  forgot that the monitor ends and restarts a stuck episode in the same tick while the
  window still contains the jump. Lesson: write the expectation from the reference code,
  not from intuition; the golden test against the real monitor now pins it.
- Curriculum threshold was logged after the iteration (always the next value; smoke showed
  0.0 at iteration 0). Fixed to log the value in effect while the rollout was collected.
- NaN as "no stuck onset" broke the gymnasium determinism check (see decisions).

## Negative checks (each test went red, code restored)

- Motor torque x1.5 in the wheel rows of the solve -> energy test red (gain exceeded motor
  work by 3.7 mJ).
- `p_dp_restricted` 0.7 -> B0 band test red.
- ENTRAPPED dated at the HALT instead of the onset -> label-horizon test red.
- Stuck hold counted like `baselines.StuckDetector` (n * dt instead of (n - 1) * dt) ->
  golden monitor test red (both cases).

## Only verifiable on the real rover / open questions

- **The speed stuck rule practically never fires in sim.** With speed-fix noise 0.03 m/s (1
  sigma) the estimate stays under the 0.03 m/s threshold for only ~73 % of the steps of a
  rover that is truly at rest, so 3 s continuous never happens; stuck B0 rovers in the loose
  cells end as 60 s timeouts, never ENTRAPPED (sensitivity table). Also, backward escapes
  move > 0.1 m and so reset the ENTRAPPED window. If the rover's estimator is as noisy, the
  kill-test outcome "ENTRAPPED" will be rare there too. Needs M7 (FIX-B) and a protocol call.
- Sim B0 / CriticGate escape on `baselines.StuckDetector` (obs-based, `cmd_v > 0.1`, max
  slip, n * dt hold), the rover on orion_safety FLAG_STUCK (|cmd v|, driven-direction slip,
  (n - 1) * dt hold). Requested from FIX-B: `B0Controller.act(obs, stuck=None)` and
  `CriticGate.gate(obs, action, rho, stuck=None)` taking an external flag, so sim can pass
  `info['soft_stuck']` exactly as the rover passes FLAG_STUCK.
- `synth_bag.py` (not mine) writes the true tilt to /imu/data; it does not add the new IMU
  bias.
- Whether the derived patch factors and the 40-60 % band produce a teacher that both drives
  and avoids traps can only be seen in a full run (`b0_completable`, teacher success).

## Test commands and results

- `PYTHONPATH=$PWD/learning python3 -m pytest -q learning/tests`: 75 passed (~65 s).
- `python3 -m orion_learning.train --config configs/smoke.yaml`: finished in 37 s.
- `flake8` (learning/.flake8) on orion_learning, tests, scripts: clean.
- `scripts/verify.sh unit orion_safety orion_policy orion_experiments`: 181 tests, 4 errors,
  0 failures; the 4 errors are `orion_experiments/test/test_analysis_bags.py` setup:
  `bag_fixture.py` builds PolicyStatus without the `critic_gating` field added in c5ab004. No
  robot_ws file is touched by this branch, and robot_ws only imports `orion_learning.obs_spec`.
