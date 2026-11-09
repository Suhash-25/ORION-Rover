# FIX-B (fix-shared): one estimator, one history, one escape library

Branch `claude/fix-shared` (from 1d5c6ed, merged with `claude/integration` at dd0c407).
Findings: B2, B3, M4, M5, M6, M7. New `OBS_SPEC_HASH` = **f28f136d0518c63d**.

## What was built

- **M5 / M7, shared estimator** (`learning/orion_learning/estimator.py`). This is the only copy of:
  - the body-speed Kalman filter (`kf_predict`, `kf_update`);
  - confidence `c = exp(-t_since_fix / 3) * clip(1 - kf_std / 0.1, 0, 1)`;
  - the exact low-pass `alpha = 1 - exp(-dt / tau)`, with a masked hold;
  - the per-wheel effort model along the direction of rotation (WP3 convention);
  - the informative mask (|w| >= 0.5 rad/s, |yaw| <= 0.2 rad/s; outside it the value is held);
  - the sinkage proxy and the least-squares fit.

  The env, `orion_terrain` (filters, speed, effort_model, fit_effort_model) and the env calibration all call it. The effort-model YAML no longer carries its own window: on load it refuses a file fitted on a different window. `TerrainState.time_since_fix` is the fix's age on the header-stamp clock, clamped to >= 0, and inf before the first fix.
- **B2, history** (`history.py`). Row k = [obs_k, executed a_(k-1)]. The action is taken from obs `prev_cmd`, so the row cannot disagree with its own observation. Rows are pushed before inference and zero-padded. The convention strings are in `obs_spec.SEMANTICS` and are part of the hash. `PolicyCore` uses the same class.
- **B3, escape library** (`escape_library.py`). E is a tuple of named manoeuvres made of `(duration_s, v, w)` segments, with `MAX_ESCAPE_ATTEMPTS = 3`. Its users:
  - `baselines` (B0, CriticGate);
  - the risk labels;
  - `orion_policy.sequences`, where `ESCAPE_LIBRARY is E`.

  B0 is a single implementation (`SlipGovernor` + `_EscapeArbiter`); `orion_policy.BaselineB0` is a thin adapter around it. The stuck rule (`StuckRule`, `stuck_conditions`) mirrors `orion_safety`'s rule:
  - |cmd| above the speed threshold;
  - slip measured in the driven direction;
  - the speed condition counts when confidence > 0.3 **or** the fix is < 0.5 s old;
  - first true sample counts as 0 s.

  The env's FLAG_STUCK, which FIX-A already ported from the monitor, now takes its conditions and thresholds from this rule. `B0Controller.act` and `CriticGate.gate` accept `stuck=`, so simulation can pass `info['soft_stuck']` the same way the rover passes FLAG_STUCK.
- **M6** (`orion_terrain/clock.py`). The node logs a throttled warning when |stamp - receipt| > 0.1 s for 1 s on the external pose, IMU or joint states.
- **M4** (`orion_terrain/test/test_golden_replay.py`). A synthetic bag (heavy-loose, seed 3, forced 0.5-2 s dropouts) is replayed through `TerrainEstimator` and `ObservationBuilder`. Results:
  - slip RMS vs `truth.csv` on settled-fix steps: 0.021 (limit 0.05);
  - proxy RMS vs the env: 0.006 (limit 0.05);
  - per-field observation tolerances (table in the test): measured `prev_cmd`, `roll_pitch` and `confidence` 0.0; `body_speed` 0.049; `slip_side` 0.057.
- `orion_policy/test/test_learning_parity.py`:
  - runtime B0 (PolicyCore + escape plans + attempt limit) equals learning `B0Controller` + `rate_limit` to within 1e-6 on a scripted trace covering governor cuts, all four first manoeuvres of E and the give-up;
  - the adapter input is bit-identical to learning `History`.

## Decisions (rejected options)

- **E[0] = the 7 s settle-reverse-wiggle-creep sequence.** The four short learning manoeuvres were rejected as the first escape. In env experiments (B0 stuck states re-run per manoeuvre) it was best or tied in every condition. Success after a stuck flag was:
  - train without patch: 0.85 for every option except forward-wiggle (1.00 = all failed);
  - original training distribution: 0.96 for all except forward-wiggle (1.00).
  The others are kept as E[1..4] in that order. Forward-wiggle comes last because it was worst everywhere.
- **The history action comes from obs `prev_cmd`** (rejected: a separate `prev_action` argument). The old form let the action column and the observation disagree. A `push(obs, prev_action)` call still works but ignores the argument and warns once. This shim is temporary.
- **The runtime keeps the rate-limit origin across policy <-> escape channel switches** (it used to reset to 0). The reset was a step change bigger than `MAX_LINEAR_ACCEL` and did not happen in training. IDLE still stops at once.
- **The env keeps FIX-A's counter-based monitor port.** Only its instantaneous conditions now come from `baselines`. My NaN-timer version was dropped in the merge.

## Mistakes

- The first golden-replay tolerances assumed the env's speed-fix noise matched the runtime's. It is about 3x larger, so the tolerances for `body_speed` and `slip_side` had to be widened.
- After merging integration, the replay broke three ways:
  - B0 now finishes in about 7 s;
  - the IMU bias was missing from the synthetic bag;
  - the `i_ref` factor was not applied.

  All three are fixed in `synth_bag.py` and the test.
- A refit of the effort-model coefficients is not seed-stable (c0 and c1 trade off inside the window). The test now compares predictions within 0.04 * ref.

## What is rover-only

- `ClockSkewMonitor`, effort-source provenance and the YAML.
- `PolicyCore` IDLE-on-give-up. In sim the rover stops via the arbiter instead: the runtime zeroes the command at once, the sim rate-limits it to 0.

## Open questions and requests to others

1. `student.py`, `risk.py`, `evaluate.py`: call `hist.push(obs)` (drop `prev_action`) and pass `stuck=info['soft_stuck']` to B0 / CriticGate.
   - With the soft-stuck flag passed in, the obs-based detector differs from the monitor in one case only: the first ~0.4 s of a dropout when confidence <= 0.3 (the observation has no `time_since_fix`).
   - `risk.py` uses `branches=4` while `len(E) = 5`.
2. **Speed stuck rule in sim (coordinator request 2).** On a stalled rover (light-packed, 256 rovers, 30 s):

   | Speed-fix noise | P(\|raw fix\| < 0.03) | P(\|KF speed\| < 0.03) | Rovers flagged within 30 s |
   |---|---|---|---|
   | 0.03 m/s (current) | 0.684 | 0.845 | 1.6 % |
   | 0.02 m/s | — | 0.965 | 95.7 % (median at 6.5 s) |
   | 0.015 m/s | — | — | 100 % (at 3.1 s) |

   - At the current noise the KF speed's own scatter is 0.021 m/s, so the KF only partly helps. A 0.25 s low-pass of the KF speed would give 0.999.
   - The runtime estimator (1 mm camera poses) has 0.0075 m/s RMS error on the replay, so the env's 0.03 m/s fix noise looks about 3x too pessimistic.
   - Suggestion for FIX-A: lower `noise_speed_fix` to about 0.015 m/s, or measure it on the rover. `orion_safety` thresholds are unchanged.
3. **Critic visits are escape-heavy.** With the 7 s E[0], CriticGate visits are about 51 % escape. `test_risk` now needs a 20 s rollout before retreat reaches 2 %.
4. `PolicyCore` gives up into IDLE with a zero command. `orion_safety`'s hard-stuck rule needs |cmd| > 0.1, so it will not HALT a rover that has already stopped. The trial then waits for the operator.
5. `docs/` still describe wheel-odometry confidence. Owner: docs.

## Tests (commands from the worktree root)

- `python3 rosrun.py learning python3 -m pytest -q tests` → **99 passed**. `orion_learning.__file__` is in the worktree.
- `cd robot_ws && colcon build --parallel-workers 2 && colcon test --packages-skip orion_sim_tests --ctest-args -E xmllint && colcon test-result` → **473 tests, 0 errors, 0 failures, 12 skipped**. Of these, `orion_policy` 71 and `orion_terrain` 77.
- **Negative checks** (each run on a deliberately broken copy, then reverted):
  - Euler low-pass gain `dt/tau` → `test_estimator::test_lowpass_is_exact...` and `orion_terrain test_slip::test_low_pass_time_constant` fail.
  - History pushed after inference → `test_learning_parity::test_history_adapter_input...` and `test_core::test_adapter_sees_current_obs...` fail.
  - Old channel-switch rate-limit reset → `test_learning_parity::test_runtime_b0...` fails at the first escape tick.
