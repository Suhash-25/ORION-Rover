# FIX-C: experiments, safety and docs fixes from the council review (2026-11-09)

Branch `claude/fix-experiments` from `1d5c6ed`. Findings closed: M2 (rover side), M3, M6
(runner side), M7 (safety side), M8, M1 (statement), m1, m2, m3, m7, m9, m11
(`docs/engineering-log/council/2026-11-09-integration-review.md`).

## What I built

- **M3, Phase-0 rule** (`orion_experiments/analysis.py`). `EFFECT_SIZE_GATE = False`: GO on the
  one-sided Fisher / Mann-Whitney tests at the pre-registered 0.01 / 0.04 alone. Every row now
  carries the failure-rate rise with a Newcombe hybrid-score 95 % CI (`newcombe_interval`,
  `wilson_interval`) and the median-speed drop with a seeded percentile-bootstrap 95 % CI
  (`speed_drop_interval`). `effect_size_gate=True` (CLI `--effect-size-gate`) restores the
  >= 0.25 / >= 30 % gated rule. The result also names `go_cells`, flags `two_factor_only` when
  a GO rests on heavy-loose alone (ballast and soil both change), and counts SAFETY outcomes
  per cell; the report says SAFETY counts as failure and depends on the I_peak GUESS.
- **M3, `orion_experiments/power.py`** (CLI `phase0_power`). Draws whole two-stage sessions
  (failures Bernoulli; failure speed = U(0, 1.5 m) / 60 s, success speed log-normal, every
  failure slower than every success) and applies the phase0_decision rules vectorised (both
  stage-1 loose cells x both metrics, stage-2 pooling). A test checks the vectorised rule
  against `phase0_decision` session by session for all five rule variants and every decision
  path. 20000 sessions per scenario, seed 20261109, about 4 s for the whole report. Also exact
  plain-Fisher power (reproduces the paper's 0.49 / 0.71 / 0.85) and main-study power.
- **m7, pooled primary** (`main_study`): student_critic vs B0 pooled over the loose cells,
  one-sided Fisher at 0.05, unadjusted, role `primary`; the 16 per-cell tests are `secondary`
  with Holm over that family only. Both carry the rate reduction with a Newcombe CI.
- **M2, onset labels** (`entrapment_onset`, `trial_metrics`). t_entrap = the first FLAG_STUCK of
  the stuck episode that ended in the hard-stuck HALT. Episode = the first FLAG_STUCK run still
  high at, or starting after, the last time the rover was >= 0.1 m from where it halted
  (position from the pose track, else integrated TerrainState.body_speed, else the last run).
  Label 1 if t_entrap - t <= 10 s; censoring as before. Recorded per trial:
  `entrapment_onset_s`, `onset_source`.
- **M6, trial runner**: StartTrial refused unless TerrainState.body_speed_source has been
  EXTERNAL with no gap > 0.25 s for `external_speed_hold_s` (2 s); `require_external_speed`
  (default true, launch argument too). The message names the source, the staleness or the
  short hold. metadata.json records the check.
- **M7, orion_safety**: the stuck-by-speed rule trusts the body speed without the confidence
  term when the fix is younger than `stuck_fresh_fix_s` (0.5 s); fix age = time_since_fix +
  age of the TerrainState message. `on_terrain` takes `time_since_fix` (default inf).
- **m11, safety_node**: a steady-clock 1 Hz timer warns (throttled 5 s) while use_sim_time is set
  and the node clock is still 0 after `clock_warn_after_s` (5 s). `SafetyNode(**kwargs)` for
  tests; orion_safety tests get ROS domain 96.
- **m9, `orion_experiments/imu_check.py`** (CLI `imu_check`) and BRINGUP step 10: static a_x bias,
  integral of (a_x - bias) over the acceleration to cruise vs camera speed, pass within 5 %.
- **Docs**: SPEC §8 and §9, RESEARCH.md Sample Size and Part 2B (method table, M1 statement),
  SAFETY.md decisions 13-16 and a known-gaps table, BRINGUP step 10, INTERFACES.md, twist_mux.yaml
  comment (m2).
- **Pre-existing bug fixed**: `test/bag_fixture.py` did not set PolicyStatus.critic_gating (added
  to the message in c5ab004), so all 4 bag tests errored at HEAD.

## Decisions

- **M3 default implemented as instructed, with a finding against it.** The simulation shows the
  ungated rule with the speed test has false GO 0.062 (H0 10 %) and 0.063 (H0 30 %), above the
  0.05 the 0.01 + 0.04 split was supposed to bound: that split is Bonferroni over two looks for
  ONE test each, but the rule runs 4 tests at stage 1 and 2 at stage 2. Options with numbers are
  in SPEC §9; my recommendation is ungated, failure rate only (`--no-speed`): power 0.89-0.91,
  false GO 0.017-0.029. I did not change the default because the integrator and the user own
  that choice; it is one switch either way.
- **Speed CI = bootstrap of the median ratio**, not Hodges-Lehmann on log speeds: effective speed
  can be 0 or negative (progress / 60 s for a rover that stopped), so a log scale is undefined.
  Seed fixed so reports are reproducible.
- **Power simulation calls scipy's mannwhitneyu with axis=1** (same function, same 'auto' method
  as the analysis, which is asymptotic with continuity correction for n > 8) and Fisher via a
  lookup table built from `analysis.fisher_one_sided`, instead of re-deriving either test.
- **Speed model**: with slowdown = 1 the Mann-Whitney test only sees the failure counts (ranks),
  so the log-normal success-speed guess does not affect any number; stated in the docstring and
  SPEC. A real slowdown of successful runs would add power. Rejected: inventing a loose-soil
  slowdown for the headline numbers.
- **Primary main-study test outside the Holm family**: tested alone at 0.05 (pre-registered
  primary); the 16 per-cell tests are a separate secondary family. Rejected: including the
  primary in the Holm family (it would cost it most of its power for no inferential gain).
- **M2 episode definition from the rover's displacement, not from the monitor's internal episode
  clock.** The monitor restarts its episode whenever the 20 s window still contains >= 0.1 m of
  progress, so its "episode start" lags the physical onset by up to 20 s after a fast approach.
  The label must mark when motion stopped, so the definition uses "last time >= 0.1 m from the
  halt position". ENTRAPPED without any FLAG_STUCK (operator outcome) falls back to the HALT time,
  recorded as `onset_source = 'halt'`.
- **M6 checks only at StartTrial**, not during the trial: a mid-trial camera dropout is a normal
  event the estimator handles; the bag records it. Subscription with sensor-data QoS so it
  matches best-effort and reliable publishers.
- **M7 counts any non-IMU fix** (time_since_fix covers EXTERNAL and LIDAR_ODOM): both measure the
  body, not the wheels. The message's own age is added so a TerrainState that stops arriving
  does not stay "fresh" for the 0.5 s terrain timeout.
- **m1 not gated, documented**: gating ESTOP_CLEAR on the reset needs a new interface from
  orion_safety to orion_hardware (not my package). SAFETY.md decision 14 records why it is safe
  today (priority-255 zeros while orion_safety's ESTOP stays latched) and the residual risk
  (orion_safety dead).
- **m2 comment corrected against the twist_mux source.** First draft said a timeout would
  "release a pressed e-stop"; `/opt/rosenv/include/twist_mux/topic_handle.hpp` shows
  `isLocked() = hasExpired() || data`, so an expired lock LOCKS. With a hand-published
  `/orion/estop` any timeout would lock the rover 0.5 s after each release; hence 0.0 until a
  repeating driver exists.
- **m9 as a tool, not a doc snippet**: a tested `imu_check` instead of untested inline Python in
  BRINGUP. The check integrates over the acceleration window only (2.5 s), not the whole 2 m
  run: double integration over 10 s turns a 0.01 m/s^2 bias residual into 0.5 m of error.

## Mistakes made and fixed

- **Wrong twist_mux lock semantics in the first comment** (above). Root cause: wrote from memory.
  Lesson: read the installed header before documenting third-party behaviour.
- **Wrong expected distance in the imu_check test** (2.05 m instead of 2.15 m): miscounted the
  cruise time of my own synthetic run. Caught by the test on its first run.
- **flake8 import order (I100)** in two test files: google style sorts `import numpy` among the
  `from orion_experiments ...` lines by module name.

## Tests (sandbox, RoboStack Jazzy)

- orion_experiments direct pytest: test_statistics 21, test_analysis_bags 5, test_power 5,
  test_imu_check 4, test_trial_runner 21, plus flake8 / pep257 / protocol.
- orion_safety direct pytest: 75 passed (test_monitor 61 incl. 6 new, test_safety_node 3 new).
- Negative checks (each restored afterwards): labels from t1 instead of t_entrap -> the bag label
  test fails; ungated rule forced to require the rise -> the M3 switch test fails; the
  vectorised gate inverted in power.py -> the equivalence test fails; the StartTrial gate
  disabled -> 4 runner tests fail; fresh-fix trust removed -> the M7 test fails; fix age without
  message age -> the age test fails; warning counter removed -> the m11 test fails; bias not
  subtracted in imu_check -> 3 tests fail.
- Full regression: colcon build (10 packages) + `colcon test --packages-skip orion_sim_tests
  --ctest-args -E xmllint`: 461 tests, 0 errors, 0 failures, 12 skipped. learning
  `python3 -m pytest -q`: 55 passed.

## Only verifiable on the real rover

- The I_peak GUESS decides how many trials end SAFETY; the per-cell SAFETY counts show it.
- Whether the camera clock offset is small enough for EXTERNAL to stay fresh (M6 gate).
- IMU scale and bias (BRINGUP step 10).
- Fresh-fix stuck rule with real camera latency (the fix age assumes the terrain node's
  time_since_fix is on the same clock as its stamps).

## Open questions / needs from other packages

- **User / integrator**: the M3 switch (SPEC §9); the default exceeds the 0.05 false-GO bound.
- **SPEC §5** (not mine): the stuck row should read "body speed < 0.03 m/s for 3 s with a fix
  < 0.5 s old or confidence > 0.3".
- **learning/orion_learning/baselines.py StuckDetector** mirrors the stuck rule with the
  confidence term only; the observation has no time_since_fix, so sim B0 and the rover's stuck
  rule now differ when the confidence is low with a fresh fix. Either add a fix-age feature or
  accept and document it (FIX-B's confidence change narrows the gap).
- **M2 sim side**: the env's critic labels should also count from the onset of the stuck episode.
- **m1 proper fix**: orion_hardware should send ESTOP_CLEAR only after /orion/estop_reset
  succeeds (needs an orion_safety -> orion_hardware signal).
