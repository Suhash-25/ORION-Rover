# WP5: orion_policy and orion_experiments

Branch `claude/wp5-policy`, based on 9e2754a. ROS_DOMAIN_ID 75 (Gazebo test and orion_policy
unit tests), 95 (orion_experiments unit tests).

## What was built

### orion_policy (ament_python)
- `observation.py`: `ObservationBuilder` caches /joint_states (matched by joint name to
  FL FR RL RR), /imu/data (roll, pitch from the quaternion; gravity removed from the specific
  force with `g * (-sin p, sin r cos p, cos r cos p)`; yaw rate) and TerrainState, then calls
  `obs_spec.build_observation` (never re-implemented). Missing or stale inputs raise
  `InputNotReady` (limits: joints 0.3 s, IMU 0.5 s, terrain 0.5 s, matching orion_safety's
  heartbeat rule). `History` keeps 40 rows of [obs, action], oldest first, zero-padded.
- `artefact.py`: loads the ONNX contract directory. Refuses (`ArtefactError`) on a manifest
  or file `obs_spec_hash` different from the installed one, any sha256 mismatch, a file with
  no sha256 entry, a file name that is not a plain name in the directory, metadata
  (`model_kind`, dims, `control_hz`, `controller`) that does not match, or graph I/O names
  that differ from the contract. Zero latent without an adapter; NaN risk without a critic.
  Uses only onnxruntime at runtime (metadata via `get_modelmeta()`), 1 thread per session.
- `controllers.py`: `Controller` interface, `BaselineB0`, `OnnxController`.
- `sequences.py`: the escape library E (settle 1 s, reverse 0.15 m/s 2 s, 4 x 0.5 s yaw
  wiggle at 0.6 rad/s, creep 0.12 m/s 2 s), the probe (0.1 m/s straight, 1 s), and
  `EntryPath` (last 2 s of executed commands) for the retreat.
- `shaping.py`: gating bands, `scale_to_limits`, `CommandShaper` (obs_spec.rate_limit then the
  SafetyStatus limits).
- `core.py`: `PolicyCore`, the ROS-free mode machine (IDLE, BASELINE, POLICY, PROBE, ESCAPE).
- `policy_node.py`: ROS wiring; 20 Hz timer on the node clock (sim time in Gazebo);
  /cmd_vel/policy, /cmd_vel/escape, /orion/policy_status, service /orion/set_mode.
- `config/policy.yaml`, `launch/policy.launch.py`.

### orion_experiments (ament_python)
- `protocol.py` + `plan_session` CLI: protocol YAML (session, seed, cells, controllers, n,
  randomisation blocked|full, duration, cone depth bands), seeded plan, CSV output.
  `config/phase0_protocol.yaml` (80 trials) and `config/main_study_protocol.yaml` (400).
- `recorder.py`: `ros2 bag record -s mcap` subprocess of the SPEC topic list (incl.
  /orion/ground_truth_pose), SIGINT then SIGTERM/SIGKILL on stop.
- `trial_runner.py`: StartTrial/StopTrial on /orion/start_trial, /orion/stop_trial.
  Starts the bag, waits `record_warmup_s`, sets the controller (SetMode BASELINE for B0,
  POLICY otherwise), starts the timeout. Ends on timeout (TIMEOUT), FLAG_HARD_STUCK
  (ENTRAPPED), HALT/ESTOP (SAFETY), ground-truth progress >= 1.5 m along the start heading
  (SUCCESS) or StopTrial. Then IDLE, stop the bag, write metadata.json next to it.
- `metadata.py`: git SHA + dirty flag, ONNX sha256 recomputed from disk, atomic JSON.
- `analysis.py` (no ROS needed): SPEC section 8 metrics from bags via `rosbags`, Phase-0
  GO/NO-GO, Holm-corrected main study, critic calibration (ECE, Brier, reliability table);
  writes trials.csv, tests.csv, calibration.csv, report.md. CLI `analyse_trials`.

### orion_sim_tests
- `test/test_policy.py` + CMake line (domain 75, partition orion_test_policy).

## Decisions (and options rejected)
1. **prev_cmd and the history action are the command actually published** (after rate limit
   and safety scaling), in obs_spec action units. Rejected: the raw policy output, which is
   not what the wheels were told during probes, escapes and safety limiting. WP4's env must
   do the same (open question below).
2. **Adapter input is the history up to the previous tick**; the new row is pushed after the
   command is published (RMA convention). Zero rows pad the first 2 s.
3. **Safety limits scale v and w by one common factor** (keeps the turn radius) instead of
   clamping each separately. Both limits still hold.
4. **Stale/missing SafetyStatus, HALT/ESTOP, stale inputs: immediate zero** on the active
   channel (not rate limited) with a logged reason. PolicyStatus has no reason field, so the
   reason is in the node log (first occurrence at once, then throttled to 2 s).
5. **Channel hand-over**: leaving PROBE/ESCAPE publishes one zero on /cmd_vel/escape, else
   twist_mux (priority 120 > 100) would hold the last escape command for its 0.5 s timeout.
   IDLE publishes one zero then nothing (publishing zeros at priority 100 would block Nav2).
6. **B0**: "cuts v 20 %/s" read as the speed fraction falling by 0.20 of cruise per second
   (linear), recovering 0.05/s, triggered by max(slip_side) > 0.3 (strict). Floor 0.4
   (0.12 m/s) so the command stays above orion_safety's 0.1 m/s stuck gate; with a floor of
   0 the governor could hide an entrapment. Heading open loop (w = 0).
7. **Gating**: 0.3 <= rho < 0.5 caps v at 0.2 m/s and starts a probe on entry into the band
   (at most once per 5 s, against chattering at 0.3); 0.5 <= rho <= 0.7 retreats along the
   last 2 s of executed commands, reversed and negated, slowed to 0.2 m/s with time
   stretched so curvature and arc length are kept; with no recorded path it escalates to
   ESCAPE (holding still would deadlock: the stuck rule needs cmd v > 0.1). rho > 0.7 ESCAPE.
   Retreat runs on /cmd_vel/policy in mode POLICY (no RETREAT mode exists in PolicyStatus).
8. **Escape attempts**: at most 3 escapes without a clean driving tick in between, then IDLE
   and orion_safety's hard-stuck rule ends the trial. Covers both stuck-flag and rho escapes.
9. **SetMode** accepts IDLE/BASELINE/POLICY (POLICY fails without model_dir) and one-shot
   PROBE/ESCAPE that return to the previous driving mode.
10. **Recorder is a `ros2 bag record` subprocess**, not rosbag2_py.Recorder in-process:
    Recorder.record() blocks and owns its own context/signals, and a recorder crash must not
    kill the runner and the metadata. mcap storage, explicit.
11. **Runner drives the controller** (SetMode) after a 2 s recorder warm-up, so recording
    always starts before motion and the timeout stops the rover.
12. **Ballast check**: heavy-* cells need ballast_kg > 0, light-* need 0, else StartTrial is
    refused (cheap guard against mislabelled trials).
13. **Analysis definitions** (pre-registered in the analysis.py docstring): ABORT and
    cone-out-of-band trials are invalid; failure = TIMEOUT/ENTRAPPED/SAFETY; effective speed
    divides by 60 s for non-successes (a rover that gives up early is not rewarded);
    time-to-entrapment censored at 60 s; calibration labels censored in the last 10 s of
    non-entrapped trials; energy electrical (V_bat * sum |duty * I|) with RoverHealth,
    mechanical sum |tau * w| otherwise. Protocol randomisation default "blocked".
14. **Main-study family** (`COMPARISONS`): student < B2, student_critic < student,
    student < B0, B2 < B0 in each cell, one-sided Fisher, Holm over all of them.

## Mistakes and fixes
- **rclpy logger severity**: `log = info if ok else warn; log(...)` crashed the SetMode
  service ("Logger severity cannot be changed between calls"): rclpy ties severity to the
  call site. Caught by test_policy_node; now two call sites. Lesson: test the ROS node, not
  only the pure core.
- **Unit-test cross-talk under colcon**: colcon tests packages in parallel; the policy node
  test's /orion/set_mode server answered the trial runner test (which expected none) and the
  runner test's PolicyStatus reached the policy test. Sequential pytest runs hid it. Fix:
  each package's conftest sets its own ROS_DOMAIN_ID (75, 95).
- **ns rounding in the analysis window**: a sample exactly at the trial end fell outside
  `t <= t1` after bag timestamps were rounded to integer ns; 1 us tolerance.
- **pep257 convention**: ament's pep257 enforces D213 (summary on the second line); fixed
  all docstrings.
- Brief noted by the integrator: `tests_require` is ignored by this setuptools; both
  packages use `extras_require={'test': ['pytest']}`.

## Tests (commands and results)
- `colcon build --parallel-workers 2 --packages-select orion_policy orion_experiments`
  then `colcon test --packages-select orion_policy orion_experiments`:
  orion_policy 68 tests, 0 failures; orion_experiments 43 tests, 0 failures (incl. flake8,
  pep257, and a real `ros2 bag record` round trip read back with rosbags).
- `colcon test --packages-select orion_sim_tests --ctest-args -R test_policy`: 3 tests
  (2 + shutdown), 0 failures; Gazebo up and the rover driving within ~16 s.
- Negative checks (each restored afterwards): sha256 check disabled -> 2 artefact tests
  red; stale-safety check disabled -> stale-safety test red; joint mapping by position
  instead of name -> 2 builder tests red; 0.7 assigned to ESCAPE -> threshold test red;
  Fisher two-sided instead of one-sided -> 2 Phase-0 tests red; runner treats FLAG_STUCK as
  entrapment -> runner test red; effective speed divided by elapsed -> analysis test red;
  sim test with initial_mode IDLE -> 2 of 3 launch tests red.
- `scripts/verify.sh unit ...` fails in this sandbox before testing: its
  `colcon build --symlink-install` hits "option --editable not recognized" for every
  ament_python package (setuptools 84 in the RoboStack env). Not a package bug; the plain
  build above works and CI's ros:jazzy setuptools supports it.

## Only verifiable on the real rover
- effort_max in A (the current equivalent of tau_max), IMU mounting and gravity removal
  with real vibration, the B0 constants (0.3 m/s, 20 %/s, 5 %/s, floor) and the escape
  library in real sand, probe/retreat usefulness, recorder warm-up time on the rover's
  computer, cone-depth bands, the electrical energy estimate.

## Open questions
1. WP4 env must use the same prev_cmd/history-action convention (executed command) and the
   same zero padding, the same escape library E (rho is defined under E), and the same B0
   (`learning/orion_learning/baselines.py` vs `orion_policy.controllers.BaselineB0`:
   keep one).
2. student vs student_critic differ only by orion_policy's `critic_gating` parameter, which
   neither PolicyStatus nor the trial metadata can see. Suggest a PolicyStatus field (or
   the runner reading the parameter) so a mislaunched B3 cannot be recorded as
   student_critic.
3. RoverHealth has no firmware hash; the runner takes it as a parameter. A
   `fw_git_hash` field (from PONG) would make it automatic.
4. SPEC 6 "retreat" is not a PolicyStatus mode; retreat is logged only.
