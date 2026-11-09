# WP3 orion_terrain (branch claude/wp3-terrain)

WP3 agent, 2026-11-09. Package: `robot_ws/src/orion_terrain` (ament_python). Contract:
docs/SPEC.md section 4.

## What was built

| file | content |
| --- | --- |
| `orion_terrain/imu.py` | quaternion -> rotation, gravity compensation, roll/pitch; frame conventions documented |
| `orion_terrain/speed.py` | `BodySpeedKalman` (1-D on v_x), `PoseDifferentiator` (5-sample S-G), `confidence()`, `SpeedSource` |
| `orion_terrain/effort_model.py` | `EffortModel` (YAML, residual, informative window, sinkage proxy), least-squares fit |
| `orion_terrain/filters.py` | `LowPass` (tau, per-entry hold mask), `WheelAccel` (dw/dt, shared by runtime and fit tool) |
| `orion_terrain/estimator.py` | `TerrainEstimator` (one code path for node, replay and tests), `estimate_from_messages()` |
| `orion_terrain/fit_effort_model.py` | calibration CLI on rosbag2 via `rosbags` (no ROS needed) |
| `orion_terrain/terrain_node.py` | ROS wrapper, 20 Hz `/orion/terrain_state` |
| `config/terrain.yaml`, `launch/terrain.launch.py` | parameters (mirror of `params.py`, test-enforced); `sim:=true` |
| `config/effort_model_sim.yaml` | labelled UNCALIBRATED zero model (see "Gazebo findings") |
| `test/` | 68 unit + lint tests; `test/sim/` Gazebo launch test and calibration driver |

Offline use (learning-env bags, any rosbag2):

```python
from orion_terrain.estimator import estimate_from_messages
from orion_terrain.fit_effort_model import read_bag
out = estimate_from_messages(read_bag(bag_dir, topics), sim=True, effort_model=model)
```

`msg` may be a ROS message, a rosbags message or a nested dict with ROS field names; input
order does not matter (sorted by (t, topic)); output at 20 Hz.

## Test commands and results

- `cd robot_ws && colcon build --parallel-workers 2 --packages-select orion_msgs orion_terrain`
  then `colcon test --packages-select orion_terrain && colcon test-result --verbose`:
  **68 tests, 0 errors, 0 failures, 0 skipped** (incl. flake8, pep257, enum-vs-msg check).
- Gazebo: `cd robot_ws/src/orion_terrain && ORION_RUN_SIM=1 ROS_DOMAIN_ID=73
  GZ_PARTITION=orion_terrain_test python3 -m pytest test/sim/test_terrain_gazebo.py -s`:
  **1 passed** (3 checks + shutdown), 36 s. Reported numbers: TerrainState 20.00 Hz (sim time);
  IMU_ONLY straight drive body_speed 0.219 m/s vs odom 0.302; EXTERNAL (relayed Gazebo pose)
  body_speed 0.300 vs odom 0.304, slip_side mean -0.001, max |slip_side| 0.005,
  confidence min 0.83 / mean 0.99.
- Replay (synthetic trajectory, 0.02 m/s^2 bias, 1-1.5 s occlusions): slip_side RMS < 0.05
  (asserted), body speed RMS < 0.01 m/s while fixed.
- Negative checks (each sabotage made the targeted tests fail, then reverted): gravity sign
  flipped (7 IMU tests red); lidar fused while external fresh (priority test red); joints by
  index (3 mapping tests red); wheel odometry fused as a fix (2 tests red); residual hold
  removed (hold test red); delayed-fix propagation removed (propagation test red).

## Decisions (and options rejected)

1. **Effort model along the direction of rotation.** SPEC: `I_hat = c0 + c1|w| + c2 dw/dt +
   c3 sin(pitch) sign(w)`, `e = I - I_hat`. With signed effort (amps / N.m are signed in
   /joint_states) reversing out of sand would read as *negative* sinkage. Adopted, with
   d = sign(w): `d I = c0 + c1|w| + c2 d dw/dt + c3 d sin(pitch)` (= the physical signed
   model `I = d c0 + c1 w + c2 dw/dt + c3 sin(pitch)`), residual `e = d I - I_hat` (+ = extra
   load in both directions). Rejected: literal SPEC form (breaks escape manoeuvres); unsigned
   |I| (loses deceleration).
2. **Residual only inside the model's window, held outside it.** Per wheel, the residual low-pass
   is updated only while |w| >= deadband (0.5 rad/s) and |yaw rate| <= max_yaw_rate (0.2 rad/s);
   both are stored in the model and used by the fit too. Reason: Gazebo data (below) showed the
   first design (d = sign(I) at standstill) reading sinkage ~0.6 at rest, and skid-steer scrub
   dominating effort while turning. Holding (not zeroing) keeps the last sinkage estimate when
   the rover stops in sand. Rejected: a turning regressor c4 |w_z| (not in SPEC, and the
   learning env's proxy would have to change too; raised as an open question).
3. **Slope term only when identifiable**: if std(sin pitch) < 0.01 in the bag, c3 := 0,
   `slope_identified: false` in the YAML and a warning (flat-ground lstsq gave c3 ~ -370).
4. **Centre S-G derivative + forward propagation.** The 5-point quadratic end-point derivative
   amplifies pose noise 22.3/h vs 6.3/h at the centre; the centre value is 2 samples old, so
   the Kalman update adds the IMU-integrated velocity change since then. Because the camera
   pose is stamped with image time, this also compensates camera latency.
5. **Lower-priority fixes are not fused while a higher one is fresh** (no double counting).
   Fresh = newest fix younger than its timeout (external 0.25 s, lidar 0.5 s, IMU 0.5 s).
   EXTERNAL needs 5 poses after a gap > 0.25 s (0.17 s at the camera's 30 Hz).
6. **Kept the 1-D filter** (SPEC) although [v, accel-bias] would bridge dropouts far better:
   the policy is trained on the same degraded estimate (F4), so runtime must match what the
   learning env simulates. Quantified: constant 0.02 m/s^2 bias, 5 s dropout -> slip_side RMS
   0.08 over the run (0.05 on samples with confidence >= 0.3) vs 0.009 without bias.
7. **Confidence edge cases**: no fix ever -> time_since_fix = inf, c = 0. Wheel speed from
   /diff_drive_controller/odom, falling back to r mean(w) from /joint_states; no wheel data
   at all -> c = 0.
8. **Effort-source gate**: sinkage only with a calibrated model whose `effort_source` equals
   the live one (SIM_TORQUE in sim; CURRENT / DUTY_ESTIMATE from RoverHealth.current_sensed;
   no RoverHealth yet -> NONE). Otherwise sinkage 0, effort_source NONE, residual zeros, one
   warning per reason.
9. **Model files**: sim `config/effort_model_sim.yaml` (shipped), hardware
   `config/effort_model.yaml` (SPEC path; written by the fit tool, deliberately not shipped,
   a test asserts its absence). The node picks by `sim` unless `effort_model_path` is set.
10. **Same dw/dt estimator in the fit and at runtime** (finite difference + 0.05 s low-pass).
    Time base: header stamps (receipt time if zero); publish tick on the node clock. Slip and
    residual are filtered at the joint-state rate (100 Hz), not the 20 Hz output.

## Gazebo findings (measured, 2026-11-09)

- **Effort**: velocity-commanded joints report ~0 N.m during smooth straight driving (even at
  8 rad/s^2) and saturated +-6 N.m spikes in scrub/contact transients. 196 s calibration bag
  -> R^2 0.03-0.11; the fitted model put sinkage > 0.1 on rigid ground 12 % of the time
  (p99 0.57). Shipped a labelled zero model (`calibrated: false`) -> NONE in sim. This is the
  brief's fallback; nothing physical exists to fit on rigid ground (D-003).
- **IMU**: from rest to 0.3 m/s the raw IMU integral (sum f_x dt) is 0.230 m/s while the true
  speed is 0.299 (`gz model -m orion -p` over 5 s) and odometry 0.300. The estimator reproduces
  the raw integral (node 0.227, offline replay 0.226), so it is the simulated IMU signal: the
  wheel speed steps, the body is accelerated by physics-rate contact impulses, and a 100 Hz
  IMU reporting instantaneous acceleration aliases part of them away (in smooth sweeps f_x
  stayed at 0 +- 0.03 while odometry swung +-0.4 m/s). IMU-only slip ~ 0 is therefore not
  achievable in this Gazebo setup; the launch test reports it and checks slip ~ 0 with the
  true model pose relayed as /orion/ground_truth_pose instead. A real MEMS IMU low-pass
  filters internally, but the same effect argues for a hardware ODR/DLPF check.

## Mistakes made and fixed

- *caplog saw no records* (a ROS pytest plugin reconfigures logging). Fix: the estimator takes
  a logger; tests pass `RecordingLogger`. Lesson: inject loggers.
- *Replay showed slip RMS 0.3*: my synthetic bias random walk (0.02/sqrt(s)) reached
  ~0.15 m/s^2. Lesson: check a generator's statistics before blaming the estimator.
- *Gazebo test was collected by plain pytest* despite `collect_ignore`: launch_testing
  registers a `tryfirst` `pytest_ignore_collect` that returns False (not None), which ends the
  firstresult chain. Fix: my own `tryfirst` hook in test/conftest.py. Lesson: verify a skip
  with `--collect-only`.
- *Stalled-wheel rule* (d = sign(I) below the deadband) looked right on paper and was wrong
  on data (noise -> sinkage 0.6 at rest). Lesson: run new signal processing on recorded
  data before shipping it.
- *Orphaned ROS processes*: killing `ros2 launch` / `ros2 run` with SIGKILL left bridges
  alive; the next run in the same GZ partition received duplicate /imu/data (431k msgs in 17 s).
  Also `ros2 bag record` started as a background job of a non-interactive shell ignores
  SIGINT, so it never wrote metadata (fixed with `ros2 bag reindex` + `ros2 bag convert`).
  Fix: `setsid` per launch and kill the process group; check for leftovers by ROS_DOMAIN_ID.
  All orphans were in my own domain (73) and were killed.

## Sandbox notes

- `colcon build --symlink-install` fails for ament_python here (`option --editable not
  recognized`, setuptools 84 + colcon-python-setup-py 0.2.9), so `scripts/verify.sh unit
  orion_terrain` fails at the build step in the sandbox; plain `colcon build` works and CI
  (`ci.yml`, no symlink install) is not affected.
- setuptools ignores `tests_require`; `extras_require={'test': ['pytest']}` is used.

## Only verifiable on the real rover

- IMU mounting vs base_link, orientation quality (1 deg tilt error = 0.17 m/s^2 into a_x) and
  the IMU's internal low-pass (see the Gazebo IMU finding).
- Clock sync between the overhead-camera PC and the rover (header stamps are compared).
- The hardware effort model: 5 min packed-light bag + stall test -> `fit_effort_model`; the
  turning-scrub level on sand and whether 0.2 rad/s is the right max_yaw_rate.
- rf2o twist convention (assumed body-frame forward speed in twist.linear.x).

## Open questions

- **Confidence penalises real slip**: `|v_wheel - v_x|` is large whenever the wheels slip,
  even with a perfect fix. A stuck rover with rim speed >= 0.5 m/s gets c = 0, which disables
  the safety rule "body speed < 0.03 m/s for 3 s (confidence > 0.3)"; the slip_side > 0.8 rule
  still fires. Suggest the council keeps this knowingly or bases the second factor on the
  Kalman std instead.
- **Turning resistance** is not in the SPEC effort model; the residual is held while turning.
  Should the model gain a scrub term (and the learning env's proxy too)?
- 1-D vs 2-state filter (decision 6): switch both runtime and env together if at all.
