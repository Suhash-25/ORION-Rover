# FINAL (final-gaps): e-stop clear, give-up, ONNX in sim, Round 7 hooks

Branch `claude/final-gaps` (from 9802739). Items: safety m1, council m10, the policy give-up,
Round 7 readiness, the confidence formula in the docs, firmware build, and the integrator's
drive-train launch arguments.

## What was built

- **Launch arguments (integrator item, d657e1b).** `robot.launch.py` declares `port`, `baud`,
  `encoder_cpr`, `gear_ratio`, `kt`, `motor_resistance`, `brake_on_idle` (later also
  `perturbation_max_rad_s`) with the xacro defaults and forwards every one to the xacro
  `Command`. `test_robot_launch.py` renders the URDF with a non-default port and encoder_cpr,
  checks every argument with distinct values, and checks that the launch defaults equal the
  xacro defaults. `config/99-orion.rules` is a udev example (`/dev/orion_fw` by Teensy USB
  serial); BRINGUP section 0 references it.
- **m1, ESTOP_CLEAR (bf05671).** `orion_hardware` subscribes `/orion/safety_status`.
  `EstopLatch` (`safety_gates.hpp`) sends ESTOP_CLEAR only when:
  - `/orion/estop` is false;
  - SafetyStatus showed STATE_ESTOP after the latch;
  - SafetyStatus then left STATE_ESTOP, and the gap between the last ESTOP status and the
    first non-ESTOP status was < `safety_timeout_s` (0.5 s).

  A gap longer than that means orion_safety restarted and forgot the e-stop, so the latch
  stays. A refused reset keeps it too. A new press voids an earlier reset. While latched,
  write() repeats ESTOP.
- **Round 7 (a), WheelPerturbation (bf05671).** New `orion_msgs/WheelPerturbation` (header,
  `float32[4] offsets` rad/s in URDF joint order, `duration_s`). `PerturbationGate` behaves as
  follows:
  - it is disabled when `perturbation_max_rad_s` is 0 (the default);
  - it rejects requests that exceed the max or contain NaN, and requests made while
    SafetyStatus is not NOMINAL/LIMITED or is stale;
  - `duration_s <= 0` cancels;
  - HALT, ESTOP, a stale SafetyStatus and on_deactivate cancel an active perturbation;
  - write() adds the offsets to the commanded wheel velocities until it expires.

  No detector was built.
- **Policy give-up (26c94bd, D-013).** `PolicyStatus.escapes_exhausted` is latched from the
  give-up until the next SetMode. On its rising edge the trial runner ends the trial
  ENTRAPPED ('policy gave up'). analysis dates it at the onset of the stuck episode (M2
  anchor), falling back to the give-up stamp, and records `entrapment_source`
  (`hard_stuck` / `policy_gave_up` / `operator`). Simulation counts it the same way:
  `evaluate.py` outcome `gave_up`, and the critic branch labels use
  `risk.entrapment_events`. Documented in SPEC section 8 and INTERFACES.
- **m10 (2aaacd3).** `orion_sim_tests/test_policy_onnx` (domain 76) exports adapter, policy and
  critic ONNX with orion_policy's fixture into a temp dir. The policy is constant: biases are
  atanh of the action, weights 0. orion_policy loads it in MODE_POLICY and drives through
  twist_mux. The test asserts:
  - odometry > 1 m in 10 s;
  - mode POLICY throughout;
  - `obs_spec_hash`, and `policy_sha` = manifest sha256[:16];
  - command 0.3 m/s;
  - a non-zero latent and a risk in [0, 1].
- **Round 7 (b).** `test_drive.test_2c` measures header-stamp rates over >= 2 s of sim time
  and asserts >= 90 Hz for `/joint_states` and `/imu/data`. Measured: see Tests. The rates
  come from controllers.yaml `update_rate: 100` and imu.xacro `update_rate` 100. They are
  documented in INTERFACES as required 100 Hz.
- **Docs (e573dfa).** The confidence c = exp(-t_since_fix/3) * clip(1 - kf_std/0.1, 0, 1)
  and the < 0.5 s fresh-fix rule are now in SPEC sections 4 and 5, SAFETY decision 13,
  RESEARCH section 6.1 and the method table, and learning/README.

## Decisions (rejected options)

- **The clear signal is SafetyStatus, not the EstopReset service.** Rejected: calling the
  service from the hardware plugin, or watching the operator's call. SafetyStatus is already
  published at 50 Hz, it carries the state orion_safety actually accepted, and it needs no
  new interface. The gap rule covers orion_safety restarting.
- **Give-up ends the trial ENTRAPPED; the controller does not HALT.** Rejected:
  - letting orion_safety HALT a stopped rover (it would need a new rule on PolicyStatus);
  - keeping TIMEOUT (it hides failures and differs between controllers).

  The rising edge alone ends the trial, so a latched flag left over from the previous trial
  cannot end the next one.
- **Constant-action ONNX fixture** instead of a trained artefact. The expected motion is
  known, and there is no training dependency in CI. The adapter and critic still run.
- **The perturbation adds to the command** and never replaces it. It is applied after
  ros2_control limits, so `perturbation_max_rad_s` is its only bound.

## Mistakes

- The first launch-forwarding test passed with `port` not forwarded, because the xacro
  default matched. Every argument now gets a distinct non-default value.
- The first EstopLatch cleared after an orion_safety restart (a fresh node publishes
  NOMINAL). This was found in review and fixed with the 0.5 s continuity rule and a test.
- The gtest loops used `spin_some`, which returns at once, so the timing tests were
  meaningless. A time-based `spin(d)` fixes this.

## What is rover-only, and open questions

- The real serial port, the udev serial number (`REPLACE_WITH_SERIAL`) and the IMU driver
  rate (it must be set to 100 Hz) can only be checked on the rover.
- Calibration: critic-triggered give-ups also count as ENTRAPPED. This should be checked
  against real trials.
- Pre-existing parity gap: the sim student and B2 have no escape arbiter, while rover
  POLICY mode escapes on FLAG_STUCK.
- **Firmware (item 6) is blocked.** PlatformIO 6.2.0 installs from PyPI. Both
  `pio run -e teensy41` and `pio test -e native` fail at "Platform Manager: Installing ...
  HTTPClientError". The proxy answers `CONNECT tunnel failed, response 403` for
  api.registry.platformio.org, dl.registry.platformio.org and github.com. The host CMake
  build plus ctest of the firmware logic pass (2/2). No firmware files were changed.

## Tests

- `cd robot_ws && colcon build --parallel-workers 2 && colcon test --parallel-workers 2
  --ctest-args -E xmllint && colcon test-result --all` (with `PYTHONPATH=<worktree>/learning`)
  → **529 tests, 0 errors, 0 failures, 14 skipped** (the skips are cppcheck). This includes
  all five orion_sim_tests launch tests (drive 7, nav 2, policy 3, policy_onnx 3, safety 5).
  Other counts:
  - orion_hardware: safety_gates 12, orion_system 10, integration 6;
  - orion_bringup: test_robot_launch 3;
  - orion_experiments 71, orion_policy 73.
- Sim sensor rates (test_drive test_2c, sim time): `/joint_states` 100.0 Hz, `/imu/data`
  100.0 Hz.
- `python3 -m pytest -q learning/tests` (worktree `learning` on PYTHONPATH) → **101 passed**.
- **Negative checks** (each run on a deliberately broken copy, then reverted):
  - on_estop sends ESTOP_CLEAR on topic release → integration "ESTOP_CLEAR sent without the
    reset" fails, and two OrionSystem gtests fail;
  - launch arguments not forwarded to xacro → 2 test_robot_launch tests fail;
  - the trial runner ends on the level of `escapes_exhausted` instead of its rising edge →
    `test_policy_give_up_is_entrapped` fails;
  - `POLICY_ACTION = (0, 0)` → test_policy_onnx fails with "odometry advanced only 0.00 m";
  - `MIN_SENSOR_RATE_HZ = 150` → test_drive fails with "/joint_states at 100.0 Hz".
