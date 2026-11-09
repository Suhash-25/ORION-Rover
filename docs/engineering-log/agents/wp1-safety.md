# WP1: safety, twist_mux, launch rewiring (2026-11-09)

Branch `claude/wp1-safety` from `9e2754a`. Sandbox: RoboStack Jazzy, ROS_DOMAIN_ID 71,
GZ_PARTITION `orion_wp1_*`.

## What I built

- **orion_safety** (new ament_python package). `monitor.py`: every SPEC section 5 rule as a pure
  Python state machine with an injected clock (tilt LIMITED / HALT latched, current LIMITED /
  HALT with cool-down, host heartbeat with startup grace, geofence with 1 s prediction and an
  odom fallback, stuck, hard-stuck, firmware faults, latched e-stop + reset). `geometry.py`:
  quaternion / accelerometer tilt, polygon signed distance, unicycle prediction. `safety_node.py`:
  thin rclpy wrapper, 50 Hz SafetyStatus, zero twists on `/cmd_vel/safety` (HALT, ESTOP) and
  `/cmd_vel/estop` (ESTOP), service `/orion/estop_reset`. Config `safety.yaml` (rover) +
  `safety_sim.yaml` (Gazebo overrides).
- **twist_mux**: `orion_control/config/twist_mux.yaml` + `launch/twist_mux.launch.py`; output
  remapped to `/diff_drive_controller/cmd_vel`.
- **Launch**: `gazebo.launch.py` starts twist_mux and orion_safety and takes
  `hardware:=gazebo|serial` (passed to xacro); teleop -> `/cmd_vel/teleop`; Nav2 controller and
  behavior servers -> `/cmd_vel/nav`; new `robot.launch.py` for the real rover;
  `use_sim_time` passed through `controllers.launch.py` / `teleop.launch.py`.
- **Tests**: 66 unit tests in orion_safety (55 rule tests, 6 geometry, 3 config, flake8, pep257);
  `orion_sim_tests/test/test_safety.py` (headless Gazebo); `test_drive.py` now drives through
  `/cmd_vel/teleop`.
- **Docs**: `docs/INTERFACES.md`, `docs/SAFETY.md`, `docs/BRINGUP.md`.

## Decisions (full list with reasons in docs/SAFETY.md "Decisions")

- **twist_mux does TwistStamped natively.** Installed twist_mux is 4.5.0. Its headers
  (`/opt/rosenv/include/twist_mux/topic_handle.hpp`) have `VelocityStampedTopicHandle` and a
  `cmd_pub_stamped_` publisher; the binary contains the parameter `use_stamped` with the message
  `"use_stamped" is not declared as parameter, defaulting to "true"`. Older 4.x builds defaulted
  to false, so the YAML sets `use_stamped: true` explicitly. Verified at runtime:
  `ros2 topic info /diff_drive_controller/cmd_vel` -> `geometry_msgs/msg/TwistStamped`, 1
  publisher (twist_mux). Timeouts use twist_mux's receive time, not the header stamp; headers pass
  through, so every source must stamp with its node clock. No alternative mux was needed.
- **Two e-stop paths**: the twist_mux lock on raw `/orion/estop` (works with orion_safety dead)
  plus orion_safety's latched zero at priority 255 on `/cmd_vel/estop` (stays stopped if someone
  publishes `false`). Rejected: having orion_safety republish `/orion/estop` itself (two writers
  on one topic, and its own `true` would block the reset).
- **Reset needs `/orion/estop` false** (else the mux lock stays on while the status says NOMINAL)
  and, for a tilt latch, a fresh IMU reading <= 18 deg.
- **Startup grace**: required inputs never seen trip HALT after `startup_grace_s` (10 s) counted
  from the node's first tick; once seen, the 0.3 / 0.5 s timeouts apply; HALT clears itself when
  the input returns. Tested: never-seen -> HALT at 10 s and clears on arrival; IMU at 8 s +
  joint_states at 9.5 s -> never leaves NOMINAL; seen-then-lost inside the grace -> HALT.
- **Sim thresholds**: Gazebo effort is N.m. Measured with a probe script (headless empty.sdf,
  max abs effort over 4 wheels): straight 0.5 m/s ~0 N.m; spin 1 rad/s median 5.3, p95 6.0,
  max 6.1; transients to 6.7 N.m. The 6 A I_cont would LIMIT every spin in sim, so
  `safety_sim.yaml` uses 10 / 20 N.m. Geofence off in Gazebo (worlds are not the sandbox).
- **Hard-stuck window** counts from the start of a stuck episode and uses ground-truth pose or
  integrated terrain body speed, never wheel odometry.
- **Geofence frame**: the integrator added orion_ground_truth publishing `/orion/ground_truth_pose`
  in frame `sandbox`; the node checks `header.frame_id == geofence_frame` (`sandbox`) and ignores
  (with a throttled error) poses in another frame.
- **CMake test env**: `test_safety.py` uses ROS_DOMAIN_ID 71 / GZ_PARTITION `orion_wp1_safety`,
  the range assigned to this WP, so it cannot collide with other agents' tests.

## Mistakes and fixes

1. **rclpy logger severity crash** (found by a ROS smoke run, not by unit tests). I picked
   `info` or `warn` into one variable and called it from one line; rclpy raises
   `ValueError: Logger severity cannot be changed between calls` the second time a call site
   changes severity, which killed the node on its first NOMINAL -> HALT transition. Root cause:
   the node wrapper had no test; the pure logic did. Fix: separate `info` / `warn` calls. Lesson:
   run every node once for real (smoke script with twist_mux + safety_node, no Gazebo) before
   trusting the unit tests; the Gazebo test now exercises NOMINAL -> ESTOP -> NOMINAL logging.
2. **Unit-test arithmetic errors** in my own tests (arc prediction radius v/w; feeding a new tilt
   value without a tick before asking for a reset). Caught on the first run; tests corrected, not
   the code.
3. **Mutation left behind in install/**: for the sim negative check I edited the *installed*
   copies (not source). `colcon build` afterwards did **not** restore `safety_node.py`
   (setuptools only copies files newer than the build copy). Detected with `grep MUTANT`; fixed
   with `rm -rf install/orion_safety build/orion_safety` and a rebuild, then reran the sim tests.
   Lesson: after mutating installed files, delete the package's build and install dirs.
4. A `pkill -f 'ros2 launch orion_bringup'` cleanup was (rightly) refused by the sandbox: it would
   have killed other agents' launches. All later runs start my processes with `setsid` and kill
   only that process group plus gz servers whose environment has my GZ_PARTITION.

## Sandbox notes

- `colcon build --symlink-install` fails for ament_python packages here
  (`error: option --editable not recognized`, setuptools 84.2.1 with colcon-core 0.19). So
  `scripts/verify.sh` (which uses `--symlink-install`) cannot build orion_safety in this sandbox;
  I built with `colcon build --parallel-workers 2` (no symlink). CI's apt setuptools is older.
- `tests_require` is ignored by this setuptools (coordinator note); setup.py uses
  `extras_require={'test': ['pytest']}`; colcon ran 66 pytest tests.

## Test commands and results (all in robot_ws, sandbox env, ROS_DOMAIN_ID=71)

| Command | Result |
| --- | --- |
| `colcon build --parallel-workers 2` | 6 packages finished |
| `colcon test --packages-select orion_safety orion_control orion_bringup orion_description orion_msgs orion_sim_tests --ctest-args -E "xmllint\|test_drive\|test_nav"` then `colcon test-result --all` | 106 tests, 0 errors, 0 failures (orion_safety pytest 66/66; test_safety launch test 5/5) |
| `colcon test --packages-select orion_sim_tests --ctest-args -R test_safety` | 4/4 + post-shutdown 1/1; e-stop -> wheels < 0.1 rad/s in 0.011 s sim |
| `ROS_DOMAIN_ID=71 GZ_PARTITION=orion_wp1_drive launch_test src/orion_sim_tests/test/test_drive.py` | Ran 5 tests OK + shutdown OK |
| `ROS_DOMAIN_ID=71 GZ_PARTITION=orion_wp1_nav launch_test src/orion_sim_tests/test/test_nav.py` | Ran 1 test OK (goal reached via /cmd_vel/nav -> twist_mux) + cleanup OK |

drive and nav were run with `launch_test` under my domain instead of `colcon test`, because
their CMake ENV (61 / 62, `orion_test_drive` / `orion_test_nav`) is shared with every other agent
running the same tests at the same time.

### Negative checks (tests can fail)

- Unit: made `/orion/estop false` clear the latch and replaced the grace check with `if True:`
  -> `test_estop_latches_and_needs_exact_phrase`, `test_heartbeat_grace_when_input_never_arrives`
  and (after strengthening it to check every phase) `test_heartbeat_slow_launch_never_halts`
  failed. Restored -> 55/55.
- Sim: removed both stop paths in the installed copies (no zero twists from orion_safety, no
  twist_mux lock) -> `test_3_estop_stops_wheels_while_policy_publishes` failed with "wheels
  still turning 3 s after the e-stop". Restored (clean rebuild) -> pass.

## Only verifiable on the real rover

Physical e-stop and firmware ESTOP frame; 200 ms firmware watchdog; firmware current clamp;
real I_cont / I_peak in A; IMU orientation vs the accelerometer fallback (and its error under
acceleration); the wheel-odometry geofence fallback's drift; ground-truth latency vs the 0.5 s
pose timeout; `robot.launch.py` as a whole (needs orion_hardware's `hardware:=serial` xacro branch
and real sensor drivers). Checklist: docs/BRINGUP.md.

## Open questions

- Should a dead `/orion/ground_truth_pose` during a trial (it was alive, then stopped) be a
  trial abort? Today it is ignored by design (optional input) and the geofence falls back to odom.
- SPEC gives no limits for geofence LIMITED; I used 0.2 m/s and 0.5 rad/s (same as tilt).
- Sim-time clock jumps (Gazebo reset) are not handled specially: timers would see a negative dt
  (clamped to 0) and stale inputs until new messages arrive. Nothing resets Gazebo today.
- Accelerometer-based tilt is only a fallback; if the real IMU has no orientation output, a
  complementary filter would be better than raw gravity direction.
