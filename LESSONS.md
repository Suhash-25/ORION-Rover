# Lessons

A running log of reasons things didn't work out — why a feature got blocked, why a fix was
rejected, why an issue was marked won't-fix. Regenerated automatically (`hx feature block`,
`hx issue reject`, `hx issue wontfix`, or `hx lesson add`) — read this before repeating the same
dead end a future session might otherwise walk into again.

## 2026-11-09: Build path mismatch from stale CMake cache

- Source: manual (manual)
- The recovered machine had CMake cache pointing to /home/suhash/ARES/robot_ws, but current clone is at /Desktop/ARES/robot_ws. Old build/install/log directories contain stale paths. Mitigation: Always use isolated build directories with colcon --build-base build_handoff --install-base install_handoff --symlink-install. Never reuse stale build artifacts from previous paths.

## 2026-11-09: Teleop topic subscription mismatch in ros2_control

- Source: manual (manual)
- ros2_control configuration may reference wrong topic names for velocity commands. Teleop publishes to /cmd_vel (twist) but controller YAML must explicitly map this. Symptom: rover does not respond to keyboard input. Debug: ros2 topic list (verify /cmd_vel exists), ros2 control list_controllers (verify diff_drive_controller active), ros2 topic echo /cmd_vel (verify commands being published), grep cmd_vel in robot_ws/src/orion_control (verify YAML subscription). Do not assume topic wiring is correct; always validate explicitly.

## 2026-11-09: Verification that cannot fail is worse than none

- Source: manual (Claude)
- feature_list.json checks were `echo`/`print` commands, so `hx verify` marked broken SLAM and Nav2 features passing. Every check now runs `scripts/verify.sh` (real colcon build/tests). Break the code once to prove a test can go red. Details: docs/engineering-log/MISTAKES.md M-001.

## 2026-11-09: Jazzy diff_drive_controller only accepts TwistStamped

- Source: manual (Claude)
- Nav2 published Twist on /cmd_vel, so it never moved the robot. `use_stamped_vel` no longer exists in Jazzy. Fix: `enable_stamped_cmd_vel: true` on controller_server, behavior_server and velocity_smoother, remapped to /diff_drive_controller/cmd_vel. Guarded by orion_sim_tests/test_nav.py. (M-003)

## 2026-11-09: A clean checkout is the only honest build

- Source: manual (Claude)
- main failed `colcon build` on a fresh clone (CMake installed non-existent dirs) while old local install/ folders hid it. CI builds from scratch on every push. (M-002)

## 2026-11-09: Simulation must not depend on the internet

- Source: manual (Claude)
- Worlds pulled Sun and Ground Plane from Gazebo Fuel; offline the robot fell through nothing. Now inline SDF. (M-004)

## 2026-11-09: gz sim leaves its server running after launch shutdown

- Source: manual (Claude)
- The ruby `gz sim` wrapper dies, the server survives and answers the next test. Each launch test gets its own ROS_DOMAIN_ID and GZ_PARTITION and kills leftovers. (M-005)
