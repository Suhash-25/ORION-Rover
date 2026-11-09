# ORION interfaces

The single list of every topic, service, action and frame ORION uses: type, who publishes, who
subscribes, rate and units. If it is not here, it is not a contract. Changing a row is a decision:
log it in `engineering-log/DECISIONS.md`. Background: M-003 (Nav2 could never move the robot
because the topic contract lived in nobody's head).

Conventions: REP 103 (x forward, y left, z up; SI units) and REP 105 frames. Wheel order in every
array is **FL, FR, RL, RR**. Every velocity command is `geometry_msgs/msg/TwistStamped`, stamped
with the publisher's node clock (sim time in Gazebo): Jazzy's `diff_drive_controller` drops
commands older than `cmd_vel_timeout` (0.5 s) and rejects plain `Twist`.

Status column: **live** = runs today in `gazebo.launch.py`; **planned** = contract for a package
that does not exist yet (see `feature_list.json`).

## Command path

```
 /cmd_vel/estop   255  orion_safety (zero, while ESTOP) ─┐
 /cmd_vel/teleop  200  teleop_twist_keyboard / operator  │
 /cmd_vel/safety  150  orion_safety (zero, HALT/ESTOP)   ├─ twist_mux ─ /diff_drive_controller/cmd_vel ─ diff_drive_controller
 /cmd_vel/escape  120  orion_policy escape behaviour     │      ▲                                              │ 100 Hz
 /cmd_vel/policy  100  orion_policy (B0 / learned)       │      │ lock (priority 255)                          ▼
 /cmd_vel/nav      50  Nav2 controller/behavior server ──┘  /orion/estop                     ros2_control hardware
                                                                                (gz_ros2_control | orion_hardware -> firmware)
```

twist_mux (`orion_control/config/twist_mux.yaml`, `use_stamped: true`) forwards the
highest-priority input that published within its 0.5 s timeout. While `/orion/estop` is `true`
every input below 255 is masked. The policy and escape sources must additionally obey the limits
in `/orion/safety_status` themselves; twist_mux does not scale commands (docs/SAFETY.md).

## Topics

| Topic | Type | Publisher | Subscribers | Rate | Units / notes | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `/cmd_vel/estop` | geometry_msgs/TwistStamped | orion_safety | twist_mux (prio 255) | 50 Hz while ESTOP, else silent | always zero | live |
| `/cmd_vel/teleop` | geometry_msgs/TwistStamped | teleop_twist_keyboard (`orion_control/teleop.launch.py`), operator tools | twist_mux (prio 200) | on key press | m/s, rad/s | live |
| `/cmd_vel/safety` | geometry_msgs/TwistStamped | orion_safety | twist_mux (prio 150) | 50 Hz while HALT or ESTOP, else silent | always zero | live |
| `/cmd_vel/escape` | geometry_msgs/TwistStamped | orion_policy (escape) | twist_mux (prio 120) | 20 Hz while escaping | m/s, rad/s; within SafetyStatus limits | planned |
| `/cmd_vel/policy` | geometry_msgs/TwistStamped | orion_policy (B0 baseline or ONNX policy) | twist_mux (prio 100) | 20 Hz | v = 0.5 a0 m/s, w = 1.0 a1 rad/s, rate-limited 1.0 m/s^2 / 2.0 rad/s^2, then scaled to SafetyStatus limits (SPEC section 3) | planned |
| `/cmd_vel/nav` | geometry_msgs/TwistStamped | Nav2 `controller_server`, `behavior_server` (remapped from `cmd_vel`) | twist_mux (prio 50) | 20 Hz (controller_frequency) | m/s, rad/s | live |
| `/diff_drive_controller/cmd_vel` | geometry_msgs/TwistStamped | **twist_mux only** (`cmd_vel_out` remapped) | diff_drive_controller; orion_safety (geofence prediction, stuck rule) | as inputs arrive | m/s, rad/s. Nothing else may publish here | live |
| `/orion/estop` | std_msgs/Bool | operator / physical e-stop driver (must be **reliable** QoS) | twist_mux (lock, timeout 0.0 = fail-open for a dead publisher until a repeating driver exists), orion_safety, orion_hardware (true: ESTOP frame every cycle; false alone clears nothing: ESTOP_CLEAR only after orion_safety accepted `/orion/estop_reset`, docs/SAFETY.md decision 14) | on change; a button driver should repeat at >= 10 Hz | `true` latches ESTOP in orion_safety; `false` is required before a reset | live |
| `/orion/safety_status` | orion_msgs/SafetyStatus | orion_safety | orion_policy, orion_experiments, orion_hardware (ESTOP_CLEAR only once the state leaves ESTOP after a latch; cancels a wheel perturbation on HALT / ESTOP or when older than 0.5 s) | 50 Hz | state, flags, max_linear_speed m/s, max_angular_speed rad/s, max_current A per wheel (N.m in sim) | live |
| `/orion/terrain_state` | orion_msgs/TerrainState | orion_terrain | orion_safety (stuck, hard-stuck; optional; a fix younger than 0.5 s replaces the confidence test), orion_policy, orion_experiments trial runner (StartTrial needs body_speed_source EXTERNAL for 2 s) | 20 Hz | slip [-1, 1], effort_residual A (N.m sim), body_speed m/s, confidence [0, 1], time_since_fix s (SPEC section 4) | planned |
| `/orion/policy_status` | orion_msgs/PolicyStatus | orion_policy | orion_experiments (ends the trial ENTRAPPED on a rising escapes_exhausted) | 20 Hz | mode, entrapment_risk [0, 1], latent, policy_sha, obs_spec_hash, critic_gating, escapes_exhausted (escape library used up: IDLE, zero command, latched until the next SetMode; SPEC section 8) | live (orion_policy) |
| `/orion/wheel_perturbation` | orion_msgs/WheelPerturbation | fault-injection tool (Round 7, none yet) | orion_hardware (added to the commanded wheel velocities in write()) | on demand | offsets[4] rad/s per wheel FL FR RL RR (URDF sign), duration_s s. Off unless hardware parameter `perturbation_max_rad_s` > 0 (xacro / robot.launch.py argument, default 0); rejected above the bound or unless SafetyStatus is fresh and NOMINAL / LIMITED; cancelled by HALT, ESTOP, `/orion/estop`, stale SafetyStatus, deactivation; duration_s <= 0 cancels (docs/PROTOCOL.md) | reserved (hardware only; no Gazebo equivalent) |
| `/orion/rover_health` | orion_msgs/RoverHealth | orion_hardware | orion_safety (firmware faults; optional), orion_experiments | 10 Hz | battery_v V, fw_fault_flags bits, duty [-1, 1] | planned |
| `/orion/ground_truth_pose` | geometry_msgs/PoseStamped | orion_ground_truth (overhead camera + AprilTag) | orion_safety (geofence, hard-stuck progress; optional), orion_terrain | >= 20 Hz | m, frame `sandbox` | planned |
| `/odom_rf2o` | nav_msgs/Odometry | rf2o_laser_odometry | orion_terrain | 10 Hz (scan rate) | m, m/s; body-speed fallback (SPEC F4) | planned |
| `/joint_states` | sensor_msgs/JointState | joint_state_broadcaster | robot_state_publisher, orion_safety (heartbeat, current), orion_terrain | 100 Hz (controller_manager `update_rate: 100` in orion_control/config/controllers.yaml; joint_state_broadcaster publishes every update). **Required >= 100 Hz** (Round 7 wheel-fault detection); orion_sim_tests test_drive asserts >= 90 Hz in sim time from header stamps (measured 100.0 Hz) | position rad, velocity rad/s, **effort = A on hardware, N.m in Gazebo** | live |
| `/diff_drive_controller/odom` | nav_msgs/Odometry | diff_drive_controller | Nav2, AMCL, orion_safety (geofence fallback only) | 50 Hz | wheel odometry, frame `odom` -> `base_link`. Not body velocity under slip (SPEC F4) | live |
| `/imu/data` | sensor_msgs/Imu | Gazebo IMU via ros_gz_bridge; real IMU driver (hardware list pending) | orion_safety (heartbeat, tilt), orion_terrain | 100 Hz sim (`update_rate` 100 in orion_description/urdf/sensors/imu.xacro; test_drive asserts >= 90 Hz sim time). **Required 100 Hz** on the rover too (Round 7): choose the IMU driver rate accordingly | frame `imu_link`; orientation (if provided), angular_velocity rad/s, linear_acceleration m/s^2 (reads +9.8 z at rest) | live (sim) |
| `/scan` | sensor_msgs/LaserScan | Gazebo gpu_lidar via ros_gz_bridge; real lidar driver (pending) | slam_toolbox, AMCL, Nav2 costmaps, rf2o | 10 Hz | frame `lidar_link`, 720 samples, 360 deg, 0.12-12 m | live (sim) |
| `/clock` | rosgraph_msgs/Clock | ros_gz_bridge `parameter_bridge` (orion_bringup gazebo.launch.py) | every node with use_sim_time | sim step | sim only | live (sim) |
| `/robot_description` | std_msgs/String | robot_state_publisher | gz spawn (`create`), ros2_control_node (real rover) | latched | URDF from xacro with `hardware:=gazebo|serial` | live |
| `/tf`, `/tf_static` | tf2_msgs/TFMessage | robot_state_publisher (static + wheels), diff_drive_controller (`odom` -> `base_link`), AMCL (`map` -> `odom`) | everyone | 50-100 Hz | see Frames | live |
| `/map` | nav_msgs/OccupancyGrid | nav2 map_server / slam_toolbox | AMCL, Nav2 | latched | `orion_bringup/maps/orion_map.yaml` | live |
| `/initialpose` | geometry_msgs/PoseWithCovarianceStamped | operator / RViz / test | AMCL | on demand | frame `map` | live |

## Services and actions

| Name | Type | Server | Notes | Status |
| --- | --- | --- | --- | --- |
| `/orion/estop_reset` | orion_msgs/srv/EstopReset | orion_safety | `confirm` must equal `"CLEAR ESTOP"` exactly; refused while `/orion/estop` is still `true` or while a latched tilt HALT is still tilted. Clears ESTOP, tilt HALT and hard-stuck HALT | live |
| `/orion/set_mode` | orion_msgs/srv/SetMode | orion_policy | PolicyStatus.MODE_* | planned |
| `/orion/start_trial` | orion_msgs/srv/StartTrial | orion_experiments | starts the bag + metadata; refused unless TerrainState.body_speed_source has been EXTERNAL continuously for 2 s (parameter require_external_speed, council M6) | planned |
| `/orion/stop_trial` | orion_msgs/srv/StopTrial | orion_experiments | operator outcome; the runner itself ends a trial ENTRAPPED when SafetyStatus has FLAG_HARD_STUCK or PolicyStatus.escapes_exhausted rises | planned |
| `navigate_to_pose` | nav2_msgs/action/NavigateToPose | Nav2 bt_navigator | goals in `map` | live |
| `/lifecycle_manager_navigation/is_active` | std_srvs/srv/Trigger | Nav2 lifecycle manager | goals are rejected until true (M-006) | live |
| `/controller_manager/*` | controller_manager_msgs | controller_manager | spawners | live |

## Frames

| Frame | Parent | Published by | Notes |
| --- | --- | --- | --- |
| `map` | - | map_server / slam_toolbox | fixed world frame of the saved map |
| `odom` | `map` | AMCL (`map` -> `odom`) | continuous, drifts |
| `base_link` | `odom` | diff_drive_controller (`enable_odom_tf: true`) | chassis centre at wheel-axle height; x forward |
| `front_left_wheel`, `front_right_wheel`, `rear_left_wheel`, `rear_right_wheel` | `base_link` | robot_state_publisher from `/joint_states` | wheels at x = +-0.25 m, y = +-0.26 m (track 0.52 m), r = 0.12 m |
| `sensor_mast_link` | `base_link` | robot_state_publisher (static) | |
| `lidar_link` | `sensor_mast_link` | robot_state_publisher (static) | 0.15 m above the mast; measure on the rover (BRINGUP step 7) |
| `imu_link` | `base_link` | robot_state_publisher (static) | 0.05 m above base_link; axes must match base_link (BRINGUP step 6) |
| `sandbox` | - | orion_ground_truth | ground-truth frame of the 2 x 2 m sandbox; the geofence polygon is in this frame. Not connected to the TF tree (yet) |

## Units summary

| Quantity | Unit |
| --- | --- |
| linear speed | m/s |
| angular speed | rad/s |
| wheel position / speed | rad, rad/s |
| wheel effort | **A (motor current) on the rover, N.m (torque) in Gazebo**; TerrainState.effort_source says which |
| angles in config files | deg (named `*_deg`), everywhere else rad |
| time | s; stamps from the node clock (sim time in Gazebo) |
