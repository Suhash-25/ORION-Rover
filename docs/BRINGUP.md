# Real-rover bring-up checklist

Run this, in order, before the rover drives on its own and again after any hardware change.
Each step has the exact command and a pass criterion; record the measured value in the sign-off
table at the end and commit it. Nothing here can be checked in Gazebo.

Conventions: REP 103 (x forward, y left, z up). Wheel order FL, FR, RL, RR. Wheel radius
r = 0.12 m and track b = 0.52 m are the URDF / `controllers.yaml` values until step 3 and step 8
replace them with measurements.

## 0. Setup

- Rover **on blocks** (wheels off the ground) for steps 1-4 and 6; physical e-stop within reach.
- One terminal per command below; every terminal: `source robot_ws/install/setup.bash`.
- The IMU and lidar drivers depend on the hardware list (docs/HARDWARE.md). Until they exist,
  orion_safety HALTs 10 s after start because `/imu/data` never arrives; that is intended.

- Give the firmware board a stable name: install the udev rule
  `robot_ws/src/orion_bringup/config/99-orion.rules` (instructions inside; it links
  `/dev/orion_fw` by USB serial number) and check `ls -l /dev/orion_fw`.
- The drive-train values are launch arguments of `robot.launch.py`, forwarded to the xacro:
  `port`, `baud`, `encoder_cpr`, `gear_ratio`, `kt`, `motor_resistance`, `brake_on_idle`
  (defaults are the GUESSES in `ros2_control.xacro`; `ros2 launch orion_bringup robot.launch.py
  --show-args` lists them). Steps 3 and 5 replace the guesses with measurements.

```bash
ros2 launch orion_bringup robot.launch.py port:=/dev/orion_fw \
  encoder_cpr:=<counts> gear_ratio:=<ratio> motor_resistance:=<ohm> \
  imu_launch:=/path/to/imu_driver.launch.py lidar_launch:=/path/to/lidar_driver.launch.py
ros2 control list_controllers      # pass: joint_state_broadcaster and diff_drive_controller active
ros2 topic echo /orion/safety_status --field state   # pass: 0 (NOMINAL)
```

Teleop commands below go to `/cmd_vel/teleop` (twist_mux priority 200, stamped with `now`):

```bash
# drive: replace X and W, Ctrl-C to stop (the mux and the controller time out after 0.5 s)
ros2 topic pub -r 10 /cmd_vel/teleop geometry_msgs/msg/TwistStamped \
  "{header: {stamp: now, frame_id: base_link}, twist: {linear: {x: X}, angular: {z: W}}}"
```

## 1. Physical e-stop

1. Drive on blocks at X = 0.2, W = 0.
2. Press the physical e-stop.

Pass: all wheels stop within 0.5 s (phone video at 60 fps: <= 30 frames from press to stop);
`ros2 topic echo /joint_states --field velocity` reads 0 within 0.5 s; motors stay unpowered
while the teleop command keeps publishing. Release: motors must **not** restart until the host
reset (`/orion/estop` false + `/orion/estop_reset`).

Then the software e-stop:

```bash
ros2 topic pub --once /orion/estop std_msgs/msg/Bool "{data: true}"
```

Pass: wheels stop within 0.5 s while teleop is still publishing; SafetyStatus.state = 3.
Clear: `ros2 topic pub --once /orion/estop std_msgs/msg/Bool "{data: false}"`. Pass: RoverHealth
`fw_fault_flags` still has ESTOP_LATCHED (16): the firmware stays latched until the reset.
Then `ros2 service call /orion/estop_reset orion_msgs/srv/EstopReset "{confirm: 'CLEAR ESTOP'}"`.
Pass: ESTOP_LATCHED clears within 0.1 s and the orion_hardware log says "orion_safety accepted
the e-stop reset: ESTOP_CLEAR sent to firmware".

## 2. Wheel direction

```bash
ros2 topic echo /joint_states --field velocity     # while publishing X = 0.1, W = 0
```

Pass: all four velocities **positive**, each within 10 % of v / r = 0.83 rad/s, and every wheel
visibly turns so the rover would move forward. Then X = 0, W = 0.5: left wheels (FL, RL)
negative, right wheels (FR, RR) positive, magnitude ~ 0.5 x 0.26 / 0.12 = 1.08 rad/s.
Fail: flip the motor polarity / encoder sign in the firmware config, never in the URDF.

## 3. Encoder ticks per revolution and effective radius (2 m)

a. On blocks, mark one wheel and the chassis. Read positions, turn the wheel by hand exactly
10 revolutions forward, read again:

```bash
ros2 topic echo --once /joint_states --field position
```

Pass: that wheel's position changed by 62.83 rad +- 1 % (10 x 2 pi). Otherwise
ticks_per_rev_new = ticks_per_rev_old x delta_reported / 62.83. Repeat for every wheel.

b. On flat concrete, tape a 2.000 m straight line. Read positions, drive X = 0.1 until the
reference mark is exactly on the 2.000 m line, read again.

Pass: every wheel's delta = 2.000 / 0.12 = 16.67 rad +- 2 %. Otherwise the effective radius is
r_eff = 2.000 / mean(delta) m: set `wheel_radius` in `orion_control/config/controllers.yaml`
(and the URDF constant) and log the change.

## 4. Current in amps

The rover's `/joint_states` effort must be motor **current in A** (Gazebo uses N.m).

```bash
ros2 topic echo /joint_states --field effort
ros2 topic echo /orion/rover_health --field current_sensed
```

1. On blocks at X = 0.3: note the free-running current per wheel (I_free).
2. On the ground at X = 0.3 with a DC clamp meter on one motor lead: compare.

Pass: `current_sensed: true` and the reported effort matches the clamp meter within 10 % or
0.2 A, whichever is larger. If there is no current sensor, `current_sensed: false` and
TerrainState.effort_source = 2 (DUTY_ESTIMATE). Record I_free, the stall current from the motor
datasheet, and pick I_cont / I_peak for `orion_safety/config/safety.yaml` (SPEC section 9).

## 5. Cable-pull watchdog

1. On blocks, drive at X = 0.2.
2. Pull the USB/serial cable between host and motor controller.

Pass: the firmware stops driving within 200 ms (video: <= 12 frames at 60 fps from the cable
leaving to the wheels starting to coast); wheels do not restart when the cable is plugged back
in until new CMD frames arrive; after reconnect `ros2 topic echo --once /orion/rover_health
--field watchdog_trips` has incremented. On the host, orion_safety reports HALT with
FLAG_HEARTBEAT (`/joint_states` stale) within 0.3 s.

## 6. IMU axes

```bash
ros2 topic echo /imu/data --field linear_acceleration
ros2 topic echo /imu/data --field angular_velocity
ros2 topic echo /orion/safety_status --field reason
```

Pass, all of:
- Level: z = +9.8 +- 0.3, abs(x) and abs(y) < 0.3 m/s^2.
- Nose up (front wheels on a 10 cm block): x > 0 (pitch negative in REP 103).
- Left side up: y > 0 (roll positive).
- Turning left (W = +0.5 on the ground): angular_velocity z > 0.
- Tilt by hand past 18 deg: reason says `tilt`, state 1 (LIMITED). Past 25 deg for more than
  0.2 s: state 2 (HALT) that stays after levelling, until `/orion/estop_reset`.
- `header.frame_id` = `imu_link`. If the IMU is mounted rotated, fix `rpy` in
  `orion_description/urdf/sensors/imu.xacro`, not the driver signs.

## 7. Lidar offset

Measure the lidar's optical centre relative to `base_link` (chassis centre at axle height) with
a tape, and compare with the URDF:

```bash
ros2 run tf2_ros tf2_echo base_link lidar_link
```

Pass: URDF within 1 cm of the tape in x, y, z; fix `lidar.xacro` / `mast.xacro` otherwise.
Then stand a flat board across the +x axis at a taped distance D = 1.000 m from `base_link`:

```bash
ros2 topic echo --once --csv /scan | python3 -c "
import math, sys
f = sys.stdin.read().strip().split(',')
amin, inc, ranges = float(f[3]), float(f[5]), [float(r) for r in f[10:]]
i = round((0.0 - amin) / inc)
print('range at 0 deg:', ranges[i])"
```

Pass: range at 0 deg = D - x_lidar +- 0.02 m, and the board's closest point is within +-1 deg
of 0 deg (yaw mounting error; fix `rpy` of the lidar joint otherwise).

## 8. 360 deg spin: wheel_separation_multiplier

Skid steer scrubs when turning, so the effective track is wider than 0.52 m. On the trial
surface (packed sand) **and** on concrete for reference:

1. Mark the rover's heading on the floor. Read wheel positions:
   `ros2 topic echo --once /joint_states --field position`.
2. Turn in place with X = 0, W = 0.5 until the rover has turned exactly 360 deg (stop on the
   mark). Read positions again.
3. Odometry's rotation is theta_odom = r x (dphi_R - dphi_L) / (0.52 x m_old), with dphi_L, dphi_R
   the mean position change of the left (FL, RL) and right (FR, RR) wheels and m_old the
   current `wheel_separation_multiplier`.
4. Set m_new = m_old x theta_odom / (2 pi) in `orion_control/config/controllers.yaml`.

Pass after the update: repeat the spin and
`ros2 topic echo --once /diff_drive_controller/odom --field pose.pose.orientation` before/after
shows 360 deg +- 5 deg (yaw back to the start value within 5 deg). Record both surfaces' values;
the experiments use the sand value.

## 9. Slip near 0 on concrete

Needs orion_terrain and a body-speed source (ground truth or lidar odometry).

```bash
ros2 topic echo /orion/terrain_state --field slip_side      # while driving X = 0.2 straight
ros2 topic echo /orion/terrain_state --field confidence
```

Pass: after 1 s of steady driving on concrete, both `slip_side` values stay within +-0.05 for
3 s with confidence > 0.5, and orion_safety never raises FLAG_STUCK. A biased slip (e.g. +0.1 on
concrete) means r or the speed source is wrong: redo step 3 before trusting any sand result.

## 10. IMU forward acceleration vs camera truth (council m9)

Needs orion_ground_truth (overhead camera + AprilTag) running, on the trial surface, on flat
ground. The terrain estimator predicts body speed from the IMU's forward acceleration between
camera fixes, so a scaled or biased a_x becomes speed drift in every fix dropout. (Gazebo's IMU
integrates 23 % low; that is a simulator artefact and is why this needs the real rover.)

```bash
ros2 bag record -s mcap -o imu_check /imu/data /orion/ground_truth_pose
# in another terminal, after the recording has started:
#   1. stand still for 10 s (static bias), rover level, nobody touching it;
#   2. drive straight at X = 0.2, W = 0 for about 10 s (2 m), then stop; Ctrl-C the recording.
ros2 run orion_experiments imu_check imu_check     # --static-s 10 --window-s 2 are the defaults
```

The tool takes the static bias of a_x from the first 10 s (it refuses if the camera saw the
rover move > 5 mm then), integrates (a_x - bias) over the acceleration to cruise speed (from
0.5 s before motion to 2 s after) and divides by the camera speed at the end of that window.

Pass: `PASS`, i.e. integrated IMU speed within 5 % of the camera speed; static bias within
+-0.3 m/s^2 (step 6) and the printed run length about 2 m. Fail: check the IMU's
accelerometer range / scale setting and its mounting (`imu.xacro` rpy), then repeat. Record
the ratio; the terrain estimator's IMU prediction is not trusted until this passes.

## Sign-off

| Step | Measured | Pass? | Date / operator | Commit with config change |
| --- | --- | --- | --- | --- |
| 1 physical e-stop | stop time __ s | | | |
| 1 software e-stop | stop time __ s | | | |
| 2 wheel direction | FL __ FR __ RL __ RR __ rad/s | | | |
| 3a ticks per rev | __ rad per 10 rev (per wheel) | | | |
| 3b effective radius | r_eff = __ m | | | |
| 4 current | effort __ A vs meter __ A; I_free __ A | | | |
| 5 cable-pull watchdog | __ ms | | | |
| 6 IMU axes | signs ok / tilt LIMITED / HALT latched | | | |
| 7 lidar offset | x __ y __ z __ m; range error __ m; yaw __ deg | | | |
| 8 spin multiplier | sand __ / concrete __ | | | |
| 9 slip on concrete | mean slip L __ R __ | | | |
| 10 IMU vs camera | bias __ m/s^2; ratio __ (pass within 0.95-1.05) | | | |
