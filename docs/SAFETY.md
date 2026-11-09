# ORION safety

What stops the rover, where it is enforced, and how to clear it. The contract is
`docs/SPEC.md` section 5; this file records how it is implemented and the choices made where the
SPEC was silent. Code: `robot_ws/src/orion_safety` (logic in `monitor.py`, no ROS; the node in
`safety_node.py` only moves data in and out). Thresholds: `orion_safety/config/safety.yaml`
(rover) overlaid by `safety_sim.yaml` in Gazebo.

## Layers

| Layer | Enforces | Survives |
| --- | --- | --- |
| **Firmware** (motor controller, planned with orion_hardware) | 200 ms CMD watchdog (targets 0, PID off, coast); per-wheel current clamp at 1.2 I_peak for 100 ms (clamp duty, fault flag); ESTOP frame | host crash, ROS crash, USB unplugged |
| **Physical e-stop** | cuts motor power (hardware list pending) | everything above |
| **twist_mux** | one command source at a time by priority; `/orion/estop` lock masks every input below 255 | orion_safety crash (the lock acts on the raw topic) |
| **orion_safety** (host, 50 Hz) | publishes limits on `/orion/safety_status`; takes the mux with zero twists on HALT / ESTOP | policy, Nav2 or teleop bugs |
| **Command sources** (orion_policy) | obey `max_linear_speed`, `max_angular_speed` from SafetyStatus | - |

Host software can never be the last line: anything orion_safety does needs a running host. The
firmware watchdog and the physical e-stop are what make a host crash safe.

## States

| State | Meaning | Twists from orion_safety | Limits published |
| --- | --- | --- | --- |
| NOMINAL | nothing wrong (flags may still be set, e.g. STUCK) | none | 0.5 m/s, 1.0 rad/s, I_peak |
| LIMITED | slow down | none: sources must obey the limits | minimum over active rules |
| HALT | stop | zero on `/cmd_vel/safety` (prio 150) at 50 Hz | 0, 0, 0 |
| ESTOP | latched emergency stop | zero on `/cmd_vel/safety` and `/cmd_vel/estop` (prio 255) at 50 Hz | 0, 0, 0 |

Several rules can fire at once: the state is the most severe, every limit is the minimum, flags
are OR-ed and `reason` lists all of them, most severe first.

Teleop (priority 200) overrides a HALT on purpose: an operator must be able to drive the rover
back inside the geofence or off a slope. Nothing overrides ESTOP.

## Rules

Times are measured on the node clock from when a message **arrives** (not its header stamp), so a
sensor with a wrong clock cannot fake freshness. All durations are "continuously true for".

| Rule | Enter | Action | Exit |
| --- | --- | --- | --- |
| tilt LIMITED | max(abs(roll), abs(pitch)) > 18 deg | v <= 0.2 m/s, w <= 0.5 rad/s, FLAG_TILT | immediately at <= 18 deg |
| tilt HALT | > 25 deg for 0.2 s | HALT, FLAG_TILT, **latched** | reset service, accepted only when tilt <= 18 deg on a fresh IMU sample |
| current LIMITED | any wheel abs(effort) > I_cont (6 A GUESS) for 2 s | max_current = I_cont, v <= 0.3 m/s, FLAG_OVERCURRENT | all wheels <= I_cont for 2 s (release hold, SPEC silent) |
| current HALT | any wheel > I_peak (12 A GUESS) for 0.3 s | HALT, FLAG_OVERCURRENT | 5 s cool-down counted from the moment all wheels are <= I_peak; restarts if one exceeds it again; then NOMINAL without reset |
| heartbeat | `/joint_states` older than 0.3 s or `/imu/data` older than 0.5 s | HALT, FLAG_HEARTBEAT | as soon as both are fresh again |
| startup grace | a required input never received 10 s after orion_safety started | HALT, FLAG_HEARTBEAT ("since start") | when it arrives |
| geofence LIMITED | pose predicted 1 s ahead (unicycle arc with the twist on `/diff_drive_controller/cmd_vel`) is less than the margin inside the polygon | v <= 0.2, w <= 0.5, FLAG_GEOFENCE | prediction back inside |
| geofence HALT | pose itself less than the margin (0.15 m) inside the polygon, or outside | HALT, FLAG_GEOFENCE | back inside the margin (teleop can drive it back) |
| stuck | abs(cmd v) > 0.1 m/s and abs(body speed) < 0.03 m/s for 3 s, with the body speed trusted: a fix younger than 0.5 s (`time_since_fix` + message age) **or** confidence > 0.3; **or** side slip > 0.8 in the driven direction for 3 s | FLAG_STUCK only; orion_policy arms ESCAPE | immediately when neither holds |
| hard-stuck | stuck, the stuck episode is >= 20 s old, and net displacement over the last 20 s < 0.1 m | HALT, FLAG_HARD_STUCK, reason "ENTRAPPED", **latched** | reset service |
| firmware fault | RoverHealth.fw_fault_flags != 0 | FLAG_FIRMWARE_FAULT; HALT if a bit in `firmware_halt_mask` (ENCODER, UNDERVOLTAGE, ESTOP_LATCHED) | flags clear |
| e-stop | `/orion/estop` = true | ESTOP, **latched**; twist_mux lock | `/orion/estop` = false **and** reset with `confirm: "CLEAR ESTOP"` |

### Inputs

| Input | Required? | If missing |
| --- | --- | --- |
| `/joint_states`, `/imu/data` | required | HALT after the timeout once seen, or after the 10 s grace if never seen |
| `/orion/terrain_state` | optional | stuck and hard-stuck rules inactive (uses body_speed, confidence, slip_side, time_since_fix) |
| `/orion/ground_truth_pose` (frame `sandbox`) | optional, preferred | geofence falls back to wheel odometry; hard-stuck progress falls back to integrated terrain body speed |
| `/diff_drive_controller/odom` | optional fallback | geofence inactive when there is no pose at all |
| `/diff_drive_controller/cmd_vel` | optional | zero command assumed (no prediction, no stuck) |
| `/orion/rover_health` | optional | firmware-fault rule inactive |

Optional inputs older than their timeout (0.5 s; health 1 s) are treated as absent and never trip
anything. That is deliberate: the safety node must not halt the rover because an estimator is not
running. The price is that a dead ground-truth camera silently disables the precise geofence;
orion_experiments must check that the topic is alive before starting a trial.

## Decisions where the SPEC was silent or ambiguous

1. **Two stop paths for ESTOP.** twist_mux's lock acts on the raw `/orion/estop`, so it works
   even if orion_safety is dead. orion_safety additionally latches and publishes zeros at
   priority 255 on `/cmd_vel/estop`, so the rover stays stopped even if someone publishes
   `false` before the reset. Without the zeros, a lock alone leaves the drive controller on its
   last command until its 0.5 s `cmd_vel_timeout`. Checked: with both paths removed,
   `test_safety.py` fails ("wheels still turning 3 s after the e-stop").
2. **Reset requires `/orion/estop` = false.** Otherwise the reset would succeed while the
   twist_mux lock is still engaged, and the published state would be a lie.
3. **"lock" on tilt HALT = latched until reset.** A rover that tipped past 25 deg needs a human
   look before it moves again. Hard-stuck is latched for the same reason (the trial is over).
   One reset service clears all three latches, as the SPEC names only EstopReset.
4. **Current HALT cool-down** starts when the current is back under I_peak (it drops as soon as
   the wheels stop, so in practice 5 s after the HALT).
5. **Current LIMITED release hold** of 2 s, symmetric with the entry, so it does not toggle
   every tick around I_cont.
6. **Startup grace.** Required inputs that were never seen trip only 10 s after start; once seen,
   the normal 0.3 / 0.5 s timeouts apply. This lets launch order vary (Gazebo controllers take
   seconds) without halting forever, and still catches a sensor that never comes up. Before the
   grace runs out the rover can be driven without tilt protection (the brief's requirement);
   in practice it cannot move before `/joint_states` exists anyway.
7. **Geofence prediction uses the commanded twist**, not measured velocity. Under slip the body
   moves slower than commanded, so the prediction is conservative, and it needs no velocity
   estimate.
8. **Wheel odometry fallback for the geofence** assumes the rover started at `odom_start_pose`
   in the sandbox frame and adds 0.10 m to the margin. Under slip wheel odometry *overestimates*
   forward travel, which errs on the safe side when driving at a wall, but heading drift does not;
   treat it as a coarse guard. It is never used for hard-stuck progress (spinning wheels would
   look like progress, SPEC F4).
9. **Stuck by slip is direction-aware.** Slip is signed (+ spinning when driving forward), so
   when reversing, spinning shows as negative slip. With abs(cmd v) > 0.1 the rule tests
   `slip * sign(cmd v) > 0.8`; otherwise (turning in place) `abs(slip) > 0.8`. A locked wheel
   skidding forward (slip ~ -1 while driving forward) is not stuck.
10. **Hard-stuck episode.** The 20 s window counts only from the moment the stuck flag first rose
    and ends when the rover makes >= 0.1 m net progress. A rover that was parked for minutes and
    then gets stuck is not declared entrapped after 3 s, and escape manoeuvres that toggle the
    stuck flag do not restart the clock.
11. **Firmware faults.** OVERCURRENT and WATCHDOG are handled by the firmware and transient: flag
    only. ENCODER, UNDERVOLTAGE and ESTOP_LATCHED halt while present.
12. **Gazebo overrides** (`safety_sim.yaml`): effort is torque in N.m in Gazebo, measured up to
    6.7 N.m in an in-place spin, so the sim uses 10 / 20 N.m instead of 6 / 12 A; the geofence is
    off because the Gazebo worlds are not the sandbox and have no ground-truth pose.
13. **Stuck by speed trusts a fresh fix without confidence** (council M7, `stuck_fresh_fix_s`
    0.5 s). The SPEC confidence is c = exp(-t/3)(1 - |v_wheel - v_x| / 0.5). A rover that is
    stuck while its wheels spin at >= 0.5 m/s rim speed has |v_wheel - v_x| >= 0.5, so c = 0
    and the speed rule went blind exactly when it mattered. A body-speed fix (overhead camera,
    or lidar odometry: TerrainState.time_since_fix counts any non-IMU fix) younger than 0.5 s
    measures the body directly, so the rule only needs the confidence when the fix is stale.
    The fix age is `time_since_fix` plus the age of the TerrainState message itself, so a
    message that stops arriving does not stay "fresh" for the 0.5 s terrain timeout. Stale fix
    and low confidence: the rule stays disabled, as before (an IMU-only speed must not declare
    the rover stuck). FIX-B changes the confidence formula itself (Kalman std instead of the
    wheel disagreement); this rule does not depend on which formula is used.
14. **`/orion/estop` false sends ESTOP_CLEAR to the firmware at once** (council m1). In
    `orion_hardware/src/orion_system.cpp` (`on_estop`), true sends the ESTOP frame and repeats
    it every cycle; false sends ESTOP_CLEAR immediately, before anyone calls
    `/orion/estop_reset`. So the firmware is ready to drive again before the host-side reset.
    It is still safe because nothing commands the wheels until the reset: orion_safety keeps
    its ESTOP latched and publishes zero twists at priority 255 on `/cmd_vel/estop` at 50 Hz,
    which beats every other twist_mux input, so the drive controller sends zero wheel targets.
    The residual risk is a dead orion_safety: then releasing the e-stop topic lets the
    highest remaining source (teleop, policy, Nav2) drive at once, without the reset. Gating
    ESTOP_CLEAR on the reset would need a service or topic from orion_safety to
    orion_hardware; not done, recorded here.
15. **twist_mux e-stop lock timeout 0.0 is fail-open for a dead e-stop publisher** (council
    m2). With timeout 0.0 the lock acts on the last `/orion/estop` value forever, so a
    publisher that dies (or never starts) leaves every input unmasked and nothing notices.
    twist_mux treats an expired lock as locked, so a timeout would be fail-safe, but today
    `/orion/estop` is published once by hand; any timeout would lock the rover 0.5 s after
    each release. When an e-stop driver republishes `/orion/estop` at >= 10 Hz, set the lock
    timeout to 0.5 s (`orion_control/config/twist_mux.yaml` says the same).
16. **A sim clock that never starts is reported** (council m11). With `use_sim_time` and no
    `/clock`, the node clock reads 0, the 50 Hz ROS-time timer never fires and no
    SafetyStatus is published (orion_policy then refuses to drive, so this fails safe but
    silently). A steady-clock timer logs a throttled warning once the clock is still 0 after
    `clock_warn_after_s` (5 s).

## Known gaps (threat list)

What the safety chain does **not** cover. Each item is deliberate or waiting for hardware.

| Gap | Consequence | Mitigation now | Fix |
| --- | --- | --- | --- |
| Nav2 (`/cmd_vel/nav`) and teleop (`/cmd_vel/teleop`) are not gated by SafetyStatus (council m3) | they ignore LIMITED speed caps and stuck flags; only orion_policy scales its commands to `max_linear_speed` / `max_angular_speed` | HALT masks Nav2 (priority 50 < 150); teleop overrides HALT on purpose (200 > 150, see States); ESTOP masks both; the firmware current clamp and watchdog still apply | a twist filter between twist_mux and the controller that applies SafetyStatus limits to every source |
| Dead e-stop publisher (m2) | pressing a software e-stop does nothing; the twist_mux lock stays open | physical e-stop; operator watches SafetyStatus | repeating e-stop driver, then lock timeout 0.5 s (decision 15) |
| ESTOP_CLEAR before the reset (m1) | firmware re-enabled before the host-side reset | priority-255 zeros from orion_safety until EstopReset (decision 14) | gate ESTOP_CLEAR on the reset |
| Optional inputs silently absent | no geofence without a pose, no stuck rule without TerrainState | orion_experiments refuses to start a trial without EXTERNAL body speed for 2 s (council M6) | - |
| I_cont / I_peak are GUESSES | a current HALT (trial outcome SAFETY, counted as a failure in the kill test) may fire too early or too late | values in safety.yaml flagged GUESS; BRINGUP step 4 | measured motor and driver limits |

## Operating

```bash
# E-stop from a terminal (reliable QoS is the ros2 CLI default)
ros2 topic pub --once /orion/estop std_msgs/msg/Bool "{data: true}"
# Release and reset
ros2 topic pub --once /orion/estop std_msgs/msg/Bool "{data: false}"
ros2 service call /orion/estop_reset orion_msgs/srv/EstopReset "{confirm: 'CLEAR ESTOP'}"
# Watch the state
ros2 topic echo /orion/safety_status --field reason
```

## Tests

- `orion_safety/test/test_monitor.py`: every rule above, entry and exit, durations to one tick,
  cool-down, latches and reset, grace period and launch order, priority of simultaneous rules,
  the fresh-fix stuck rule (spinning at 0.5 m/s rim speed with confidence 0 and a 0.05 s old
  fix flags STUCK; stale fix and low confidence do not; message age counts).
- `orion_safety/test/test_safety_node.py`: the stalled-clock warning with and without
  `use_sim_time`.
- `orion_sim_tests/test/test_safety.py` (headless Gazebo): policy drives through twist_mux;
  e-stop stops the wheels within 0.5 s sim while the policy keeps publishing; reset is refused
  with the wrong phrase or while the e-stop is held, then the policy drives again.

## Only verifiable on the real rover

Firmware watchdog and current clamp, physical e-stop, real I_cont / I_peak, IMU orientation vs
accelerometer fallback, ground-truth latency. See `docs/BRINGUP.md`.
