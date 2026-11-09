# ORION build spec (council output, 2026-11-09)

Written by the planning council (Fable 5.1) after a critical review of the software plan.
This is the contract the packages are built against. Values marked GUESS must be replaced
with measured values once the hardware is known (see docs/HARDWARE.md). Changes to this file
are decisions: log them in docs/engineering-log/DECISIONS.md.

## 1. Council critique and the fixes adopted

- **F1 Recoverability margin has no ground truth on the real rover.** Renamed to
  **entrapment risk** rho = P(hard-stuck or belly-out within T_h = 10 s | observations, history,
  under the escape library E). Labelled in the learning env by branched rollouts (clone state,
  run E for 10 s, K = 4 branches, label = fraction that fail). On the rover it is never trained,
  only calibrated: reliability diagram, ECE and Brier score over the trial set. Never call it a
  "distance to entrapment".
- **F2 Training in a Bekker sim to fix over-trust in Earth sims is circular unless model-form
  error is randomised.** Per-episode multiplicative noise on thrust and resistance
  (LogU(0.7, 1.4) each); randomised gravity-soil coupling exponent beta (the disputed effect);
  policy inputs restricted to measured quantities; 8-D adaptation latent, and on the rover a
  linear probe on z-hat must separate the 4 cells (AUC > 0.8) or the adaptation claim is dropped.
- **F3 Kill test underpowered.** Pre-registered: 4 cells x 20 trials, primary failure-rate
  metric with one-sided Fisher exact test, secondary effective speed with Mann-Whitney. It can
  answer "does loose sand break a packed-tuned controller", not the ballast x soil interaction.
- **F4 Wheel odometry is not body velocity.** Under slip it is the numerator of the slip
  ratio. Sandbox truth = overhead camera + AprilTag (/orion/ground_truth_pose, >= 20 Hz);
  onboard fallback = lidar odometry (rf2o, /odom_rf2o); IMU only bridges gaps; confidence decays
  with time since the last fix. Policy is trained with the same degraded estimate.
- **F5 Current is not sinkage.** Use an effort residual against a flat-ground model calibrated
  on packed-light driving. If there is no current sensor, duty x V_bat / R is the fallback,
  flagged in TerrainState.effort_source.
- **F6 RMA is the right family** (cheap, no on-robot gradient steps, diagnosable latent), but the
  privileged vector is functional traction descriptors, not raw Bekker parameters, and the
  robust non-adaptive PPO baseline (B2) is mandatory. If the student can't beat B2 in sim, the
  adaptation claim is dropped before hardware.
- **F7 Nothing scientific is testable in Gazebo.** Replay harness: learning/ writes synthetic
  rosbag2 files from env rollouts; orion_terrain and orion_policy CI tests run on them and must
  reproduce env slip/sinkage. obs_spec hash embedded in ONNX metadata and checked at load.
- **F8 Safety as just another twist source is not enough.** Firmware enforces the hard current
  limit and 200 ms watchdog. orion_safety publishes limits that policy and escape obey, and only
  takes the mux on HALT/ESTOP. Stuck detection arms escape; it does not halt.

## 2. orion_msgs

See robot_ws/src/orion_msgs (TerrainState, SafetyStatus, PolicyStatus, RoverHealth; SetMode,
StartTrial, StopTrial, EstopReset). /joint_states effort = amps on hardware, N.m in Gazebo.

## 3. Observation spec

learning/orion_learning/obs_spec.py is the single source. Policy rate 20 Hz, ros2_control
100 Hz, firmware PID 200 Hz. History H = 40 steps (2 s). 20 observation floats per step
(prev cmd v/w, 4 wheel speeds, 4 wheel efforts, body accel x/y, yaw rate, roll, pitch, body
speed, speed confidence, slip left/right, sinkage proxy), normalised and clipped to [-1.5, 1.5].
Action: 2 floats in [-1, 1] -> v = 0.5 a0 m/s, w = 1.0 a1 rad/s, rate-limited 1.0 m/s^2 and
2.0 rad/s^2, published as TwistStamped on /cmd_vel/policy, then scaled to SafetyStatus limits.
Privileged (teacher only, 10): per side [DP coeff at s = 0.3, slip at DP = 0.1 W, z0/r, Rc/W],
m_total/14, g/9.81.

## 4. Terrain estimator (orion_terrain, 20 Hz)

- Side wheel-centre speed: v_L = v_x - w_z b/2, v_R = v_x + w_z b/2, b = 0.52 m.
- Slip per wheel: s_i = (r w_i - v_side) / max(|r w_i|, |v_side|, 0.02); + spinning, - skidding.
- Body speed priority: EXTERNAL > LIDAR_ODOM > IMU_ONLY. 1-D Kalman on v_x: predict with
  gravity-compensated a_x, Q = (0.1 m/s^2)^2 dt; update R = (0.02 m/s)^2 external,
  (0.05 m/s)^2 lidar. Wheel odometry is never a body-speed source.
- Confidence c = exp(-t_since_fix / 3 s) * (1 - min(1, |v_x,wheel - v_x| / 0.5)).
- Effort model I_hat = c0 + c1 |w| + c2 dw/dt + c3 sin(pitch) sign(w), least squares on
  5 min packed-light driving, stored in orion_terrain/config/effort_model.yaml with git SHA.
  Residual e_i = I_i - I_hat_i; sinkage_proxy = clip(mean(e_i) / I_ref, 0, 1),
  I_ref = I_stall - I_free.
- Low-pass tau = 0.25 s on slip and residual.

## 5. Safety rules (orion_safety, 50 Hz, thresholds in config/safety.yaml)

| rule | threshold | action |
| --- | --- | --- |
| tilt LIMITED | roll or pitch > 18 deg | v <= 0.2 m/s, w <= 0.5 rad/s |
| tilt HALT | > 25 deg for 0.2 s | HALT, zero twist on /cmd_vel/safety, lock |
| current LIMITED | any wheel > I_cont (GUESS 6 A) for 2 s | max_current = I_cont, v <= 0.3 |
| current HALT | any wheel > I_peak (GUESS 12 A) for 0.3 s | HALT, 5 s cool-down, then NOMINAL |
| firmware hard limit | 1.2 I_peak for 100 ms | clamp duty, fault flag (firmware only) |
| heartbeat (firmware) | no CMD frame for 200 ms | targets 0, PID off, coast |
| heartbeat (host) | /joint_states older than 300 ms or /imu/data older than 500 ms | HALT |
| geofence | keep-in polygon (2 x 2 m minus 0.15 m); predicted pose in 1 s outside -> LIMITED; outside -> HALT | |
| stuck | cmd v > 0.1 and body speed < 0.03 m/s for 3 s (confidence > 0.3), or slip_side > 0.8 for 3 s | flag stuck, policy arms ESCAPE; no halt |
| hard-stuck | stuck and < 0.1 m net progress over 20 s | HALT, outcome ENTRAPPED |
| e-stop | /orion/estop true (latched) | ESTOP, mux lock, firmware ESTOP frame; clear via EstopReset |

twist_mux inputs /cmd_vel/{estop,teleop,safety,escape,policy,nav}, priorities
255/200/150/120/100/50, timeouts 0.5 s, lock /orion/estop, output
/diff_drive_controller/cmd_vel (TwistStamped).

## 6. Learning environment (learning/orion_learning, NumPy, vectorised, dt 0.05 s, 4 substeps)

State per env: pose, body velocity (v_x, v_y, w_z), 4 wheel speeds, 4 sinkages, parameters.
Load W_i = m g / 4 with longitudinal transfer m h a_x / (2L), h = 0.15 m (GUESS), L = 0.50 m.

- Static sinkage z0 = [3W / ((3 - n) b (k_c/b + k_phi_eff) sqrt(D))]^(2/(2n+1)), b = 0.06 m, D = 0.24 m.
- Slip sinkage (GUESS): z_target = z0 (1 + a_s |s| / (1 - |s| + 0.05)), a_s in [0.3, 2.0].
  Sticky: z <- max(z_target, z - 0.3 |v_wheel| dt).
- Contact length l = r acos(1 - z/r), area A = b l.
- Thrust (Janosi-Hanamoto): H = (A c + W tan phi)(1 - (K/(|s| l))(1 - exp(-|s| l / K))) sign(s);
  for |s| < 1e-3 use H = (A c + W tan phi) |s| l / (2K) sign(s).
- Compaction resistance Rc = b (k_c/b + k_phi_eff) z^(n+1) / (n+1). Bulldozing omitted.
- Net wheel force F = m_fH H - m_fR Rc, m_fH, m_fR ~ LogU(0.7, 1.4).
- Lateral F_y = -mu_y W tanh(v_y / 0.05), mu_y in [0.3, 0.8].
- Wheel J dw/dt = tau - r H, J = 0.011 kg m^2, tau = clip(PI(w_cmd - w), +-tau_max),
  tau_max = 3.0 N m (GUESS). Torque fault: one wheel x U(0.3, 1) with p = 0.2.
- Body: m dv_x = sum F + m v_y w_z; m dv_y = sum F_y - m v_x w_z;
  I_z dw_z = sum(+-F b/2) + sum(+-F_y L/2), I_z = 0.9 kg m^2.
- Gravity coupling (GUESS, the disputed effect): k_phi_eff = k_phi (g/9.81)^beta, beta ~ U(0, 1).
  g ~ LogU(1.62, 9.81) in training.

Randomisation: k_c [0.5, 15] kN/m^(n+1); k_phi [400, 2500] kN/m^(n+2); n [0.7, 1.3];
c [0, 2.0] kPa; phi [25, 40] deg; K [0.01, 0.04] m; mass [12, 30] kg; per-side heterogeneity
p = 0.5; one loose patch of radius U(0.3, 0.6) m; action latency 1-4 steps; observation noise
(wheel 0.05 rad/s, effort 0.03 tau_max, accel 0.1 m/s^2, gyro 0.02 rad/s, speed fix 0.03 m/s);
fix dropouts U(0, 6) s with p = 0.3, bias random walk 0.02 m/s^2.

Termination: belly-out when any z >= 0.08 m; hard-stuck when |v_x| < 0.02 m/s for 10 s while
|r w_cmd| > 0.1 m/s. Episode 60 s, goal 1.5 m ahead (heading U(-30, 30) deg), 2 x 2 m fence.
Reward (GUESS weights): +10 progress, +10 goal, -20 belly-out, -10 hard-stuck, -0.02/step,
-0.5 mean slip^2, -1.0 sinkage proxy, -0.05 |da|^2, -5 leaving the fence.

Training: teacher PPO with privileged encoder (10 -> 8); student TCN over 40-step history ->
z-hat, DAgger-style; critic -> rho with BCE on branched-rollout labels, temperature-scaled,
ECE reported. Runtime gating: rho < 0.3 explore; 0.3-0.5 cap 0.2 m/s and probe; 0.5-0.7
retreat; > 0.7 ESCAPE. Exports: policy, adaptation module, critic ONNX + obs_spec_hash.

Baselines: B0 non-RL (cruise 0.3 m/s, slip governor: slip_side > 0.3 cuts v 20 %/s, recovers
5 %/s; stuck -> scripted escape). B1 oracle teacher (sim only). B2 robust PPO without z.
B3 student without critic gating.

## 7. Serial protocol

Frame [type u8][seq u8][len u8][payload <= 64 B][crc16 LE], CRC-16/CCITT-FALSE over
type..payload, COBS-encoded, 0x00 delimiter, little-endian. 921600 baud preferred, 115200 min.
CMD 0x01 (100 Hz): int16[4] w_target mrad/s, u8 flags (enable, brake_on_idle).
STATE 0x02 (100 Hz): u32 t_us, int32[4] position mrad, int16[4] w mrad/s, int16[4] current mA
(0x7FFF = n/a), int16[4] duty +-10000, u16 vbat mV, u8 faults, u8 status.
SET_PARAM 0x03 / PARAM_ACK 0x04: u8 id, f32 value. PING 0x05 / PONG 0x06: proto_ver,
fw_git_hash, param_crc. ESTOP 0x07 / ESTOP_CLEAR 0x08 (magic 0x0DE5C0DE). LOG 0x09 ASCII.
Firmware: 200 Hz PID, 200 ms watchdog, per-wheel current clamp. Host read(): STATE older than
100 ms -> ERROR.

## 8. Experiment protocol

Trial: 60 s traverse to a goal line 1.5 m away in the 2 x 2 m sandbox; cells
{light, heavy (+8 kg, GUESS)} x {packed, loose}. Soil prep recorded (rake 10 cm; packed = 3
plate-tamp passes; 1 kg drop-cone depth at 3 points, cell invalid outside its band).
Randomised order. Metrics: success, time-to-goal, entrapped, time-to-entrapment (censored at
60 s), recovery, distance, effective speed, mean slip, mean sinkage proxy, energy, max rho,
critic calibration. Metadata: git SHA, firmware hash, ONNX sha256, obs_spec_hash, cell,
ballast, cone depths, operator, timestamp, bag path.

Failure = outcome TIMEOUT, ENTRAPPED or SAFETY (pre-registered). SAFETY includes the current
HALT, whose threshold I_peak (12 A) is a GUESS until the motor drivers are known (section 5), so
the report lists SAFETY outcomes per cell next to every test. Critic calibration label at time
t (council M2): in an ENTRAPPED trial 1 if t_entrap - t <= 10 s, where t_entrap is the first
FLAG_STUCK of the stuck episode that ended in the hard-stuck HALT (not the HALT, which comes
~23 s after motion stops); in other trials 0 when >= 10 s of trial remain, else censored.

Phase-0 kill test (pre-registered, two-stage; D-008): B0 tuned on light-packed (cell C).
Stage 1 (pilot): 20 trials in each of the 4 cells. Early GO if for at least one loose cell the
failure rate is higher than in light-packed with one-sided Fisher exact p < 0.01, or the
effective speed is lower with one-sided Mann-Whitney p < 0.01. Stage 2 (if no early GO): extend
light-packed (C) and light-loose (D) to 40 trials each and test C vs D once more at p < 0.04.
GO if stage 2 passes; otherwise NO-GO and pivot to self-rescue. The effect sizes are reported
next to the decision, descriptively: failure-rate rise with a Newcombe hybrid-score 95 % CI and
median-speed drop with a seeded bootstrap 95 % CI. This is the council M3 rule; the switch
`EFFECT_SIZE_GATE` in orion_experiments/analysis.py (CLI `--effect-size-gate`) restores the
original D-008 rule, which also required a rise >= 0.25 or a drop >= 30 % (open decision,
section 9). heavy-loose differs from light-packed in two factors (ballast and soil), so a GO
that rests on heavy-loose alone is reported as a two-factor result: H1 is the C -> D transfer.

Power and false-GO rate of the implemented rule, simulated by `ros2 run orion_experiments
phase0_power` (20000 sessions per row, seed 20261109, Monte-Carlo SE <= 0.004; failures
Bernoulli; a failed trial is always slower than a successful one, so the speed test only sees
the failure counts unless successful runs also slow down on loose soil). C = light-packed,
D = light-loose, B = heavy-loose; stage-1 early-GO probability in brackets:

| rule | power, C 10 % / D 40 % / B 40 % | power, B 10 % | false GO, all cells 10 % | all 30 % |
| --- | --- | --- | --- | --- |
| ungated, failures only (**implemented default**; speed reported, not decisive) | 0.91 (stage 1: 0.51) | 0.89 | 0.017 | 0.029 |
| ungated, failures + speed (`--speed-in-decision`) | 0.93 (0.58) | 0.91 | **0.062** | **0.063** |
| ungated, every test at alpha / 2 | 0.87 (0.45) | 0.85 | 0.031 | 0.033 |
| gated, failures + speed (`--effect-size-gate`) | 0.81 (0.55) | 0.75 | 0.002 | 0.027 |
| gated, failures only | 0.80 (0.51) | 0.75 | 0.001 | 0.017 |

The 0.01 / 0.04 split bounds the false-GO rate at 0.05 only for one test per look. The rule
runs four tests at stage 1 (2 loose cells x 2 metrics) and two at stage 2, so the ungated rule
with the speed test exceeds 0.05 (0.062-0.063). The failure-only or alpha / 2 variants keep it
below 0.05, which is why the default decides on the failure rate alone (`SPEED_IN_DECISION =
False`); effective speed is still reported with its CI. See section 9. Reference (not the implemented rule): a plain two-sided Fisher test
at 0.05 on 10 % vs 40 % has exact power 0.49 at 20 per cell, 0.71 at 30, 0.85 at 40.

Main study: 4 cells x 4 controllers (B0, B2, student, student + critic) x 25 = 400 trials.
Primary (pre-registered, council m7): student + critic fails less often than B0, both pooled
over light-loose and heavy-loose (50 vs 50), one-sided Fisher at 0.05, unadjusted. Secondary:
the 4 comparisons in each of the 4 cells (16 tests), one-sided Fisher, Holm-corrected. Exact
power at n = 25 per cell: for B0 40 % vs 20 %, 0.07 at Holm's first step (0.05 / 16) and 0.35
unadjusted; 40 % vs 10 %: 0.29 and 0.73. The pooled primary at 50 vs 50 has 0.64 (40 % vs
20 %) and 0.96 (40 % vs 10 %). Per-cell results are descriptive unless the effect is large.

## 9. Open decisions for the user

- **Phase-0 decision rule (council M3), awaiting the user's choice.** Implemented default (b):
  GO on the one-sided failure-rate test at 0.01 / 0.04 alone, effect sizes and speed reported
  with CIs (`EFFECT_SIZE_GATE = False`, `SPEED_IN_DECISION = False`). Options, with the
  simulated numbers of section 8 (power at 10 % vs 40 %, false GO under H0):
  (a) failures + speed (`--speed-in-decision`): power 0.91-0.93, false GO 0.062-0.063, above
  the 0.05 the design promised; (b) ungated, failure rate only (the default): power
  0.89-0.91, false GO 0.017-0.029; (c) ungated, every test at alpha / 2: power 0.85-0.87, false
  GO 0.031-0.033; (d) the gated D-008 rule (`--effect-size-gate`): power 0.75-0.81, false GO
  0.002-0.027. The implementer's recommendation is (b): almost the same power as (a), and the
  0.05 bound holds. Whatever is chosen must be fixed before the first stage-1 trial.

- Ballast mass for "heavy"; I_cont / I_peak after the hardware inventory.
- Is an overhead camera + AprilTag acceptable for ground truth? Without it, slip truth is
  lidar-odometry grade (2-5 cm/s) and the kill test should use failure rate only.
