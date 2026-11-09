# ORION hardware inventory (to be filled in)

The rover's parts list is **not known yet**. The software is written so that only the places
listed here change when it is: the firmware HAL (`firmware/src/`), a few xacro arguments and
firmware parameters. Every value the code currently uses for an unknown is a GUESS and is
listed below with where it lives. Please answer each question (edit this file: replace
`ANSWER:` lines) and the matching config is a small, mechanical change.

## Where the answers go

| what | file | how |
| --- | --- | --- |
| pins, motor/encoder polarity, current-sense and battery scaling | `firmware/src/board_config.hpp` | edit, reflash |
| a different microcontroller | `firmware/src/main.cpp` (implement `orion_fw::Hal`, ~120 lines) + `platformio.ini` env | new board file |
| serial port, encoder CPR, gear ratio, kt, motor resistance, brake/coast | xacro args of `orion_description/urdf/control/ros2_control.xacro` (`hardware:=serial port:=... encoder_cpr:=... gear_ratio:=... kt:=... motor_resistance:=... brake_on_idle:=...`) | launch argument |
| wheel-to-firmware slot mapping and sign | per-joint `motor_index` / `direction` in the same xacro | edit |
| PID gains, firmware current limit, undervoltage threshold | firmware params (docs/PROTOCOL.md section 3), defaults in `firmware/lib/orion_core/params.hpp` | edit defaults, reflash |
| I_cont / I_peak used by the host safety rules | `orion_safety/config/safety.yaml` (owned by orion_safety) | edit |
| wheel radius / separation | `orion_control/config/controllers.yaml`, `orion_description/urdf/common/constants.xacro` | edit |

---

## A. Microcontroller

**Q:** Which board drives the motors? (model, or "none yet")
ANSWER:

**Why it matters:** the HAL (PWM, encoder decoding, ADC, serial) is board-specific; everything
else is portable and host-tested.

**Default chosen: Teensy 4.1** (`firmware/src/main.cpp`, `[env:teensy41]`). Reasons: 600 MHz
Cortex-M7 with a double-precision FPU (the 200 Hz PID for 4 wheels is ~1 % CPU); native USB at
480 Mbit/s so the 921600-baud budget is irrelevant; every pin is interrupt-capable (4
quadrature encoders on interrupts) and it has 4 hardware quadrature decoders (ENC1-4) if the
count rate is high; 12-bit PWM at 20 kHz on many pins; mature PlatformIO support. Rejected
defaults: ESP32 (good PCNT encoder units, but the ADC is non-linear and Wi-Fi/RTOS jitter is
noise we do not need), Arduino Mega/Uno (8-bit, no FPU, too few interrupt pins for 8 encoder
channels), RP2040 (fine, PIO quadrature decoders; a reasonable alternative). **The Teensy pin
map is a placeholder** - every pin in `board_config.hpp` is marked CONFIRM. The board build
(`pio run -e teensy41`) has not been compiled in CI or the sandbox (no toolchain download);
`main.cpp` was only syntax-checked against stub headers.

## B. Motor drivers

**Q1:** Driver model(s) and interface: PWM + DIR, IN1/IN2 (H-bridge), RC-PWM/servo, or
serial/CAN (e.g. RoboClaw, ODrive, VESC)?
ANSWER:

**Q2:** Can the driver **coast** (outputs floating) as well as **brake** (outputs shorted)? Which
pin state does what?
ANSWER:

**Q3:** Logic voltage (3.3 V or 5 V inputs) and maximum PWM frequency.
ANSWER:

**Why it matters:** `Hal::set_motor(wheel, duty, brake)` is written for PWM + DIR. The SPEC's
watchdog says "coast unless brake_on_idle"; drivers of the Cytron MDD10A class brake whenever
PWM is low and **cannot coast**, so on those the watchdog always brakes (stated in
`main.cpp`). A smart driver (RoboClaw/ODrive) would replace the PID too: then the firmware core
becomes a protocol bridge and the PID params are unused - tell us before building that.

**Changes:** `board_config.hpp` (`kMotors[].pwm/dir`, `kPwmFrequencyHz`), `set_motor()` for a
different interface, `brake_on_idle` xacro arg.

## C. Current sensing (most important for the research)

**Q1:** Is motor current measured per wheel? With what (driver current-sense output,
ACS712/ACS723, INA226/INA240 shunt amplifier, smart driver telemetry)?
ANSWER:

**Q2:** Sensor scaling: volts per amp, zero-current offset, bandwidth / filtering, which pin.
ANSWER:

**Why it matters:**
1. **Sinkage signal (SPEC F5):** the terrain estimator's sinkage proxy is an effort residual.
   With a sensor, `/joint_states` effort is measured amps. Without one, the host estimates
   `I = (duty * V_bat - kt * gear_ratio * w) / R` and `RoverHealth.current_sensed` is false, so
   orion_terrain flags `effort_source` as estimated. The estimate cannot see load the PID has not
   yet reacted to, and with `kt = 0` (default) it is the SPEC's plain `duty * V_bat / R`, which
   grows with speed even on flat ground; the residual model's `c1 |w|` term absorbs part of that.
2. **Firmware hard current limit (SPEC section 5, F8):** the per-wheel clamp at 1.2 x I_peak for
   100 ms needs a measured current. **Without sensing the firmware cannot enforce it**; the only
   protection left is `max_duty` and the driver's own limit. For a sand-entrapment study, where
   stalled wheels are the expected case, we strongly recommend a current sensor per wheel or a
   driver with a hardware current limit.

**Changes:** `board_config.hpp` `kMotors[].current_pin`, `kCurrentOffsetV`,
`kCurrentAmpsPerV`. Nothing on the host: it detects `0x7FFF` per wheel.

## D. Motors and gearbox

**Q1:** Motor model, nominal voltage, gear ratio (motor revs per wheel rev).
ANSWER:

**Q2:** Datasheet or measured: terminal resistance R (ohm), torque constant kt (N m/A at the
motor shaft) or speed constant, no-load speed, no-load current, stall current.
ANSWER:

**Why it matters:** `gear_ratio` converts encoder counts to wheel radians (wrong value = wrong
odometry and wrong speed control); `R` and `kt` set the effort estimate when current is not
sensed; stall current sets the realistic `I_peak` (SPEC GUESS 12 A) and `I_cont` (GUESS 6 A),
and the firmware limit `current_limit_a` = 1.2 x I_peak (default 14.4 A). No-load speed sets the
feed-forward gain `kff` ~= 1 / (free wheel speed in rad/s at full duty).

**Measure R:** lock the wheel, apply a low duty `d`, read battery voltage `V` and current `I`:
`R = d * V / I`. **Measure kt (= ke in SI):** spin unloaded at duty `d`, read wheel speed `w`
and current `I`: `kt = (d * V - I * R) / (gear_ratio * w)`.

**Changes:** xacro args `gear_ratio`, `kt`, `motor_resistance`; firmware param defaults
`kff`, `kp`, `ki`, `current_limit_a`; safety.yaml I_cont / I_peak.

## E. Encoders

**Q1:** Type (quadrature magnetic / optical, hall-only), mounted on the motor shaft or the
wheel shaft?
ANSWER:

**Q2:** Counts per revolution as specified (lines/PPR) and whether that is before or after x4
quadrature decoding. Supply and output voltage (3.3 V / 5 V).
ANSWER:

**Why it matters:** `encoder_cpr` is **counts per motor revolution after x4 decoding** (a
500-line encoder is 2000). If the encoder is on the wheel shaft, set `gear_ratio:=1`. 5 V
encoder outputs need level shifting for a Teensy 4.1 (3.3 V, **not 5 V tolerant**). Count rate:
interrupt decoding on the Teensy is comfortable up to roughly 100-200 k counts/s in total;
`cpr x motor rpm / 60 x 4 wheels` above that needs the hardware ENC modules (QuadEncoder
library) instead of the PJRC `Encoder` library. Positive wheel rotation (URDF axis +y, i.e.
forward on both sides) must count up: fix per wheel with `invert_encoder` / `invert_motor` in
`board_config.hpp` (the firmware PID needs duty and counts to agree), then with the joint
`direction` param if the host still sees a sign flip.

**Changes:** xacro `encoder_cpr` (pushed to the firmware on every configure),
`board_config.hpp` pins and invert flags.

## F. Battery and power

**Q1:** Chemistry, cell count, nominal and minimum voltage, capacity.
ANSWER:

**Q2:** Is battery voltage wired to an ADC pin (through which divider)?
ANSWER:

**Q3:** Is there a physical e-stop that cuts motor power (independent of software)?
ANSWER:

**Why it matters:** `vbat_min_v` (default 10.5 V = 3S LiPo at 3.5 V/cell) drives the
UNDERVOLTAGE fault; the effort estimate uses the measured vbat or `nominal_battery_v`
(default 12 V). The software e-stop (`/orion/estop` -> firmware latch) depends on the USB link
and the firmware; a hardware e-stop in the motor power line is the only stop that survives a
crashed computer, and it is required before field trials.

**Changes:** `board_config.hpp` `kVbatPin`, `kVbatDividerRatio`; firmware param `vbat_min_v`;
OrionSystem param `nominal_battery_v`.

## G. IMU

**Q:** Model, interface (I2C/SPI/USB), mounting position and orientation, ROS 2 driver, rate.
ANSWER:

**Why it matters:** tilt safety (18/25 deg) and the host heartbeat need `/imu/data`; the
terrain estimator bridges speed-fix gaps with gravity-compensated acceleration (SPEC section
4). The URDF has `imu_link` on `base_link`: its pose must match the real mounting. An IMU on
the microcontroller would need a new protocol message (not in v1); a USB/I2C IMU on the
onboard computer with its own ROS driver is the simpler path.

**Changes:** `orion_description/urdf/sensors/imu.xacro` pose; bringup launch for the driver.

## H. Lidar

**Q:** Model, interface, ROS 2 driver, mounting height, scan rate.
ANSWER:

**Why it matters:** lidar odometry (rf2o, `/odom_rf2o`) is the onboard body-speed source when
the overhead camera is not used (SPEC F4); its quality bounds the slip estimate.

**Changes:** `orion_description/urdf/sensors/lidar.xacro` pose; bringup launch.

## I. Onboard computer

**Q:** Model (Raspberry Pi 5, Jetson Orin, NUC, ...), OS, how the microcontroller connects
(USB / UART pins), power source.
ANSWER:

**Why it matters:** ROS 2 Jazzy needs Ubuntu 24.04; the control loop runs at 100 Hz in
`ros2_control_node`. Give the microcontroller a stable device name with a udev rule (e.g.
`/dev/orion_fw` by USB serial number) and pass it as `port:=/dev/orion_fw`; the user needs
the `dialout` group. A UART on Pi GPIO is 3.3 V and works at 921600 with `kUseUart = true`.

## J. Geometry and mass

**Q:** Measured wheel radius (loaded, on sand), track width (wheel centre to wheel centre),
wheelbase, total mass with and without ballast.
ANSWER:

**Why it matters:** wheel radius and separation set the diff-drive kinematics and every slip
number (SPEC section 4 uses r = 0.12 m, b = 0.52 m); mass enters the learning environment.

---

## Bring-up checklist (once A-F are answered)

1. Fill `board_config.hpp`, flash with `pio run -e <board> -t upload`. Wheels off the ground.
2. Rehearse without the board: `socat pty,raw,echo=0,link=/tmp/fw pty,raw,echo=0,link=/tmp/host`,
   `ros2 run orion_hardware fake_firmware --port /tmp/fw`, then launch with `port:=/tmp/host`.
   With the board, firmware LOG lines appear in the controller manager output
   (`[firmware] ORION firmware up` after a reset).
3. Launch with `hardware:=serial port:=<dev> encoder_cpr:=... gear_ratio:=...`. Configure must
   log `Firmware xxxxxxxx, protocol v1`; it fails loudly on no PONG or a rejected param.
4. Polarity: command +0.2 m/s; every wheel must turn forward and `/joint_states` velocities must
   all be positive. Fix motor/encoder inversion in `board_config.hpp` (if a wheel runs away at
   full duty, the encoder and motor signs disagree: stop and fix `invert_encoder`).
5. Spin each wheel by hand 10 turns: position must change by 62.8 rad (checks `encoder_cpr`
   and `gear_ratio`).
6. Watchdog: unplug USB while driving slowly; wheels stop within 200 ms.
7. E-stop: publish `/orion/estop true`; wheels brake and stay stopped until it goes false.
8. Current: stall one wheel by hand at low duty; check the measured current, then (carefully,
   briefly) confirm the OVERCURRENT fault and duty fold-back at the configured limit.
9. Tune `kp`/`ki`/`kff` on the ground: step 0 -> 2 rad/s should settle in < 0.3 s without
   > 20 % overshoot (the host-test plant does this with the defaults; the real one may not).
