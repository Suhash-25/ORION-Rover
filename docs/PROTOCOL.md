# ORION host <-> firmware serial protocol (v1)

The contract between `orion_hardware/OrionSystem` (host, ros2_control) and the wheel firmware.
It refines docs/SPEC.md section 7; where this file adds something the SPEC does not say, the
addition is marked **[ext]** and recorded in docs/engineering-log/agents/wp2-hardware.md.

Single source of truth in code: `firmware/lib/orion_proto/orion_proto.hpp` (C++, header-only,
compiled into both the firmware and the host plugin). The Python reference
(`robot_ws/src/orion_hardware/orion_hardware_py/protocol.py`) mirrors it, and both are checked
against the same golden vectors (`firmware/lib/orion_proto/test_vectors.json`).

## 1. Physical link

| item | value |
| --- | --- |
| transport | USB CDC (Teensy native USB) or a 3.3 V UART |
| baud | 921600 preferred, 115200 minimum (ignored by USB CDC) |
| format | 8N1, raw, no flow control |
| byte order | little-endian for every multi-byte field |
| float | IEEE-754 binary32 |

Bandwidth at 100 Hz: STATE is 55 B on the wire (5.5 kB/s = 55 kbit/s with start/stop bits),
CMD is 16 B (16 kbit/s). 115200 baud carries this at ~50 % load per direction; 921600 leaves
room for LOG and future fields.

## 2. Framing

```
raw   = [type u8][seq u8][len u8][payload: len bytes, 0..64][crc16 u16 LE]
wire  = COBS(raw) 0x00
```

- **CRC-16/CCITT-FALSE**: poly 0x1021, init 0xFFFF, no reflection, no final XOR, computed over
  `type..payload` (header and payload, not the CRC). Check value: `"123456789"` -> `0x29B1`.
- **COBS**: canonical encoding (no empty trailing block after a full 254-byte block). The
  encoded block never contains 0x00; a single 0x00 ends each frame. Maximum raw frame 69 B,
  encoded 70 B, on the wire 71 B.
- **seq**: per-direction 8-bit counter, +1 per frame sent, wraps 255 -> 0. The firmware counts
  gaps in the CMD stream (`seq_gap = (seq - prev - 1) mod 256`) and reports them in STATS.
- **len** must equal the actual payload length; each message type has one fixed payload size
  (LOG: 0..64). A frame of the right CRC but the wrong size for its type is dropped and counted
  as a framing error.

### Receiver rules (both sides)

| input | action | counted as |
| --- | --- | --- |
| 0x00 with an empty buffer | ignored (idle / resync delimiter) | - |
| more than 70 bytes before a 0x00 | discard through the next 0x00 | framing error |
| COBS block malformed (bad code byte, embedded zero) | drop | framing error |
| fewer than 5 decoded bytes, or `len` != block size - 5, or `len` > 64 | drop | framing error |
| CRC mismatch | drop | CRC error |
| valid frame, unknown type or wrong payload size | drop | framing error |

Any byte sequence is safe to feed to the decoder (fuzzed with 200 000 random bytes in
`firmware/test/test_proto` and 50 000 in `test_protocol.py`, under ASan/UBSan for C++). After
garbage, the first 0x00 resynchronises the stream.

### Worked example (from the golden vectors)

CMD, seq 7, targets FL +1000, FR -1000, RL +32767, RR -32768 mrad/s, flags enable|brake:

```
raw  01 07 09 | e8 03 18 fc ff 7f 00 80 03 | 64 f4          (crc 0xF464, LE)
wire 0a 01 07 09 e8 03 18 fc ff 7f 05 80 03 64 f4 00
```

PING, seq 3: `wire 03 05 03 03 3f 72 00`.

## 3. Messages

Wheel order everywhere: **0 FL, 1 FR, 2 RL, 3 RR** (firmware slots; the URDF maps joints to
slots with the per-joint `motor_index` param). Positive = the wheel's URDF-positive direction
after the board's polarity config and the joint's `direction` param.

| type | name | dir | rate | payload bytes |
| --- | --- | --- | --- | --- |
| 0x01 | CMD | host -> fw | 100 Hz (controller rate) | 9 |
| 0x02 | STATE | fw -> host | 100 Hz | 48 |
| 0x03 | SET_PARAM | host -> fw | on configure | 5 |
| 0x04 | PARAM_ACK | fw -> host | reply | 6 **[ext: +status]** |
| 0x05 | PING | host -> fw | on configure | 0 |
| 0x06 | PONG | fw -> host | reply | 7 |
| 0x07 | ESTOP | host -> fw | on e-stop, then every cycle while engaged | 0 |
| 0x08 | ESTOP_CLEAR | host -> fw | once on release | 4 |
| 0x09 | LOG | fw -> host | events | 0..64 ASCII |
| 0x0A | STATS **[ext]** | fw -> host | 1 Hz | 24 |

### CMD (0x01), 9 B

| offset | type | field | unit |
| --- | --- | --- | --- |
| 0 | int16[4] | w_target | mrad/s (+-32.767 rad/s), saturated by the host |
| 8 | u8 | flags | bit0 ENABLE, bit1 BRAKE_ON_IDLE |

ENABLE clear: PID off, motors idle (coast, or brake if BRAKE_ON_IDLE). Every valid CMD feeds
the watchdog, enabled or not, unless the e-stop is latched.

### STATE (0x02), 48 B

| offset | type | field | unit / meaning |
| --- | --- | --- | --- |
| 0 | u32 | t_us | firmware micros() when sent; wraps every 71.6 min (host unwraps) |
| 4 | int32[4] | position | wheel angle, mrad; wraps at int32 (~2.1e6 rad, ~59 h at 10 rad/s) |
| 20 | int16[4] | velocity | mrad/s, low-pass filtered (tau param 10) |
| 28 | int16[4] | current | mA, signed like duty; **0x7FFF = not sensed** |
| 36 | int16[4] | duty | +-10000 = +-100 % PWM |
| 44 | u16 | vbat | mV; 0 when not sensed |
| 46 | u8 | faults | bitmask below (same values as `orion_msgs/RoverHealth.FAULT_*`) |
| 47 | u8 | status | bitmask below |

faults: 1 WATCHDOG (no CMD for 200 ms, also set at boot until the first CMD), 2 OVERCURRENT
(current clamp active), 4 ENCODER (|duty| >= 0.3 for 0.5 s with no encoder count change),
8 UNDERVOLTAGE (vbat < vbat_min, 0.2 V hysteresis), 16 ESTOP_LATCHED.

status: 1 ENABLED (PID running), 2 CURRENT_SENSED (all four wheels), 4 VBAT_SENSED,
8 BRAKE_ON_IDLE.

### SET_PARAM (0x03), 5 B / PARAM_ACK (0x04), 6 B

SET_PARAM: `u8 id, f32 value`. PARAM_ACK: `u8 id, f32 value, u8 status` where value is the
value now in force (the old one on rejection) and status is 0 OK, 1 unknown id,
2 out of range / non-finite (rejected). **[ext]** The SPEC lists only `id, value`; without a
status the host cannot tell a rejection from an accepted value that happens to be equal.

| id | name | unit | default (GUESS) | range |
| --- | --- | --- | --- | --- |
| 0 | kp | duty per rad/s | 0.10 | 0..10 |
| 1 | ki | duty per rad | 1.0 | 0..100 |
| 2 | kd | duty per rad/s^2 | 0 | 0..1 |
| 3 | kff | duty per rad/s of target | 0.15 | 0..10 |
| 4 | max_duty | - | 1.0 | 0..1 |
| 5 | current_limit_a | A per wheel | 14.4 (1.2 x I_peak 12 A) | 0.5..40 |
| 6 | current_limit_time_s | s | 0.1 | 0..1 |
| 7 | encoder_cpr | counts per motor rev, after x4 decoding | 2048 | 1..1e6 |
| 8 | gear_ratio | motor revs per wheel rev | 30 | 0.01..1000 |
| 9 | vbat_min_v | V | 10.5 | 0..60 |
| 10 | vel_filter_tau_s | s | 0.01 | 0..0.5 |

Params live in RAM and reset to defaults at boot. The host pushes `encoder_cpr` and
`gear_ratio` from the URDF on every configure and fails configure unless both are acked OK
with the exact value. The watchdog period (200 ms) is deliberately **not** a parameter.

### PING (0x05), 0 B / PONG (0x06), 7 B

PONG: `u8 proto_ver (=1), u32 fw_git_hash (first 8 hex digits of the commit, 0 if unknown),
u16 param_crc` (CRC-16/CCITT-FALSE over all 11 params as f32 LE in id order). The host
refuses to configure on a protocol version mismatch.

### ESTOP (0x07), 0 B / ESTOP_CLEAR (0x08), 4 B

ESTOP latches: motors to duty 0 with brake, targets zeroed, CMD ignored (and not feeding the
watchdog) until ESTOP_CLEAR with `u32 magic = 0x0DE5C0DE`. A wrong magic or wrong length is
ignored (and logged). After a clear the firmware stays idle until a **fresh** CMD arrives; it
never resumes a pre-e-stop command. The latch is RAM only: a firmware reset clears it, so the
host re-sends ESTOP every write() cycle while `/orion/estop` is true.

### LOG (0x09), 0..64 B

ASCII text, not NUL-terminated. Events: boot, watchdog trip, e-stop latched/cleared, rejected
clear. The host prints them at INFO as `[firmware] ...`.

### STATS (0x0A), 24 B **[ext]**

`u32 rx_frames_ok, rx_crc_errors, rx_framing_errors, cmd_seq_gaps, watchdog_trips,
overcurrent_trips`, cumulative since boot, sent at 1 Hz. The SPEC asks the firmware to count
CRC errors and watchdog trips (RoverHealth reports them) but STATE has no room for counters;
a separate low-rate frame keeps STATE at the SPEC layout.

## 4. Timing and safety behaviour

| behaviour | value | where enforced | test |
| --- | --- | --- | --- |
| PID | 200 Hz fixed grid (resyncs if > 1 period late) | firmware | `ControlCore.StepResponse*` |
| STATE | 100 Hz | firmware | `Link.StateAt100HzAndStatsAt1Hz` |
| watchdog | no valid CMD for 200 ms -> targets 0, PID off, coast unless BRAKE_ON_IDLE; WATCHDOG fault; trip counted | firmware | `Watchdog.*` (200 +- 10 ms, also across the micros() wrap) |
| current clamp | \|I\| > current_limit_a for current_limit_time_s -> duty cap folds back x0.9 per 5 ms step while over, recovers 0.5/s; OVERCURRENT fault until fully recovered | firmware (needs current sensing) | `CurrentLimit.*` |
| e-stop | latch until magic clear | firmware | `Estop.*` |
| stale STATE | read() returns ERROR when the newest STATE is older than 100 ms (`state_timeout_s`) | host | `ReadReturnsErrorWhenStateIsOlderThan100ms`, `test_silent_firmware_deactivates_hardware` |
| serial device lost | reader thread stops; read()/write() return ERROR | host | `TransportFailureIsReported` |
| write failure | frame dropped and counted (non-blocking write); next CMD in 10 ms | host | `DroppedWritesAreCountedNotBlocking` |

Without current sensing (`current == 0x7FFF`) the firmware **cannot** enforce the hard current
limit; `max_duty` is then the only electrical protection (see docs/HARDWARE.md, question C).

What the controller manager does on ERROR: in ros2_control 4.x, a read() or write() ERROR calls
`on_error()` on OrionSystem (sends a disable CMD best effort, closes the port) and the
component leaves the active state; controllers claiming its interfaces are deactivated. The
firmware watchdog stops the wheels within 200 ms regardless. Recovery: power-cycle or fix the
link, then configure + activate the hardware component again.

## 5. Host-side mapping (orion_hardware)

| ros2_control | from firmware |
| --- | --- |
| `<joint>/position` (rad) | `direction * position_mrad / 1000` |
| `<joint>/velocity` (rad/s) | `direction * velocity_mrad_s / 1000` |
| `<joint>/effort` (A) | measured: `direction * current_ma / 1000`; not sensed: `direction * (duty * V_bat - kt * gear_ratio * w) / R` (SPEC F5; `kt = 0` gives exactly `duty * V_bat / R`; V_bat = `nominal_battery_v` if not sensed) |
| CMD target (mrad/s) | `round(1000 * direction * velocity_command)`, saturated to int16, NaN -> 0 |

`/orion/rover_health` (`orion_msgs/RoverHealth`, 10 Hz): `battery_v` (NaN if not sensed),
`fw_fault_flags` = STATE.faults, `fw_uptime_ms` from the unwrapped t_us, `crc_errors` = host
rx CRC+framing errors + firmware rx CRC+framing errors, `watchdog_trips` from STATS, `duty[4]`
in slot order FL FR RL RR, `current_sensed` = no wheel reports 0x7FFF.

`/orion/estop` (`std_msgs/Bool`, any reliable publisher, e.g. orion_safety at 50 Hz): a false
-> true edge sends ESTOP immediately and every write() cycle after it instead of CMD; a
true -> false edge sends one ESTOP_CLEAR. The host never clears a latch it did not see set.

## 6. Changing the protocol

1. Edit `orion_proto.hpp` and `protocol.py` together, bump `kProtocolVersion`/`PROTOCOL_VERSION`.
2. `python3 -m orion_hardware_py.vectors firmware/lib/orion_proto/test_vectors.json`
   (run from `robot_ws/src/orion_hardware`); `test_committed_vectors_match_the_generator`
   fails until you do.
3. Update this file. Flash the firmware and the host together; PONG rejects mixed versions.
