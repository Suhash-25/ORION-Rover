# WP2 hardware: firmware, protocol, orion_hardware (engineering log)

Branch `claude/wp2-hardware`, base `9e2754a`. ROS_DOMAIN_ID 72 for every ROS test.

## What was built

- `firmware/lib/orion_proto/orion_proto.hpp`: the one header-only codec (COBS, CRC-16/CCITT-FALSE,
  fixed-size payload structs, streaming decoder with error counters). Compiled into the firmware
  and into `orion_hardware`. Golden vectors in `test_vectors.json`.
- `firmware/lib/orion_core/`: portable control core behind a 5-method HAL (`set_motor`,
  `encoder_count`, `read_current` -> "not available", `read_battery_voltage`, `micros`,
  `serial_write`): 200 Hz PID (feed-forward, derivative on measurement, conditional
  integration + clamped integrator, output clamp), 200 ms watchdog, per-wheel current clamp with
  fault flag, e-stop latch with magic clear, CMD seq-gap counting, encoder-stall and
  undervoltage faults, STATE 100 Hz, STATS 1 Hz, PING/PONG, SET_PARAM/PARAM_ACK.
- `firmware/src/`: Teensy 4.1 example board (pins marked CONFIRM), `platformio.ini` with
  `native` and `teensy41` envs, `tools/git_hash.py`.
- `firmware/test/`: GoogleTest host tests (43 cases) with a fake HAL (DC-motor plant, fake
  clock), built by `firmware/CMakeLists.txt` under ASan/UBSan, and also by orion_hardware's
  colcon tests so CI runs them.
- `robot_ws/src/orion_hardware/`: `OrionSystem` (SystemInterface plugin), `FirmwareLink`
  (reader thread, thread-safe senders), `SerialTransport` (termios, poll, non-blocking writes),
  pure conversions/config parsing; Python reference `orion_hardware_py/protocol.py`, vector
  generator, `fake_firmware` (also `ros2 run orion_hardware fake_firmware --port ...`).
- `orion_description/urdf/control/ros2_control.xacro`: `hardware:=gazebo|serial` (default
  gazebo, unchanged output), serial params `port baud encoder_cpr gear_ratio kt
  motor_resistance brake_on_idle`, per-joint `motor_index`/`direction`; an unknown value aborts
  xacro with a clear message.
- `docs/PROTOCOL.md`, `docs/HARDWARE.md`.

### Jazzy SystemInterface API used (hardware_interface 4.49.0, from the installed headers)

```cpp
CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params);
CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state);
CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state);
CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous_state);
CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state);
CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state);
CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state);
hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period);
hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period);
```

`on_init(const HardwareInfo &)` still exists in 4.49 but is `[[deprecated]]`, so the params
overload is used (the base `SystemInterface::on_init(params)` is called first). Interfaces are
exported by the framework from the URDF; values go through `set_state(name, v)` /
`get_command<double>(name)`. RoverHealth and the `/orion/estop` subscription use the
component's default node, `get_node()`; the integration test proves the controller manager's
executor spins it (the e-stop callback fires inside `ros2_control_node`). Minimum version: the
API above (tested on 4.49.0); older Jazzy releases without `HardwareComponentInterfaceParams`
or `get_node()` will not compile.

## Decisions (and options rejected)

1. **Teensy 4.1 as the example board.** FPU, native USB, all pins interrupt-capable, 4 hardware
   quadrature decoders. Rejected ESP32 (ADC, RTOS jitter), AVR (no FPU, too few interrupt
   pins). RP2040 is a fine alternative. Rationale in HARDWARE.md section A.
2. **Protocol extensions** (PROTOCOL.md, marked [ext]): PARAM_ACK gets a `status` byte (a
   rejected value is otherwise indistinguishable from an accepted one); new `STATS` 0x0A at
   1 Hz carries firmware counters (RoverHealth needs CRC errors and watchdog trips; STATE has no
   room and the SPEC fixes its layout). PONG field sizes chosen: `u8, u32, u16`; param_crc is
   CRC-16 over the params, reusing the codec's CRC.
3. **STATE.faults bit values equal RoverHealth.FAULT_*** so the host copies them unchanged.
4. **Watchdog is a constant (200 ms), not a SET_PARAM.** A safety limit should not be settable
   over the link it protects. Boot counts as "watchdog tripped" (motors idle, flag set) but not
   as a trip. CMD while e-stopped does not feed the watchdog.
5. **E-stop:** brake (not coast) on latch; after a valid clear the firmware waits for a fresh
   CMD instead of resuming. Host: `/orion/estop` (`std_msgs/Bool`, default reliable/volatile
   QoS so it accepts volatile and transient-local publishers) false->true sends ESTOP at once
   and in every write() cycle instead of CMD (survives lost frames and firmware resets);
   true->false sends one ESTOP_CLEAR. The host never clears a latch it did not see set.
6. **Current clamp needs measured current.** Rejected using a duty-based estimate for the hard
   limit: `duty * V / R` is far above the real current at speed and would clamp normal driving.
   Without sensing the firmware says so (status bit, 0x7FFF) and HARDWARE.md says the hard
   limit is then not enforced. Clamp = duty-cap fold-back x0.9 per 5 ms step while over the
   limit, recovery 0.5/s; fault flag until fully recovered.
7. **Encoder fault heuristic:** |duty| >= 0.3 for 0.5 s with no count change. Without it the
   RoverHealth FAULT_ENCODER bit would be dead.
8. **Effort fallback** `I = (duty * V_bat - kt * gear * w) / R`; `kt` default 0 gives exactly
   the SPEC's `duty * V_bat / R`; setting `kt` adds back-EMF correction (makes the
   requested `kt` param meaningful). `nominal_battery_v` (12 V) when vbat is not measured.
9. **Host pushes `encoder_cpr` and `gear_ratio` on configure** and fails unless both are acked
   OK with the exact value: the URDF is the single place they are set.
10. **Health topic `/orion/rover_health`** (matches the name orion_terrain's params already use).
    `battery_v` = NaN when not sensed. `crc_errors` sums host and firmware rx CRC+framing errors.
11. **Non-blocking serial writes; a frame that cannot be queued is dropped and counted.** A late
    CMD is worse than a lost one (the next follows in 10 ms; the firmware watchdog covers longer).
12. **Firmware tests built inside orion_hardware's colcon test** (`ORION_FIRMWARE_DIR` CMake
    cache var, default `../../../firmware`) so CI covers firmware code without PlatformIO and
    without changing ci.yml.
13. **Fake firmware is a first-order wheel model,** not a copy of the control law (that is
    tested in C++). Rejected running the C++ core in the fake: two implementations of the
    contract is the point of the cross-check.
14. Rejected adding `orion_hardware` as an exec_depend of orion_description: orion_hardware
    test-depends on orion_description, which would make a colcon dependency cycle. The launch
    package that uses `hardware:=serial` must depend on orion_hardware instead.

## Mistakes made and fixed

- **Anti-windup test could not fail.** Mutation (conditional integration removed) still passed,
  because the integrator clamp alone bounded the integral when feed-forward was 0. Root cause:
  test exercised only one of two mechanisms. Added `IntegratorHoldsWhenOtherTermsAlreadySaturate`
  (feed-forward saturates on its own); the mutation now fails. Lesson: mutate each mechanism.
- **Signed-overflow UB in the fake HAL** (`count + offset` near INT32_MAX) - spotted while
  writing the wrap test; now unsigned arithmetic like real decoder hardware.
- **Integration test sent cmd_vel once, then waited 0.5 s:** diff_drive's 0.5 s
  `cmd_vel_timeout` zeroed the targets, so the CMD check saw zeros. Fixed with a 20 Hz timer.
- **Teardown order:** socat was killed before the fake firmware, whose thread then died on EIO.
  The fake now stops on OSError and the fixture closes it first.
- **Lint:** flake8 A003 (dataclass field `id` shadows a builtin -> `param_id`, vectors and C++
  test updated); I100 from `sys.path` hacks before imports -> `test/conftest.py`.
- **Integration flake under load:** 2 of 3 back-to-back colcon runs failed all 5 cases while
  four other agents were building/simulating (load average ~14 on 4 CPUs); direct reruns passed
  (3/3, then 3/3 via colcon). The all-five pattern matches the hardware erroring right after
  activation, most likely a >100 ms gap from the Python fake-firmware thread. Not proven. Mitigated: the
  integration test injects `state_timeout_s = 0.5` (the 100 ms limit stays covered by the unit
  test with an injected mock), and a failure now prints the controller_manager log.

## Tests (commands and results, 2026-11-09)

- Firmware host build: `cmake -S firmware -B <build> -DCMAKE_PREFIX_PATH=/opt/rosenv &&
  cmake --build <build> -j2 && ctest --test-dir <build>` -> 2/2 ctest targets, test_core 29/29,
  test_proto 14/14, 0 compiler warnings (`-Wall -Wextra -Wpedantic -Wconversion`, ASan+UBSan).
- `ORION_OFFLINE=1 ORION_ROS_SETUP=/tmp/claude-0/rosenv.sh scripts/verify.sh unit
  orion_hardware orion_description orion_control` -> exit 0, "160 tests, 0 errors, 0 failures,
  12 skipped". orion_hardware: test_conversions (8), test_firmware_link (9), test_orion_system
  (6), test_firmware_proto (14), test_firmware_core (29), test_protocol_py (13),
  test_fake_firmware (4), test_xacro_serial (3), test_integration (5, ~6 s), cppcheck, flake8,
  lint_cmake, pep257, uncrustify all pass. orion_description test_urdf passes with the new xacro.
- Negative checks (each went red, then reverted):
  watchdog 300 ms -> 3 Watchdog tests fail; conditional integration removed -> 1 PID test
  fails; current fold-back removed -> StalledWheelIsClampedAfterLimitTime fails; e-stop not
  latched -> LatchesBrakesAndIgnoresCommands fails; seq gap off by one -> 2 tests fail; CRC init
  0x0000 -> 3 golden-vector tests fail; host staleness check removed -> test_orion_system and
  test_integration fail; joint `direction` ignored in write() -> test_orion_system fails.
- Not verified: `pio test -e native` and `pio run -e teensy41` (PlatformIO installs from PyPI
  but its registry is blocked: `HTTPClientError` installing the native platform). `main.cpp`
  was only syntax-checked with g++ against stub `Arduino.h`/`Encoder.h`.

## Only verifiable on the real rover

Board build and flashing; every pin; motor/encoder polarity; whether the driver can coast;
encoder count rate vs interrupt decoding; PWM frequency vs driver; current-sense scaling and
noise (and therefore the clamp thresholds); PID gains (the defaults are tuned on a simulated
plant); USB latency/jitter and the 100 ms staleness margin on the onboard computer; e-stop
stop distance; behaviour after a firmware brown-out (latch lost -> host re-sends ESTOP).

## Open questions

- orion_safety: please publish `/orion/estop` as `std_msgs/Bool` continuously (twist_mux lock
  needs it anyway) and drop it to false only after a successful EstopReset; OrionSystem turns
  the edges into ESTOP / ESTOP_CLEAR.
- orion_bringup: a real-robot launch (robot_state_publisher with `hardware:=serial ...`,
  `ros2_control_node` with orion_control's controllers.yaml, spawners) and an exec_depend on
  orion_hardware. `gazebo_control.xacro` is still included with `hardware:=serial`; the
  `<gazebo>` block is ignored by ros2_control_node, but orion_description could skip it.
- Hardware inventory: all of docs/HARDWARE.md.
