// Hardware abstraction layer: the only thing a new board has to implement.
//
// Everything above this interface (PID, watchdog, current limit, e-stop, protocol) is portable
// C++ and is unit-tested on the host with a fake HAL (firmware/test/test_core/fake_hal.hpp).
#ifndef ORION_CORE_HAL_HPP_
#define ORION_CORE_HAL_HPP_

#include <cstddef>
#include <cstdint>

namespace orion_fw
{

class Hal
{
public:
  virtual ~Hal() = default;

  /// Drives one motor. `duty` in [-1, 1]; positive must make the encoder count up and turn the
  /// wheel in its URDF-positive direction (fix polarity in the board config, not here).
  /// When duty == 0, `brake` true shorts the motor terminals, false lets it coast. Drivers that
  /// cannot coast may brake in both cases (document it in the board file).
  virtual void set_motor(size_t wheel, float duty, bool brake) = 0;

  /// Raw quadrature decoder count. May wrap around at the int32 limits.
  virtual int32_t encoder_count(size_t wheel) = 0;

  /// Motor current in A, signed like duty. Returns false when this board has no current sensing.
  virtual bool read_current(size_t wheel, float & amps) = 0;

  /// Battery voltage in V. Returns false when the board does not measure it.
  virtual bool read_battery_voltage(float & volts) = 0;

  /// Free-running microsecond clock. Wraps every ~71.6 min; the core only uses wrap-safe
  /// differences, and the host unwraps STATE.t_us, so no millisecond clock is needed.
  virtual uint32_t micros() = 0;

  /// Queues bytes on the host link. Must not block for longer than ~1 ms.
  virtual void serial_write(const uint8_t * data, size_t len) = 0;
};

}  // namespace orion_fw

#endif  // ORION_CORE_HAL_HPP_
