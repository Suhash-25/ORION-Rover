// Runtime-settable firmware parameters (SET_PARAM / PARAM_ACK). Defaults are GUESSES until the
// hardware is known (docs/HARDWARE.md); the host pushes encoder CPR and gear ratio on configure.
#ifndef ORION_CORE_PARAMS_HPP_
#define ORION_CORE_PARAMS_HPP_

#include <cmath>
#include <cstdint>

#include "orion_proto.hpp"

namespace orion_fw
{

struct ParamSpec
{
  const char * name;
  float default_value;
  float min_value;
  float max_value;
};

// Indexed by orion_proto::param_id. The current-limit ceiling (40 A) is the most a host can ask
// for; lower it in the board config if the driver or wiring is rated lower.
constexpr ParamSpec kParamSpecs[orion_proto::param_id::kCount] = {
  {"kp", 0.10f, 0.0f, 10.0f},
  {"ki", 1.0f, 0.0f, 100.0f},
  {"kd", 0.0f, 0.0f, 1.0f},
  {"kff", 0.15f, 0.0f, 10.0f},
  {"max_duty", 1.0f, 0.0f, 1.0f},
  {"current_limit_a", 14.4f, 0.5f, 40.0f},  // 1.2 x I_peak (12 A, GUESS), SPEC section 5
  {"current_limit_time_s", 0.1f, 0.0f, 1.0f},
  {"encoder_cpr", 2048.0f, 1.0f, 1.0e6f},
  {"gear_ratio", 30.0f, 0.01f, 1000.0f},
  {"vbat_min_v", 10.5f, 0.0f, 60.0f},  // 3S LiPo at 3.5 V/cell, GUESS
  {"vel_filter_tau_s", 0.01f, 0.0f, 0.5f},
};

class Params
{
public:
  Params()
  {
    for (uint8_t i = 0; i < orion_proto::param_id::kCount; ++i) {
      values_[i] = kParamSpecs[i].default_value;
    }
  }

  /// Applies a value if `id` exists and `value` is finite and within range.
  /// Returns an orion_proto::param_status code; on rejection the old value is kept.
  uint8_t set(uint8_t id, float value)
  {
    if (id >= orion_proto::param_id::kCount) {
      return orion_proto::param_status::kUnknownId;
    }
    const ParamSpec & s = kParamSpecs[id];
    if (!std::isfinite(value) || value < s.min_value || value > s.max_value) {
      return orion_proto::param_status::kOutOfRange;
    }
    values_[id] = value;
    return orion_proto::param_status::kOk;
  }

  /// Current value, or NaN for an unknown id.
  float get(uint8_t id) const
  {
    return id < orion_proto::param_id::kCount ? values_[id] : NAN;
  }

  /// CRC-16/CCITT-FALSE over all values as f32 LE in id order (reported in PONG).
  uint16_t crc() const
  {
    uint16_t crc = 0xFFFF;
    for (uint8_t i = 0; i < orion_proto::param_id::kCount; ++i) {
      uint8_t b[4];
      orion_proto::put_f32(b, values_[i]);
      crc = orion_proto::crc16_ccitt_false(b, sizeof(b), crc);
    }
    return crc;
  }

private:
  float values_[orion_proto::param_id::kCount];
};

}  // namespace orion_fw

#endif  // ORION_CORE_PARAMS_HPP_
