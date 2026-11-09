// ORION host <-> firmware serial protocol (docs/PROTOCOL.md, docs/SPEC.md section 7).
//
// Header-only and dependency-free (no heap, no exceptions, no STL containers) so the same file
// compiles for the microcontroller and for the ros2_control plugin on the host.
//
// Wire format of one frame:
//   COBS( [type u8][seq u8][len u8][payload, len <= 64 B][crc16 u16 LE] ) 0x00
// The CRC is CRC-16/CCITT-FALSE over type..payload. All multi-byte fields are little-endian.
#ifndef ORION_PROTO_HPP_
#define ORION_PROTO_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace orion_proto
{

/// Bumped on any incompatible change to framing or to a payload layout.
constexpr uint8_t kProtocolVersion = 1;

constexpr size_t kMaxPayload = 64;
constexpr size_t kHeaderSize = 3;   // type, seq, len
constexpr size_t kCrcSize = 2;
constexpr size_t kMaxRawFrame = kHeaderSize + kMaxPayload + kCrcSize;            // 69
constexpr size_t kMaxEncodedFrame = kMaxRawFrame + (kMaxRawFrame + 253) / 254;  // 70
constexpr size_t kMaxWireFrame = kMaxEncodedFrame + 1;                           // + 0x00

/// Message types. STATS (0x0A) is an ORION extension to SPEC section 7 (see PROTOCOL.md).
enum MsgType : uint8_t
{
  MSG_CMD = 0x01,
  MSG_STATE = 0x02,
  MSG_SET_PARAM = 0x03,
  MSG_PARAM_ACK = 0x04,
  MSG_PING = 0x05,
  MSG_PONG = 0x06,
  MSG_ESTOP = 0x07,
  MSG_ESTOP_CLEAR = 0x08,
  MSG_LOG = 0x09,
  MSG_STATS = 0x0A,
};

constexpr uint32_t kEstopClearMagic = 0x0DE5C0DEu;
constexpr int16_t kCurrentNotAvailable = 0x7FFF;  // STATE current field when not sensed
constexpr int16_t kDutyFullScale = 10000;         // STATE duty: +-10000 = +-100 %
constexpr size_t kNumWheels = 4;                  // order FL, FR, RL, RR

/// STATE.faults bits. Values equal orion_msgs/RoverHealth FAULT_* on purpose.
namespace fault
{
constexpr uint8_t kWatchdog = 1;
constexpr uint8_t kOvercurrent = 2;
constexpr uint8_t kEncoder = 4;
constexpr uint8_t kUndervoltage = 8;
constexpr uint8_t kEstopLatched = 16;
}  // namespace fault

/// STATE.status bits.
namespace status
{
constexpr uint8_t kEnabled = 1;        // PID running (enabled CMD, no watchdog, no e-stop)
constexpr uint8_t kCurrentSensed = 2;  // current fields are measured, not 0x7FFF
constexpr uint8_t kVbatSensed = 4;     // vbat field is measured
constexpr uint8_t kBrakeOnIdle = 8;    // idle motors are shorted (brake), not left floating
}  // namespace status

/// CMD.flags bits.
namespace cmd_flag
{
constexpr uint8_t kEnable = 1;
constexpr uint8_t kBrakeOnIdle = 2;
}  // namespace cmd_flag

/// SET_PARAM ids. Units and ranges: firmware/lib/orion_core/params.hpp and PROTOCOL.md.
namespace param_id
{
constexpr uint8_t kKp = 0;                // duty per rad/s of wheel speed error
constexpr uint8_t kKi = 1;                // duty per rad of integrated error
constexpr uint8_t kKd = 2;                // duty per rad/s^2 (derivative on measurement)
constexpr uint8_t kKff = 3;               // duty per rad/s of target (feed-forward)
constexpr uint8_t kMaxDuty = 4;           // [0, 1]
constexpr uint8_t kCurrentLimitA = 5;     // firmware hard limit per wheel, A
constexpr uint8_t kCurrentLimitTimeS = 6; // time over the limit before clamping, s
constexpr uint8_t kEncoderCpr = 7;        // decoder counts per motor revolution (after x4)
constexpr uint8_t kGearRatio = 8;         // motor revolutions per wheel revolution
constexpr uint8_t kVbatMinV = 9;          // undervoltage fault threshold, V
constexpr uint8_t kVelFilterTauS = 10;    // wheel speed low-pass time constant, s
constexpr uint8_t kCount = 11;
}  // namespace param_id

/// PARAM_ACK.status values.
namespace param_status
{
constexpr uint8_t kOk = 0;
constexpr uint8_t kUnknownId = 1;
constexpr uint8_t kOutOfRange = 2;  // rejected; value field carries the unchanged current value
}  // namespace param_status

// ------------------------------------------------------------------------------------------
// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no xor-out.
// Check value: crc16("123456789") == 0x29B1.
// ------------------------------------------------------------------------------------------
inline uint16_t crc16_ccitt_false(const uint8_t * data, size_t len, uint16_t crc = 0xFFFF)
{
  for (size_t i = 0; i < len; ++i) {
    crc = static_cast<uint16_t>(crc ^ (static_cast<uint16_t>(data[i]) << 8));
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) :
        static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

// ------------------------------------------------------------------------------------------
// COBS (Consistent Overhead Byte Stuffing). Output never contains 0x00.
// ------------------------------------------------------------------------------------------

/// Encodes `len` bytes. Returns the encoded size, or 0 if `out_cap` is too small.
inline size_t cobs_encode(const uint8_t * in, size_t len, uint8_t * out, size_t out_cap)
{
  if (out_cap == 0) {
    return 0;
  }
  size_t code_idx = 0;
  size_t o = 1;
  uint8_t code = 1;
  for (size_t i = 0; i < len; ++i) {
    if (in[i] == 0) {
      out[code_idx] = code;
      code = 1;
      code_idx = o++;
      if (code_idx >= out_cap) {
        return 0;
      }
    } else {
      if (o >= out_cap) {
        return 0;
      }
      out[o++] = in[i];
      ++code;
      if (code == 0xFF && i + 1 < len) {  // full 254-byte block; canonical: no empty tail
        out[code_idx] = code;
        code = 1;
        code_idx = o++;
        if (code_idx >= out_cap) {
          return 0;
        }
      }
    }
  }
  out[code_idx] = code;
  return o;
}

/// Decodes one COBS block (without the 0x00 delimiter). Returns the decoded size, or 0 on a
/// malformed block (embedded zero, code pointing past the end) or if `out_cap` is too small.
inline size_t cobs_decode(const uint8_t * in, size_t len, uint8_t * out, size_t out_cap)
{
  size_t i = 0;
  size_t o = 0;
  while (i < len) {
    const uint8_t code = in[i++];
    if (code == 0) {
      return 0;
    }
    for (uint8_t k = 1; k < code; ++k) {
      if (i >= len || in[i] == 0 || o >= out_cap) {
        return 0;
      }
      out[o++] = in[i++];
    }
    if (code != 0xFF && i < len) {
      if (o >= out_cap) {
        return 0;
      }
      out[o++] = 0;
    }
  }
  return o;
}

// ------------------------------------------------------------------------------------------
// Little-endian field access.
// ------------------------------------------------------------------------------------------
inline void put_u16(uint8_t * p, uint16_t v)
{
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}
inline void put_u32(uint8_t * p, uint32_t v)
{
  for (int i = 0; i < 4; ++i) {
    p[i] = static_cast<uint8_t>(v >> (8 * i));
  }
}
inline void put_i16(uint8_t * p, int16_t v) {put_u16(p, static_cast<uint16_t>(v));}
inline void put_i32(uint8_t * p, int32_t v) {put_u32(p, static_cast<uint32_t>(v));}
inline void put_f32(uint8_t * p, float v)
{
  uint32_t u;
  static_assert(sizeof(u) == sizeof(v), "IEEE-754 binary32 required");
  std::memcpy(&u, &v, sizeof(u));
  put_u32(p, u);
}
inline uint16_t get_u16(const uint8_t * p)
{
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
inline uint32_t get_u32(const uint8_t * p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline int16_t get_i16(const uint8_t * p) {return static_cast<int16_t>(get_u16(p));}
inline int32_t get_i32(const uint8_t * p) {return static_cast<int32_t>(get_u32(p));}
inline float get_f32(const uint8_t * p)
{
  const uint32_t u = get_u32(p);
  float v;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

// ------------------------------------------------------------------------------------------
// Frames.
// ------------------------------------------------------------------------------------------
struct Frame
{
  uint8_t type = 0;
  uint8_t seq = 0;
  uint8_t len = 0;
  uint8_t payload[kMaxPayload] = {};
};

/// Builds one wire frame (COBS block + trailing 0x00). Returns bytes written, or 0 if the payload
/// is longer than kMaxPayload or `out_cap` < needed (kMaxWireFrame always suffices).
inline size_t encode_frame(
  uint8_t type, uint8_t seq, const uint8_t * payload, size_t len, uint8_t * out, size_t out_cap)
{
  if (len > kMaxPayload) {
    return 0;
  }
  uint8_t raw[kMaxRawFrame];
  raw[0] = type;
  raw[1] = seq;
  raw[2] = static_cast<uint8_t>(len);
  if (len > 0) {
    std::memcpy(raw + kHeaderSize, payload, len);
  }
  const uint16_t crc = crc16_ccitt_false(raw, kHeaderSize + len);
  put_u16(raw + kHeaderSize + len, crc);
  const size_t n = cobs_encode(raw, kHeaderSize + len + kCrcSize, out, out_cap);
  if (n == 0 || n >= out_cap) {
    return 0;
  }
  out[n] = 0x00;
  return n + 1;
}

enum class ParseResult : uint8_t
{
  kOk,
  kTooShort,        // fewer than header + CRC bytes
  kLengthMismatch,  // len field disagrees with the block size, or len > kMaxPayload
  kBadCrc,
};

/// Validates an already COBS-decoded block and copies it into `out`.
inline ParseResult parse_raw_frame(const uint8_t * raw, size_t n, Frame & out)
{
  if (n < kHeaderSize + kCrcSize) {
    return ParseResult::kTooShort;
  }
  const size_t len = raw[2];
  if (len > kMaxPayload || n != kHeaderSize + len + kCrcSize) {
    return ParseResult::kLengthMismatch;
  }
  if (crc16_ccitt_false(raw, kHeaderSize + len) != get_u16(raw + kHeaderSize + len)) {
    return ParseResult::kBadCrc;
  }
  out.type = raw[0];
  out.seq = raw[1];
  out.len = static_cast<uint8_t>(len);
  if (len > 0) {
    std::memcpy(out.payload, raw + kHeaderSize, len);
  }
  return ParseResult::kOk;
}

/// Streaming receiver: feed it every byte from the link. Any byte sequence is safe.
struct DecoderCounters
{
  uint32_t frames_ok = 0;
  uint32_t crc_errors = 0;      // well-formed block, wrong CRC
  uint32_t framing_errors = 0;  // bad COBS, wrong length, oversize block
};

class FrameDecoder
{
public:
  /// Consumes one byte. Returns true when it completed a valid frame, now in frame().
  bool push(uint8_t byte)
  {
    if (byte != 0x00) {
      if (n_ < sizeof(buf_)) {
        buf_[n_++] = byte;
      } else {
        overflow_ = true;  // keep discarding until the next delimiter
      }
      return false;
    }
    const size_t n = n_;
    const bool overflow = overflow_;
    n_ = 0;
    overflow_ = false;
    if (n == 0 && !overflow) {
      return false;  // idle delimiter, not an error
    }
    if (overflow) {
      ++counters_.framing_errors;
      return false;
    }
    uint8_t raw[kMaxEncodedFrame];
    const size_t m = cobs_decode(buf_, n, raw, sizeof(raw));
    if (m == 0) {
      ++counters_.framing_errors;
      return false;
    }
    switch (parse_raw_frame(raw, m, frame_)) {
      case ParseResult::kOk:
        ++counters_.frames_ok;
        return true;
      case ParseResult::kBadCrc:
        ++counters_.crc_errors;
        return false;
      default:
        ++counters_.framing_errors;
        return false;
    }
  }

  const Frame & frame() const {return frame_;}
  const DecoderCounters & counters() const {return counters_;}

private:
  uint8_t buf_[kMaxEncodedFrame] = {};
  size_t n_ = 0;
  bool overflow_ = false;
  Frame frame_;
  DecoderCounters counters_;
};

// ------------------------------------------------------------------------------------------
// Payloads. pack() writes exactly kSize bytes; unpack() requires len == kSize.
// ------------------------------------------------------------------------------------------

/// CMD (host -> fw, 100 Hz): wheel speed targets.
struct CmdMsg
{
  static constexpr uint8_t kType = MSG_CMD;
  static constexpr size_t kSize = 9;
  int16_t w_target_mrad_s[kNumWheels] = {0, 0, 0, 0};  // mrad/s, FL FR RL RR
  uint8_t flags = 0;                                   // cmd_flag::*

  void pack(uint8_t * p) const
  {
    for (size_t i = 0; i < kNumWheels; ++i) {
      put_i16(p + 2 * i, w_target_mrad_s[i]);
    }
    p[8] = flags;
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    for (size_t i = 0; i < kNumWheels; ++i) {
      w_target_mrad_s[i] = get_i16(p + 2 * i);
    }
    flags = p[8];
    return true;
  }
};

/// STATE (fw -> host, 100 Hz): wheel measurements.
struct StateMsg
{
  static constexpr uint8_t kType = MSG_STATE;
  static constexpr size_t kSize = 48;
  uint32_t t_us = 0;                         // firmware micros() at sampling, wraps
  int32_t position_mrad[kNumWheels] = {};    // wheel angle, mrad, wraps at int32
  int16_t velocity_mrad_s[kNumWheels] = {};  // mrad/s
  int16_t current_ma[kNumWheels] = {};       // mA, kCurrentNotAvailable if not sensed
  int16_t duty[kNumWheels] = {};             // +-kDutyFullScale
  uint16_t vbat_mv = 0;                      // mV, 0 if not sensed
  uint8_t faults = 0;                        // fault::*
  uint8_t status = 0;                        // status::*

  void pack(uint8_t * p) const
  {
    put_u32(p, t_us);
    for (size_t i = 0; i < kNumWheels; ++i) {
      put_i32(p + 4 + 4 * i, position_mrad[i]);
      put_i16(p + 20 + 2 * i, velocity_mrad_s[i]);
      put_i16(p + 28 + 2 * i, current_ma[i]);
      put_i16(p + 36 + 2 * i, duty[i]);
    }
    put_u16(p + 44, vbat_mv);
    p[46] = faults;
    p[47] = status;
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    t_us = get_u32(p);
    for (size_t i = 0; i < kNumWheels; ++i) {
      position_mrad[i] = get_i32(p + 4 + 4 * i);
      velocity_mrad_s[i] = get_i16(p + 20 + 2 * i);
      current_ma[i] = get_i16(p + 28 + 2 * i);
      duty[i] = get_i16(p + 36 + 2 * i);
    }
    vbat_mv = get_u16(p + 44);
    faults = p[46];
    status = p[47];
    return true;
  }
};

/// SET_PARAM (host -> fw).
struct SetParamMsg
{
  static constexpr uint8_t kType = MSG_SET_PARAM;
  static constexpr size_t kSize = 5;
  uint8_t id = 0;
  float value = 0.0f;

  void pack(uint8_t * p) const
  {
    p[0] = id;
    put_f32(p + 1, value);
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    id = p[0];
    value = get_f32(p + 1);
    return true;
  }
};

/// PARAM_ACK (fw -> host): the value now in force. `status` is an ORION extension.
struct ParamAckMsg
{
  static constexpr uint8_t kType = MSG_PARAM_ACK;
  static constexpr size_t kSize = 6;
  uint8_t id = 0;
  float value = 0.0f;
  uint8_t status = param_status::kOk;

  void pack(uint8_t * p) const
  {
    p[0] = id;
    put_f32(p + 1, value);
    p[5] = status;
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    id = p[0];
    value = get_f32(p + 1);
    status = p[5];
    return true;
  }
};

/// PONG (fw -> host), answer to an empty PING.
struct PongMsg
{
  static constexpr uint8_t kType = MSG_PONG;
  static constexpr size_t kSize = 7;
  uint8_t proto_ver = kProtocolVersion;
  uint32_t fw_git_hash = 0;  // first 8 hex digits of the firmware commit
  uint16_t param_crc = 0;    // CRC-16/CCITT-FALSE over all params as f32 LE, in id order

  void pack(uint8_t * p) const
  {
    p[0] = proto_ver;
    put_u32(p + 1, fw_git_hash);
    put_u16(p + 5, param_crc);
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    proto_ver = p[0];
    fw_git_hash = get_u32(p + 1);
    param_crc = get_u16(p + 5);
    return true;
  }
};

/// ESTOP_CLEAR (host -> fw). Ignored unless magic == kEstopClearMagic.
struct EstopClearMsg
{
  static constexpr uint8_t kType = MSG_ESTOP_CLEAR;
  static constexpr size_t kSize = 4;
  uint32_t magic = kEstopClearMagic;

  void pack(uint8_t * p) const {put_u32(p, magic);}
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    magic = get_u32(p);
    return true;
  }
};

/// STATS (fw -> host, 1 Hz): cumulative firmware counters since boot. ORION extension.
struct StatsMsg
{
  static constexpr uint8_t kType = MSG_STATS;
  static constexpr size_t kSize = 24;
  uint32_t rx_frames_ok = 0;
  uint32_t rx_crc_errors = 0;
  uint32_t rx_framing_errors = 0;
  uint32_t cmd_seq_gaps = 0;  // CMD frames missing according to the seq counter
  uint32_t watchdog_trips = 0;
  uint32_t overcurrent_trips = 0;

  void pack(uint8_t * p) const
  {
    put_u32(p, rx_frames_ok);
    put_u32(p + 4, rx_crc_errors);
    put_u32(p + 8, rx_framing_errors);
    put_u32(p + 12, cmd_seq_gaps);
    put_u32(p + 16, watchdog_trips);
    put_u32(p + 20, overcurrent_trips);
  }
  bool unpack(const uint8_t * p, size_t len)
  {
    if (len != kSize) {
      return false;
    }
    rx_frames_ok = get_u32(p);
    rx_crc_errors = get_u32(p + 4);
    rx_framing_errors = get_u32(p + 8);
    cmd_seq_gaps = get_u32(p + 12);
    watchdog_trips = get_u32(p + 16);
    overcurrent_trips = get_u32(p + 20);
    return true;
  }
};

/// Packs a message struct and frames it. Returns bytes written to `out` (0 on error).
template<typename Msg>
inline size_t encode_msg(const Msg & msg, uint8_t seq, uint8_t * out, size_t out_cap)
{
  uint8_t payload[Msg::kSize];
  msg.pack(payload);
  return encode_frame(Msg::kType, seq, payload, Msg::kSize, out, out_cap);
}

/// Unpacks `frame` into `msg` if the type and length match.
template<typename Msg>
inline bool decode_msg(const Frame & frame, Msg & msg)
{
  return frame.type == Msg::kType && msg.unpack(frame.payload, frame.len);
}

/// Unsigned 8-bit gap between consecutive sequence numbers (0 = in order).
inline uint8_t seq_gap(uint8_t previous, uint8_t current)
{
  return static_cast<uint8_t>(current - previous - 1);
}

}  // namespace orion_proto

#endif  // ORION_PROTO_HPP_
