// Host side of the firmware link: a background reader thread that decodes frames and keeps the
// latest STATE/STATS, plus thread-safe frame senders. ROS-free so it is unit-testable.
#ifndef ORION_HARDWARE__FIRMWARE_LINK_HPP_
#define ORION_HARDWARE__FIRMWARE_LINK_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "orion_hardware/conversions.hpp"
#include "orion_hardware/transport.hpp"
#include "orion_proto.hpp"

namespace orion_hardware
{

class FirmwareLink
{
public:
  using Clock = std::chrono::steady_clock;
  using NowFn = std::function<Clock::time_point ()>;
  using LogFn = std::function<void (const std::string &)>;

  struct StateSample
  {
    orion_proto::StateMsg msg;
    Clock::time_point rx_time;  // host time the frame was decoded
    uint64_t fw_time_us = 0;    // STATE.t_us unwrapped to 64 bits
    uint64_t count = 0;         // number of STATE frames received so far
  };

  struct Counters
  {
    uint32_t rx_frames_ok = 0;
    uint32_t rx_crc_errors = 0;
    uint32_t rx_framing_errors = 0;  // includes valid frames with a bad payload size
    uint32_t tx_frames = 0;
    uint32_t tx_dropped = 0;         // frames the transport could not queue
  };

  /// `now` is injectable for tests; `on_log` receives firmware LOG text (reader thread).
  explicit FirmwareLink(
    std::unique_ptr<Transport> transport, NowFn now = Clock::now, LogFn on_log = nullptr);
  ~FirmwareLink();

  FirmwareLink(const FirmwareLink &) = delete;
  FirmwareLink & operator=(const FirmwareLink &) = delete;

  /// Opens the transport and starts the reader thread.
  bool start(std::string & error);
  /// Stops the reader thread and closes the transport. Idempotent.
  void stop();

  bool send_cmd(const orion_proto::CmdMsg & cmd);
  bool send_estop();
  bool send_estop_clear();

  /// PING and wait for PONG. Returns nullopt on timeout.
  std::optional<orion_proto::PongMsg> ping(std::chrono::milliseconds timeout);
  /// SET_PARAM and wait for the matching PARAM_ACK. Returns nullopt on timeout.
  std::optional<orion_proto::ParamAckMsg> set_param(
    uint8_t id, float value, std::chrono::milliseconds timeout);
  /// Waits for a STATE received after this call. Returns false on timeout.
  bool wait_for_state(std::chrono::milliseconds timeout);

  std::optional<StateSample> latest_state() const;
  std::optional<orion_proto::StatsMsg> latest_stats() const;
  Counters counters() const;
  /// True once the transport reported a hard failure (device gone).
  bool failed() const {return failed_.load();}

  /// Feeds raw bytes as if read from the transport (used by the reader thread and by tests).
  void ingest(const uint8_t * data, size_t len);

private:
  template<typename Msg>
  bool send(const Msg & msg);
  bool send_raw(uint8_t type, const uint8_t * payload, size_t len);
  void reader_loop();
  void handle_frame(const orion_proto::Frame & frame);

  std::unique_ptr<Transport> transport_;
  NowFn now_;
  LogFn on_log_;

  std::thread reader_;
  std::atomic<bool> running_{false};
  std::atomic<bool> failed_{false};

  std::mutex tx_mutex_;
  uint8_t tx_seq_ = 0;

  mutable std::mutex rx_mutex_;
  std::condition_variable rx_cv_;
  orion_proto::FrameDecoder decoder_;
  uint32_t bad_payload_ = 0;
  std::atomic<uint32_t> tx_frames_{0};
  std::atomic<uint32_t> tx_dropped_{0};
  std::optional<StateSample> state_;
  std::optional<orion_proto::StatsMsg> stats_;
  std::optional<orion_proto::PongMsg> pong_;
  std::optional<orion_proto::ParamAckMsg> ack_;
  uint64_t state_count_ = 0;
  uint64_t pong_count_ = 0;
  uint64_t ack_count_ = 0;
  MicrosUnwrapper fw_clock_;
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__FIRMWARE_LINK_HPP_
