#include "orion_hardware/firmware_link.hpp"

#include <string>
#include <utility>

namespace orion_hardware
{

namespace op = orion_proto;

FirmwareLink::FirmwareLink(std::unique_ptr<Transport> transport, NowFn now, LogFn on_log)
: transport_(std::move(transport)), now_(std::move(now)), on_log_(std::move(on_log)) {}

FirmwareLink::~FirmwareLink() {stop();}

bool FirmwareLink::start(std::string & error)
{
  stop();
  if (!transport_->open(error)) {
    return false;
  }
  failed_ = false;
  running_ = true;
  reader_ = std::thread(&FirmwareLink::reader_loop, this);
  return true;
}

void FirmwareLink::stop()
{
  running_ = false;
  if (reader_.joinable()) {
    reader_.join();
  }
  transport_->close();
}

void FirmwareLink::reader_loop()
{
  uint8_t buf[512];
  while (running_) {
    const long n = transport_->read_some(buf, sizeof(buf), 20);
    if (n < 0) {
      failed_ = true;
      rx_cv_.notify_all();
      if (on_log_) {
        on_log_("serial device failed; reader stopped");
      }
      return;
    }
    if (n > 0) {
      ingest(buf, static_cast<size_t>(n));
    }
  }
}

void FirmwareLink::ingest(const uint8_t * data, size_t len)
{
  for (size_t i = 0; i < len; ++i) {
    bool got = false;
    op::Frame frame;
    {
      std::lock_guard<std::mutex> lock(rx_mutex_);
      got = decoder_.push(data[i]);
      if (got) {
        frame = decoder_.frame();
      }
    }
    if (got) {
      handle_frame(frame);
    }
  }
}

void FirmwareLink::handle_frame(const op::Frame & frame)
{
  if (frame.type == op::MSG_LOG) {
    if (on_log_) {
      on_log_(std::string(reinterpret_cast<const char *>(frame.payload), frame.len));
    }
    return;
  }
  std::unique_lock<std::mutex> lock(rx_mutex_);
  switch (frame.type) {
    case op::MSG_STATE: {
        op::StateMsg msg;
        if (!msg.unpack(frame.payload, frame.len)) {
          break;
        }
        state_ = StateSample{msg, now_(), fw_clock_.update(msg.t_us), ++state_count_};
        lock.unlock();
        rx_cv_.notify_all();
        return;
      }
    case op::MSG_STATS: {
        op::StatsMsg msg;
        if (!msg.unpack(frame.payload, frame.len)) {
          break;
        }
        stats_ = msg;
        return;
      }
    case op::MSG_PONG: {
        op::PongMsg msg;
        if (!msg.unpack(frame.payload, frame.len)) {
          break;
        }
        pong_ = msg;
        ++pong_count_;
        lock.unlock();
        rx_cv_.notify_all();
        return;
      }
    case op::MSG_PARAM_ACK: {
        op::ParamAckMsg msg;
        if (!msg.unpack(frame.payload, frame.len)) {
          break;
        }
        ack_ = msg;
        ++ack_count_;
        lock.unlock();
        rx_cv_.notify_all();
        return;
      }
    default:
      break;
  }
  ++bad_payload_;  // unknown type, host-to-firmware type, or wrong payload size
}

template<typename Msg>
bool FirmwareLink::send(const Msg & msg)
{
  uint8_t payload[Msg::kSize];
  msg.pack(payload);
  return send_raw(Msg::kType, payload, Msg::kSize);
}

bool FirmwareLink::send_raw(uint8_t type, const uint8_t * payload, size_t len)
{
  uint8_t wire[op::kMaxWireFrame];
  std::lock_guard<std::mutex> lock(tx_mutex_);
  const size_t n = op::encode_frame(type, tx_seq_++, payload, len, wire, sizeof(wire));
  const bool ok = n > 0 && transport_->write_all(wire, n);
  ++(ok ? tx_frames_ : tx_dropped_);
  return ok;
}

bool FirmwareLink::send_cmd(const op::CmdMsg & cmd) {return send(cmd);}

bool FirmwareLink::send_estop() {return send_raw(op::MSG_ESTOP, nullptr, 0);}

bool FirmwareLink::send_estop_clear() {return send(op::EstopClearMsg{});}

std::optional<op::PongMsg> FirmwareLink::ping(std::chrono::milliseconds timeout)
{
  uint64_t before;
  {
    std::lock_guard<std::mutex> lock(rx_mutex_);
    before = pong_count_;
  }
  if (!send_raw(op::MSG_PING, nullptr, 0)) {
    return std::nullopt;
  }
  std::unique_lock<std::mutex> lock(rx_mutex_);
  if (!rx_cv_.wait_for(lock, timeout, [&] {return pong_count_ > before || failed_;}) ||
    pong_count_ == before)
  {
    return std::nullopt;
  }
  return pong_;
}

std::optional<op::ParamAckMsg> FirmwareLink::set_param(
  uint8_t id, float value, std::chrono::milliseconds timeout)
{
  uint64_t seen;
  {
    std::lock_guard<std::mutex> lock(rx_mutex_);
    seen = ack_count_;
  }
  op::SetParamMsg req;
  req.id = id;
  req.value = value;
  if (!send(req)) {
    return std::nullopt;
  }
  const auto deadline = Clock::now() + timeout;
  std::unique_lock<std::mutex> lock(rx_mutex_);
  while (true) {
    if (!rx_cv_.wait_until(lock, deadline, [&] {return ack_count_ > seen || failed_;}) ||
      ack_count_ == seen)
    {
      return std::nullopt;
    }
    seen = ack_count_;
    if (ack_ && ack_->id == id) {
      return ack_;
    }
  }
}

bool FirmwareLink::wait_for_state(std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(rx_mutex_);
  const uint64_t before = state_count_;
  return rx_cv_.wait_for(lock, timeout, [&] {return state_count_ > before || failed_;}) &&
         state_count_ > before;
}

std::optional<FirmwareLink::StateSample> FirmwareLink::latest_state() const
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  return state_;
}

std::optional<op::StatsMsg> FirmwareLink::latest_stats() const
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  return stats_;
}

FirmwareLink::Counters FirmwareLink::counters() const
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  Counters c;
  const op::DecoderCounters & d = decoder_.counters();
  c.rx_frames_ok = d.frames_ok;
  c.rx_crc_errors = d.crc_errors;
  c.rx_framing_errors = d.framing_errors + bad_payload_;
  c.tx_frames = tx_frames_.load();
  c.tx_dropped = tx_dropped_.load();
  return c;
}

}  // namespace orion_hardware
