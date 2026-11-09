// In-memory Transport for unit tests. A test plays firmware through `respond` (called for each
// frame the host writes) and can stream STATE frames every 10 ms.
#ifndef MOCK_TRANSPORT_HPP_
#define MOCK_TRANSPORT_HPP_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "orion_hardware/transport.hpp"
#include "orion_proto.hpp"

namespace orion_hardware_test
{

namespace op = orion_proto;

/// In-memory transport. `respond` lets a test play firmware for frames the host writes.
class MockTransport : public orion_hardware::Transport
{
public:
  struct Shared
  {
    std::mutex m;
    std::deque<uint8_t> to_host;
    std::vector<uint8_t> from_host;
    std::atomic<bool> fail{false};
    std::atomic<bool> accept_writes{true};
    std::function<void(const op::Frame &, Shared &)> respond;
    op::FrameDecoder decoder;
    std::atomic<bool> stream_state{false};  // emit `state` every 10 ms from read_some()
    op::StateMsg state;
    std::chrono::steady_clock::time_point last_state{};
    uint8_t seq = 0;

    template<typename Msg>
    void push(const Msg & msg)
    {
      uint8_t wire[op::kMaxWireFrame];
      const size_t n = op::encode_msg(msg, seq++, wire, sizeof(wire));
      to_host.insert(to_host.end(), wire, wire + n);
    }
  };

  explicit MockTransport(std::shared_ptr<Shared> s)
  : s_(std::move(s)) {}

  bool open(std::string &) override {return true;}
  long read_some(uint8_t * buf, size_t cap, int timeout_ms) override
  {
    if (s_->fail) {
      return -1;
    }
    {
      std::lock_guard<std::mutex> lock(s_->m);
      const auto now = std::chrono::steady_clock::now();
      if (s_->stream_state && now - s_->last_state >= std::chrono::milliseconds(10)) {
        s_->last_state = now;
        s_->state.t_us += 10000;
        s_->push(s_->state);
      }
      if (!s_->to_host.empty()) {
        size_t n = 0;
        while (n < cap && !s_->to_host.empty()) {
          buf[n++] = s_->to_host.front();
          s_->to_host.pop_front();
        }
        return static_cast<long>(n);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(std::min(timeout_ms, 2)));
    return 0;
  }
  bool write_all(const uint8_t * data, size_t len) override
  {
    if (!s_->accept_writes) {
      return false;
    }
    std::lock_guard<std::mutex> lock(s_->m);
    s_->from_host.insert(s_->from_host.end(), data, data + len);
    for (size_t i = 0; i < len; ++i) {
      if (s_->decoder.push(data[i]) && s_->respond) {
        s_->respond(s_->decoder.frame(), *s_);
      }
    }
    return true;
  }
  void close() override {}

private:
  std::shared_ptr<Shared> s_;
};

}  // namespace orion_hardware_test

#endif  // MOCK_TRANSPORT_HPP_
