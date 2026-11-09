// Byte transport under FirmwareLink: a POSIX serial port on the rover, a mock in unit tests.
#ifndef ORION_HARDWARE__TRANSPORT_HPP_
#define ORION_HARDWARE__TRANSPORT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

namespace orion_hardware
{

class Transport
{
public:
  virtual ~Transport() = default;

  /// Opens the device. Returns false and sets `error` on failure.
  virtual bool open(std::string & error) = 0;

  /// Waits up to `timeout_ms` for data. Returns bytes read (> 0), 0 on timeout, or -1 if the
  /// device failed (unplugged, closed); after -1 the transport stays failed until reopened.
  virtual long read_some(uint8_t * buffer, size_t capacity, int timeout_ms) = 0;

  /// Writes without blocking. Returns false if the bytes could not all be queued; the caller
  /// drops the frame (a late command is worse than a lost one: the next arrives in 10 ms).
  virtual bool write_all(const uint8_t * data, size_t len) = 0;

  virtual void close() = 0;
};

/// termios serial port, raw 8N1, non-blocking writes, poll()-based reads.
class SerialTransport : public Transport
{
public:
  SerialTransport(std::string port, int baud);
  ~SerialTransport() override;

  bool open(std::string & error) override;
  long read_some(uint8_t * buffer, size_t capacity, int timeout_ms) override;
  bool write_all(const uint8_t * data, size_t len) override;
  void close() override;

private:
  std::string port_;
  int baud_;
  int fd_ = -1;
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__TRANSPORT_HPP_
