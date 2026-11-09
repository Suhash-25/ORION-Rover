#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include "orion_hardware/transport.hpp"

namespace orion_hardware
{

namespace
{

bool baud_constant(int baud, speed_t & out)
{
  switch (baud) {
    case 9600: out = B9600; return true;
    case 19200: out = B19200; return true;
    case 38400: out = B38400; return true;
    case 57600: out = B57600; return true;
    case 115200: out = B115200; return true;
    case 230400: out = B230400; return true;
    case 460800: out = B460800; return true;
    case 921600: out = B921600; return true;
    case 1000000: out = B1000000; return true;
    case 2000000: out = B2000000; return true;
    default: return false;
  }
}

std::string errno_text(const std::string & what)
{
  return what + ": " + std::strerror(errno);
}

}  // namespace

SerialTransport::SerialTransport(std::string port, int baud)
: port_(std::move(port)), baud_(baud) {}

SerialTransport::~SerialTransport() {close();}

bool SerialTransport::open(std::string & error)
{
  close();
  speed_t speed;
  if (!baud_constant(baud_, speed)) {
    error = "unsupported baud rate " + std::to_string(baud_);
    return false;
  }
  fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) {
    error = errno_text("open " + port_);
    return false;
  }
  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    error = errno_text("tcgetattr " + port_);
    close();
    return false;
  }
  cfmakeraw(&tio);
  tio.c_cflag |= CLOCAL | CREAD;
  tio.c_cflag &= ~static_cast<tcflag_t>(CSTOPB | CRTSCTS);
  tio.c_cc[VMIN] = 0;
  tio.c_cc[VTIME] = 0;
  cfsetispeed(&tio, speed);
  cfsetospeed(&tio, speed);
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    error = errno_text("tcsetattr " + port_);
    close();
    return false;
  }
  tcflush(fd_, TCIOFLUSH);
  return true;
}

long SerialTransport::read_some(uint8_t * buffer, size_t capacity, int timeout_ms)
{
  if (fd_ < 0) {
    return -1;
  }
  pollfd pfd{fd_, POLLIN, 0};
  const int ready = ::poll(&pfd, 1, timeout_ms);
  if (ready < 0) {
    return errno == EINTR ? 0 : -1;
  }
  if (ready == 0) {
    return 0;
  }
  if ((pfd.revents & POLLIN) == 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
    return -1;
  }
  const ssize_t n = ::read(fd_, buffer, capacity);
  if (n > 0) {
    return static_cast<long>(n);
  }
  if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
    return 0;
  }
  return -1;  // EOF or EIO: the device went away
}

bool SerialTransport::write_all(const uint8_t * data, size_t len)
{
  if (fd_ < 0) {
    return false;
  }
  size_t done = 0;
  while (done < len) {
    const ssize_t n = ::write(fd_, data + done, len - done);
    if (n > 0) {
      done += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      return false;  // EAGAIN (output buffer full) or a real error
    }
  }
  return true;
}

void SerialTransport::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

}  // namespace orion_hardware
