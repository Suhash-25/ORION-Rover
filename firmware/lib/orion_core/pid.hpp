// Wheel speed PID with feed-forward, derivative on measurement and conditional-integration
// anti-windup. Units: target/measured in rad/s, output in duty [-1, 1], dt in s.
#ifndef ORION_CORE_PID_HPP_
#define ORION_CORE_PID_HPP_

namespace orion_fw
{

struct PidGains
{
  float kp = 0.0f;   // duty per rad/s
  float ki = 0.0f;   // duty per rad
  float kd = 0.0f;   // duty per rad/s^2
  float kff = 0.0f;  // duty per rad/s of target
};

class Pid
{
public:
  /// Forget the integrator and derivative history (used whenever control is re-enabled).
  void reset()
  {
    integral_ = 0.0f;
    last_output_ = 0.0f;
    has_previous_ = false;
  }

  /// One control step. `limit` (>= 0) clamps the output to [-limit, limit]. The integrator only
  /// moves when that does not push a saturated output further into saturation, and is itself
  /// kept within [-limit, limit], so it cannot wind up while the output is clamped.
  float update(const PidGains & g, float target, float measured, float dt, float limit)
  {
    if (dt <= 0.0f) {
      return clamp(last_output_, limit);
    }
    const float error = target - measured;
    const float derivative = has_previous_ ? -(measured - previous_measured_) / dt : 0.0f;
    previous_measured_ = measured;
    has_previous_ = true;

    const float base = g.kff * target + g.kp * error + g.kd * derivative;
    const float candidate = clamp(integral_ + g.ki * error * dt, limit);
    const float unclamped = base + candidate;
    const bool pushes_high = unclamped > limit && error > 0.0f;
    const bool pushes_low = unclamped < -limit && error < 0.0f;
    if (!pushes_high && !pushes_low) {
      integral_ = candidate;
    }
    last_output_ = clamp(base + integral_, limit);
    return last_output_;
  }

  float integral() const {return integral_;}

private:
  static float clamp(float v, float limit)
  {
    return v > limit ? limit : (v < -limit ? -limit : v);
  }

  float integral_ = 0.0f;
  float previous_measured_ = 0.0f;
  float last_output_ = 0.0f;
  bool has_previous_ = false;
};

}  // namespace orion_fw

#endif  // ORION_CORE_PID_HPP_
