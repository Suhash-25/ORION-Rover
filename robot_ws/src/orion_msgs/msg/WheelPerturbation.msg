# Wheel-velocity perturbation for fault-injection experiments (reserved for Round 7).
# orion_hardware adds `offsets` to the wheel velocities the controller commands, for
# `duration_s` from receipt, then stops on its own. Off unless the hardware parameter
# perturbation_max_rad_s > 0 (default 0). Rejected when any |offset| exceeds that bound, when the
# latest SafetyStatus is not NOMINAL/LIMITED or older than 0.5 s, or while e-stopped. An active
# perturbation is cancelled by SafetyStatus HALT or ESTOP, a stale SafetyStatus, /orion/estop or
# deactivation of the hardware. A new message replaces the active one.

std_msgs/Header header
float32[4] offsets           # rad/s per wheel, FL FR RL RR, URDF joint sign (+ drives forward)
float32 duration_s           # s from receipt; <= 0 cancels the active perturbation
