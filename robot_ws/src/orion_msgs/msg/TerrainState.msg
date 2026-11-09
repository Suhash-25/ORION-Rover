# Terrain estimate from orion_terrain at 20 Hz. Wheel order everywhere: FL, FR, RL, RR.
uint8 SPEED_SOURCE_NONE=0
uint8 SPEED_SOURCE_IMU_ONLY=1
uint8 SPEED_SOURCE_LIDAR_ODOM=2
uint8 SPEED_SOURCE_EXTERNAL=3
uint8 EFFORT_SOURCE_NONE=0
uint8 EFFORT_SOURCE_CURRENT=1
uint8 EFFORT_SOURCE_DUTY_ESTIMATE=2
uint8 EFFORT_SOURCE_SIM_TORQUE=3

std_msgs/Header header
float32[4] slip              # (r*w - v_side) / max(|r*w|, |v_side|, 0.02); + spinning, - skidding; [-1, 1]
float32[2] slip_side         # left, right mean slip
float32[4] effort_residual   # measured effort - flat-ground model (A on hardware, N.m in sim)
float32 sinkage_proxy        # [0, 1], clip(mean(effort_residual) / I_ref, 0, 1)
float32 body_speed           # m/s, longitudinal body speed estimate used for slip
float32 body_yaw_rate        # rad/s, gyro z
uint8 body_speed_source      # SPEED_SOURCE_*
uint8 effort_source          # EFFORT_SOURCE_*
float32 confidence           # [0, 1] confidence in body_speed (and so in slip)
float32 time_since_fix       # s since the last non-IMU velocity fix
