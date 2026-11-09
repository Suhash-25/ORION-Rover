# Low-level health from orion_hardware at 10 Hz.
uint8 FAULT_WATCHDOG=1
uint8 FAULT_OVERCURRENT=2
uint8 FAULT_ENCODER=4
uint8 FAULT_UNDERVOLTAGE=8
uint8 FAULT_ESTOP_LATCHED=16

std_msgs/Header header
float32 battery_v            # V
uint8 fw_fault_flags         # bitwise OR of FAULT_*
uint32 fw_uptime_ms
uint32 crc_errors            # cumulative frames dropped for bad CRC / framing
uint32 watchdog_trips        # cumulative firmware watchdog trips
float32[4] duty              # [-1, 1] PWM duty per wheel, FL FR RL RR
bool current_sensed          # false when effort is estimated from duty instead of measured
string fw_git_hash           # 8 hex digits of the firmware commit from the PONG handshake
