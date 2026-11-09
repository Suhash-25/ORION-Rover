# Safety state from orion_safety at 50 Hz. All command sources must obey the limits.
uint8 STATE_NOMINAL=0
uint8 STATE_LIMITED=1
uint8 STATE_HALT=2
uint8 STATE_ESTOP=3
uint8 FLAG_TILT=1
uint8 FLAG_OVERCURRENT=2
uint8 FLAG_HEARTBEAT=4
uint8 FLAG_GEOFENCE=8
uint8 FLAG_STUCK=16
uint8 FLAG_FIRMWARE_FAULT=32
uint8 FLAG_HARD_STUCK=64

std_msgs/Header header
uint8 state                  # STATE_*
uint8 flags                  # bitwise OR of FLAG_*
float32 max_linear_speed     # m/s, limit for every command source
float32 max_angular_speed    # rad/s
float32 max_current          # A per wheel
string reason                # human-readable, empty when nominal
