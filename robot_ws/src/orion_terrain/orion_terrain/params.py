"""
Tuning parameters of the terrain estimator (docs/SPEC.md section 4).

The dataclass defaults are the single source of truth; config/terrain.yaml mirrors them for
the ROS node and a unit test keeps the two identical. Robot geometry (wheel radius, track)
is not a parameter here: it comes from orion_learning.obs_spec.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


@dataclass(frozen=True)
class TerrainParams:
    """All estimator parameters. Units are in the field names."""

    # Output
    rate_hz: float = 20.0                     # TerrainState publish rate
    # Body-speed Kalman filter (1-D on v_x)
    q_accel_mps2: float = 0.1                 # process noise: Q = q^2 dt
    r_external_mps: float = 0.02              # measurement std, overhead camera fix
    r_lidar_mps: float = 0.05                 # measurement std, rf2o lidar odometry
    initial_speed_std_mps: float = 0.1        # prior std at start-up (rover assumed at rest)
    gravity_mps2: float = 9.81                # removed from the IMU specific force
    # External pose differentiation
    sg_window: int = 5                        # Savitzky-Golay window (samples), quadratic
    # Staleness: a source is used only while its newest sample is younger than this
    external_timeout_s: float = 0.25
    lidar_timeout_s: float = 0.5
    imu_timeout_s: float = 0.5
    wheel_odom_timeout_s: float = 0.5
    # Confidence c = exp(-t_since_fix / tau) * (1 - min(1, |v_wheel - v_x| / v_disagree))
    confidence_tau_s: float = 3.0
    confidence_disagree_mps: float = 0.5
    # Filters
    slip_tau_s: float = 0.25                  # low-pass on slip and effort residual
    wheel_accel_tau_s: float = 0.05           # low-pass on the finite-difference dw/dt

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> 'TerrainParams':
        """Build from a dict, rejecting unknown keys (catches typos in YAML)."""
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f'unknown terrain parameters: {sorted(unknown)}')
        defaults = cls()
        typed = {k: type(getattr(defaults, k))(v) for k, v in values.items()}
        return cls(**typed)


# Topic names (docs/SPEC.md, docs/INTERFACES.md). Keys are the roles used internally.
DEFAULT_TOPICS = {
    'joint_states': '/joint_states',
    'imu': '/imu/data',
    'wheel_odom': '/diff_drive_controller/odom',
    'ground_truth_pose': '/orion/ground_truth_pose',
    'lidar_odom': '/odom_rf2o',
    'rover_health': '/orion/rover_health',
    'terrain_state': '/orion/terrain_state',
}

# Joint name of each wheel, in obs_spec.WHEEL_ORDER (FL, FR, RL, RR).
DEFAULT_JOINT_NAMES = (
    'front_left_wheel_joint',
    'front_right_wheel_joint',
    'rear_left_wheel_joint',
    'rear_right_wheel_joint',
)
