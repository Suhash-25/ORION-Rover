"""The single definition of what the ORION policy sees and does.

Imported by the training code (learning/) and by the rover runtime (orion_terrain,
orion_policy). If training and runtime disagree on anything here, the policy silently
gets garbage on the real rover, so nothing else may re-define these layouts.

Changing anything below changes OBS_SPEC_HASH, and orion_policy refuses to load an ONNX
model exported with a different hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

# --- Robot geometry (must match orion_description/urdf/common/constants.xacro) -------------
WHEEL_RADIUS_M = 0.12
TRACK_WIDTH_M = 0.52          # wheel_separation used by diff_drive_controller
WHEELBASE_M = 0.50
NOMINAL_MASS_KG = 14.0
WHEEL_ORDER = ('FL', 'FR', 'RL', 'RR')
LEFT_WHEELS = (0, 2)
RIGHT_WHEELS = (1, 3)

# --- Timing ----------------------------------------------------------------------------------
CONTROL_HZ = 20
HISTORY_LEN = 40              # steps fed to the adaptation module (2 s)
LATENT_DIM = 8

# --- Action ------------------------------------------------------------------------------------
MAX_LINEAR_SPEED = 0.5        # m/s at action = 1
MAX_ANGULAR_SPEED = 1.0       # rad/s at action = 1
MAX_LINEAR_ACCEL = 1.0        # m/s^2 rate limit applied after the policy
MAX_ANGULAR_ACCEL = 2.0       # rad/s^2
ACTION_DIM = 2

# --- Normalisation scales ----------------------------------------------------------------------
WHEEL_SPEED_SCALE = MAX_LINEAR_SPEED / WHEEL_RADIUS_M   # rad/s
ACCEL_SCALE = 9.81            # m/s^2, constant for every gravity level
YAW_RATE_SCALE = 2.0          # rad/s
TILT_SCALE = 0.5              # rad
OBS_CLIP = 1.5
SLIP_FLOOR_MPS = 0.02


@dataclass(frozen=True)
class Field:
    name: str
    size: int
    unit: str
    source: str


OBS_FIELDS: tuple[Field, ...] = (
    Field('prev_cmd', 2, 'v m/s / 0.5, w rad/s / 1.0', 'own last action'),
    Field('wheel_speed', 4, 'rad/s / WHEEL_SPEED_SCALE', '/joint_states velocity'),
    Field('wheel_effort', 4, 'effort / effort_max', '/joint_states effort'),
    Field('body_accel_xy', 2, 'gravity-removed m/s^2 / 9.81', '/imu/data'),
    Field('yaw_rate', 1, 'rad/s / 2.0', '/imu/data angular_velocity.z'),
    Field('roll_pitch', 2, 'rad / 0.5', '/imu/data orientation'),
    Field('body_speed', 1, 'm/s / 0.5', 'TerrainState.body_speed'),
    Field('speed_confidence', 1, '[0, 1]', 'TerrainState.confidence'),
    Field('slip_side', 2, '[-1, 1]', 'TerrainState.slip_side'),
    Field('sinkage_proxy', 1, '[0, 1]', 'TerrainState.sinkage_proxy'),
)
OBS_DIM = sum(f.size for f in OBS_FIELDS)

PRIVILEGED_FIELDS: tuple[Field, ...] = (
    Field('dp_coeff_at_s30', 2, 'drawbar pull / W at 30% slip, per side', 'env'),
    Field('slip_at_dp10', 2, 'slip at drawbar pull = 0.1 W, per side', 'env'),
    Field('static_sinkage_over_r', 2, 'z0 / r, per side', 'env'),
    Field('resistance_over_w', 2, 'Rc / W, per side', 'env'),
    Field('mass_ratio', 1, 'm_total / 14 kg', 'env'),
    Field('gravity_ratio', 1, 'g / 9.81', 'env'),
)
PRIVILEGED_DIM = sum(f.size for f in PRIVILEGED_FIELDS)


def _layout() -> dict:
    return {
        'obs': [(f.name, f.size, f.unit) for f in OBS_FIELDS],
        'privileged': [(f.name, f.size, f.unit) for f in PRIVILEGED_FIELDS],
        'scales': [WHEEL_RADIUS_M, TRACK_WIDTH_M, MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
                   WHEEL_SPEED_SCALE, ACCEL_SCALE, YAW_RATE_SCALE, TILT_SCALE, OBS_CLIP],
        'timing': [CONTROL_HZ, HISTORY_LEN, LATENT_DIM, ACTION_DIM],
        'wheel_order': WHEEL_ORDER,
    }


OBS_SPEC_HASH = hashlib.sha256(
    json.dumps(_layout(), sort_keys=True).encode()).hexdigest()[:16]


def field_slice(name: str) -> slice:
    """Index range of an observation field, e.g. obs[field_slice('slip_side')]."""  # noqa: D402
    start = 0
    for f in OBS_FIELDS:
        if f.name == name:
            return slice(start, start + f.size)
        start += f.size
    raise KeyError(name)


def wheel_slip(wheel_speed: Sequence[float], body_speed: float, yaw_rate: float) -> np.ndarray:
    """Per-wheel longitudinal slip, FL FR RL RR, bounded to [-1, 1].

    s_i = (r w_i - v_side) / max(|r w_i|, |v_side|, 0.02); positive = spinning.
    Also accepts a batch: wheel_speed (..., 4), body_speed and yaw_rate (...).
    """
    w = np.asarray(wheel_speed, dtype=np.float64)
    v = np.asarray(body_speed, dtype=np.float64)[..., None]
    half = np.asarray(yaw_rate, dtype=np.float64)[..., None] * TRACK_WIDTH_M / 2.0
    v_left = v - half
    v_right = v + half
    v_side = np.concatenate([v_left, v_right, v_left, v_right], axis=-1)
    rim = WHEEL_RADIUS_M * w
    denom = np.maximum(np.maximum(np.abs(rim), np.abs(v_side)), SLIP_FLOOR_MPS)
    return np.clip((rim - v_side) / denom, -1.0, 1.0)


def side_mean(per_wheel: Sequence[float]) -> np.ndarray:
    """Mean of the left (FL, RL) and right (FR, RR) wheels; (..., 4) -> (..., 2)."""
    a = np.asarray(per_wheel, dtype=np.float64)
    return np.stack([a[..., list(LEFT_WHEELS)].mean(axis=-1),
                     a[..., list(RIGHT_WHEELS)].mean(axis=-1)], axis=-1)


def build_observation(
    *,
    prev_cmd: Sequence[float],
    wheel_speed: Sequence[float],
    wheel_effort: Sequence[float],
    effort_max: float,
    body_accel_xy: Sequence[float],
    yaw_rate: float,
    roll: float,
    pitch: float,
    body_speed: float,
    speed_confidence: float,
    slip_side: Sequence[float],
    sinkage_proxy: float,
) -> np.ndarray:
    """Physical units in, the normalised OBS_DIM float32 vector out.

    prev_cmd is (v m/s, w rad/s). wheel_effort is in the same unit as effort_max
    (amps on the rover, N.m in simulation). body_accel_xy must already have gravity removed.
    Also accepts a batch: every argument may carry the same leading batch shape as
    body_speed (e.g. wheel_speed (N, 4), yaw_rate (N,)); the result is then (N, OBS_DIM).
    """
    batch = np.shape(body_speed)

    def part(x, size, scale=1.0):
        return np.reshape(np.asarray(x, dtype=np.float64), batch + (size,)) / scale

    parts = [
        part(prev_cmd, 2, np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])),
        part(wheel_speed, 4, WHEEL_SPEED_SCALE),
        part(wheel_effort, 4, np.reshape(np.asarray(effort_max, dtype=np.float64),
                                         np.shape(effort_max) + (1,))),
        part(body_accel_xy, 2, ACCEL_SCALE),
        part(yaw_rate, 1, YAW_RATE_SCALE),
        np.stack(np.broadcast_arrays(np.asarray(roll, dtype=np.float64) / TILT_SCALE,
                                     np.asarray(pitch, dtype=np.float64) / TILT_SCALE), -1),
        part(body_speed, 1, MAX_LINEAR_SPEED),
        part(speed_confidence, 1),
        part(slip_side, 2),
        part(sinkage_proxy, 1),
    ]
    obs = np.concatenate(parts, axis=-1)
    if obs.shape != batch + (OBS_DIM,):
        raise ValueError(f'observation has shape {obs.shape}, expected {batch + (OBS_DIM,)}')
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def action_to_twist(action: Sequence[float]) -> tuple[float, float]:
    """Policy action in [-1, 1]^2 to (v m/s, w rad/s), before rate and safety limits."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float(a[0] * MAX_LINEAR_SPEED), float(a[1] * MAX_ANGULAR_SPEED)


def rate_limit(prev: Sequence[float], target: Sequence[float], dt: float) -> tuple[float, float]:
    """Limit the change of (v, w) to MAX_LINEAR_ACCEL / MAX_ANGULAR_ACCEL over dt seconds."""
    dv = np.clip(target[0] - prev[0], -MAX_LINEAR_ACCEL * dt, MAX_LINEAR_ACCEL * dt)
    dw = np.clip(target[1] - prev[1], -MAX_ANGULAR_ACCEL * dt, MAX_ANGULAR_ACCEL * dt)
    return float(prev[0] + dv), float(prev[1] + dw)


def describe() -> Mapping[str, object]:
    """Metadata embedded in exported ONNX files and checked at load time."""
    return {'obs_spec_hash': OBS_SPEC_HASH, 'obs_dim': OBS_DIM, 'control_hz': CONTROL_HZ,
            'history': HISTORY_LEN, 'latent_dim': LATENT_DIM, 'action_dim': ACTION_DIM}
