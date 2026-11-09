"""Vectorised ORION rover-on-soil environment (docs/SPEC.md section 6).

N rovers are simulated in lock-step with NumPy. Control period 0.05 s (20 Hz) with 4 physics
substeps. Observations are built only through obs_spec.build_observation / wheel_slip and
the shared estimator maths, so what the policy sees in training is computed exactly like on
the rover.

Physics per substep (all SI):

* wheel i at body position (x_i, y_i) (front x = +L/2, left y = +b/2) sees longitudinal
  ground speed v_x - w_z y_i and lateral speed v_y + w_z x_i;
* wheel load W_i = m g / 4 -+ m h a_x / (2 L) (front unloads under forward acceleration);
* sinkage: static part z0(W_i) immediately; the slip part is dug towards
  z_target(s_i) = z0 (1 + a_s |s| / (1 - |s| + 0.05)) at DIG_RATE (0.1 m per metre of slipped
  rim travel |r w - v|) and relaxes "stickily", z <- max(z_target, z - 0.3 |v_long| dt), only
  while the wheel travels onto fresh soil. (The SPEC jumps to z_target instantly; then the
  start-up slip transient buries 84 % of randomised rovers within 1 s, median 0.3 s, before
  any controller can react. With the dig rate the median is 2.8 s. See the engineering log.)
* longitudinal force F_i = m_fH H_i - m_fR Rc_i tanh(v_long,i / 0.01 m/s) (Rc opposes travel);
* lateral force Fy_i = -mu_y W_i tanh(v_lat,i / 0.05 m/s), with the per-wheel lateral speed,
  which gives the skid-steer yaw resistance;
* wheel J dw/dt = tau - r m_fH H, tau from a PI speed loop clipped to +-tau_max (x the
  torque-fault factor);
* body m dv_x = sum F + m v_y w_z, m dv_y = sum Fy - m v_x w_z,
  I_z dw_z = sum(-y_i F_i + x_i Fy_i), I_z = 0.9 kg m^2 x m / 14 kg.

The contact is very stiff near standstill (the slip denominator floors at 0.02 m/s and Rc and
Fy switch direction within 1-5 cm/s), so explicit Euler is unstable at dt = 12.5 ms. Each
substep writes every contact force as a non-negative secant coefficient times a velocity
(thrust = k (r w - v), Rc = c v, Fy = -d v_lat, coefficients taken at the current state) and
solves the 7 velocities [w1..w4, v_x, v_y, w_z] implicitly with one batched 7x7 solve. The
scheme is dissipative, so velocities never overshoot through zero.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from orion_learning import baselines
from orion_learning import estimator as est
from orion_learning import obs_spec as S
from orion_learning import soil as soil_mod

# --- Fixed rover constants (SPEC section 6) -------------------------------------------------
DT = 1.0 / S.CONTROL_HZ               # s, control period
SUBSTEPS = 4
DT_SUB = DT / SUBSTEPS
R = S.WHEEL_RADIUS_M
TRACK = S.TRACK_WIDTH_M
WHEELBASE = S.WHEELBASE_M
CG_HEIGHT_M = 0.15                    # GUESS
WHEEL_INERTIA = 0.011                 # kg m^2
YAW_INERTIA_NOMINAL = 0.9             # kg m^2 at the nominal 14 kg, scaled with mass
TAU_MAX = 3.0                         # N m, GUESS; also the effort normaliser
sim_effort_max = TAU_MAX              # effort_max passed to build_observation in simulation
PI_KP = 0.5                           # N m s / rad, wheel speed loop (stands in for firmware PID)
PI_KI = 5.0                           # N m / rad
RC_SPEED_EPS = 0.01                   # m/s, regularises the direction of Rc near standstill
LATERAL_SPEED_EPS = 0.05              # m/s, from the SPEC lateral force law
SINKAGE_RELAX = 0.3                   # sticky sinkage relaxation per metre travelled
DIG_RATE = 0.1                        # GUESS: slip sinkage dug per metre of slipped rim travel
BELLY_OUT_Z = 0.08                    # m
HARD_STUCK_SPEED = 0.02               # m/s
HARD_STUCK_CMD = 0.1                  # m/s rim speed
HARD_STUCK_TIME = 10.0                # s
FENCE_HALF = 1.0                      # m, 2 x 2 m sandbox centred at the origin
START_XY = (-0.85, 0.0)               # m, start pose; goal line 1.5 m ahead along goal_dir
GOAL_DIR = 0.0                        # rad, goal line normal (world x)

WHEEL_X = np.array([1, 1, -1, -1]) * WHEELBASE / 2      # FL FR RL RR
WHEEL_Y = np.array([1, -1, 1, -1]) * TRACK / 2
WHEEL_SIDE = np.array([0, 1, 0, 1])                       # 0 left, 1 right
LOAD_TRANSFER_SIGN = np.array([-1.0, -1.0, 1.0, 1.0])     # front unloads when accelerating

# Flat-ground effort model in N.m, fitted on packed-light driving with
# `python -m orion_learning.env --calibrate` (tests check the fit is still current).
SIM_EFFORT_MODEL = est.EffortModel(np.array([0.30494, 0.14439, 0.04765, 0.0]), ref=2.69506)


def _logu(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


@dataclass(frozen=True)
class SoilRanges:
    """Sampling ranges of one soil (k in kN units, c in kPa, phi in deg, K in m)."""

    k_c_kn: tuple[float, float] = (0.5, 15.0)        # kN/m^(n+1), log-uniform
    k_phi_kn: tuple[float, float] = (400.0, 2500.0)  # kN/m^(n+2), log-uniform
    n: tuple[float, float] = (0.7, 1.3)
    c_kpa: tuple[float, float] = (0.0, 2.0)
    phi_deg: tuple[float, float] = (25.0, 40.0)
    K_m: tuple[float, float] = (0.01, 0.04)
    a_s: tuple[float, float] = (0.3, 2.0)

    def sample(self, rng, size) -> np.ndarray:
        """(size, 7) array ordered as soil.SOIL_KEYS, SI units."""
        return np.stack([
            _logu(rng, *self.k_c_kn, size) * 1e3,
            _logu(rng, *self.k_phi_kn, size) * 1e3,
            rng.uniform(*self.n, size),
            rng.uniform(*self.c_kpa, size) * 1e3,
            np.radians(rng.uniform(*self.phi_deg, size)),
            rng.uniform(*self.K_m, size),
            rng.uniform(*self.a_s, size),
        ], axis=-1)


@dataclass(frozen=True)
class EnvConfig:
    """Randomisation ranges and task constants (SPEC section 6). Defaults = training."""

    soil: SoilRanges = field(default_factory=SoilRanges)
    p_side_hetero: float = 0.5          # right side gets an independent soil sample
    loose_patch: bool = True
    patch_radius_m: tuple[float, float] = (0.3, 0.6)
    patch_soil: SoilRanges = field(default_factory=lambda: SoilRanges(
        k_c_kn=(0.5, 3.0), k_phi_kn=(400.0, 1000.0), n=(1.0, 1.3), c_kpa=(0.0, 0.5),
        phi_deg=(25.0, 30.0), K_m=(0.02, 0.04)))
    beta: tuple[float, float] = (0.0, 1.0)
    gravity: tuple[float, float] = (1.62, 9.81)     # m/s^2, log-uniform
    mass_kg: tuple[float, float] = (12.0, 30.0)
    m_fH: tuple[float, float] = (0.7, 1.4)          # log-uniform
    m_fR: tuple[float, float] = (0.7, 1.4)          # log-uniform
    mu_y: tuple[float, float] = (0.3, 0.8)
    latency_steps: tuple[int, int] = (1, 4)         # inclusive
    p_torque_fault: float = 0.2
    torque_fault_factor: tuple[float, float] = (0.3, 1.0)
    heading_deg: tuple[float, float] = (-30.0, 30.0)
    # Sensor noise (1 sigma) and speed-fix dropouts.
    noise_wheel_speed: float = 0.05     # rad/s
    noise_effort_frac: float = 0.03     # x tau_max
    noise_accel: float = 0.1            # m/s^2
    noise_gyro: float = 0.02            # rad/s
    noise_speed_fix: float = 0.03       # m/s
    accel_bias_rw: float = 0.02         # m/s^2 per sqrt(s)
    p_fix_dropout: float = 0.3          # probability an episode has fix dropouts
    dropout_rate_hz: float = 0.1        # dropout onsets per second in such an episode
    dropout_duration_s: tuple[float, float] = (0.0, 6.0)
    # Task.
    episode_s: float = 60.0
    goal_m: float = 1.5
    # Reward weights (GUESS, SPEC section 6). Shaping terms are per second of driving.
    w_progress: float = 10.0
    w_goal: float = 10.0
    w_belly_out: float = -20.0
    w_hard_stuck: float = -10.0
    w_step: float = -0.02
    w_slip: float = -0.5
    w_sinkage: float = -1.0
    w_action_rate: float = -0.05
    w_fence: float = -5.0

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_s / DT))


# --- Sandbox cells (SPEC section 8), as degenerate sampling ranges --------------------------
CELL_MASS_KG = {'light': 14.0, 'heavy': 22.0}     # heavy = +8 kg ballast (GUESS)
CELL_SOIL = {
    # Packed: strong, peak steady DP/W ~ 0.37 at 14 kg.
    'packed': SoilRanges(k_c_kn=(12.0, 12.0), k_phi_kn=(2500.0, 2500.0), n=(0.8, 0.8),
                         c_kpa=(2.0, 2.0), phi_deg=(38.0, 38.0), K_m=(0.01, 0.01),
                         a_s=(0.3, 0.3)),
    # Loose: marginal, peak steady DP/W ~ 0.08 at 14 kg and ~ 0.04 at 22 kg, at s ~ 0.35-0.4;
    # driving past that slip digs the wheels in.
    'loose': SoilRanges(k_c_kn=(2.0, 2.0), k_phi_kn=(1200.0, 1200.0), n=(1.2, 1.2),
                        c_kpa=(1.0, 1.0), phi_deg=(34.0, 34.0), K_m=(0.015, 0.015),
                        a_s=(0.5, 0.5)),
}
CELL_BETA = 0.5


def cell_config(cell: str, gravity: float = 9.81, base: EnvConfig | None = None) -> EnvConfig:
    """Return the EnvConfig of a sandbox cell, e.g. 'light-packed', at the given gravity.

    Soil, mass and gravity are fixed; model-form multipliers are 1, no loose patch, no side
    heterogeneity and no torque faults. Sensor noise, latency and fix dropouts stay random.
    """
    mass_name, soil_name = cell.split('-')
    m = CELL_MASS_KG[mass_name]
    return replace(base or EnvConfig(), soil=CELL_SOIL[soil_name], p_side_hetero=0.0,
                   loose_patch=False, beta=(CELL_BETA, CELL_BETA), gravity=(gravity, gravity),
                   mass_kg=(m, m), m_fH=(1.0, 1.0), m_fR=(1.0, 1.0), mu_y=(0.5, 0.5),
                   p_torque_fault=0.0)


CELLS = ('light-packed', 'light-loose', 'heavy-packed', 'heavy-loose')


def _tanh_over_x(v, eps):
    """tanh(v / eps) / v, with its limit 1 / eps at v = 0 (non-negative, smooth)."""
    small = np.abs(v) < 1e-6 * eps
    vs = np.where(small, 1.0, v)
    return np.where(small, 1.0 / eps, np.tanh(vs / eps) / vs)


def physical_slip(w, v_long):
    """Physics slip of each wheel, same law as obs_spec.wheel_slip; positive = spinning."""
    rim = R * w
    den = np.maximum(np.maximum(np.abs(rim), np.abs(v_long)), S.SLIP_FLOOR_MPS)
    return np.clip((rim - v_long) / den, -1.0, 1.0)


def rate_limit_batch(prev, target, dt=DT):
    """Vectorised obs_spec.rate_limit: (N, 2) twists (v m/s, w rad/s)."""
    lim = np.array([S.MAX_LINEAR_ACCEL, S.MAX_ANGULAR_ACCEL]) * dt
    return prev + np.clip(target - prev, -lim, lim)


def action_to_twist_batch(action):
    """Vectorised obs_spec.action_to_twist: (N, 2) actions in [-1, 1] -> (v, w)."""
    return np.clip(action, -1.0, 1.0) * np.array([S.MAX_LINEAR_SPEED, S.MAX_ANGULAR_SPEED])


class OrionEnv:
    """N rovers on randomised soil. Deterministic given (config, seed).

    reset() -> (obs (N, 20) float32, info); step(action (N, 2)) -> (obs, reward, terminated,
    truncated, info). With auto_reset (default) finished rovers are re-sampled and the
    returned obs belongs to the new episode; the last observation of the old one is in
    info['final_obs'] (and info['final_privileged']).

    All per-rover state lives in self.state, a dict of arrays with leading dimension N, so
    clone_state()/restore_state() and tile_state() are plain array copies. The two random
    generators (parameters, noise) are part of the cloned state.
    """

    def __init__(self, num_envs: int, config: EnvConfig | None = None, seed: int = 0,
                 auto_reset: bool = True, record_substeps: bool = False):
        self.num_envs = int(num_envs)
        self.config = config or EnvConfig()
        self.auto_reset = auto_reset
        self.record_substeps = record_substeps
        self.effort_model = SIM_EFFORT_MODEL
        self.seed(seed)
        self.state: dict[str, np.ndarray] = {}

    # --- seeding and state -------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        ss = np.random.SeedSequence(int(seed))
        a, b = ss.spawn(2)
        self.rng_params = np.random.Generator(np.random.PCG64(a))
        self.rng_noise = np.random.Generator(np.random.PCG64(b))

    def clone_state(self) -> dict[str, Any]:
        """Deep copy of everything that determines the future, including RNG states."""
        return {'arrays': {k: v.copy() for k, v in self.state.items()},
                'rng_params': copy.deepcopy(self.rng_params.bit_generator.state),
                'rng_noise': copy.deepcopy(self.rng_noise.bit_generator.state)}

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        arrays = snapshot['arrays']
        n = next(iter(arrays.values())).shape[0]
        if n != self.num_envs:
            raise ValueError(f'snapshot has {n} envs, env has {self.num_envs}')
        self.state = {k: v.copy() for k, v in arrays.items()}
        self.rng_params.bit_generator.state = copy.deepcopy(snapshot['rng_params'])
        self.rng_noise.bit_generator.state = copy.deepcopy(snapshot['rng_noise'])

    @staticmethod
    def tile_state(snapshot: dict[str, Any], repeats: int,
                   rows: np.ndarray | None = None) -> dict[str, Any]:
        """Snapshot of len(rows) * repeats rovers for branched rollouts.

        Each selected rover is repeated `repeats` times consecutively (branch b of rover i is
        row i * repeats + b). RNG states are kept; give the branch env its own seed afterwards
        so branches see different noise.
        """
        arrays = snapshot['arrays']
        if rows is None:
            rows = np.arange(next(iter(arrays.values())).shape[0])
        return {'arrays': {k: np.repeat(v[rows], repeats, axis=0) for k, v in arrays.items()},
                'rng_params': snapshot['rng_params'], 'rng_noise': snapshot['rng_noise']}

    # --- episode sampling --------------------------------------------------------------------
    def _sample_episode(self, idx: np.ndarray) -> None:
        cfg, rng, k = self.config, self.rng_params, len(idx)
        st = self.state
        left = cfg.soil.sample(rng, k)
        other = cfg.soil.sample(rng, k)
        hetero = rng.random(k) < cfg.p_side_hetero
        right = np.where(hetero[:, None], other, left)
        patch = cfg.patch_soil.sample(rng, k)
        st['soil'][idx] = np.stack([left, right, patch], axis=1)
        st['beta'][idx] = rng.uniform(*cfg.beta, k)
        st['g'][idx] = _logu(rng, *cfg.gravity, k)
        st['mass'][idx] = rng.uniform(*cfg.mass_kg, k)
        st['m_fH'][idx] = _logu(rng, *cfg.m_fH, k)
        st['m_fR'][idx] = _logu(rng, *cfg.m_fR, k)
        st['mu_y'][idx] = rng.uniform(*cfg.mu_y, k)
        st['latency'][idx] = rng.integers(cfg.latency_steps[0], cfg.latency_steps[1] + 1, k)
        fault = np.ones((k, 4))
        has_fault = rng.random(k) < cfg.p_torque_fault
        wheel = rng.integers(0, 4, k)
        factor = rng.uniform(*cfg.torque_fault_factor, k)
        fault[np.arange(k), wheel] = np.where(has_fault, factor, 1.0)
        st['fault'][idx] = fault
        heading = np.radians(rng.uniform(*cfg.heading_deg, k))
        # Loose patch somewhere between start and goal line, within the fence.
        st['patch_on'][idx] = cfg.loose_patch
        st['patch_r'][idx] = rng.uniform(*cfg.patch_radius_m, k)
        st['patch_xy'][idx] = np.stack([rng.uniform(-0.6, 0.5, k), rng.uniform(-0.4, 0.4, k)], 1)
        st['dropout_prone'][idx] = rng.random(k) < cfg.p_fix_dropout

        # Static descriptors of each soil set under the static wheel load.
        W = st['mass'][idx] * st['g'][idx] / 4.0
        soil = soil_mod.Soil.from_stacked(st['soil'][idx])            # fields (k, 3)
        st['desc'][idx] = soil_mod.traction_descriptors(
            W[:, None], soil, st['g'][idx, None], st['beta'][idx, None],
            st['m_fH'][idx, None], st['m_fR'][idx, None])

        # Kinematic state: at rest at the start pose, wheels at static sinkage.
        st['pose'][idx] = np.stack([np.full(k, START_XY[0]), np.full(k, START_XY[1]),
                                    GOAL_DIR + heading], 1)
        st['vel'][idx] = 0.0
        st['w'][idx] = 0.0
        st['wheel_angle'][idx] = 0.0
        st['pi_int'][idx] = 0.0
        st['acc'][idx] = 0.0
        st['tau_cmd'][idx] = 0.0
        st['cmd'][idx] = 0.0
        st['act_buf'][idx] = 0.0
        st['prev_action'][idx] = 0.0
        wheel_soil = soil_mod.Soil.from_stacked(np.take_along_axis(
            st['soil'][idx], self._wheel_soil_index(idx)[:, :, None], axis=1))
        st['z'][idx] = soil_mod.static_sinkage(W[:, None], wheel_soil, st['g'][idx, None],
                                               st['beta'][idx, None])
        # Estimator.
        st['v_est'][idx] = 0.0
        st['P'][idx] = FIX0_VAR
        st['t_since_fix'][idx] = 0.0
        st['bias'][idx] = 0.0
        st['dropout_left'][idx] = 0.0
        st['slip_lp'][idx] = 0.0
        st['resid_lp'][idx] = 0.0
        st['w_meas_prev'][idx] = 0.0
        # Episode bookkeeping.
        st['t'][idx] = 0
        st['stuck_t'][idx] = 0.0
        st['ss_speed_t'][idx] = np.nan       # held time of each stuck condition, NaN = false
        st['ss_slip_t'][idx] = np.nan
        st['progress'][idx] = 0.0
        st['ep_return'][idx] = 0.0

    def _wheel_soil_index(self, idx: np.ndarray) -> np.ndarray:
        """(k, 4) index into the soil sets (0 left, 1 right, 2 patch) for each wheel."""
        st = self.state
        pose = st['pose'][idx]
        c, s = np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3])
        wx = pose[:, 0:1] + c * WHEEL_X - s * WHEEL_Y
        wy = pose[:, 1:2] + s * WHEEL_X + c * WHEEL_Y
        d2 = (wx - st['patch_xy'][idx, 0:1]) ** 2 + (wy - st['patch_xy'][idx, 1:2]) ** 2
        in_patch = st['patch_on'][idx, None] & (d2 <= st['patch_r'][idx, None] ** 2)
        return np.where(in_patch, 2, WHEEL_SIDE[None, :])

    def _alloc(self) -> None:
        n = self.num_envs
        f = lambda *shape: np.zeros((n,) + shape)  # noqa: E731
        self.state = {
            'soil': f(3, len(soil_mod.SOIL_KEYS)), 'desc': f(3, 4), 'beta': f(), 'g': f(),
            'mass': f(), 'm_fH': f(), 'm_fR': f(), 'mu_y': f(),
            'latency': np.zeros(n, dtype=np.int64), 'fault': f(4),
            'patch_on': np.zeros(n, dtype=bool), 'patch_r': f(), 'patch_xy': f(2),
            'dropout_prone': np.zeros(n, dtype=bool),
            'pose': f(3), 'vel': f(3), 'w': f(4), 'z': f(4), 'wheel_angle': f(4),
            'pi_int': f(4), 'acc': f(2), 'tau_cmd': f(4), 'cmd': f(2), 'act_buf': f(4, 2),
            'prev_action': f(2),
            'v_est': f(), 'P': f(), 't_since_fix': f(), 'bias': f(2), 'dropout_left': f(),
            'slip_lp': f(2), 'resid_lp': f(4), 'w_meas_prev': f(4),
            't': np.zeros(n, dtype=np.int64), 'stuck_t': f(), 'ss_speed_t': f(),
            'ss_slip_t': f(), 'progress': f(), 'ep_return': f(),
        }

    # --- public API ----------------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed(seed)
        self._alloc()
        idx = np.arange(self.num_envs)
        self._sample_episode(idx)
        obs, sensors = self._observe(idx, init=True)
        return obs, {'privileged': self.privileged(), 'sensors': sensors}

    def privileged(self) -> np.ndarray:
        """(N, 10) privileged vector in PRIVILEGED_FIELDS order for the current wheel soils."""
        st = self.state
        idx = np.arange(self.num_envs)
        which = self._wheel_soil_index(idx)                               # (N, 4)
        per_wheel = np.take_along_axis(st['desc'], which[:, :, None], axis=1)   # (N, 4, 4)
        side = S.side_mean(np.moveaxis(per_wheel, 1, 2))                  # (N, 4 desc, 2)
        return np.concatenate([side.reshape(self.num_envs, 8),
                               (st['mass'] / S.NOMINAL_MASS_KG)[:, None],
                               (st['g'] / soil_mod.G_EARTH)[:, None]], axis=1).astype(np.float32)

    def step(self, action):
        cfg, st = self.config, self.state
        n = self.num_envs
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(n, 2), -1.0, 1.0)
        da = action - st['prev_action']
        st['prev_action'] = action
        st['cmd'] = rate_limit_batch(st['cmd'], action_to_twist_batch(action))
        st['act_buf'] = np.concatenate([st['cmd'][:, None], st['act_buf'][:, :-1]], axis=1)
        applied = st['act_buf'][np.arange(n), st['latency'] - 1]          # (N, 2) v, w
        w_cmd = (applied[:, :1] - applied[:, 1:2] * WHEEL_Y[None, :]) / R

        soil_idx = self._wheel_soil_index(np.arange(n))
        wsoil = soil_mod.Soil.from_stacked(np.take_along_axis(
            st['soil'], soil_idx[:, :, None], axis=1))
        z_max = st['z'].max(axis=1)
        subs = []
        for _ in range(SUBSTEPS):
            self._substep(w_cmd, wsoil)
            z_max = np.maximum(z_max, st['z'].max(axis=1))
            if self.record_substeps:
                subs.append(self._substep_record())

        # Termination and task progress (true state).
        pose = st['pose']
        d = (pose[:, 0] - START_XY[0]) * np.cos(GOAL_DIR) + (pose[:, 1] - START_XY[1]) * np.sin(
            GOAL_DIR)
        dprog = d - st['progress']
        st['progress'] = d
        st['t'] += 1
        vx = st['vel'][:, 0]
        commanding = np.max(np.abs(R * w_cmd), axis=1) > HARD_STUCK_CMD
        st['stuck_t'] = np.where(commanding & (np.abs(vx) < HARD_STUCK_SPEED),
                                 st['stuck_t'] + DT, 0.0)
        success = d >= cfg.goal_m
        belly = z_max >= BELLY_OUT_Z
        hard_stuck = st['stuck_t'] >= HARD_STUCK_TIME - 1e-9    # sums of DT: round-off
        fence = np.any(np.abs(pose[:, :2]) > FENCE_HALF, axis=1)
        failed = (belly | hard_stuck | fence) & ~success
        terminated = success | failed
        truncated = (st['t'] >= cfg.max_steps) & ~terminated

        obs, sensors = self._observe()
        slip_true = physical_slip(st['w'], vx[:, None] - st['vel'][:, 2:3] * WHEEL_Y[None, :])

        # Reward (weights SPEC section 6; shaping terms scaled per second).
        remaining = cfg.max_steps - st['t']
        reward = (cfg.w_progress * dprog / cfg.goal_m + cfg.w_step
                  + DT * (cfg.w_slip * np.mean(slip_true ** 2, axis=1)
                          + cfg.w_sinkage * np.mean(st['z'], axis=1) / BELLY_OUT_Z
                          + cfg.w_action_rate * np.sum(da ** 2, axis=1))
                  + cfg.w_goal * success + cfg.w_belly_out * (belly & ~success)
                  + cfg.w_hard_stuck * (hard_stuck & ~success & ~belly)
                  + cfg.w_fence * (fence & ~success & ~belly & ~hard_stuck)
                  # A failure also forfeits the step penalty of the time left, so ending an
                  # episode early by failing is never cheaper than driving on.
                  + cfg.w_step * remaining * failed)
        st['ep_return'] += reward

        # Soft-stuck flag on estimated quantities: orion_safety's stuck rule (SPEC section 5).
        no_motion, spin = baselines.stuck_conditions(
            st['cmd'][:, 0], sensors['body_speed'], sensors['confidence'], sensors['slip_side'],
            sensors['time_since_fix'])
        st['ss_speed_t'] = baselines.held_for(st['ss_speed_t'], no_motion, DT)
        st['ss_slip_t'] = baselines.held_for(st['ss_slip_t'], spin, DT)
        hold = baselines.STUCK_RULE.hold_s - 1e-9
        soft_stuck = (st['ss_speed_t'] >= hold) | (st['ss_slip_t'] >= hold)

        elapsed = st['t'] * DT
        info = {
            'success': success, 'belly_out': belly & ~success, 'hard_stuck': hard_stuck & ~success,
            'out_of_fence': fence & ~success, 'soft_stuck': soft_stuck,
            'slip_true': slip_true, 'sinkage': st['z'].copy(), 'body_speed_true': vx.copy(),
            'progress': d.copy(), 'sensors': sensors, 'applied_twist': applied.copy(),
            'cmd_twist': st['cmd'].copy(),
        }
        if self.record_substeps:
            info['substeps'] = subs
        done = terminated | truncated
        if done.any():
            info['final_obs'] = obs.copy()
            info['final_privileged'] = self.privileged()
            info['episode_return'] = st['ep_return'].copy()
            info['episode_length'] = st['t'].copy()
            info['effective_speed'] = np.maximum(d, 0.0) / elapsed
            info['episode_time'] = elapsed
            if self.auto_reset:
                idx = np.flatnonzero(done)
                self._sample_episode(idx)
                obs[idx], _ = self._observe(idx, init=True)
        info['privileged'] = self.privileged()
        return obs, reward.astype(np.float32), terminated, truncated, info

    # --- physics -------------------------------------------------------------------------------
    def _substep(self, w_cmd, wsoil) -> None:
        """Advance all rovers by DT_SUB with the secant-implicit scheme (module docstring)."""
        st = self.state
        n = self.num_envs
        dt = DT_SUB
        m, g, beta = st['mass'], st['g'], st['beta'][:, None]
        vx, vy, wz = st['vel'][:, 0], st['vel'][:, 1], st['vel'][:, 2]
        w = st['w']
        v_long = vx[:, None] - wz[:, None] * WHEEL_Y
        v_lat = vy[:, None] + wz[:, None] * WHEEL_X
        u = R * w - v_long                                    # slip velocity at the rim, m/s

        # Wheel loads with longitudinal load transfer.
        W_static = (m * g / 4.0)[:, None]
        W = W_static + LOAD_TRANSFER_SIGN * (m * CG_HEIGHT_M * st['acc'][:, 0]
                                             / (2.0 * WHEELBASE))[:, None]
        W = np.maximum(W, 0.05 * W_static)

        # Sinkage: static part instantly, slip part dug at DIG_RATE x slipped rim travel,
        # relaxing (sticky) only as the wheel travels onto fresh soil.
        s = physical_slip(w, v_long)
        z0 = soil_mod.static_sinkage(W, wsoil, g[:, None], beta)
        z_t = np.minimum(soil_mod.slip_sinkage_target(z0, s, wsoil.a_s), R)
        z = st['z']
        z_up = np.minimum(z_t, np.maximum(z + DIG_RATE * np.abs(u) * dt, z0))
        z_down = np.maximum(z_t, z - SINKAGE_RELAX * np.abs(v_long) * dt)
        z = np.where(z_t > z, z_up, z_down)
        st['z'] = z

        # Secant "conductances": every contact force is written as coefficient x velocity
        # with a non-negative coefficient taken at the current state, then the velocities are
        # solved implicitly. Dissipative and unconditionally stable, no overshoot through 0.
        H = st['m_fH'][:, None] * soil_mod.thrust(s, z, W, wsoil)
        den = np.maximum(np.maximum(np.abs(R * w), np.abs(v_long)), S.SLIP_FLOOR_MPS)
        hmax = st['m_fH'][:, None] * soil_mod.max_thrust(z, W, wsoil)
        k_small = hmax * soil_mod.contact_length(z) / (2.0 * wsoil.K * den)
        k_thrust = np.where(np.abs(u) > 1e-9, H / np.where(u == 0.0, 1.0, u), k_small)
        rc = st['m_fR'][:, None] * soil_mod.compaction_resistance(z, wsoil, g[:, None], beta)
        c_rc = rc * _tanh_over_x(v_long, RC_SPEED_EPS)
        d_lat = st['mu_y'][:, None] * W * _tanh_over_x(v_lat, LATERAL_SPEED_EPS)

        # PI wheel-speed loop with clamping anti-windup; implicit in w while unsaturated.
        err = w_cmd - w
        tau_pi = PI_KP * err + st['pi_int']
        sat = np.abs(tau_pi) > TAU_MAX
        tau_cmd = np.clip(tau_pi, -TAU_MAX, TAU_MAX)
        fault = st['fault']
        kp = np.where(sat, 0.0, PI_KP * fault)                  # tau = tau0 - kp * dw
        tau0 = fault * tau_cmd

        # Linear system A x_new = b for x = [w1..w4, v_x, v_y, w_z]:
        #   J (w_i' - w_i) = dt [tau0_i - kp_i (w_i' - w_i) - r k_i (r w_i' - vl_i')]
        #   m (vx' - vx)   = dt [sum k_i (r w_i' - vl_i') - c_i vl_i' + m vy wz]
        #   m (vy' - vy)   = dt [-sum d_i vt_i' - m vx wz]
        #   Iz (wz' - wz)  = dt [sum -y_i (k_i (r w_i' - vl_i') - c_i vl_i') - x_i d_i vt_i']
        # with vl_i' = vx' - y_i wz' and vt_i' = vy' + x_i wz'.
        Iz = YAW_INERTIA_NOMINAL * m / S.NOMINAL_MASS_KG
        k, c, d = k_thrust, c_rc, d_lat
        y, x = WHEEL_Y, WHEEL_X
        i4 = np.arange(4)
        A = np.zeros((n, 7, 7))
        b = np.empty((n, 7))
        A[:, i4, i4] = WHEEL_INERTIA + dt * (kp + R * R * k)
        A[:, i4, 4] = -dt * R * k
        A[:, i4, 6] = dt * R * k * y
        b[:, :4] = WHEEL_INERTIA * w + dt * (tau0 + kp * w)
        kc = k + c
        A[:, 4, i4] = -dt * R * k
        A[:, 4, 4] = m + dt * kc.sum(1)
        A[:, 4, 6] = -dt * (kc * y).sum(1)
        b[:, 4] = m * vx + dt * m * vy * wz
        A[:, 5, 5] = m + dt * d.sum(1)
        A[:, 5, 6] = dt * (d * x).sum(1)
        b[:, 5] = m * vy - dt * m * vx * wz
        A[:, 6, i4] = dt * R * k * y
        A[:, 6, 4] = -dt * (kc * y).sum(1)
        A[:, 6, 5] = dt * (d * x).sum(1)
        A[:, 6, 6] = Iz + dt * ((kc * y * y).sum(1) + (d * x * x).sum(1))
        b[:, 6] = Iz * wz
        sol = np.linalg.solve(A, b[..., None])[..., 0]

        w_new, vel_new = sol[:, :4], sol[:, 4:]
        st['pi_int'] = np.clip(np.where(sat, st['pi_int'], st['pi_int'] + PI_KI * err * dt),
                               -TAU_MAX, TAU_MAX)
        # Motor torque the controller asked for (what the current sensor reflects).
        st['tau_cmd'] = np.where(sat, tau_cmd, PI_KP * (w_cmd - w_new) + (tau_pi - PI_KP * err))
        # Body-frame acceleration (what an accelerometer sees, gravity removed).
        st['acc'] = np.stack([(vel_new[:, 0] - vx) / dt - vy * wz,
                              (vel_new[:, 1] - vy) / dt + vx * wz], 1)
        st['w'] = w_new
        st['vel'] = vel_new
        yaw = st['pose'][:, 2]
        vxn, vyn, wzn = vel_new[:, 0], vel_new[:, 1], vel_new[:, 2]
        st['pose'] = st['pose'] + dt * np.stack([vxn * np.cos(yaw) - vyn * np.sin(yaw),
                                                 vxn * np.sin(yaw) + vyn * np.cos(yaw), wzn], 1)
        st['wheel_angle'] = st['wheel_angle'] + w_new * dt

    def tilt(self, idx: np.ndarray | None = None):
        """(roll, pitch) in rad from the wheel sinkage pattern (flat sandbox floor).

        Positive pitch = nose down (front sinks deeper); positive roll = right side down.
        """
        z = self.state['z'] if idx is None else self.state['z'][idx]
        pitch = np.arctan((z[:, :2].mean(1) - z[:, 2:].mean(1)) / WHEELBASE)
        roll = np.arctan((z[:, [1, 3]].mean(1) - z[:, [0, 2]].mean(1)) / TRACK)
        return roll, pitch

    def _substep_record(self) -> dict[str, np.ndarray]:
        st = self.state
        roll, pitch = self.tilt()
        return {'pose': st['pose'].copy(), 'vel': st['vel'].copy(), 'w': st['w'].copy(),
                'wheel_angle': st['wheel_angle'].copy(), 'tau_cmd': st['tau_cmd'].copy(),
                'acc': st['acc'].copy(), 'roll': roll, 'pitch': pitch, 'z': st['z'].copy()}

    # --- sensors and estimator -----------------------------------------------------------------
    def _observe(self, idx: np.ndarray | None = None, init: bool = False):
        """Advance sensors + estimator of rovers idx by one period and build observations.

        Returns (obs (len(idx), 20), sensors dict).

        With init (fresh episode) the estimator starts from a fix instead of being advanced:
        no low-pass history, last fix at t = 0.
        """
        cfg, st, rng = self.config, self.state, self.rng_noise
        if idx is None:
            idx = np.arange(self.num_envs)
        n = len(idx)
        w_true, vel = st['w'][idx], st['vel'][idx]
        w_meas = w_true + rng.normal(0.0, cfg.noise_wheel_speed, (n, 4))
        eff_meas = st['tau_cmd'][idx] + rng.normal(0.0, cfg.noise_effort_frac * TAU_MAX, (n, 4))
        bias = st['bias'][idx] + rng.normal(0.0, cfg.accel_bias_rw * np.sqrt(DT), (n, 2))
        acc_meas = st['acc'][idx] + bias + rng.normal(0.0, cfg.noise_accel, (n, 2))
        yaw_meas = vel[:, 2] + rng.normal(0.0, cfg.noise_gyro, n)
        fix_v = vel[:, 0] + rng.normal(0.0, cfg.noise_speed_fix, n)
        onset = rng.random(n) < cfg.dropout_rate_hz * DT
        dur = rng.uniform(*cfg.dropout_duration_s, n)
        left = st['dropout_left'][idx]
        start = st['dropout_prone'][idx] & (left <= 0.0) & onset & (not init)
        left = np.where(start, dur, np.maximum(left - DT, 0.0))
        has_fix = left <= 0.0
        roll, pitch = self.tilt(idx)
        # Effort residual per wheel; only informative samples move its low-pass (else held),
        # exactly as orion_terrain does.
        informative = est.informative_mask(w_meas, yaw_meas)
        if init:
            v, P = fix_v, np.full(n, FIX0_VAR)
            t_fix = np.zeros(n)
            slip = S.side_mean(S.wheel_slip(w_meas, v, yaw_meas))
            resid = self.effort_model.residual(eff_meas, w_meas, np.zeros((n, 4)), pitch)
            slip_lp, resid_lp = slip, np.where(informative, resid, 0.0)
        else:
            v, P = est.kf_predict(st['v_est'][idx], st['P'][idx], acc_meas[:, 0], DT)
            v, P = est.kf_update(v, P, fix_v, has_fix)
            t_fix = np.where(has_fix, 0.0, st['t_since_fix'][idx] + DT)
            slip = S.side_mean(S.wheel_slip(w_meas, v, yaw_meas))
            dw_dt = (w_meas - st['w_meas_prev'][idx]) / DT
            resid = self.effort_model.residual(eff_meas, w_meas, dw_dt, pitch)
            slip_lp = est.lowpass(st['slip_lp'][idx], slip, DT)
            resid_lp = est.lowpass(st['resid_lp'][idx], resid, DT, mask=informative)
        st['bias'][idx], st['dropout_left'][idx] = bias, left
        st['v_est'][idx], st['P'][idx], st['t_since_fix'][idx] = v, P, t_fix
        st['slip_lp'][idx], st['resid_lp'][idx] = slip_lp, resid_lp
        st['w_meas_prev'][idx] = w_meas
        conf = est.speed_confidence(t_fix, np.sqrt(P))
        proxy = self.effort_model.proxy(resid_lp)
        obs = S.build_observation(
            prev_cmd=st['cmd'][idx], wheel_speed=w_meas, wheel_effort=eff_meas,
            effort_max=sim_effort_max, body_accel_xy=acc_meas, yaw_rate=yaw_meas, roll=roll,
            pitch=pitch, body_speed=v, speed_confidence=conf, slip_side=slip_lp,
            sinkage_proxy=proxy)
        sensors = {'wheel_speed': w_meas, 'wheel_effort': eff_meas, 'accel': acc_meas,
                   'yaw_rate': yaw_meas, 'roll': roll, 'pitch': pitch, 'body_speed': v,
                   'confidence': conf, 'slip_side': slip_lp, 'sinkage_proxy': proxy,
                   'has_fix': has_fix, 'fix_speed': fix_v, 'accel_bias': bias,
                   'time_since_fix': t_fix, 'speed_std': np.sqrt(P)}
        return obs, sensors


FIX0_VAR = est.FIX_SIGMA_EXTERNAL ** 2     # m^2/s^2, speed variance right after a fix


def calibrate_effort_model(seed: int = 0, num_envs: int = 32, steps: int = 400,
                           cell: str = 'light-packed') -> est.EffortModel:
    """Fit the flat-ground effort model the way the rover does (SPEC section 4).

    Drives `num_envs` rovers in the given cell with random piecewise-constant commands
    (held 1 s), and fits est.fit_effort_model (the fit orion_terrain uses, informative samples
    only) on the measured (noisy) wheel speeds and efforts, pooled over the 4 identical wheels.
    The slope term c3 is fixed at 0: the simulated sandbox is flat, pitch only comes from
    uneven sinkage, so c3 is unidentifiable here (a free fit returns a large noise-driven
    value). ref = tau_max - c0 (stall minus free effort).
    """
    env = OrionEnv(num_envs, cell_config(cell), seed=seed, auto_reset=True)
    env.effort_model = est.EffortModel(np.zeros(4), TAU_MAX)
    env.reset()
    rng = np.random.default_rng(seed)
    data = {'w': [], 'dw_dt': [], 'effort': [], 'pitch': [], 'yaw_rate': []}
    w_prev = None
    action = np.zeros((num_envs, 2))
    for k in range(steps):
        if k % S.CONTROL_HZ == 0:
            action = np.stack([rng.uniform(-0.8, 0.8, num_envs),
                               rng.uniform(-0.5, 0.5, num_envs)], 1)
        _, _, _, _, info = env.step(action)
        sens = info['sensors']
        w = sens['wheel_speed']
        if w_prev is not None:
            for key, value in (('w', w), ('dw_dt', (w - w_prev) / DT),
                               ('effort', sens['wheel_effort']), ('pitch', sens['pitch']),
                               ('yaw_rate', sens['yaw_rate'])):
                data[key].append(value)
        w_prev = w
    per_wheel = {k: np.concatenate(data[k]).reshape(-1, 1) for k in ('w', 'dw_dt', 'effort')}
    per_rover = {k: np.repeat(np.concatenate(data[k]), 4) for k in ('pitch', 'yaw_rate')}
    fit = est.fit_effort_model(**per_wheel, **per_rover, fit_slope=False)
    c = fit.coefficients[0]
    return est.EffortModel(np.round(c, 5), ref=round(float(TAU_MAX - c[0]), 5))


if __name__ == '__main__':  # pragma: no cover - maintenance entry point
    import argparse
    ap = argparse.ArgumentParser(description='ORION env maintenance tools')
    ap.add_argument('--calibrate', action='store_true',
                    help='refit SIM_EFFORT_MODEL on light-packed driving and print it')
    if ap.parse_args().calibrate:
        m = calibrate_effort_model()
        print(f'coefficients={m.coefficients.tolist()}, ref={m.ref}')
