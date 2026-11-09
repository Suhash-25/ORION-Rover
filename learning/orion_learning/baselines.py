"""Non-learned controllers on the policy observation vector (SPEC sections 5 and 6).

Pure NumPy, no torch, vectorised over a leading batch of N rovers (N = 1 on the rover), so
orion_policy can import and run exactly the code evaluated in simulation.

* StuckDetector: the SPEC section 5 stuck rule, computed from the observation.
* EscapeLibrary: the scripted escape library E (sequences of timed (v, w) segments).
* B0Controller: cruise at 0.3 m/s, slip governor, stuck -> escape.
* CriticGate: runtime gating of a learned policy by the entrapment-risk critic rho.

All controllers output policy actions in [-1, 1]^2 (v / 0.5 m/s, w / 1.0 rad/s); the runtime
then rate-limits and applies the safety limits, as for the learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion_learning import obs_spec as S

DT = 1.0 / S.CONTROL_HZ

_PREV_CMD = S.field_slice('prev_cmd')
_BODY_SPEED = S.field_slice('body_speed')
_CONF = S.field_slice('speed_confidence')
_SLIP = S.field_slice('slip_side')


def twist_to_action(v, w):
    """(v m/s, w rad/s) -> policy action in [-1, 1]^2, shape (..., 2)."""
    a = np.stack(np.broadcast_arrays(np.asarray(v, dtype=np.float64) / S.MAX_LINEAR_SPEED,
                                     np.asarray(w, dtype=np.float64) / S.MAX_ANGULAR_SPEED), -1)
    return np.clip(a, -1.0, 1.0)


@dataclass(frozen=True)
class StuckRule:
    """SPEC section 5 stuck rule (thresholds from orion_safety config/safety.yaml)."""

    cmd_v_min: float = 0.1        # m/s commanded
    speed_max: float = 0.03       # m/s estimated body speed
    confidence_min: float = 0.3
    slip_max: float = 0.8         # slip_side
    hold_s: float = 3.0


class StuckDetector:
    """SPEC section 5 stuck flag computed from the observation vector only.

    Stuck when cmd v > 0.1 and body speed < 0.03 m/s (confidence > 0.3) for 3 s, or
    max(slip_side) > 0.8 for 3 s.
    """

    def __init__(self, num_envs: int = 1, rule: StuckRule = StuckRule(), dt: float = DT):
        self.rule, self.dt = rule, dt
        self.t_speed = np.zeros(num_envs)
        self.t_slip = np.zeros(num_envs)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.t_speed[m] = 0.0
        self.t_slip[m] = 0.0

    def update(self, obs) -> np.ndarray:
        obs = np.atleast_2d(obs)
        r = self.rule
        cmd_v = obs[:, _PREV_CMD][:, 0] * S.MAX_LINEAR_SPEED
        speed = obs[:, _BODY_SPEED][:, 0] * S.MAX_LINEAR_SPEED
        confident = obs[:, _CONF][:, 0] > r.confidence_min
        slow = (cmd_v > r.cmd_v_min) & (speed < r.speed_max) & confident
        spin = obs[:, _SLIP].max(axis=1) > r.slip_max
        self.t_speed = np.where(slow, self.t_speed + self.dt, 0.0)
        self.t_slip = np.where(spin, self.t_slip + self.dt, 0.0)
        hold = r.hold_s - 1e-9          # timers are sums of dt: guard float round-off
        return (self.t_speed >= hold) | (self.t_slip >= hold)


# Escape library E: each manoeuvre is a sequence of (duration s, v m/s, w rad/s) segments.
ESCAPE_LIBRARY: tuple[tuple[tuple[float, float, float], ...], ...] = (
    ((2.0, -0.20, 0.0),),                                            # straight reverse
    ((0.6, -0.15, 0.0), (0.6, 0.15, 0.0)) * 3,                       # rocking
    ((0.75, 0.10, 0.8), (0.75, 0.10, -0.8)) * 2,                     # forward wiggle
    ((1.5, -0.20, 0.6), (1.5, -0.20, -0.6)),                         # reverse S-arc
)


class EscapeLibrary:
    """Plays escape manoeuvres; each new escape of a rover uses the next manoeuvre (cyclic)."""

    def __init__(self, num_envs: int = 1, library=ESCAPE_LIBRARY, dt: float = DT,
                 start_index=None):
        self.library, self.dt = library, dt
        self.n_moves = len(library)
        self.durations = [np.cumsum([seg[0] for seg in m]) for m in library]
        self.active = np.zeros(num_envs, dtype=bool)
        self.move = np.zeros(num_envs, dtype=np.int64) if start_index is None else \
            np.asarray(start_index, dtype=np.int64) % self.n_moves
        self.t = np.zeros(num_envs)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.active[m] = False
        self.t[m] = 0.0

    def start(self, mask) -> None:
        """Begin the next manoeuvre for rovers in mask that are not already escaping."""
        new = np.asarray(mask, dtype=bool) & ~self.active
        self.active |= new
        self.t[new] = 0.0

    def act(self) -> tuple[np.ndarray, np.ndarray]:
        """Twist (v, w) of each escaping rover for this step (zeros where inactive).

        Advances time; a rover whose manoeuvre ends becomes inactive and moves on to the
        next manoeuvre.
        """
        v = np.zeros(len(self.active))
        w = np.zeros(len(self.active))
        for i in np.flatnonzero(self.active):
            move = self.library[self.move[i]]
            seg = int(np.searchsorted(self.durations[self.move[i]], self.t[i], side='right'))
            seg = min(seg, len(move) - 1)
            v[i], w[i] = move[seg][1], move[seg][2]
            self.t[i] += self.dt
            if self.t[i] >= self.durations[self.move[i]][-1] - 1e-9:
                self.active[i] = False
                self.move[i] = (self.move[i] + 1) % self.n_moves
        return v, w


@dataclass(frozen=True)
class B0Params:
    cruise_v: float = 0.3         # m/s
    governor_slip: float = 0.3    # slip_side above which v is cut
    cut_per_s: float = 0.20       # fraction of cruise removed per second while slipping
    recover_per_s: float = 0.05   # fraction recovered per second otherwise
    min_scale: float = 0.4        # keeps cmd v (0.12 m/s) above the stuck rule's 0.1 m/s
    scale_after_escape: float = 0.5


class B0Controller:
    """B0: cruise 0.3 m/s straight, slip governor, scripted escape when stuck.

    Governor: while max(slip_side) > 0.3 the speed scale drops by 20 % of cruise per second,
    otherwise it recovers by 5 % per second, within [min_scale, 1]. Stuck (SPEC section 5
    rule) starts the next escape manoeuvre; afterwards cruise resumes at scale 0.5.
    """

    def __init__(self, num_envs: int = 1, params: B0Params = B0Params(), dt: float = DT,
                 escape_start_index=None):
        self.p, self.dt, self.num_envs = params, dt, num_envs
        self.scale = np.ones(num_envs)
        self.stuck = StuckDetector(num_envs, dt=dt)
        self.escape = EscapeLibrary(num_envs, dt=dt, start_index=escape_start_index)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.scale[m] = 1.0
        self.stuck.reset(mask)
        self.escape.reset(mask)

    def act(self, obs) -> np.ndarray:
        obs = np.atleast_2d(obs)
        p = self.p
        slipping = obs[:, _SLIP].max(axis=1) > p.governor_slip
        rate = np.where(slipping, -p.cut_per_s, p.recover_per_s)
        self.scale = np.clip(self.scale + rate * self.dt, p.min_scale, 1.0)
        stuck = self.stuck.update(obs)
        self.escape.start(stuck)
        escaping = self.escape.active.copy()
        v_esc, w_esc = self.escape.act()
        finished = escaping & ~self.escape.active
        self.scale[finished] = p.scale_after_escape
        self.stuck.reset(escaping)
        v = np.where(escaping, v_esc, p.cruise_v * self.scale)
        w = np.where(escaping, w_esc, 0.0)
        return twist_to_action(v, w).astype(np.float32)


@dataclass(frozen=True)
class GateThresholds:
    """SPEC section 6 runtime gating of the student by the critic's rho."""

    probe: float = 0.3            # rho >= 0.3: cap speed and probe
    retreat: float = 0.5          # rho >= 0.5: back off
    escape: float = 0.7           # rho > 0.7: run the escape library
    probe_v_max: float = 0.2      # m/s
    retreat_v: float = -0.15      # m/s


MODE_EXPLORE, MODE_PROBE, MODE_RETREAT, MODE_ESCAPE = 0, 1, 2, 3


class CriticGate:
    """Applies rho-gating to a learned policy's action (vectorised).

    rho < 0.3 explore (policy action unchanged); 0.3 <= rho < 0.5 probe (forward speed capped
    at 0.2 m/s); 0.5 <= rho <= 0.7 retreat (straight reverse at 0.15 m/s); rho > 0.7 or the
    stuck rule fires -> ESCAPE: the next escape manoeuvre runs to completion.
    Returns the gated action and the mode per rover (MODE_* constants).
    """

    def __init__(self, num_envs: int = 1, thresholds: GateThresholds = GateThresholds(),
                 dt: float = DT):
        self.th = thresholds
        self.stuck = StuckDetector(num_envs, dt=dt)
        self.escape = EscapeLibrary(num_envs, dt=dt)

    def reset(self, mask=None) -> None:
        self.stuck.reset(mask)
        self.escape.reset(mask)

    def gate(self, obs, action, rho) -> tuple[np.ndarray, np.ndarray]:
        obs = np.atleast_2d(obs)
        action = np.atleast_2d(np.asarray(action, dtype=np.float64)).copy()
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        th = self.th
        stuck = self.stuck.update(obs)
        self.escape.start(stuck | (rho > th.escape))
        escaping = self.escape.active.copy()
        v_esc, w_esc = self.escape.act()
        self.stuck.reset(escaping)
        mode = np.where(rho >= th.retreat, MODE_RETREAT,
                        np.where(rho >= th.probe, MODE_PROBE, MODE_EXPLORE))
        mode = np.where(escaping, MODE_ESCAPE, mode)
        cap = th.probe_v_max / S.MAX_LINEAR_SPEED
        probe = mode == MODE_PROBE
        action[probe, 0] = np.minimum(action[probe, 0], cap)
        retreat = mode == MODE_RETREAT
        action[retreat] = twist_to_action(th.retreat_v, 0.0)
        action[escaping] = twist_to_action(v_esc[escaping], w_esc[escaping])
        return np.clip(action, -1.0, 1.0).astype(np.float32), mode
