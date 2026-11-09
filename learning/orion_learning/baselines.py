"""Non-learned controllers on the policy observation vector (SPEC sections 5 and 6).

Pure NumPy, no torch, vectorised over a leading batch of N rovers (N = 1 on the rover), so
orion_policy imports and runs the same code that is evaluated in simulation.

* StuckRule / StuckDetector: the SPEC section 5 stuck rule with orion_safety's semantics.
* EscapeLibrary: plays the manoeuvres of the escape library E (orion_learning.escape_library).
* SlipGovernor: B0's speed governor. orion_policy's BaselineB0 is a thin adapter around it.
* B0Controller: B0 as the rover runs it: governor, stuck -> escape, give up after
  MAX_ESCAPE_ATTEMPTS. On the rover the same logic is split between orion_safety (FLAG_STUCK),
  orion_policy's mode machine (escape, attempts) and BaselineB0 (governor); a test in
  orion_policy checks that both produce the same commands.
* CriticGate: runtime gating of a learned policy by the entrapment-risk critic rho.

All controllers output policy actions in [-1, 1]^2 (v / 0.5 m/s, w / 1.0 rad/s); the runtime
then rate-limits and applies the safety limits, as for the learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion_learning import obs_spec as S
from orion_learning.escape_library import E, MAX_ESCAPE_ATTEMPTS, Manoeuvre, manoeuvre_ticks

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


# --- stuck rule -----------------------------------------------------------------------------
@dataclass(frozen=True)
class StuckRule:
    """SPEC section 5 stuck rule; values = orion_safety config/safety.yaml stuck_*."""

    cmd_v_min: float = 0.1        # m/s, |commanded v| above which the rover is driving
    speed_max: float = 0.03       # m/s, |body speed| below which it is not moving
    confidence_min: float = 0.3   # speed rule needs confidence > this ...
    fresh_fix_s: float = 0.5      # ... unless the last speed fix is younger than this (s)
    slip_max: float = 0.8         # slip_side (in the driven direction) above which it spins
    hold_s: float = 3.0           # s a condition must hold continuously


STUCK_RULE = StuckRule()


def stuck_conditions(cmd_v, body_speed, confidence, slip_side, time_since_fix=None,
                     rule: StuckRule = STUCK_RULE) -> tuple[np.ndarray, np.ndarray]:
    """The two instantaneous stuck conditions of orion_safety's monitor, (no_motion, spinning).

    cmd_v (...) m/s is the command sent to the drive (obs prev_cmd), body_speed (...) m/s,
    confidence (...) in [0, 1], slip_side (..., 2). time_since_fix (...) s is optional: the
    observation does not carry it, and without it the speed rule needs the confidence (it then
    differs from the monitor only while 0.3 >= confidence and the fix is < 0.5 s old, about
    0.1 s into a fix dropout). Spinning is judged in the driven direction (slip is negative when
    reversing); without a drive command, on |slip|.
    """
    cmd_v = np.asarray(cmd_v, dtype=np.float64)
    slip = np.asarray(slip_side, dtype=np.float64)
    driving = np.abs(cmd_v) > rule.cmd_v_min
    trusted = np.asarray(confidence) > rule.confidence_min
    if time_since_fix is not None:
        trusted = trusted | (np.asarray(time_since_fix) < rule.fresh_fix_s)
    no_motion = driving & (np.abs(np.asarray(body_speed)) < rule.speed_max) & trusted
    sign = np.where(cmd_v < 0.0, -1.0, 1.0)[..., None]
    spin = np.where(driving, np.max(sign * slip, axis=-1), np.max(np.abs(slip), axis=-1))
    return no_motion, spin > rule.slip_max


def held_for(elapsed, condition, dt):
    """Time (s) a condition has been continuously true, NaN while false (orion_safety _Held).

    The first true sample counts as 0 s, so a 3 s hold fires on the sample 3 s after onset.
    """
    return np.where(condition, np.where(np.isnan(elapsed), 0.0, elapsed + dt), np.nan)


class StuckDetector:
    """SPEC section 5 stuck flag from the observation vector, as orion_safety computes it.

    Stuck when |cmd v| > 0.1 and |body speed| < 0.03 m/s (confidence > 0.3 or a fix < 0.5 s
    old) for 3 s, or slip_side in the driven direction > 0.8 for 3 s. Like the monitor it keeps
    running during escapes: a rover still stuck when an escape ends is flagged again at once.
    """

    def __init__(self, num_envs: int = 1, rule: StuckRule = STUCK_RULE, dt: float = DT):
        self.rule, self.dt = rule, dt
        self.t_speed = np.full(num_envs, np.nan)
        self.t_slip = np.full(num_envs, np.nan)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.t_speed[m] = np.nan
        self.t_slip[m] = np.nan

    def update(self, obs, time_since_fix=None) -> np.ndarray:
        obs = np.atleast_2d(obs)
        no_motion, spin = stuck_conditions(
            obs[:, _PREV_CMD][:, 0] * S.MAX_LINEAR_SPEED,
            obs[:, _BODY_SPEED][:, 0] * S.MAX_LINEAR_SPEED, obs[:, _CONF][:, 0],
            obs[:, _SLIP], time_since_fix, self.rule)
        self.t_speed = held_for(self.t_speed, no_motion, self.dt)
        self.t_slip = held_for(self.t_slip, spin, self.dt)
        hold = self.rule.hold_s - 1e-9          # timers are sums of dt: guard float round-off
        return (self.t_speed >= hold) | (self.t_slip >= hold)


# --- escape library player -------------------------------------------------------------------
class EscapeLibrary:
    """Plays escape manoeuvres; each new escape of a rover uses the next manoeuvre (cyclic).

    library: manoeuvres (default E). start_index: first manoeuvre per rover (default 0), also
    restored by reset(). Each manoeuvre is played tick by tick (escape_library.manoeuvre_ticks),
    exactly as orion_policy plays it.
    """

    def __init__(self, num_envs: int = 1, library: tuple[Manoeuvre, ...] = E, dt: float = DT,
                 start_index=None):
        self.library, self.dt = library, dt
        self.n_moves = len(library)
        self.ticks = [np.array(manoeuvre_ticks(m, dt)) for m in library]
        self.first = (np.zeros(num_envs, dtype=np.int64) if start_index is None else
                      np.asarray(start_index, dtype=np.int64) % self.n_moves)
        self.move = self.first.copy()
        self.active = np.zeros(num_envs, dtype=bool)
        self.k = np.zeros(num_envs, dtype=np.int64)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.active[m] = False
        self.k[m] = 0
        self.move[m] = self.first[m]

    def start(self, mask) -> None:
        """Begin the next manoeuvre for rovers in mask that are not already escaping."""
        new = np.asarray(mask, dtype=bool) & ~self.active
        self.active |= new
        self.k[new] = 0

    def act(self) -> tuple[np.ndarray, np.ndarray]:
        """Twist (v, w) of each escaping rover for this tick (zeros where inactive).

        Advances one tick; a rover whose manoeuvre ends becomes inactive and moves on to the
        next manoeuvre.
        """
        v = np.zeros(len(self.active))
        w = np.zeros(len(self.active))
        for i in np.flatnonzero(self.active):
            ticks = self.ticks[self.move[i]]
            v[i], w[i] = ticks[self.k[i]]
            self.k[i] += 1
            if self.k[i] >= len(ticks):
                self.active[i] = False
                self.move[i] = (self.move[i] + 1) % self.n_moves
        return v, w


class _EscapeArbiter:
    """Stuck/trigger -> next escape manoeuvre, with orion_policy's attempt limit.

    An escape starts when triggered and not already escaping. A rover that triggers again
    after MAX_ESCAPE_ATTEMPTS escapes with no clean (untriggered) driving tick in between gives
    up: it stops until reset (on the rover: IDLE and PolicyStatus.escapes_exhausted, which
    the trial runner records as ENTRAPPED; SPEC section 8).
    """

    def __init__(self, num_envs: int, dt: float, start_index=None,
                 max_attempts: int = MAX_ESCAPE_ATTEMPTS):
        self.escape = EscapeLibrary(num_envs, dt=dt, start_index=start_index)
        self.max_attempts = max_attempts
        self.attempts = np.zeros(num_envs, dtype=np.int64)
        self.gave_up = np.zeros(num_envs, dtype=bool)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.escape.reset(mask)
        self.attempts[m] = 0
        self.gave_up[m] = False

    def step(self, trigger):
        """One tick. Returns (escaping, finished, driving, v_esc, w_esc) per rover.

        escaping: an escape manoeuvre supplies this tick's command; finished: its last tick;
        driving: the caller's own controller drives (neither escaping nor given up).
        """
        trigger = np.asarray(trigger, dtype=bool)
        start = trigger & ~self.escape.active & ~self.gave_up
        give_up = start & (self.attempts >= self.max_attempts)
        self.gave_up |= give_up
        start &= ~give_up
        self.attempts[start] += 1
        self.escape.start(start)
        escaping = self.escape.active.copy()
        v_esc, w_esc = self.escape.act()
        finished = escaping & ~self.escape.active
        driving = ~escaping & ~self.gave_up
        self.attempts[driving & ~trigger] = 0
        return escaping, finished, driving, v_esc, w_esc


# --- B0 ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class B0Params:
    cruise_v: float = 0.3         # m/s
    governor_slip: float = 0.3    # max(slip_side) above which v is cut
    cut_per_s: float = 0.20       # fraction of cruise removed per second while slipping
    recover_per_s: float = 0.05   # fraction recovered per second otherwise
    min_scale: float = 0.4        # keeps cmd v (0.12 m/s) above the stuck rule's 0.1 m/s
    scale_after_escape: float = 0.5


class SlipGovernor:
    """B0's speed: cruise 0.3 m/s straight, scaled by a slip governor (vectorised).

    While max(slip_side) > 0.3 the scale drops by 20 % of cruise per second, otherwise it
    recovers by 5 % per second, within [min_scale, 1]; after an escape it restarts at 0.5.
    """

    def __init__(self, num_envs: int = 1, params: B0Params = B0Params(), dt: float = DT):
        self.p, self.dt = params, dt
        self.scale = np.ones(num_envs)

    def reset(self, mask=None) -> None:
        self.scale[slice(None) if mask is None else mask] = 1.0

    def after_escape(self, mask=None) -> None:
        self.scale[slice(None) if mask is None else mask] = self.p.scale_after_escape

    def step(self, slip_side, mask=None) -> np.ndarray:
        """Update the scale of the rovers in mask (all if None); return the cruise v (m/s)."""
        p = self.p
        slipping = np.max(np.atleast_2d(slip_side), axis=-1) > p.governor_slip
        rate = np.where(slipping, -p.cut_per_s, p.recover_per_s)
        new = np.clip(self.scale + rate * self.dt, p.min_scale, 1.0)
        self.scale = new if mask is None else np.where(mask, new, self.scale)
        return p.cruise_v * self.scale


class B0Controller:
    """B0: slip-governed cruise, scripted escape (library E) when stuck, as on the rover.

    stuck: pass the stuck flag when it is known (the env's info['soft_stuck'], orion_safety's
    FLAG_STUCK); otherwise it is computed from the observation by StuckDetector.
    """

    def __init__(self, num_envs: int = 1, params: B0Params = B0Params(), dt: float = DT,
                 escape_start_index=None):
        self.governor = SlipGovernor(num_envs, params, dt)
        self.stuck = StuckDetector(num_envs, dt=dt)
        self.arbiter = _EscapeArbiter(num_envs, dt, escape_start_index)
        self.escape = self.arbiter.escape

    def reset(self, mask=None) -> None:
        self.governor.reset(mask)
        self.stuck.reset(mask)
        self.arbiter.reset(mask)

    def act(self, obs, stuck=None) -> np.ndarray:
        obs = np.atleast_2d(obs)
        flag = self.stuck.update(obs)
        if stuck is not None:
            flag = np.asarray(stuck, dtype=bool).reshape(-1)
        escaping, finished, driving, v_esc, w_esc = self.arbiter.step(flag)
        v = self.governor.step(obs[:, _SLIP], mask=driving)
        self.governor.after_escape(finished)
        v = np.where(escaping, v_esc, np.where(driving, v, 0.0))
        w = np.where(escaping, w_esc, 0.0)
        return twist_to_action(v, w).astype(np.float32)


# --- critic gating ------------------------------------------------------------------------------
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
    stuck rule fires -> ESCAPE: the next escape manoeuvre runs to completion (attempt limit as
    for B0; a rover that gives up is stopped). Returns the gated action and the mode per rover.
    """

    def __init__(self, num_envs: int = 1, thresholds: GateThresholds = GateThresholds(),
                 dt: float = DT):
        self.th = thresholds
        self.stuck = StuckDetector(num_envs, dt=dt)
        self.arbiter = _EscapeArbiter(num_envs, dt)
        self.escape = self.arbiter.escape

    def reset(self, mask=None) -> None:
        self.stuck.reset(mask)
        self.arbiter.reset(mask)

    def gate(self, obs, action, rho, stuck=None) -> tuple[np.ndarray, np.ndarray]:
        obs = np.atleast_2d(obs)
        action = np.atleast_2d(np.asarray(action, dtype=np.float64)).copy()
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        th = self.th
        flag = self.stuck.update(obs)
        if stuck is not None:
            flag = np.asarray(stuck, dtype=bool).reshape(-1)
        escaping, _, driving, v_esc, w_esc = self.arbiter.step(flag | (rho > th.escape))
        mode = np.where(rho >= th.retreat, MODE_RETREAT,
                        np.where(rho >= th.probe, MODE_PROBE, MODE_EXPLORE))
        mode = np.where(escaping | ~driving, MODE_ESCAPE, mode)
        cap = th.probe_v_max / S.MAX_LINEAR_SPEED
        probe = mode == MODE_PROBE
        action[probe, 0] = np.minimum(action[probe, 0], cap)
        retreat = mode == MODE_RETREAT
        action[retreat] = twist_to_action(th.retreat_v, 0.0)
        action[escaping] = twist_to_action(v_esc[escaping], w_esc[escaping])
        action[~escaping & ~driving] = 0.0
        return np.clip(action, -1.0, 1.0).astype(np.float32), mode
