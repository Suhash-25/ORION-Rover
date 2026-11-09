"""
The orion_policy mode machine, free of ROS so every rule can be unit-tested.

One call to PolicyCore.tick() is one 20 Hz control step. It decides what to publish on which
twist channel ('policy' -> /cmd_vel/policy, 'escape' -> /cmd_vel/escape) and what to report in
PolicyStatus.

Modes (PolicyStatus.MODE_*):
  IDLE      publishes nothing (one zero when entered from a driving mode).
  BASELINE  B0 on /cmd_vel/policy.
  POLICY    ONNX controller on /cmd_vel/policy with critic gating:
            rho < 0.3 run the policy; 0.3-0.5 cap v at 0.2 m/s and run a probe (on entry to
            the band, at most once per probe_cooldown_s); 0.5-0.7 retreat along the entry
            path (escape if no path was recorded); > 0.7 ESCAPE.
  PROBE     0.1 m/s straight for 1 s on /cmd_vel/escape, then back to POLICY.
  ESCAPE    the next manoeuvre of the escape library E (orion_learning.escape_library,
            cyclic from E[0] after each SetMode) on /cmd_vel/escape, then back to the mode it
            interrupted. Armed by SafetyStatus FLAG_STUCK, by rho > 0.7 or by SetMode. After
            max_escape_attempts escapes without a clean driving tick in between the core
            gives up and goes to IDLE (orion_safety's hard-stuck rule then ends the trial).

The adapter's history (orion_learning.history, the training code) gets [obs, previous
executed command] pushed every tick that builds an observation, BEFORE inference, and is
cleared when a driving mode is selected, as at the start of a training episode.

Every published command is rate-limited (obs_spec.rate_limit) and scaled to the latest
SafetyStatus limits. With no SafetyStatus newer than safety_timeout_s, in HALT/ESTOP, or with
a missing/stale input the core publishes zero on the active channel and says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
from orion_learning import obs_spec
from orion_learning.escape_library import MAX_ESCAPE_ATTEMPTS
from orion_learning.history import History

from .controllers import ControlInput
from .controllers import Controller
from .observation import InputNotReady
from .observation import ObservationBuilder
from .sequences import EntryPath
from .sequences import escape_plan
from .sequences import Plan
from .sequences import probe_plan
from .shaping import Band
from .shaping import classify
from .shaping import CommandShaper

# Same values as orion_msgs/msg/PolicyStatus and SafetyStatus (checked by test_policy_node).
MODE_IDLE, MODE_BASELINE, MODE_POLICY, MODE_PROBE, MODE_ESCAPE = range(5)
MODE_NAMES = {MODE_IDLE: 'IDLE', MODE_BASELINE: 'BASELINE', MODE_POLICY: 'POLICY',
              MODE_PROBE: 'PROBE', MODE_ESCAPE: 'ESCAPE'}
STATE_HALT = 2
FLAG_STUCK = 16

POLICY_CHANNEL = 'policy'
ESCAPE_CHANNEL = 'escape'


@dataclass
class CoreConfig:
    dt: float = 1.0 / obs_spec.CONTROL_HZ   # s per tick
    safety_timeout_s: float = 0.5
    critic_gating: bool = True
    risk_thresholds: tuple = (0.3, 0.5, 0.7)
    caution_max_speed: float = 0.2          # m/s in the 0.3-0.5 band, also the retreat cap
    probe_cooldown_s: float = 5.0
    retreat_path_s: float = 2.0             # s of entry path kept for a retreat
    max_escape_attempts: int = MAX_ESCAPE_ATTEMPTS


@dataclass
class SafetyInput:
    """Latest SafetyStatus; t = receive time on the node clock (s)."""

    t: float
    state: int
    flags: int
    max_linear_speed: float
    max_angular_speed: float
    reason: str = ''


@dataclass
class TickResult:
    """
    What one tick decided.

    publish: (channel, v m/s, w rad/s) to send in order. reason: why the command is held
    at zero or what changed ('' when driving normally).
    """

    mode: int
    publish: list = field(default_factory=list)
    risk: float = float('nan')
    latent: np.ndarray = field(
        default_factory=lambda: np.zeros(obs_spec.LATENT_DIM, dtype=np.float32))
    policy_sha: str = ''
    reason: str = ''


class PolicyCore:

    def __init__(self, config: CoreConfig, baseline: Controller,
                 policy: Optional[Controller] = None):
        self.cfg = config
        self.baseline = baseline
        self.policy = policy
        self.mode = MODE_IDLE
        self.base_mode = MODE_IDLE          # the driving mode PROBE/ESCAPE return to
        self.shaper = CommandShaper(config.dt)
        self.history = History(1)
        self.entry_path = EntryPath(config.dt, config.retreat_path_s)
        self.escape_attempts = 0
        self._escape_index = 0               # next manoeuvre of E
        self._plan: Optional[Plan] = None    # PROBE / ESCAPE plan
        self._retreat: Optional[Plan] = None
        self._last_band = Band.EXPLORE
        self._last_probe_tick = -10**9
        self._tick = 0
        self._pending_zero: list = []       # channels to zero before the next command
        self._active_channel: Optional[str] = None

    # --- mode requests ----------------------------------------------------------------------

    def request_mode(self, mode: int) -> tuple[bool, str]:
        """
        Handle SetMode.

        IDLE/BASELINE/POLICY select the driving mode; PROBE/ESCAPE run once and return to it.
        """
        if mode not in MODE_NAMES:
            return False, f'unknown mode {mode}'
        if mode == MODE_POLICY and self.policy is None:
            return False, 'no ONNX controller loaded (set model_dir); POLICY unavailable'
        self._retreat = None
        if mode in (MODE_IDLE, MODE_BASELINE, MODE_POLICY):
            self._plan = None
            self.base_mode = mode
            self.escape_attempts = 0
            self._escape_index = 0
            self.history.reset()
            self.entry_path.clear()
            self._last_band = Band.EXPLORE
            if mode == MODE_BASELINE:
                self.baseline.reset()
            elif mode == MODE_POLICY:
                self.policy.reset()
            self._set_mode(mode)
        elif mode == MODE_PROBE:
            self._start_probe()
        else:
            self.escape_attempts = 0
            self._start_escape()
        return True, f'mode {MODE_NAMES[self.mode]}'

    def _set_mode(self, mode: int) -> None:
        new_channel = self._channel_for(mode)
        if self._active_channel is not None and self._active_channel != new_channel:
            # twist_mux would hold the old channel's last command until its 0.5 s timeout.
            self._pending_zero.append(self._active_channel)
            if new_channel is None:
                self.shaper.hold()      # IDLE: stop now
            # policy <-> escape keeps the rate-limit origin: the rover is still moving at the
            # last command, and in training (B0Controller, CriticGate) an escape is
            # rate-limited from it in the same way.
        self._active_channel = new_channel
        self.mode = mode

    @staticmethod
    def _channel_for(mode: int) -> Optional[str]:
        if mode == MODE_IDLE:
            return None
        return ESCAPE_CHANNEL if mode in (MODE_PROBE, MODE_ESCAPE) else POLICY_CHANNEL

    def _start_probe(self) -> None:
        self._plan = probe_plan(self.cfg.dt)
        self._last_probe_tick = self._tick
        self._set_mode(MODE_PROBE)

    def _start_escape(self) -> bool:
        """Start the escape library; False (and IDLE) once the attempts are used up."""
        self._retreat = None
        if self.escape_attempts >= self.cfg.max_escape_attempts:
            self._plan = None
            self.base_mode = MODE_IDLE
            self._set_mode(MODE_IDLE)
            return False
        self.escape_attempts += 1
        self._plan = escape_plan(self.cfg.dt, self._escape_index)
        self._escape_index += 1
        self._set_mode(MODE_ESCAPE)
        return True

    def _finish_plan(self) -> None:
        if self.mode == MODE_ESCAPE and self.base_mode == MODE_BASELINE:
            self.baseline.after_escape()
        self._plan = None
        self._set_mode(self.base_mode)

    # --- control tick ---------------------------------------------------------------------

    def tick(self, now: float, builder: ObservationBuilder,
             safety: Optional[SafetyInput]) -> TickResult:
        """One control step at time now (s, node clock)."""
        self._tick += 1
        result = TickResult(mode=self.mode)
        result.publish = [(ch, 0.0, 0.0) for ch in self._pending_zero]
        self._pending_zero = []
        if self.mode == MODE_IDLE:
            return result

        if safety is None:
            return self._hold(result, 'no SafetyStatus received: holding zero')
        age = now - safety.t
        if age > self.cfg.safety_timeout_s:
            return self._hold(result, f'SafetyStatus is {age:.2f} s old '
                              f'(> {self.cfg.safety_timeout_s} s): holding zero')
        if safety.state >= STATE_HALT:
            self._plan = None
            self._retreat = None
            self._set_mode(self.base_mode)
            result.publish += [(ch, 0.0, 0.0) for ch in self._pending_zero]
            self._pending_zero = []
            if self.mode == MODE_IDLE:
                return result
            return self._hold(result, f'safety state {safety.state} ({safety.reason}): '
                              'holding zero')
        try:
            obs = builder.build(now, self.shaper.last)
        except InputNotReady as e:
            return self._hold(result, f'{e}: holding zero')

        stuck = bool(safety.flags & FLAG_STUCK)
        if stuck and self.mode in (MODE_BASELINE, MODE_POLICY, MODE_PROBE):
            result.reason = 'SafetyStatus FLAG_STUCK: escape'
            if not self._start_escape():
                return self._give_up(result)

        inp = ControlInput(obs=obs, history=self.history.push(obs)[0], dt=self.cfg.dt)
        learnt = self.policy if self.base_mode == MODE_POLICY else None
        out = learnt.step(inp) if learnt is not None else None
        if out is not None:
            result.risk, result.latent = out.risk, out.latent
            result.policy_sha = learnt.policy_sha

        target, from_policy = self._target(inp, out, stuck, result)
        if target is None:
            return self._give_up(result)
        v, w = self.shaper.shape(target, safety.max_linear_speed, safety.max_angular_speed)
        result.publish += [(ch, 0.0, 0.0) for ch in self._pending_zero]
        self._pending_zero = []
        result.publish.append((self._active_channel, v, w))
        result.mode = self.mode
        if from_policy:
            self.entry_path.record(v, w)
        if self._plan is not None and self._plan.done:
            self._finish_plan()
        return result

    def _target(self, inp: ControlInput, out, stuck: bool, result: TickResult):
        """(target (v, w) or None if escapes are exhausted, whether it came from the policy)."""
        if self.mode == MODE_BASELINE:
            b0 = self.baseline.step(inp)
            if not stuck:
                self.escape_attempts = 0
            return (b0.v, b0.w), False

        if self.mode in (MODE_POLICY, MODE_PROBE) and out is not None:
            band = (classify(out.risk, self.cfg.risk_thresholds) if self.cfg.critic_gating
                    else Band.EXPLORE)
            entering = band != self._last_band
            self._last_band = band
            if band == Band.ESCAPE:
                result.reason = f'rho {out.risk:.2f} > {self.cfg.risk_thresholds[2]}: escape'
                if not self._start_escape():
                    return None, False
            elif self.mode == MODE_POLICY:
                return self._gated_policy(out, band, entering, stuck, result)

        if self._plan is not None:
            return self._plan.next_target(), False
        raise RuntimeError(f'mode {MODE_NAMES[self.mode]} has nothing to run')

    def _gated_policy(self, out, band: Band, entering: bool, stuck: bool,
                      result: TickResult):
        if self._retreat is not None:
            target = self._retreat.next_target()
            if self._retreat.done:
                self._retreat = None
                self.entry_path.clear()
            return target, False
        if band == Band.RETREAT:
            self._retreat = self.entry_path.retreat_plan(self.cfg.caution_max_speed)
            if self._retreat is None:
                result.reason = f'rho {out.risk:.2f}: retreat, but no entry path: escape'
                if not self._start_escape():
                    return None, False
                return self._plan.next_target(), False
            result.reason = f'rho {out.risk:.2f}: retreat along the entry path'
            return self._gated_policy(out, band, False, stuck, result)
        if band == Band.CAUTION:
            cooldown = round(self.cfg.probe_cooldown_s / self.cfg.dt)
            if entering and self._tick - self._last_probe_tick >= cooldown:
                result.reason = f'rho {out.risk:.2f}: probe'
                self._start_probe()
                return self._plan.next_target(), False
            cap = self.cfg.caution_max_speed
            return (min(cap, max(-cap, out.v)), out.w), True
        if not stuck:
            self.escape_attempts = 0
        return (out.v, out.w), True

    def _hold(self, result: TickResult, reason: str) -> TickResult:
        result.reason = reason
        result.mode = self.mode
        self.shaper.hold()
        if self._active_channel is not None:
            result.publish.append((self._active_channel, 0.0, 0.0))
        return result

    def _give_up(self, result: TickResult) -> TickResult:
        result.publish += [(ch, 0.0, 0.0) for ch in self._pending_zero]
        self._pending_zero = []
        result.mode = self.mode
        result.reason = (f'escape failed {self.cfg.max_escape_attempts} times: IDLE '
                         '(waiting for orion_safety hard-stuck or an operator)')
        return result
