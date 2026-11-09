"""
Controllers behind one interface: the B0 baseline and a verified ONNX controller.

A controller maps one control tick's inputs to a target twist (before rate limiting and
safety limits) plus, for learnt controllers, the critic's risk and the adapter's latent.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np
from orion_learning import obs_spec

from .artefact import Artefact


@dataclass
class ControlInput:
    """
    What a controller sees in one tick.

    obs: float32 [OBS_DIM]; history: float32 [HISTORY_LEN, OBS_DIM + ACTION_DIM] up to the
    previous tick; slip_side: TerrainState.slip_side (left, right); dt: tick period in s.
    """

    obs: np.ndarray
    history: np.ndarray
    slip_side: Sequence[float]
    dt: float


@dataclass
class ControlOutput:
    """Target (v m/s, w rad/s); risk = critic rho in [0, 1] or NaN; latent = z-hat."""

    v: float
    w: float
    risk: float = float('nan')
    latent: np.ndarray = field(
        default_factory=lambda: np.zeros(obs_spec.LATENT_DIM, dtype=np.float32))


class Controller(ABC):
    name = ''
    policy_sha = ''   # short sha256 of the ONNX policy, empty for non-learnt controllers

    def reset(self) -> None:
        """Forget internal state (called when the controller is (re)selected)."""

    @abstractmethod
    def step(self, inp: ControlInput) -> ControlOutput:
        """One control tick."""


class BaselineB0(Controller):
    """
    B0, the non-RL baseline of SPEC section 6: cruise with a slip governor.

    The commanded speed is cruise_speed * fraction. While the worse side slips more than
    slip_threshold (max of TerrainState.slip_side; positive = spinning) the fraction falls by
    cut_rate per second (0.20 = 20 % of cruise per s), otherwise it recovers by recover_rate
    per second, within [min_fraction, 1]. Heading is held open-loop (w = 0): the trial goal
    line is straight ahead. Being stuck is handled outside the controller: SafetyStatus
    FLAG_STUCK switches the mode machine to the scripted escape library (sequences.py).

    min_fraction keeps the command above the stuck detector's 0.1 m/s gate (SPEC section 5)
    so the governor cannot hide an entrapment from orion_safety.

    NOTE for the integrator: WP4 implements the same B0 logic for the simulator in
    learning/orion_learning/baselines.py. Keep one implementation and import it from the
    other (this one has no ROS dependency).
    """

    name = 'B0'

    def __init__(self, cruise_speed: float = 0.3, slip_threshold: float = 0.3,
                 cut_rate: float = 0.20, recover_rate: float = 0.05,
                 min_fraction: float = 0.4):
        if not 0.0 < min_fraction <= 1.0:
            raise ValueError('min_fraction must be in (0, 1]')
        self.cruise_speed = cruise_speed
        self.slip_threshold = slip_threshold
        self.cut_rate = cut_rate
        self.recover_rate = recover_rate
        self.min_fraction = min_fraction
        self.fraction = 1.0

    def reset(self) -> None:
        self.fraction = 1.0

    def step(self, inp: ControlInput) -> ControlOutput:
        if max(inp.slip_side) > self.slip_threshold:
            self.fraction -= self.cut_rate * inp.dt
        else:
            self.fraction += self.recover_rate * inp.dt
        self.fraction = min(1.0, max(self.min_fraction, self.fraction))
        return ControlOutput(v=self.cruise_speed * self.fraction, w=0.0)


class OnnxController(Controller):
    """A learnt controller (B2, student or teacher) from a verified artefact."""

    def __init__(self, artefact: Artefact):
        self.artefact = artefact
        self.name = artefact.controller
        self.policy_sha = artefact.policy_sha

    def step(self, inp: ControlInput) -> ControlOutput:
        latent = self.artefact.latent(inp.history)
        action = self.artefact.action(inp.obs, latent)
        v, w = obs_spec.action_to_twist(action)
        return ControlOutput(v=v, w=w, risk=self.artefact.risk(inp.obs, latent), latent=latent)
