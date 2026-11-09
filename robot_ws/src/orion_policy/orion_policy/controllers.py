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

import numpy as np
from orion_learning import obs_spec
from orion_learning.baselines import B0Params, SlipGovernor

from .artefact import Artefact

_SLIP = obs_spec.field_slice('slip_side')


@dataclass
class ControlInput:
    """
    What a controller sees in one tick.

    obs: float32 [OBS_DIM]; history: float32 [HISTORY_LEN, OBS_DIM + ACTION_DIM], the
    adapter input whose newest row is [obs, previous executed command] (orion_learning.history);
    dt: tick period in s.
    """

    obs: np.ndarray
    history: np.ndarray
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

    def after_escape(self) -> None:
        """Notify that an escape that interrupted this controller has run to its end."""

    @abstractmethod
    def step(self, inp: ControlInput) -> ControlOutput:
        """One control tick."""


class BaselineB0(Controller):
    """
    B0, the non-RL baseline of SPEC section 6, as a thin adapter around orion_learning.

    The speed is orion_learning.baselines.SlipGovernor (cruise 0.3 m/s, -20 %/s of cruise while
    the worse side's slip_side > 0.3, +5 %/s otherwise, within [0.4, 1], 0.5 after an escape),
    fed the observation's slip_side exactly as in simulation. Heading is held open-loop
    (w = 0). Being stuck is handled outside: SafetyStatus FLAG_STUCK makes the mode machine run
    the next manoeuvre of the escape library E and then call after_escape(). The simulator's
    B0Controller does the same in one class; test_learning_parity checks both give the same
    commands.
    """

    name = 'B0'

    def __init__(self, params: B0Params = B0Params(), dt: float = 1.0 / obs_spec.CONTROL_HZ):
        if not 0.0 < params.min_scale <= 1.0:
            raise ValueError('min_scale must be in (0, 1]')
        self.governor = SlipGovernor(1, params, dt)

    def reset(self) -> None:
        self.governor.reset()

    def after_escape(self) -> None:
        """Restart the governor at B0Params.scale_after_escape once an escape has ended."""
        self.governor.after_escape()

    def step(self, inp: ControlInput) -> ControlOutput:
        v = self.governor.step(np.asarray(inp.obs)[None, _SLIP])[0]
        return ControlOutput(v=float(v), w=0.0)


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
