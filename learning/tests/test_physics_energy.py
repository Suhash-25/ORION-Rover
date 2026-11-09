"""Secant-implicit integrator (council W1): kinetic energy never exceeds the motor work.

Implicit Euler on the velocities gives, per substep, x' (x' - x) = (x'^2 - x^2) / 2 +
(x' - x)^2 / 2 >= delta(x^2 / 2), so the kinetic energy gained is at most the work of the
applied motor torques at the new wheel speeds minus the (non-negative) contact dissipation.
The only non-dissipative explicit terms are the Coriolis terms m v_y w_z and -m v_x w_z,
whose work is O(dt^2) and covered by a small tolerance.
"""

from dataclasses import replace

import numpy as np

from orion_learning import env as E
from orion_learning import obs_spec as S


def _kinetic_energy(env, rec):
    st = env.state
    m = st['mass']
    iz = E.YAW_INERTIA_NOMINAL * m / S.NOMINAL_MASS_KG
    vx, vy, wz = rec['vel'][:, 0], rec['vel'][:, 1], rec['vel'][:, 2]
    return (0.5 * m * (vx ** 2 + vy ** 2) + 0.5 * iz * wz ** 2
            + 0.5 * E.WHEEL_INERTIA * np.sum(rec['w'] ** 2, axis=1))


def test_kinetic_energy_gain_never_exceeds_motor_work():
    n = 32
    cfg = replace(E.EnvConfig(), p_torque_fault=0.5)      # include weak motors
    env = E.OrionEnv(n, cfg, seed=4, auto_reset=False, record_substeps=True)
    env.reset()
    rng = np.random.default_rng(0)
    ke = _kinetic_energy(env, {'vel': env.state['vel'], 'w': env.state['w']})
    worst, motor_total, gain_total = -np.inf, np.zeros(n), np.zeros(n)
    action = np.zeros((n, 2))
    for k in range(300):
        if k % 20 == 0:                                   # new random command every 1 s
            action = rng.uniform(-1.0, 1.0, (n, 2))
        # Without auto-reset a terminated rover keeps being simulated: the physics (all this
        # test checks) stays valid after a belly-out or leaving the fence.
        info = env.step(action)[-1]
        for rec in info['substeps']:
            # Applied torque = torque-fault factor x the commanded (recorded) torque.
            motor = E.DT_SUB * np.sum(env.state['fault'] * rec['tau_cmd'] * rec['w'], axis=1)
            ke_new = _kinetic_energy(env, rec)
            gain = ke_new - ke
            worst = max(worst, float(np.max(gain - motor - 1e-4 * np.abs(motor))))
            motor_total += motor
            gain_total += gain
            ke = ke_new
    print(f'max(gain - motor) = {worst:.3g} J, motor work per rover {motor_total.mean():.1f} J')
    assert worst <= 1e-6, f'kinetic energy gain exceeded motor work by {worst:.3g} J'
    assert np.all(gain_total <= motor_total)
    # Not vacuous: the motors did real work and most of it was dissipated in the soil.
    assert motor_total.mean() > 1.0
