"""Effort model: fit, residual sign, sinkage proxy, persistence, and the effort-source gate."""

import os

from conftest import RecordingLogger
from effort_data import drive, make_model, signed_effort, TRUE_COEFS
import numpy as np
from orion_terrain.effort_model import EffortModel, EffortSource, fit_coefficients
from orion_terrain.estimator import TerrainEstimator
from orion_terrain.params import DEFAULT_JOINT_NAMES
import pytest

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
LEVEL = (0.0, 0.0, 0.0, 1.0)


def test_fit_recovers_known_coefficients():
    _, w, dwdt, pitch = drive()
    rng = np.random.default_rng(3)
    effort = signed_effort(TRUE_COEFS, w, dwdt, pitch) + rng.normal(0, 0.01, w.shape)
    fit = fit_coefficients(w, dwdt, pitch, effort, np.zeros(len(w)), 0.5, 0.2)
    assert np.allclose(fit.coefficients, TRUE_COEFS, rtol=0.03, atol=0.005)
    assert np.all(fit.r2 > 0.99)
    assert np.all(fit.n_samples > 0.8 * len(w)) and fit.slope_identified


def test_fit_ignores_turning_samples():
    _, w, dwdt, pitch = drive()
    yaw = np.where(np.arange(len(w)) % 3 == 0, 0.8, 0.0)       # a third of the time turning
    effort = signed_effort(TRUE_COEFS, w, dwdt, pitch) + np.where(yaw > 0, 4.0, 0.0)[:, None]
    fit = fit_coefficients(w, dwdt, pitch, effort, yaw, 0.5, 0.2)
    assert np.allclose(fit.coefficients, TRUE_COEFS, rtol=0.03, atol=0.005)


def test_fit_on_flat_ground_sets_slope_term_to_zero():
    _, w, dwdt, _ = drive()
    pitch = np.full(len(w), 0.001)
    effort = signed_effort(TRUE_COEFS, w, dwdt, pitch)
    fit = fit_coefficients(w, dwdt, pitch, effort, np.zeros(len(w)), 0.5, 0.2)
    assert not fit.slope_identified
    assert np.all(fit.coefficients[:, 3] == 0.0)
    assert np.allclose(fit.coefficients[:, :3], TRUE_COEFS[:, :3], rtol=0.03, atol=0.01)


def test_fit_needs_informative_data():
    w = np.full((100, 4), 0.1)
    with pytest.raises(ValueError, match='only 0 samples'):
        fit_coefficients(w, w * 0, np.zeros(100), w, np.zeros(100), 0.5, 0.2)


def test_residual_is_zero_on_model_and_positive_for_extra_load_both_directions():
    m = make_model()
    for w in (5.0, -5.0):
        wv = np.full(4, w)
        nominal = signed_effort(TRUE_COEFS, wv, np.zeros(4), 0.1)
        assert np.allclose(m.residual(wv, np.zeros(4), 0.1, nominal), 0.0, atol=1e-12)
        loaded = nominal + np.sign(w) * 1.0          # 1 A more, pushing against the sand
        assert np.allclose(m.residual(wv, np.zeros(4), 0.1, loaded), 1.0)


def test_informative_window():
    m = make_model()
    assert list(m.informative([0.0, 0.4, 3.0, -3.0], 0.0)) == [False, False, True, True]
    assert not np.any(m.informative([3.0] * 4, 0.5))              # turning hard


def test_residual_held_at_standstill_and_while_turning():
    """Effort noise at rest and skid-steer scrub must not read as sinkage (Gazebo finding)."""
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.SIM_TORQUE), sim=True)
    w = np.full(4, 5.0)
    loaded = signed_effort(TRUE_COEFS, w, np.zeros(4), 0.0) + 1.0
    t = 0.0
    for k in range(300):                                          # at rest, noisy effort
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, 9.81))
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), np.zeros(4),
                           np.full(4, 3.0 * (-1) ** k))
    assert est.estimate(t).sinkage_proxy == 0.0
    for k in range(300, 600):                                     # driving with extra load
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, 9.81))
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), w, loaded)
    held = est.estimate(t).sinkage_proxy
    assert held == pytest.approx(0.2, abs=0.01)
    for k in range(600, 900):                                     # turning hard, huge scrub
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0.8), (0, 0, 9.81))
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), w, loaded + 5.0)
    assert est.estimate(t).sinkage_proxy == pytest.approx(held)
    for k in range(900, 1200):                                    # stopped again: held
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, 9.81))
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), np.zeros(4), np.zeros(4))
    assert est.estimate(t).sinkage_proxy == pytest.approx(held)


def test_sinkage_proxy_clipped_to_unit_interval():
    m = make_model(i_ref=2.0)
    assert m.sinkage_proxy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert m.sinkage_proxy([-3.0] * 4) == 0.0
    assert m.sinkage_proxy([9.0] * 4) == 1.0


def test_yaml_round_trip(tmp_path):
    m = make_model()
    m.r2, m.n_samples, m.git_sha = [0.99] * 4, [100] * 4, 'abc123'
    path = tmp_path / 'm.yaml'
    m.save(str(path), header='test')
    back = EffortModel.load(str(path))
    assert np.allclose(back.coefficients, m.coefficients)
    assert back.effort_source == EffortSource.CURRENT and back.git_sha == 'abc123'
    assert path.read_text().startswith('# test')


def test_yaml_missing_field_is_an_error(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('effort_model: {calibrated: true}\n')
    with pytest.raises(ValueError, match='missing'):
        EffortModel.load(str(path))


def test_calibrated_model_needs_positive_i_ref():
    with pytest.raises(ValueError):
        make_model(i_ref=0.0)


def test_shipped_configs():
    """
    The sim model is a labelled zero model; no hardware model ships.

    Gazebo effort has no fittable flat-ground signal (see the file header), so the sim model is
    uncalibrated and sinkage is refused in sim too. The hardware model must be calibrated on the
    rover.
    """
    path = os.path.join(CONFIG, 'effort_model_sim.yaml')
    sim = EffortModel.load(path)
    assert sim.effort_source == EffortSource.SIM_TORQUE and sim.effort_unit == 'N.m'
    assert not sim.calibrated and np.all(sim.coefficients == 0.0)
    assert 'UNCALIBRATED ZERO MODEL' in open(path).read()
    est = TerrainEstimator(effort_model=sim, sim=True, logger=RecordingLogger())
    assert feed_loaded(est).effort_source == EffortSource.NONE
    assert not os.path.exists(os.path.join(CONFIG, 'effort_model.yaml'))


# --- the effort-source gate in the estimator ----------------------------------------------------
def feed_loaded(est, extra=1.0, seconds=2.0):
    """Drive at 5 rad/s on level ground with `extra` effort above the model."""
    w = np.full(4, 5.0)
    effort = signed_effort(TRUE_COEFS, w, np.zeros(4), 0.0) + extra
    for k in range(int(seconds * 100)):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, 9.81))
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), w, effort)
    return est.estimate(seconds)


def test_hardware_without_calibrated_model_refuses_sinkage():
    log = RecordingLogger()
    for model in (None, make_model(calibrated=False, i_ref=0.0)):
        est = TerrainEstimator(effort_model=model, sim=False, logger=log)
        est.on_rover_health(0.0, True)
        e = feed_loaded(est)
        e = est.estimate(2.1)
        assert e.effort_source == EffortSource.NONE
        assert e.sinkage_proxy == 0.0 and np.all(e.effort_residual == 0.0)
    assert log.count('sinkage_proxy 0') == 2, 'one warning per estimator, not one per tick'


def test_hardware_with_model_needs_matching_live_effort_source():
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.CURRENT), sim=False)
    assert feed_loaded(est).effort_source == EffortSource.NONE          # no RoverHealth yet
    est.on_rover_health(2.0, False)                                     # duty estimate only
    assert est.estimate(2.0).effort_source == EffortSource.NONE
    est.on_rover_health(2.0, True)
    e = est.estimate(2.0)
    assert e.effort_source == EffortSource.CURRENT
    assert np.allclose(e.effort_residual, 1.0, atol=0.01)
    assert e.sinkage_proxy == pytest.approx(0.2, abs=0.01)              # 1 / I_ref 5


def test_duty_estimate_model():
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.DUTY_ESTIMATE))
    est.on_rover_health(0.0, False)
    assert feed_loaded(est).effort_source == EffortSource.DUTY_ESTIMATE


def test_sim_uses_sim_torque_and_rejects_hardware_models():
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.SIM_TORQUE), sim=True)
    e = feed_loaded(est, extra=2.5)
    assert e.effort_source == EffortSource.SIM_TORQUE and e.sinkage_proxy == pytest.approx(0.5)
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.CURRENT), sim=True)
    assert feed_loaded(est).effort_source == EffortSource.NONE
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.SIM_TORQUE), sim=False)
    est.on_rover_health(0.0, True)
    assert feed_loaded(est).effort_source == EffortSource.NONE
