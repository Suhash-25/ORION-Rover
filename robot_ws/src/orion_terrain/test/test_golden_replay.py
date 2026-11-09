"""
Golden replay (SPEC F7, council M4): a learning-env rollout as a rosbag2 through the runtime.

orion_learning.synth_bag writes a small bag of a B0 rollout in the heavy-loose cell (80 Hz
joint states and IMU, 20 Hz overhead-camera poses with fix dropouts, truth.csv). The bag goes
through the runtime TerrainEstimator (with the env's effort model) and orion_policy's
ObservationBuilder, evaluated at each truth.csv control step exactly as the nodes would. The
same rollout is re-run in the env (it is deterministic in its seed) to get the env's own
observation rows.

The rollout has a 5.5 s camera dropout. Slip, body speed and the observation are compared on
the steps with a settled speed fix (a fix in every 0.1 s of the last second, after a 1 s
warm-up): during the dropout both sides dead-reckon on independent IMU noise and diverge, and
both say so with speed confidence 0 (checked). Checked:

* slip_side vs truth.csv slip_left/right (true wheel speeds and body velocity): RMS < 0.05;
* sinkage proxy vs the env's proxy for the same rollout, whole run after warm-up (a proxy has no
  physical truth; the env's value is what the policy was trained on): RMS < 0.05;
* the runtime observation vs the env's observation, per field. The two sides draw independent
  sensor noise (the bag at 80 Hz, the env at 20 Hz), so the tolerance is per-field RMS (in
  normalised units) with about 2x margin over the measured difference: prev_cmd and roll/pitch
  are exact (1e-4), the rest is noise-limited (table TOLERANCE below).
"""

import csv

import numpy as np
from orion_learning import env as E
from orion_learning import obs_spec as S
from orion_learning.baselines import B0Controller
from orion_policy.observation import ObservationBuilder
from orion_terrain.effort_model import EffortModel, EffortSource
from orion_terrain.estimator import msg_get, msg_quat, msg_vec3, TerrainEstimator
from orion_terrain.fit_effort_model import read_bag
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS
import pytest

pytest.importorskip('rosbags')
synth_bag = pytest.importorskip('orion_learning.synth_bag')   # needs rosbags as well

CELL, SECONDS, SEED = 'heavy-loose', 12.0, 3
WARMUP_S = 1.0
TOLERANCE = {                       # per-field RMS(runtime - env), normalised obs units
    'prev_cmd': 1e-4, 'roll_pitch': 1e-4,
    'wheel_speed': 0.04, 'wheel_effort': 0.08, 'body_accel_xy': 0.04, 'yaw_rate': 0.03,
    'body_speed': 0.1, 'speed_confidence': 0.05, 'slip_side': 0.12, 'sinkage_proxy': 0.05,
}
# body_speed / slip_side: the env models a speed fix as truth + N(0, 0.03 m/s) every step, the
# runtime differentiates 1 mm camera poses (S-G window), so the env's own estimate is the
# noisier one (measured RMS 0.05 / 0.057 normalised).


def env_effort_model() -> EffortModel:
    """Return the env's SIM_EFFORT_MODEL as a runtime (calibrated, SIM_TORQUE) model."""
    m = E.SIM_EFFORT_MODEL
    return EffortModel(coefficients=np.tile(m.coefficients, (4, 1)), i_ref=m.ref,
                       effort_unit='N.m', effort_source=EffortSource.SIM_TORQUE, calibrated=True)


def env_rollout():
    """Re-run the rollout synth_bag recorded; returns the env's obs rows (k, OBS_DIM)."""
    env = E.OrionEnv(1, E.cell_config(CELL), seed=SEED, auto_reset=False, record_substeps=True)
    obs, _ = env.reset()
    ctl = B0Controller(1)
    rows = []
    for _ in range(int(round(SECONDS / E.DT))):
        obs, _, term, trunc, _ = env.step(ctl.act(obs))
        rows.append(obs[0])
        if term[0] or trunc[0]:
            break
    return np.array(rows)


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    bag = tmp_path_factory.mktemp('golden') / 'bag'
    synth_bag.write_bag(bag, CELL, SECONDS, seed=SEED)
    with open(bag / 'truth.csv') as f:
        truth = list(csv.DictReader(f))
    topics = {DEFAULT_TOPICS[r]: r for r in ('joint_states', 'imu', 'ground_truth_pose')}
    msgs = sorted(read_bag(str(bag), list(topics)), key=lambda m: (m[0], m[1]))

    est = TerrainEstimator(effort_model=env_effort_model(), sim=True)
    builder = ObservationBuilder(DEFAULT_JOINT_NAMES, effort_max=E.sim_effort_max)
    estimates, obs = [], []
    i = 0
    for row in truth:
        stamp = int(row['stamp_ns']) * 1e-9
        while i < len(msgs) and msgs[i][0] <= stamp + 1e-9:
            t, topic, msg = msgs[i]
            role = topics[topic]
            est.on_message(t, role, msg)
            if role == 'joint_states':
                builder.update_joint_state(t, list(msg_get(msg, 'name')),
                                           msg_get(msg, 'velocity'), msg_get(msg, 'effort'))
            elif role == 'imu':
                builder.update_imu(t, msg_quat(msg, 'orientation'),
                                   msg_vec3(msg, 'angular_velocity'),
                                   msg_vec3(msg, 'linear_acceleration'))
            i += 1
        e = est.estimate(stamp)
        builder.update_terrain(stamp, e.body_speed, e.confidence, e.slip_side, e.sinkage_proxy)
        estimates.append(e)
        obs.append(builder.build(stamp, (float(row['cmd_v']), float(row['cmd_w']))))
    env_obs = env_rollout()
    assert len(env_obs) == len(truth), 'the env re-run must be the rollout in the bag'
    warm = np.array([float(r['t']) >= WARMUP_S for r in truth])
    fresh = np.array([e.time_since_fix <= 0.1 + 1e-6 for e in estimates])
    span = int(round(WARMUP_S * S.CONTROL_HZ))
    settled = warm & np.array([fresh[max(0, k - span):k + 1].all() for k in range(len(fresh))])
    return truth, estimates, np.array(obs), env_obs, warm, settled


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_rollout_is_long_and_eventful_enough(replay):
    truth, estimates, obs, env_obs, _, settled = replay
    assert len(truth) >= 0.75 * SECONDS * S.CONTROL_HZ
    assert max(e.sinkage_proxy for e in estimates) > 0.1, 'loose soil must load the wheels'
    assert 0.15 < settled.mean() < 0.9, 'needs both settled fixes and a dropout'
    # Deep in the dropout (>= 1.5 s without a fix, camera still blind): both sides at c ~ 0.
    blind = np.array([1.5 < e.time_since_fix < np.inf and r['has_fix'] == '0'
                      for e, r in zip(estimates, truth)])
    conf = S.field_slice('speed_confidence')
    assert blind.any() and np.all(obs[blind, conf] == 0.0)
    assert np.all(env_obs[blind, conf] < 0.05)


def test_runtime_slip_matches_truth(replay):
    truth, estimates, _, _, _, settled = replay
    true = np.array([[float(r['slip_left']), float(r['slip_right'])] for r in truth])
    est = np.array([e.slip_side for e in estimates])
    err = rms((est - true)[settled])
    print(f'slip_side RMS error vs truth.csv (settled fix): {err:.4f}, '
          f'whole run incl. dropout: {rms(est - true):.4f}')
    assert err < 0.05


def test_runtime_sinkage_proxy_matches_env(replay):
    _, estimates, _, env_obs, warm, _ = replay
    runtime = np.array([e.sinkage_proxy for e in estimates])
    env = env_obs[:, S.field_slice('sinkage_proxy')][:, 0]
    err = rms((runtime - env)[warm])
    print(f'sinkage proxy RMS runtime vs env: {err:.4f} (env mean {env[warm].mean():.3f})')
    assert err < 0.05


def test_observation_builder_matches_env_rows(replay):
    _, _, obs, env_obs, _, settled = replay
    diff = (obs - env_obs)[settled]
    report = {f.name: rms(diff[:, S.field_slice(f.name)]) for f in S.OBS_FIELDS}
    print('per-field RMS runtime vs env:', {k: round(v, 4) for k, v in report.items()})
    bad = {k: v for k, v in report.items() if v > TOLERANCE[k]}
    assert not bad, f'fields beyond tolerance {TOLERANCE}: {bad}'
