"""
Trial metrics (SPEC section 8) and the pre-registered statistics. Runs without ROS.

    python3 -m orion_experiments.analysis <trials_root> --out <report_dir> [--protocol p.yaml]
    ros2 run orion_experiments analyse_trials <trials_root> --out <report_dir>

<trials_root>/<trial_id>/metadata.json + bag/ are written by the trial runner. Output:
trials.csv (one row per trial), tests.csv (every hypothesis test), calibration.csv and
report.md.

Definitions (fixed here so the analysis is pre-registered, not chosen after the data):
- valid trial: outcome is not ABORT and, when the protocol gives a band, every cone depth lies
  in the cell's band.
- failure: outcome TIMEOUT, ENTRAPPED or SAFETY. success: outcome SUCCESS.
- progress: displacement along the heading at the first pose of the trial, from
  /orion/ground_truth_pose (fallback /odom_rf2o; never wheel odometry, SPEC F4).
- effective speed: progress / time-to-goal for a success, progress / trial duration (60 s)
  otherwise, so a rover that stops early is not rewarded for stopping.
- time-to-entrapment: elapsed time to the hard-stuck HALT for ENTRAPPED, else censored at the
  trial duration.
- entrapment onset t_entrap (ENTRAPPED only): the first FLAG_STUCK of the stuck episode that
  ended in the HALT. The episode is the last run of FLAG_STUCK that is still high at, or
  starts after, the last time the rover was >= 0.1 m (hard_stuck_progress_m) away from where
  it halted; position from the pose track, else integrated TerrainState.body_speed, else the
  last FLAG_STUCK run (see entrapment_onset).
- energy: integral of V_bat * sum|duty_i * I_i| when RoverHealth is recorded (electrical,
  hardware), else of sum|tau_i * w_i| (mechanical, simulation).
- critic calibration label at time t (council M2): in an ENTRAPPED trial 1 if
  t_entrap - t <= 10 s, else 0; in any other trial 0 if at least 10 s of trial remain after t,
  otherwise censored and left out. Labelling from the HALT instead would mark the rover
  "safe" during the 23 s between motion stopping and the HALT.

Phase-0 kill test (B0 only, reference light-packed; two-stage, SPEC section 8, D-008):
- stage 1 (session phase0, 20 per cell): GO if for at least one loose cell the failure rate
  is higher with one-sided Fisher exact p < 0.01, or the effective speed is lower with
  one-sided Mann-Whitney p < 0.01. Otherwise CONTINUE (never NO-GO here).
- stage 2 (session phase0b adds 20 more in light-packed and light-loose): light-loose vs
  light-packed on all trials of both stages, same tests, p < 0.04. GO or NO-GO.
The 0.01 + 0.04 split bounds the overall false-GO rate at 0.05 (Bonferroni over two looks).
EFFECT_SIZE_GATE (council M3, an open decision for the user, SPEC section 9) selects the rule:
False (default): the tests alone decide; the failure-rate rise (Newcombe hybrid-score CI) and
the median-speed drop (bootstrap CI) are reported next to the decision, descriptively.
True: a test also needs a rise >= 0.25 / a drop >= 30 % (the original D-008 rule, which has
less power; numbers from phase0_power in SPEC section 8).
heavy-loose vs light-packed changes two factors (ballast and soil): a GO that rests on
heavy-loose alone is reported as such, because H1 is the light-packed -> light-loose transfer.
Main study: the primary test is student_critic vs B0 pooled over the loose cells (one-sided
Fisher, alpha 0.05, not adjusted). Secondary: one-sided Fisher for each comparison in
COMPARISONS within each cell, Holm-corrected over that family of 16.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

FAILURE_OUTCOMES = ('TIMEOUT', 'ENTRAPPED', 'SAFETY')
RISK_HORIZON_S = 10.0
REFERENCE_CELL = 'light-packed'
LOOSE_CELLS = ('light-loose', 'heavy-loose')
STAGE2_CELL = 'light-loose'
STAGE2_SESSION = 'phase0b'
STAGE_ALPHA = {1: 0.01, 2: 0.04}
# Council M3 switch (SPEC section 9): False = GO on the one-sided tests alone, effect sizes
# reported with CIs; True = the tests must also show a rise >= 0.25 / a drop >= 30 %.
EFFECT_SIZE_GATE = False
MIN_RATE_RISE = 0.25
MIN_SPEED_DROP = 0.30
# Cells that differ from REFERENCE_CELL in more than the soil (heavy-loose adds ballast too).
TWO_FACTOR_CELLS = ('heavy-loose',)
# (a, b): H1 = controller a fails less often than controller b, tested in every cell.
COMPARISONS = (('student', 'B2'), ('student_critic', 'student'), ('student', 'B0'),
               ('B2', 'B0'))
# Pre-registered primary main-study test (council m7): a fails less than b, pooled over
# LOOSE_CELLS. The per-cell COMPARISONS are secondary.
PRIMARY_COMPARISON = ('student_critic', 'B0')
PRIMARY_ALPHA = 0.05
HARD_STUCK_PROGRESS_M = 0.1      # orion_safety hard_stuck_progress_m (config/safety.yaml)
POSE_TOPICS = ('/orion/ground_truth_pose', '/odom_rf2o')
TOPICS = POSE_TOPICS + ('/orion/terrain_state', '/orion/safety_status', '/orion/policy_status',
                        '/joint_states', '/orion/rover_health')
FLAG_STUCK, FLAG_HARD_STUCK, MODE_ESCAPE = 16, 64, 4


# --- statistics ---------------------------------------------------------------------------

def fisher_one_sided(fail_a: int, n_a: int, fail_b: int, n_b: int, alternative: str) -> float:
    """
    One-sided Fisher exact p on [[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]].

    alternative 'greater': H1 failure rate a > b; 'less': H1 a < b.
    """
    table = [[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]]
    return float(stats.fisher_exact(table, alternative=alternative).pvalue)


def mannwhitney_less(x: Sequence[float], y: Sequence[float]) -> float:
    """One-sided Mann-Whitney U p for H1: x tends to be smaller than y (exact when possible)."""
    return float(stats.mannwhitneyu(x, y, alternative='less', method='auto').pvalue)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion k / n."""
    if n <= 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(k_a: int, n_a: int, k_b: int, n_b: int,
                      level: float = 0.95) -> tuple[float, float]:
    """Newcombe hybrid-score interval (his method 10) of the difference k_a/n_a - k_b/n_b."""
    if n_a <= 0 or n_b <= 0:
        return math.nan, math.nan
    pa, pb = k_a / n_a, k_b / n_b
    la, ua = wilson_interval(k_a, n_a, level)
    lb, ub = wilson_interval(k_b, n_b, level)
    d = pa - pb
    return (d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2),
            d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2))


def speed_drop_interval(x: Sequence[float], y: Sequence[float], level: float = 0.95,
                        n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """
    Percentile-bootstrap interval of 1 - median(x) / median(y) (x test, y reference).

    Each group is resampled on its own; resamples with median(y) <= 0 are dropped. The seed is
    fixed so the report is reproducible.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    mx = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    my = np.median(rng.choice(y, (n_boot, len(y))), axis=1)
    ok = my > 0
    if not ok.any():
        return math.nan, math.nan
    drop = 1.0 - mx[ok] / my[ok]
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(drop, [tail, 100.0 - tail])
    return float(low), float(high)


def phase0_stage(trials: pd.DataFrame) -> int:
    """2 once any trial of the stage-2 session (trial_id 'phase0b_...') is present, else 1."""
    return 2 if trials.trial_id.astype(str).str.startswith(STAGE2_SESSION + '_').any() else 1


def _safety_count(df: pd.DataFrame) -> float:
    """Count SAFETY outcomes (NaN when the frame has no outcome column)."""
    return int((df.outcome == 'SAFETY').sum()) if 'outcome' in df else math.nan


def phase0_decision(trials: pd.DataFrame, stage: Optional[int] = None,
                    effect_size_gate: bool = EFFECT_SIZE_GATE,
                    min_rate_rise: float = MIN_RATE_RISE, min_speed_drop: float = MIN_SPEED_DROP,
                    use_speed: bool = True, alpha: Optional[float] = None,
                    ci_level: float = 0.95) -> dict:
    """Two-stage GO / CONTINUE / NO-GO from valid B0 trials (see the module docstring)."""
    # Columns: trial_id, cell, controller, valid, failure, effective_speed (outcome optional).
    # stage defaults to phase0_stage(trials); alpha defaults to the pre-registered
    # STAGE_ALPHA[stage] and exists only for sensitivity analyses, never for the reported
    # decision. min_rate_rise / min_speed_drop only act when effect_size_gate is True.
    stage = phase0_stage(trials) if stage is None else stage
    if stage not in STAGE_ALPHA:
        raise ValueError(f'stage must be 1 or 2, got {stage}')
    alpha = STAGE_ALPHA[stage] if alpha is None else alpha
    b0 = trials[(trials.controller == 'B0') & trials.valid]
    ref = b0[b0.cell == REFERENCE_CELL]
    if ref.empty:
        raise ValueError(f'no valid B0 trials in the reference cell {REFERENCE_CELL}')
    rows = []
    for cell in (LOOSE_CELLS if stage == 1 else (STAGE2_CELL,)):
        test = b0[b0.cell == cell]
        if test.empty:
            continue
        fa, na, fb, nb = int(test.failure.sum()), len(test), int(ref.failure.sum()), len(ref)
        rise = fa / na - fb / nb
        rise_low, rise_high = newcombe_interval(fa, na, fb, nb, ci_level)
        p_fail = fisher_one_sided(fa, na, fb, nb, 'greater')
        big_rise = rise >= min_rate_rise or not effect_size_gate
        row = {'stage': stage, 'cell': cell, 'factors_changed': 2 if cell in TWO_FACTOR_CELLS
               else 1, 'n': na, 'ref_n': nb, 'failures': fa,
               'safety_failures': _safety_count(test), 'ref_safety_failures': _safety_count(ref),
               'failure_rate': fa / na, 'ref_failure_rate': fb / nb, 'rate_rise': rise,
               'rate_rise_ci_low': rise_low, 'rate_rise_ci_high': rise_high,
               'fisher_p': p_fail, 'alpha': alpha,
               'failure_criterion': bool(p_fail < alpha and big_rise)}
        speed_x, speed_y = test.effective_speed.dropna(), ref.effective_speed.dropna()
        if use_speed and len(speed_x) and len(speed_y) and speed_y.median() > 0:
            drop = 1.0 - speed_x.median() / speed_y.median()
            drop_low, drop_high = speed_drop_interval(speed_x, speed_y, ci_level)
            p_speed = mannwhitney_less(speed_x, speed_y)
            big_drop = drop >= min_speed_drop or not effect_size_gate
            row.update(median_speed=speed_x.median(), ref_median_speed=speed_y.median(),
                       speed_drop=drop, speed_drop_ci_low=drop_low, speed_drop_ci_high=drop_high,
                       mannwhitney_p=p_speed, speed_criterion=bool(p_speed < alpha and big_drop))
        else:
            row.update(median_speed=math.nan, ref_median_speed=math.nan, speed_drop=math.nan,
                       speed_drop_ci_low=math.nan, speed_drop_ci_high=math.nan,
                       mannwhitney_p=math.nan, speed_criterion=False)
        rows.append(row)
    table = pd.DataFrame(rows)
    if stage == 2 and table.empty:
        raise ValueError(f'stage 2 needs valid B0 trials in {STAGE2_CELL}')
    go_cells = ([] if table.empty else
                list(table.cell[table.failure_criterion | table.speed_criterion]))
    go = bool(go_cells)
    decision = 'GO' if go else ('CONTINUE' if stage == 1 else 'NO-GO')
    return {'go': go, 'decision': decision, 'stage': stage, 'alpha': alpha, 'table': table,
            'ref_n': len(ref), 'use_speed': use_speed, 'effect_size_gate': effect_size_gate,
            'go_cells': go_cells,
            'two_factor_only': go and all(c in TWO_FACTOR_CELLS for c in go_cells)}


def _fisher_row(ta: pd.DataFrame, tb: pd.DataFrame, ci_level: float) -> dict:
    """Failure counts, one-sided Fisher (a fails less than b) and the rate reduction b - a."""
    fa, fb, na, nb = int(ta.failure.sum()), int(tb.failure.sum()), len(ta), len(tb)
    low, high = newcombe_interval(fb, nb, fa, na, ci_level)
    return {'n_a': na, 'n_b': nb, 'failure_rate_a': fa / na, 'failure_rate_b': fb / nb,
            'rate_reduction': fb / nb - fa / na, 'rate_reduction_ci_low': low,
            'rate_reduction_ci_high': high,
            'fisher_p': fisher_one_sided(fa, na, fb, nb, 'less')}


def main_study(trials: pd.DataFrame, comparisons=COMPARISONS, alpha: float = 0.05,
               primary=PRIMARY_COMPARISON, primary_alpha: float = PRIMARY_ALPHA,
               ci_level: float = 0.95) -> pd.DataFrame:
    """
    Pre-registered main-study tests, one row each (column role).

    primary: a fails less than b, both pooled over LOOSE_CELLS, one-sided Fisher at
    primary_alpha with no adjustment (holm_p = fisher_p). secondary: the same test per cell
    and pair in comparisons, Holm-corrected over the secondary family only.
    """
    valid = trials[trials.valid]
    rows = []
    loose = valid[valid.cell.isin(LOOSE_CELLS)]
    ta, tb = loose[loose.controller == primary[0]], loose[loose.controller == primary[1]]
    if len(ta) and len(tb):
        row = {'role': 'primary', 'cell': '+'.join(LOOSE_CELLS), 'a': primary[0],
               'b': primary[1], **_fisher_row(ta, tb, ci_level)}
        row.update(holm_p=row['fisher_p'], reject_h0=bool(row['fisher_p'] < primary_alpha))
        rows.append(row)
    secondary = []
    for cell in sorted(valid.cell.unique()):
        in_cell = valid[valid.cell == cell]
        for a, b in comparisons:
            ta, tb = in_cell[in_cell.controller == a], in_cell[in_cell.controller == b]
            if ta.empty or tb.empty:
                continue
            secondary.append({'role': 'secondary', 'cell': cell, 'a': a, 'b': b,
                              **_fisher_row(ta, tb, ci_level)})
    if secondary:
        adjusted = holm([r['fisher_p'] for r in secondary])
        for r, p in zip(secondary, adjusted):
            r.update(holm_p=float(p), reject_h0=bool(p < alpha))
    return pd.DataFrame(rows + secondary)


def calibration(risk: Sequence[float], label: Sequence[int], n_bins: int = 10) -> dict:
    """Reliability table, expected calibration error and Brier score of rho."""
    r, y = np.asarray(risk, float), np.asarray(label, float)
    if len(r) == 0:
        return {'n': 0, 'ece': math.nan, 'brier': math.nan, 'table': pd.DataFrame()}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            conf, acc = r[mask].mean(), y[mask].mean()
            ece += mask.sum() / len(r) * abs(conf - acc)
            rows.append({'bin_low': edges[b], 'bin_high': edges[b + 1], 'n': int(mask.sum()),
                         'mean_risk': conf, 'entrapped_fraction': acc})
    return {'n': len(r), 'ece': float(ece), 'brier': float(np.mean((r - y) ** 2)),
            'table': pd.DataFrame(rows)}


# --- reading trials ------------------------------------------------------------------------

def find_msg_dir(explicit: Optional[str] = None) -> str:
    """Directory with orion_msgs .msg files: explicit, $ORION_MSGS_DIR, ament index, repo."""
    candidates = [explicit, os.environ.get('ORION_MSGS_DIR')]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('orion_msgs'), 'msg'))
    except Exception:  # no ROS, or orion_msgs not installed: fall through to the repo copy
        pass
    path = os.path.realpath(os.path.dirname(__file__))
    while os.path.dirname(path) != path:
        candidates.append(os.path.join(path, 'robot_ws', 'src', 'orion_msgs', 'msg'))
        candidates.append(os.path.join(path, 'orion_msgs', 'msg'))
        path = os.path.dirname(path)
    for c in candidates:
        if c and glob.glob(os.path.join(c, '*.msg')):
            return c
    raise FileNotFoundError('orion_msgs/msg not found; pass --msg-dir')


def make_typestore(msg_dir: str):
    """Jazzy typestore plus the orion_msgs message types."""
    from rosbags.typesys import get_types_from_msg
    from rosbags.typesys import get_typestore
    from rosbags.typesys import Stores
    typestore = get_typestore(Stores.ROS2_JAZZY)
    types = {}
    for path in sorted(glob.glob(os.path.join(msg_dir, '*.msg'))):
        name = os.path.splitext(os.path.basename(path))[0]
        types.update(get_types_from_msg(Path(path).read_text(), f'orion_msgs/msg/{name}'))
    typestore.register(types)
    return typestore


def read_bag(bag_path: str, typestore, topics: Sequence[str] = TOPICS) -> dict:
    """{topic: [(t_s, msg), ...]} in receive-time order (t = bag timestamp in s)."""
    from rosbags.highlevel import AnyReader
    out: dict = {t: [] for t in topics}
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in out]
        for conn, t_ns, raw in reader.messages(connections=conns):
            out[conn.topic].append((t_ns * 1e-9, reader.deserialize(raw, conn.msgtype)))
    return out


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _pose_track(msgs: dict):
    for topic in POSE_TOPICS:
        if msgs.get(topic):
            poses = [(t, m.pose.pose if topic == '/odom_rf2o' else m.pose)
                     for t, m in msgs[topic]]
            return topic, poses
    return None, []


def _energy(msgs: dict) -> tuple[float, Optional[str]]:
    joints = msgs.get('/joint_states', [])
    if len(joints) < 2:
        return math.nan, None
    health = msgs.get('/orion/rover_health', [])
    t_health = np.array([t for t, _ in health])
    power = []
    for t, js in joints:
        effort, vel = np.abs(np.asarray(js.effort)), np.abs(np.asarray(js.velocity))
        if len(health):
            h = health[max(0, int(np.searchsorted(t_health, t, side='right')) - 1)][1]
            n = min(len(effort), len(h.duty))
            power.append(h.battery_v * float(np.sum(np.abs(np.asarray(h.duty[:n])) * effort[:n])))
        else:
            power.append(float(np.sum(effort * vel)) if len(effort) == len(vel) else math.nan)
    t = np.array([t for t, _ in joints])
    p = np.asarray(power)
    energy = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(t)))
    return energy, ('electrical' if len(health) else 'mechanical')


def _flag_runs(samples) -> list[tuple[float, float]]:
    """[(t_first_high, t_last_high)] of every run of True in [(t, bool)] (time order)."""
    runs, start, last = [], None, None
    for t, high in samples:
        if high and start is None:
            start = t
        if high:
            last = t
        elif start is not None:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


def entrapment_onset(stuck, track, t_halt: float,
                     min_progress: float = HARD_STUCK_PROGRESS_M) -> Optional[float]:
    """
    First FLAG_STUCK (s) of the stuck episode that ended in the hard-stuck HALT at t_halt.

    stuck: [(t, flag high)] from /orion/safety_status. track: [(t, position)] with position
    an array in m (2-D pose or 1-D odometer); may be empty. The episode is the first FLAG_STUCK
    run that is still high at, or starts after, the last time the rover was >= min_progress
    away from its position at the HALT: earlier runs ended with the rover moving on. Without
    a track the last run is the episode. None when FLAG_STUCK never rose before t_halt.
    """
    runs = _flag_runs([(t, high) for t, high in stuck if t <= t_halt])
    if not runs:
        return None
    track = [(t, np.atleast_1d(np.asarray(p, float))) for t, p in track if t <= t_halt]
    if not track:
        return runs[-1][0]
    final = track[-1][1]
    far = [t for t, p in track if np.linalg.norm(p - final) >= min_progress]
    t_free = max(far, default=-math.inf)
    for start, end in runs:
        if end >= t_free:
            return start
    return runs[-1][0]      # moved after every run (not a real hard-stuck): last run


def _progress_track(msgs: dict):
    """(source, [(t, position)]): pose track, else integrated terrain body speed, else none."""
    source, poses = _pose_track(msgs)
    if poses:
        return source, [(t, (p.position.x, p.position.y)) for t, p in poses]
    terrain = msgs.get('/orion/terrain_state', [])
    if len(terrain) >= 2:
        t = np.array([s for s, _ in terrain])
        v = np.array([m.body_speed for _, m in terrain], float)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        return 'terrain_odometer', list(zip(t, x))
    return 'flags_only', []


def trial_metrics(meta: dict, msgs: dict, cone_valid: Optional[bool] = None):
    """(one row of SPEC section 8 metrics, [(rho, label)] calibration samples)."""
    t0, t1 = meta.get('start_ros_s'), meta.get('end_ros_s')
    if t0 is None or t1 is None:
        raise ValueError(f'{meta.get("trial_id")}: metadata has no start/end time')
    eps = 1e-6   # bag stamps are integer ns
    msgs = {k: [(t, m) for t, m in v if t0 - eps <= t <= t1 + eps] for k, v in msgs.items()}
    outcome, duration = meta['outcome'], float(meta['duration_s'])
    elapsed = float(meta.get('elapsed_s') or (t1 - t0))
    success, entrapped = outcome == 'SUCCESS', outcome == 'ENTRAPPED'
    row = {'trial_id': meta['trial_id'], 'cell': meta['cell'],
           'controller': meta['controller'], 'outcome': outcome,
           'valid': outcome is not None and outcome != 'ABORT' and cone_valid is not False,
           'success': success, 'failure': outcome in FAILURE_OUTCOMES,
           'time_to_goal_s': elapsed if success else math.nan, 'entrapped': entrapped,
           'time_to_entrapment_s': elapsed if entrapped else duration,
           'entrapment_observed': entrapped, 'elapsed_s': elapsed,
           'ballast_kg': meta.get('ballast_kg'), 'operator': meta.get('operator')}

    source, poses = _pose_track(msgs)
    if poses:
        xy = np.array([[p.position.x, p.position.y] for _, p in poses])
        yaw0 = _yaw(poses[0][1].orientation)
        progress = float((xy[-1] - xy[0]) @ np.array([math.cos(yaw0), math.sin(yaw0)]))
        distance = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        speed = progress / (elapsed if success else duration)
    else:
        progress = distance = speed = math.nan
    row.update(pose_source=source, progress_m=progress, distance_m=distance,
               effective_speed=speed)

    terrain = [m for _, m in msgs.get('/orion/terrain_state', [])]
    row['mean_slip'] = float(np.mean([np.mean(m.slip_side) for m in terrain])) if terrain \
        else math.nan
    row['mean_sinkage_proxy'] = float(np.mean([m.sinkage_proxy for m in terrain])) if terrain \
        else math.nan
    row['energy_j'], row['energy_source'] = _energy(msgs)

    safety = msgs.get('/orion/safety_status', [])
    stuck = [bool(m.flags & FLAG_STUCK) for _, m in safety]
    rises = sum(1 for a, b in zip([False] + stuck, stuck) if b and not a)
    falls = sum(1 for a, b in zip(stuck, stuck[1:]) if a and not b)
    modes = [m.mode for _, m in msgs.get('/orion/policy_status', [])]
    row.update(stuck_events=rises, recovered_stuck_events=falls,
               recovery=bool(rises > 0 and not entrapped),
               escapes=sum(1 for a, b in zip([None] + modes, modes)
                           if b == MODE_ESCAPE and a != MODE_ESCAPE))

    t_onset, onset_source = math.nan, None
    if entrapped:
        t_halt = next((t for t, m in safety if m.flags & FLAG_HARD_STUCK), t1)
        onset_source, track = _progress_track(msgs)
        flags = [(t, bool(m.flags & FLAG_STUCK)) for t, m in safety]
        t_onset = entrapment_onset(flags, track, t_halt)
        if t_onset is None:     # ENTRAPPED without any FLAG_STUCK (operator outcome)
            t_onset, onset_source = t_halt, 'halt'
    row.update(entrapment_onset_s=t_onset - t0, onset_source=onset_source)

    samples = []
    risks = [(t, float(m.entrapment_risk)) for t, m in msgs.get('/orion/policy_status', [])
             if math.isfinite(m.entrapment_risk)]
    row['max_risk'] = max((r for _, r in risks), default=math.nan)
    for t, r in risks:
        if entrapped:
            samples.append((r, int(t_onset - t <= RISK_HORIZON_S)))
        elif t1 - t >= RISK_HORIZON_S:
            samples.append((r, 0))
    return row, samples


def load_trials(root: str, msg_dir: Optional[str] = None, protocol=None):
    """(DataFrame of per-trial metrics, calibration samples per controller)."""
    from .protocol import cone_depth_valid
    typestore = make_typestore(find_msg_dir(msg_dir))
    rows, samples = [], {}
    for meta_path in sorted(glob.glob(os.path.join(root, '*', 'metadata.json'))):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('outcome') is None:
            print(f'skipping {meta_path}: trial never finished', file=sys.stderr)
            continue
        bag = os.path.join(os.path.dirname(meta_path), 'bag')
        msgs = read_bag(bag, typestore) if os.path.isdir(bag) else {}
        cone_ok = (cone_depth_valid(protocol, meta['cell'], meta.get('cone_depths_mm'))
                   if protocol is not None else None)
        row, s = trial_metrics(meta, msgs, cone_ok)
        row['cone_depth_valid'] = cone_ok
        rows.append(row)
        samples.setdefault(row['controller'], []).extend(s)
    return pd.DataFrame(rows), samples


# --- report ------------------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return '_none_\n'
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(
            f'{v:.4g}' if isinstance(v, float) else str(v) for v in r) + ' |')
    return '\n'.join(lines) + '\n'


def summary_table(trials: pd.DataFrame) -> pd.DataFrame:
    valid = trials[trials.valid]
    return valid.groupby(['cell', 'controller']).agg(
        n=('trial_id', 'count'), failures=('failure', 'sum'), failure_rate=('failure', 'mean'),
        entrapped=('entrapped', 'sum'), median_effective_speed=('effective_speed', 'median'),
        mean_slip=('mean_slip', 'mean'), mean_energy_j=('energy_j', 'mean')).reset_index()


def _phase0_lines(ph: dict) -> list[str]:
    alpha, gate = ph['alpha'], ph['effect_size_gate']
    rise = (f'a failure-rate rise >= {MIN_RATE_RISE:g} with ' if gate
            else 'a higher failure rate, ')
    crit = f'{rise}one-sided Fisher p < {alpha}'
    if ph['use_speed']:
        drop = (f'a median effective-speed drop >= {MIN_SPEED_DROP:.0%} with ' if gate
                else 'a lower effective speed, ')
        crit += f', or {drop}one-sided Mann-Whitney p < {alpha}'
    rule = ('effect-size gate ON (EFFECT_SIZE_GATE / --effect-size-gate)' if gate else
            'effect-size gate OFF: effect sizes and their 95 % CIs are descriptive')
    scope = ('at least one loose cell' if ph['stage'] == 1
             else f'{STAGE2_CELL}, stages 1 and 2 pooled')
    nxt = {'GO': '', 'NO-GO': ' Pivot to self-rescue (SPEC section 8).',
           'CONTINUE': (f' Run stage 2 ({STAGE2_SESSION}_protocol.yaml: 20 more trials in '
                        f'{REFERENCE_CELL} and {STAGE2_CELL}) and re-run this analysis.')}
    lines = [f'## Phase-0 kill test, stage {ph["stage"]} (pre-registered)', '',
             f'**Decision: {ph["decision"]}** (reference {REFERENCE_CELL}, n = {ph["ref_n"]}; '
             f'GO needs {crit}, in {scope}; {rule}).{nxt[ph["decision"]]}', '']
    if ph['two_factor_only']:
        lines += ['**The GO rests on heavy-loose alone.** That cell changes ballast and soil '
                  'together, so it does not isolate the light-packed -> light-loose transfer '
                  'that H1 names; report it as a two-factor result.', '']
    lines += ['SAFETY outcomes (current, tilt, heartbeat or geofence HALT) count as failures; '
              'the current HALT depends on the I_peak GUESS (SPEC section 5), so the '
              'safety_failures columns show how much of the result rests on it.', '',
              _md(ph['table'])]
    return lines


def write_report(trials: pd.DataFrame, samples: dict, out_dir: str,
                 use_speed: bool = True, effect_size_gate: bool = EFFECT_SIZE_GATE) -> dict:
    """Write trials.csv, tests.csv, calibration.csv and report.md; return the results."""
    os.makedirs(out_dir, exist_ok=True)
    trials.to_csv(os.path.join(out_dir, 'trials.csv'), index=False)
    lines = ['# ORION trial analysis', '',
             f'{len(trials)} trials, {int(trials.valid.sum())} valid '
             f'({int((~trials.valid).sum())} excluded: ABORT or cone depth out of band).', '',
             '## Per cell and controller', '', _md(summary_table(trials))]
    results: dict = {}
    tests = []
    b0 = trials[trials.controller == 'B0']
    if (b0.cell == REFERENCE_CELL).any() and b0.cell.isin(LOOSE_CELLS).any():
        ph = phase0_decision(trials, use_speed=use_speed, effect_size_gate=effect_size_gate)
        results['phase0'] = ph
        lines += _phase0_lines(ph)
        tests += [dict(test='phase0', **r) for r in ph['table'].to_dict('records')]
    ms = main_study(trials)
    if len(ms):
        results['main'] = ms
        lines += ['## Main study: failure rate, one-sided Fisher', '',
                  f'Primary (pre-registered): {PRIMARY_COMPARISON[0]} vs '
                  f'{PRIMARY_COMPARISON[1]} pooled over {" and ".join(LOOSE_CELLS)}, '
                  f'alpha {PRIMARY_ALPHA}, unadjusted. Secondary: every per-cell comparison, '
                  'Holm-corrected over the secondary family (holm_p).', '', _md(ms)]
        tests += [dict(test='main', **r) for r in ms.to_dict('records')]
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, 'tests.csv'), index=False)
    lines += ['## Critic calibration (rho vs entrapment within 10 s)', '']
    cal_rows = []
    for ctrl, s in sorted(samples.items()):
        if not s:
            continue
        cal = calibration([r for r, _ in s], [y for _, y in s])
        results.setdefault('calibration', {})[ctrl] = cal
        lines += [f'### {ctrl}: n = {cal["n"]}, ECE = {cal["ece"]:.3f}, '
                  f'Brier = {cal["brier"]:.3f}', '', _md(cal['table'])]
        cal_rows += [dict(controller=ctrl, **r) for r in cal['table'].to_dict('records')]
    if not cal_rows:
        lines.append('_no critic output recorded_')
    pd.DataFrame(cal_rows).to_csv(os.path.join(out_dir, 'calibration.csv'), index=False)
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='ORION trial metrics and statistics.')
    parser.add_argument('trials_root')
    parser.add_argument('--out', required=True, help='report directory')
    parser.add_argument('--protocol', help='session protocol YAML (cone depth bands)')
    parser.add_argument('--msg-dir', help='orion_msgs/msg directory (default: search)')
    parser.add_argument('--no-speed', action='store_true',
                        help='failure rate only (no ground truth, SPEC section 9)')
    parser.add_argument('--effect-size-gate', action='store_true', default=EFFECT_SIZE_GATE,
                        help='Phase-0 GO also needs a rise >= 0.25 / drop >= 30 %% (council '
                             'M3; SPEC section 9 open decision)')
    args = parser.parse_args(argv)
    protocol = None
    if args.protocol:
        from .protocol import load_protocol
        protocol = load_protocol(args.protocol)
    trials, samples = load_trials(args.trials_root, args.msg_dir, protocol)
    if trials.empty:
        print(f'no finished trials under {args.trials_root}', file=sys.stderr)
        return 1
    results = write_report(trials, samples, args.out, use_speed=not args.no_speed,
                           effect_size_gate=args.effect_size_gate)
    if 'phase0' in results:
        ph = results['phase0']
        print(f'Phase-0 stage {ph["stage"]}: {ph["decision"]}')
    print(f'report written to {os.path.join(args.out, "report.md")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
