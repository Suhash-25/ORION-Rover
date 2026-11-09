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
- time-to-entrapment: elapsed time for ENTRAPPED, else censored at the trial duration.
- energy: integral of V_bat * sum|duty_i * I_i| when RoverHealth is recorded (electrical,
  hardware), else of sum|tau_i * w_i| (mechanical, simulation).
- critic calibration label at time t: 1 if the trial ended ENTRAPPED within 10 s after t,
  0 if it did not and at least 10 s of trial remain; otherwise censored and left out.

Phase-0 GO / NO-GO (B0 only, reference light-packed): GO if for at least one loose cell
the failure rate rises >= 0.25 with one-sided Fisher exact p < 0.05, or the median
effective speed drops >= 30 % with one-sided Mann-Whitney p < 0.05.
Main study: one-sided Fisher on failure rate for each comparison in COMPARISONS within each
cell, Holm-corrected over the whole family.
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
# (a, b): H1 = controller a fails less often than controller b, tested in every cell.
COMPARISONS = (('student', 'B2'), ('student_critic', 'student'), ('student', 'B0'),
               ('B2', 'B0'))
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


def phase0_decision(trials: pd.DataFrame, alpha: float = 0.05, min_rate_rise: float = 0.25,
                    min_speed_drop: float = 0.30, use_speed: bool = True) -> dict:
    """GO / NO-GO from valid B0 trials (columns cell, controller, failure, effective_speed)."""
    b0 = trials[(trials.controller == 'B0') & trials.valid]
    ref = b0[b0.cell == REFERENCE_CELL]
    if ref.empty:
        raise ValueError(f'no valid B0 trials in the reference cell {REFERENCE_CELL}')
    rows = []
    for cell in LOOSE_CELLS:
        test = b0[b0.cell == cell]
        if test.empty:
            continue
        fa, na, fb, nb = int(test.failure.sum()), len(test), int(ref.failure.sum()), len(ref)
        rise = fa / na - fb / nb
        p_fail = fisher_one_sided(fa, na, fb, nb, 'greater')
        row = {'cell': cell, 'n': na, 'failures': fa, 'failure_rate': fa / na,
               'ref_failure_rate': fb / nb, 'rate_rise': rise, 'fisher_p': p_fail,
               'failure_criterion': bool(rise >= min_rate_rise and p_fail < alpha)}
        speed_x, speed_y = test.effective_speed.dropna(), ref.effective_speed.dropna()
        if use_speed and len(speed_x) and len(speed_y) and speed_y.median() > 0:
            drop = 1.0 - speed_x.median() / speed_y.median()
            p_speed = mannwhitney_less(speed_x, speed_y)
            row.update(median_speed=speed_x.median(), ref_median_speed=speed_y.median(),
                       speed_drop=drop, mannwhitney_p=p_speed,
                       speed_criterion=bool(drop >= min_speed_drop and p_speed < alpha))
        else:
            row.update(median_speed=math.nan, ref_median_speed=math.nan, speed_drop=math.nan,
                       mannwhitney_p=math.nan, speed_criterion=False)
        rows.append(row)
    table = pd.DataFrame(rows)
    go = bool(len(table) and (table.failure_criterion | table.speed_criterion).any())
    return {'go': go, 'table': table, 'ref_n': len(ref), 'use_speed': use_speed}


def main_study(trials: pd.DataFrame, comparisons=COMPARISONS, alpha: float = 0.05) -> pd.DataFrame:
    """One-sided Fisher (a fails less than b) per cell and comparison, Holm over the family."""
    valid = trials[trials.valid]
    rows = []
    for cell in sorted(valid.cell.unique()):
        in_cell = valid[valid.cell == cell]
        for a, b in comparisons:
            ta, tb = in_cell[in_cell.controller == a], in_cell[in_cell.controller == b]
            if ta.empty or tb.empty:
                continue
            fa, fb = int(ta.failure.sum()), int(tb.failure.sum())
            rows.append({'cell': cell, 'a': a, 'b': b, 'n_a': len(ta), 'n_b': len(tb),
                         'failure_rate_a': fa / len(ta), 'failure_rate_b': fb / len(tb),
                         'fisher_p': fisher_one_sided(fa, len(ta), fb, len(tb), 'less')})
    table = pd.DataFrame(rows)
    if len(table):
        table['holm_p'] = holm(table.fisher_p)
        table['reject_h0'] = table.holm_p < alpha
    return table


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

    stuck = [bool(m.flags & FLAG_STUCK) for _, m in msgs.get('/orion/safety_status', [])]
    rises = sum(1 for a, b in zip([False] + stuck, stuck) if b and not a)
    falls = sum(1 for a, b in zip(stuck, stuck[1:]) if a and not b)
    modes = [m.mode for _, m in msgs.get('/orion/policy_status', [])]
    row.update(stuck_events=rises, recovered_stuck_events=falls,
               recovery=bool(rises > 0 and not entrapped),
               escapes=sum(1 for a, b in zip([None] + modes, modes)
                           if b == MODE_ESCAPE and a != MODE_ESCAPE))

    samples = []
    risks = [(t, float(m.entrapment_risk)) for t, m in msgs.get('/orion/policy_status', [])
             if math.isfinite(m.entrapment_risk)]
    row['max_risk'] = max((r for _, r in risks), default=math.nan)
    for t, r in risks:
        if entrapped:
            samples.append((r, int(t1 - t <= RISK_HORIZON_S)))
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


def write_report(trials: pd.DataFrame, samples: dict, out_dir: str,
                 use_speed: bool = True) -> dict:
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
        ph = phase0_decision(trials, use_speed=use_speed)
        results['phase0'] = ph
        lines += ['## Phase-0 kill test (pre-registered)', '',
                  f'**Decision: {"GO" if ph["go"] else "NO-GO"}** (reference {REFERENCE_CELL}, '
                  f'n = {ph["ref_n"]}; GO needs a failure-rate rise >= 0.25 with one-sided '
                  'Fisher p < 0.05' + (', or a median effective-speed drop >= 30 % with '
                                       'one-sided Mann-Whitney p < 0.05' if use_speed else '')
                  + ', in at least one loose cell).', '', _md(ph['table'])]
        tests += [dict(test='phase0', **r) for r in ph['table'].to_dict('records')]
    ms = main_study(trials)
    if len(ms):
        results['main'] = ms
        lines += ['## Main study: failure rate, one-sided Fisher, Holm-corrected', '', _md(ms)]
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
    args = parser.parse_args(argv)
    protocol = None
    if args.protocol:
        from .protocol import load_protocol
        protocol = load_protocol(args.protocol)
    trials, samples = load_trials(args.trials_root, args.msg_dir, protocol)
    if trials.empty:
        print(f'no finished trials under {args.trials_root}', file=sys.stderr)
        return 1
    results = write_report(trials, samples, args.out, use_speed=not args.no_speed)
    if 'phase0' in results:
        print('Phase-0:', 'GO' if results['phase0']['go'] else 'NO-GO')
    print(f'report written to {os.path.join(args.out, "report.md")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
