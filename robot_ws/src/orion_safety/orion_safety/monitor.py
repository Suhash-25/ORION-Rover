"""
ORION safety rules (docs/SPEC.md section 5) as a pure-Python state machine.

No ROS imports: the node in safety_node.py feeds it inputs with timestamps and publishes the
decision. Every method takes ``now`` in seconds on one monotonic clock (the node clock: sim
time in Gazebo, system time on the rover), so tests can drive it with a fake clock.

Units: speeds m/s and rad/s, angles deg in the config and rad in the inputs, times s,
wheel effort in the platform's effort unit (A on hardware, N.m in Gazebo; see SafetyConfig).
"""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import math

from orion_safety.geometry import predict_pose
from orion_safety.geometry import signed_distance_inside

# Mirrors orion_msgs/msg/SafetyStatus (test_monitor.py checks they stay equal).
STATE_NOMINAL = 0
STATE_LIMITED = 1
STATE_HALT = 2
STATE_ESTOP = 3
FLAG_TILT = 1
FLAG_OVERCURRENT = 2
FLAG_HEARTBEAT = 4
FLAG_GEOFENCE = 8
FLAG_STUCK = 16
FLAG_FIRMWARE_FAULT = 32
FLAG_HARD_STUCK = 64

STATE_NAMES = {STATE_NOMINAL: 'NOMINAL', STATE_LIMITED: 'LIMITED', STATE_HALT: 'HALT',
               STATE_ESTOP: 'ESTOP'}

# Mirrors orion_msgs/msg/RoverHealth fault bits.
FW_FAULT_NAMES = {1: 'WATCHDOG', 2: 'OVERCURRENT', 4: 'ENCODER', 8: 'UNDERVOLTAGE',
                  16: 'ESTOP_LATCHED'}

RESET_PHRASE = 'CLEAR ESTOP'


@dataclass
class SafetyConfig:
    """Thresholds of every rule. Field names are the ROS parameter names in config/safety.yaml."""

    # Limits when nothing is wrong (the policy's action range, SPEC section 3).
    max_linear_speed: float = 0.5           # m/s
    max_angular_speed: float = 1.0          # rad/s
    # Tilt from the IMU (max of |roll|, |pitch|).
    tilt_limited_deg: float = 18.0
    tilt_limited_linear: float = 0.2        # m/s
    tilt_limited_angular: float = 0.5       # rad/s
    tilt_halt_deg: float = 25.0
    tilt_halt_s: float = 0.2
    # Wheel current from /joint_states effort. Unit = effort unit of the platform:
    # A on the rover (I_cont / I_peak are GUESSES until the motor drivers are known).
    current_cont: float = 6.0
    current_cont_s: float = 2.0
    current_limited_release_s: float = 2.0  # all wheels <= current_cont this long ends LIMITED
    current_limited_linear: float = 0.3     # m/s
    current_peak: float = 12.0
    current_peak_s: float = 0.3
    current_cooldown_s: float = 5.0         # counted from the moment all wheels <= current_peak
    # Host heartbeat on the required inputs.
    joint_states_timeout_s: float = 0.3
    imu_timeout_s: float = 0.5
    startup_grace_s: float = 10.0           # a required input never seen trips only after this
    # Optional inputs are ignored (never trip) when older than these.
    terrain_timeout_s: float = 0.5
    pose_timeout_s: float = 0.5
    command_timeout_s: float = 0.5
    health_timeout_s: float = 1.0
    # Geofence: keep-in polygon (flat [x0, y0, x1, y1, ...], m, fence frame) eroded by margin.
    geofence_enabled: bool = True
    geofence_polygon: list = field(
        default_factory=lambda: [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    geofence_margin_m: float = 0.15
    geofence_horizon_s: float = 1.0
    geofence_limited_linear: float = 0.2    # m/s
    geofence_limited_angular: float = 0.5   # rad/s
    # Wheel odometry is only a fallback pose (no /orion/ground_truth_pose): its frame starts at
    # odom_start_pose [x, y, yaw(rad)] in the fence frame and it drifts, so the margin grows.
    odom_extra_margin_m: float = 0.10
    odom_start_pose: list = field(default_factory=lambda: [-0.75, 0.0, 0.0])
    # Stuck (flag only, arms ESCAPE) and hard-stuck (HALT, outcome ENTRAPPED).
    stuck_cmd_linear: float = 0.1           # m/s
    stuck_body_speed: float = 0.03          # m/s
    stuck_min_confidence: float = 0.3
    # A body-speed fix (TerrainState.time_since_fix, aged by the message's own age) younger
    # than this is trusted without the confidence term (council M7): a stuck rover spinning
    # its wheels drives the wheel/body disagreement, and so the confidence, to 0.
    stuck_fresh_fix_s: float = 0.5
    stuck_slip: float = 0.8
    stuck_s: float = 3.0
    hard_stuck_window_s: float = 20.0
    hard_stuck_progress_m: float = 0.1
    # Firmware faults (RoverHealth.fw_fault_flags) that HALT; others are only flagged.
    # Default ENCODER | UNDERVOLTAGE | ESTOP_LATCHED. WATCHDOG and OVERCURRENT are handled by
    # the firmware itself and are transient.
    firmware_halt_mask: int = 28

    def __post_init__(self):
        """Reject inconsistent thresholds early (a typo here must not disable a rule)."""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and (math.isnan(value) or value < 0.0):
                raise ValueError(f'{f.name} must be >= 0, got {value}')
        if not self.tilt_limited_deg < self.tilt_halt_deg:
            raise ValueError('tilt_limited_deg must be below tilt_halt_deg')
        if not self.current_cont < self.current_peak:
            raise ValueError('current_cont must be below current_peak')
        poly = self.geofence_polygon
        if len(poly) < 6 or len(poly) % 2:
            raise ValueError('geofence_polygon needs >= 3 (x, y) pairs as a flat list')
        if len(self.odom_start_pose) != 3:
            raise ValueError('odom_start_pose must be [x, y, yaw]')
        if not 0 <= self.firmware_halt_mask <= 255:
            raise ValueError('firmware_halt_mask must fit in uint8')

    @property
    def polygon(self):
        p = [float(v) for v in self.geofence_polygon]
        return list(zip(p[0::2], p[1::2]))


@dataclass
class SafetyDecision:
    """What orion_safety publishes on /orion/safety_status every tick."""

    state: int
    flags: int
    max_linear_speed: float    # m/s
    max_angular_speed: float   # rad/s
    max_current: float         # effort unit per wheel (A on hardware)
    reason: str

    @property
    def stops_robot(self):
        """Whether orion_safety must publish zero twists (HALT or ESTOP)."""
        return self.state >= STATE_HALT


class _Held:
    """Tracks whether a condition has been continuously true for at least `duration` s."""

    def __init__(self, duration):
        self.duration = duration
        self.since = None

    def update(self, now, condition):
        if not condition:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        # 1e-9 s tolerance: with float stamps k * 0.05 a hold of exactly `duration` can come
        # out as 2.9999999999999982 s and fire one tick late (found by the env golden test).
        return now - self.since >= self.duration - 1e-9


class _Input:
    """Latest value of one input topic and when it arrived."""

    def __init__(self):
        self.value = None
        self.stamp = None

    def update(self, now, value):
        self.value = value
        self.stamp = now

    def fresh(self, now, timeout):
        return self.stamp is not None and now - self.stamp <= timeout

    def get(self, now, timeout):
        """Value if it arrived within timeout, else None."""
        return self.value if self.fresh(now, timeout) else None


class _Verdict:
    """Accumulates the outcome of every rule: worst state, tightest limits, all reasons."""

    def __init__(self, cfg):
        self.state = STATE_NOMINAL
        self.flags = 0
        self.max_v = cfg.max_linear_speed
        self.max_w = cfg.max_angular_speed
        self.max_i = cfg.current_peak
        self._reasons = []

    def flag(self, flag, reason, state=STATE_NOMINAL):
        self.flags |= flag
        self.state = max(self.state, state)
        self._reasons.append((state, reason))

    def limit(self, flag, reason, v=math.inf, w=math.inf, i=math.inf):
        self.flag(flag, reason, STATE_LIMITED)
        self.max_v = min(self.max_v, v)
        self.max_w = min(self.max_w, w)
        self.max_i = min(self.max_i, i)

    def decision(self):
        stop = self.state >= STATE_HALT
        reasons = [r for _, r in sorted(self._reasons, key=lambda sr: -sr[0])]
        return SafetyDecision(
            state=self.state,
            flags=self.flags,
            max_linear_speed=0.0 if stop else self.max_v,
            max_angular_speed=0.0 if stop else self.max_w,
            max_current=0.0 if stop else self.max_i,
            reason='; '.join(reasons))


class SafetyMonitor:
    """
    Evaluates every SPEC section 5 rule at each step(now).

    Latching: ESTOP, tilt HALT and hard-stuck HALT stay until request_reset() succeeds.
    Current HALT clears itself after the cool-down; heartbeat and geofence HALT clear as soon
    as their condition is gone. Stuck only raises FLAG_STUCK.
    """

    def __init__(self, config=None):
        self.cfg = config or SafetyConfig()
        cfg = self.cfg
        self._t_start = None
        self._t_last_step = None
        # Inputs.
        self._joint = _Input()      # tuple of efforts
        self._imu = _Input()        # (roll, pitch) rad
        self._terrain = _Input()    # (speed, confidence, (slip_left, slip_right), t_since_fix)
        self._ext_pose = _Input()   # (x, y, yaw) fence frame
        self._odom_pose = _Input()  # (x, y, yaw) odom frame
        self._cmd = _Input()        # (v, w) as sent to the drive controller
        self._health = _Input()     # fw_fault_flags
        self._estop_input = False
        # Latches and timers.
        self._estop_latched = False
        self._tilt_latched = False
        self._tilt_peak_deg = 0.0
        self._tilt_halt = _Held(cfg.tilt_halt_s)
        self._over_cont = _Held(cfg.current_cont_s)
        self._under_cont = _Held(cfg.current_limited_release_s)
        self._current_limited = False
        self._over_peak = _Held(cfg.current_peak_s)
        self._current_halt = False
        self._cooldown_since = None
        self._stuck_speed = _Held(cfg.stuck_s)
        self._stuck_slip = _Held(cfg.stuck_s)
        self._hard_stuck = False
        self._episode_start = None
        self._progress_source = None
        self._progress = deque()    # (t, x, y) samples over the hard-stuck window
        self._odometer = 0.0        # m, integral of terrain body speed

    # ------------------------------------------------------------------ inputs

    def on_joint_states(self, now, efforts):
        """Wheel efforts (A on hardware, N.m in sim); NaN marks a wheel without a reading."""
        self._joint.update(now, tuple(float(e) for e in efforts))

    def on_imu(self, now, roll, pitch):
        """Body roll and pitch in rad."""
        self._imu.update(now, (roll, pitch))

    def on_terrain(self, now, body_speed, confidence, slip_side, time_since_fix=math.inf):
        """
        Terrain estimate: body speed (m/s), its confidence [0, 1], (left, right) slip.

        time_since_fix: s since the last non-IMU body-speed fix (EXTERNAL or LIDAR_ODOM) when
        the estimate was made; inf or NaN when there has been none.
        """
        self._terrain.update(now, (body_speed, confidence, tuple(slip_side),
                                   float(time_since_fix)))

    def on_external_pose(self, now, x, y, yaw):
        """Ground-truth pose in the fence frame (m, rad)."""
        self._ext_pose.update(now, (x, y, yaw))

    def on_odom_pose(self, now, x, y, yaw):
        """Wheel-odometry pose in the odom frame (m, rad); fallback for the geofence only."""
        self._odom_pose.update(now, (x, y, yaw))

    def on_command(self, now, v, w):
        """Twist actually sent to the drive controller (m/s, rad/s)."""
        self._cmd.update(now, (v, w))

    def on_health(self, now, fault_flags):
        """RoverHealth.fw_fault_flags."""
        self._health.update(now, int(fault_flags))

    def on_estop(self, now, engaged):
        """/orion/estop. True latches ESTOP; False only allows a later reset."""
        self._estop_input = bool(engaged)
        if engaged:
            self._estop_latched = True

    def request_reset(self, now, confirm):
        """Clear the latched ESTOP / tilt / hard-stuck states. Returns (success, message)."""
        if confirm != RESET_PHRASE:
            return False, f'refused: confirm must be exactly "{RESET_PHRASE}"'
        if self._estop_input:
            return False, ('refused: /orion/estop is still true; release the e-stop '
                           '(publish false) first, it also holds the twist_mux lock')
        if self._tilt_latched:
            rp = self._imu.get(now, self.cfg.imu_timeout_s)
            if rp is None:
                return False, 'refused: tilt latched and no fresh /imu/data to check it'
            tilt = math.degrees(max(abs(rp[0]), abs(rp[1])))
            if tilt > self.cfg.tilt_limited_deg:
                return False, (f'refused: still tilted {tilt:.1f} deg '
                               f'(> {self.cfg.tilt_limited_deg:.0f} deg)')
        cleared = [name for name, latched in (
            ('ESTOP', self._estop_latched), ('tilt HALT', self._tilt_latched),
            ('hard-stuck HALT', self._hard_stuck)) if latched]
        self._estop_latched = False
        self._tilt_latched = False
        self._tilt_peak_deg = 0.0
        self._hard_stuck = False
        self._episode_start = None
        self._progress.clear()
        self._stuck_speed.since = None
        self._stuck_slip.since = None
        if not cleared:
            return True, 'nothing was latched'
        return True, 'cleared ' + ', '.join(cleared)

    # ------------------------------------------------------------------ rules

    def step(self, now):
        """Evaluate every rule at time now (s) and return the SafetyDecision."""
        if self._t_start is None:
            self._t_start = now
        dt = 0.0 if self._t_last_step is None else max(0.0, now - self._t_last_step)
        self._t_last_step = now
        v = _Verdict(self.cfg)
        if self._estop_latched:
            v.flag(0, 'e-stop latched (/orion/estop); clear with /orion/estop_reset',
                   STATE_ESTOP)
        self._rule_firmware(now, v)
        self._rule_heartbeat(now, v)
        self._rule_tilt(now, v)
        self._rule_current(now, v)
        stuck = self._rule_stuck(now, v)
        self._rule_hard_stuck(now, dt, stuck, v)
        self._rule_geofence(now, v)
        return v.decision()

    def _rule_firmware(self, now, v):
        flags = self._health.get(now, self.cfg.health_timeout_s)
        if not flags:
            return
        names = '|'.join(n for bit, n in FW_FAULT_NAMES.items() if flags & bit) or hex(flags)
        halts = flags & self.cfg.firmware_halt_mask
        v.flag(FLAG_FIRMWARE_FAULT, f'firmware fault {names}',
               STATE_HALT if halts else STATE_NOMINAL)

    def _rule_heartbeat(self, now, v):
        cfg = self.cfg
        for name, inp, timeout in (('/joint_states', self._joint, cfg.joint_states_timeout_s),
                                   ('/imu/data', self._imu, cfg.imu_timeout_s)):
            if inp.stamp is None:
                if now - self._t_start > cfg.startup_grace_s:
                    v.flag(FLAG_HEARTBEAT,
                           f'no {name} since start ({cfg.startup_grace_s:.0f} s grace)',
                           STATE_HALT)
            elif now - inp.stamp > timeout:
                v.flag(FLAG_HEARTBEAT,
                       f'{name} stale {now - inp.stamp:.2f} s (> {timeout:.2f} s)', STATE_HALT)

    def _rule_tilt(self, now, v):
        cfg = self.cfg
        rp = self._imu.get(now, cfg.imu_timeout_s)
        tilt = None if rp is None else math.degrees(max(abs(rp[0]), abs(rp[1])))
        over = tilt is not None and tilt > cfg.tilt_halt_deg
        if self._tilt_halt.update(now, over):
            self._tilt_latched = True
        if self._tilt_latched:
            self._tilt_peak_deg = max(self._tilt_peak_deg, tilt or 0.0)
            v.flag(FLAG_TILT, f'tilt HALT latched (peak {self._tilt_peak_deg:.1f} deg > '
                   f'{cfg.tilt_halt_deg:.0f}); reset required', STATE_HALT)
        elif tilt is not None and tilt > cfg.tilt_limited_deg:
            v.limit(FLAG_TILT, f'tilt {tilt:.1f} deg > {cfg.tilt_limited_deg:.0f}',
                    v=cfg.tilt_limited_linear, w=cfg.tilt_limited_angular)

    def _rule_current(self, now, v):
        cfg = self.cfg
        efforts = self._joint.get(now, cfg.joint_states_timeout_s)
        known = [abs(e) for e in (efforts or ()) if not math.isnan(e)]
        peak = max(known) if known else None
        above_cont = peak is not None and peak > cfg.current_cont
        above_peak = peak is not None and peak > cfg.current_peak

        if self._over_peak.update(now, above_peak):
            self._current_halt = True
        if self._current_halt:
            if above_peak:
                self._cooldown_since = None
            elif self._cooldown_since is None:
                self._cooldown_since = now
            if self._cooldown_since is not None and \
                    now - self._cooldown_since >= cfg.current_cooldown_s:
                self._current_halt = False
                self._cooldown_since = None
        if self._current_halt:
            v.flag(FLAG_OVERCURRENT, f'wheel current > {cfg.current_peak:g} for '
                   f'{cfg.current_peak_s:g} s; {cfg.current_cooldown_s:g} s cool-down',
                   STATE_HALT)

        if self._over_cont.update(now, above_cont):
            self._current_limited = True
        if self._under_cont.update(now, not above_cont):
            self._current_limited = False
        if self._current_limited:
            v.limit(FLAG_OVERCURRENT, f'wheel current > {cfg.current_cont:g} for '
                    f'{cfg.current_cont_s:g} s', v=cfg.current_limited_linear,
                    i=cfg.current_cont)

    def _command(self, now):
        return self._cmd.get(now, self.cfg.command_timeout_s) or (0.0, 0.0)

    def _rule_stuck(self, now, v):
        """Raise FLAG_STUCK; return whether the rover is stuck now."""
        cfg = self.cfg
        terrain = self._terrain.get(now, cfg.terrain_timeout_s)
        if terrain is None:
            self._stuck_speed.since = self._stuck_slip.since = None
            return False
        body_speed, confidence, slip_side, t_since_fix = terrain
        cmd_v = self._command(now)[0]
        driving = abs(cmd_v) > cfg.stuck_cmd_linear
        fix_age = t_since_fix + (now - self._terrain.stamp)
        fresh_fix = fix_age < cfg.stuck_fresh_fix_s      # False for inf and NaN
        trusted = fresh_fix or confidence > cfg.stuck_min_confidence
        no_motion = driving and abs(body_speed) < cfg.stuck_body_speed and trusted
        # Wheels spinning in the driven direction (slip is negative when reversing).
        if driving:
            spin = max(math.copysign(1.0, cmd_v) * s for s in slip_side)
        else:
            spin = max(abs(s) for s in slip_side)
        by_speed = self._stuck_speed.update(now, no_motion)
        by_slip = self._stuck_slip.update(now, spin > cfg.stuck_slip)
        if by_speed:
            basis = 'fresh fix' if fresh_fix else f'confidence {confidence:.2f}'
            v.flag(FLAG_STUCK, f'stuck: cmd {cmd_v:.2f} m/s but body {body_speed:.3f} m/s '
                   f'({basis}) for {cfg.stuck_s:g} s')
        if by_slip:
            v.flag(FLAG_STUCK, f'stuck: side slip {spin:.2f} > {cfg.stuck_slip:g} '
                   f'for {cfg.stuck_s:g} s')
        return by_speed or by_slip

    def _rule_hard_stuck(self, now, dt, stuck, v):
        """
        HALT when a stuck episode has made < hard_stuck_progress_m net progress in the window.

        Progress comes from the ground-truth pose, else from integrating the terrain body
        speed. Wheel odometry is never used: under slip it reports motion that did not happen.
        """
        cfg = self.cfg
        sample = self._progress_sample(now, dt)
        if sample is None:
            self._episode_start = None
        else:
            self._progress.append(sample)
            window_start = now - cfg.hard_stuck_window_s
            while len(self._progress) > 1 and self._progress[1][0] <= window_start:
                self._progress.popleft()
            t0, x0, y0 = self._progress[0]
            progress = math.hypot(sample[1] - x0, sample[2] - y0)
            if progress >= cfg.hard_stuck_progress_m:
                self._episode_start = None      # it is moving: any stuck episode is over
            if stuck and self._episode_start is None:
                self._episode_start = now
            if (stuck and now - self._episode_start >= cfg.hard_stuck_window_s
                    and t0 <= window_start and progress < cfg.hard_stuck_progress_m):
                self._hard_stuck = True
        if self._hard_stuck:
            v.flag(FLAG_HARD_STUCK, f'ENTRAPPED: < {cfg.hard_stuck_progress_m:g} m net '
                   f'progress in {cfg.hard_stuck_window_s:g} s while stuck; reset required',
                   STATE_HALT)

    def _progress_sample(self, now, dt):
        pose = self._ext_pose.get(now, self.cfg.pose_timeout_s)
        terrain = self._terrain.get(now, self.cfg.terrain_timeout_s)
        source = 'external' if pose is not None else (
            'odometer' if terrain is not None else None)
        if source != self._progress_source:
            self._progress_source = source
            self._progress.clear()
            self._odometer = 0.0
            dt = 0.0
        if source == 'external':
            return now, pose[0], pose[1]
        if source == 'odometer':
            self._odometer += terrain[0] * dt
            return now, self._odometer, 0.0
        return None

    def _rule_geofence(self, now, v):
        cfg = self.cfg
        if not cfg.geofence_enabled:
            return
        margin = cfg.geofence_margin_m
        pose = self._ext_pose.get(now, cfg.pose_timeout_s)
        source = 'pose'
        if pose is None:
            odom = self._odom_pose.get(now, cfg.pose_timeout_s)
            if odom is None:
                return          # optional input missing: geofence inactive, never trips
            pose = _odom_to_fence(odom, cfg.odom_start_pose)
            margin += cfg.odom_extra_margin_m
            source = 'odom'
        x, y, yaw = pose
        polygon = cfg.polygon
        depth = signed_distance_inside(x, y, polygon)
        if depth < margin:
            v.flag(FLAG_GEOFENCE, f'outside geofence ({source}: {margin - depth:.2f} m past '
                   f'the {margin:.2f} m margin)', STATE_HALT)
            return
        cmd_v, cmd_w = self._command(now)
        px, py, _ = predict_pose(x, y, yaw, cmd_v, cmd_w, cfg.geofence_horizon_s)
        if signed_distance_inside(px, py, polygon) < margin:
            v.limit(FLAG_GEOFENCE, f'geofence ahead: predicted pose in '
                    f'{cfg.geofence_horizon_s:g} s is outside ({source})',
                    v=cfg.geofence_limited_linear, w=cfg.geofence_limited_angular)


def _odom_to_fence(odom_pose, start_pose):
    """Express an odom-frame pose in the fence frame, given where odom started (x, y, yaw)."""
    x, y, yaw = odom_pose
    x0, y0, yaw0 = (float(s) for s in start_pose)
    c, s = math.cos(yaw0), math.sin(yaw0)
    return x0 + c * x - s * y, y0 + s * x + c * y, yaw0 + yaw
