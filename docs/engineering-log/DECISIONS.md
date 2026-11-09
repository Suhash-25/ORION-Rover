# Decisions

## D-001 GitHub Actions is the test machine (2026-11-09)
The user can't run ROS locally. Options: (a) push untested, (b) CI on every push.
Chose (b): `ci.yml` (build, unit tests, lint) and `sim-smoke.yml` (headless Gazebo) gate every
PR. Rejected (a): broken code accumulates silently (see M-002, M-003).

## D-002 No RL before the Phase 0 kill test (2026-11-09)
If a tuned non-RL controller doesn't degrade on loose sand / lighter loading, the research
premise is false and RL work is wasted. RL code may be written, but no RL claims are made and
no large training runs happen before the kill test result.

## D-003 Train RL in a custom soil simulator, not Gazebo (2026-11-09)
Gazebo uses rigid contact; it cannot produce sinkage or entrapment. Chose a vectorised
NumPy env with a Bekker/Wong-Reece wheel-soil model. Known bias: soil models are fitted to
Earth tests, the very thing the research questions. Sim is a prior; real sand is the evidence.

## D-004 Sandbox ROS from RoboStack; CI from official apt (2026-11-09)
The sandbox can't reach packages.ros.org, so it uses RoboStack conda packages. CI uses
`ros:jazzy-ros-base` + rosdep (what users and the rover will run). Risk: a dependency present in
RoboStack but missing from `package.xml` passes in the sandbox and fails in CI. Found once
already (navigation2/slam_toolbox were missing from orion_bringup's package.xml) and fixed.

## D-005 Keep the hx harness conventions, replace its checks (2026-11-09)
User asked to use the harness. Kept `feature_list.json` / `LESSONS.md` / progress log so
`hx` still works, but every verification calls `scripts/verify.sh` and statuses only become
`passing` through a real run.

## D-006 Lint: ament defaults minus Q000 (2026-11-09)
Existing launch files use double quotes. Ignoring the quote-style rule (Q000) avoids rewriting
every file for style; all other ament flake8 rules (imports, unused vars, tabs) are enforced.

## D-007 Sandbox ground truth from an overhead camera + AprilTags (2026-11-09, user approved)
Council finding F4: wheel odometry is the numerator of slip, not truth. Options: (a) overhead
camera + AprilTag (~5 mm, >= 20 Hz), (b) lidar odometry only (2-5 cm/s), (c) motion capture
(not available). User chose (a) at 11:28. Package orion_ground_truth: a reference tag fixed in
the sandbox plus a tag on the rover, both in the same image, so camera placement errors
cancel. Implausible detections (> 1 m/s jumps, > 0.6 rad tilt) are rejected. Lidar odometry
stays as the onboard fallback.

## D-008 Kill test is two-stage: 4 × 20 pilot, then 40 in cells C and D (2026-11-09)
The paper thread showed that 20 runs per cell has about 49% power for trap rates of 10% vs
40%, so a NO-GO after 20 runs would be weak evidence and could kill a true effect. The council
spec had 4 × 20. Options: (a) keep 4 × 20, (b) 40 in every cell (160 runs, too slow for Phase
0), (c) two-stage. Chose (c): stage 1 pilot 4 × 20 with early GO only at α = 0.01; stage 2
extends C and D to 40 each and tests at α = 0.04; NO-GO only after stage 2. Paper and SPEC now
agree. orion_experiments analysis must implement the two looks.
Approved by the user (suhas) at 12:01: "yes do it".

## D-009 Four stacked draft PRs, pushed from the user's PC (2026-11-09)
The user asked for a PR when each section is complete. Sections:
1. CI
2. Make the stack build and drive, plus the Gazebo tests
3. IMU, spec, messages, ground truth and the paper
4. The software stack

Each PR is based on the previous one, so each diff stays reviewable. The alternative, one
48-commit PR, would be unreviewable. Section 1 is red on its own by design: it adds the gate
before the fix. The cloud account (aidevops-1) has no access, so the push runs on the user's
PC with their own credentials (suhasintern), with the user's approval ("remote control to vs
code, and push from there").

## D-010 WP4 physics deviations ratified (council verdict 2, 2026-11-09)
- Dig-rate slip sinkage (sinkage grows with slipped rim travel) replaces SPEC's instantaneous
  z_target. SPEC's version buried the rover 84 % of the way in 1 s, which is unphysical.
  DIG_RATE is now a per-episode model-form parameter, LogU(0.03, 0.3) m/m (FIX-A).
- The secant-implicit integrator is kept, with an energy test (body work ≤ motor work).

## D-011 Training distribution ≠ evaluation distribution (council B1; FIX-A)
Training uses:
- a patch derived from the main soil draw, present with p = 0.5;
- mixture rejection on analytic peak DP/W, with the curriculum threshold going from 0.08 to 0;
- a restricted fraction of **0.4**, not the council's 0.7. With 0.7, B0 completed 69-78 % of
  episodes, which is outside the pre-registered 40-60 % band. With 0.4 it completes 46-52 %.

Evaluation keeps the SPEC ranges, where B0 succeeds about 9 % of the time. Effect: B0
belly-out on training episodes fell from 86 % to 43 %, and the critic label mean from 0.72 to
0.42.

## D-012 Phase-0 decision: failure rate only, no effect-size gate (council M3; user-approved)
FIX-C simulated the exact two-stage rule over 20 000 sessions:
- GO on failures or speed: false GO 0.062-0.063, which breaks the promised 0.05.
- The gated D-008 rule: power 0.75-0.81.
- Failure-only, ungated (now the default): power 0.89-0.91, false GO 0.017-0.029.

Speed is still reported with its CI. Switches: `--speed-in-decision`, `--effect-size-gate`.
The first decision card recommended "drop the gate" with speed included. It was withdrawn
after FIX-C's numbers came in, and replaced by a corrected card (M-011).
Approved by the user (suhas) at 13:28 on the corrected card: "No gate, failure rate only". It
is now pre-registered and must not change after the first stage-1 trial.

## D-013 A controller give-up ends the trial ENTRAPPED (FINAL, implementer decision)
orion_policy gives up after 3 failed escapes: IDLE, zero command. orion_safety's hard-stuck rule
needs a command > 0.1 m/s, so it never fired and the trial ran to TIMEOUT. Now
`PolicyStatus.escapes_exhausted` (new field) makes the trial runner end the trial ENTRAPPED,
dated for the critic labels at the onset of the stuck episode (M2 anchor), and the sim counts
it the same way (evaluate.py, critic branch labels). Failure rates are unchanged (TIMEOUT and
ENTRAPPED are both failures); ENTRAPPED counts, time-to-entrapment and calibration labels
change. Rejected: keep TIMEOUT (labels a stopped, trapped rover "safe" for the rest of the
trial); keep creeping forward so the hard-stuck rule fires (drives into the trap on purpose and
still needs 20 s more). Open for the user: whether critic-triggered give-ups (no FLAG_STUCK)
stay in the calibration set; the analysis records `entrapment_source` so both can be reported.
SPEC section 8.
