# Journal

Times are UTC. "Sandbox" = the cloud machine Claude works in, with ROS 2 Jazzy, Gazebo
Harmonic (gz-sim 8), Nav2 and ros2_control installed from RoboStack (conda) because the
official ROS apt servers are blocked there. CI on GitHub uses the official apt packages.

## 2026-11-09

### 10:40 Request: plan all the software
- **Asked (suhas):** "Assume the hardware is built; I have no software to test and want to just push
  code to git. Plan what code has to be written and pushed."
- **Did:** Read the repo on the user's laptop (no changes). Wrote the plan doc
  ([ORION software build plan](https://claude.ai/artifact/MprgPQtt19TxtfGRTq1acN)).
- **Plan v1:** 9 PRs. 0 CI skeleton, 1 fix existing bugs, 2 msgs + safety, 3 real driver +
  firmware, 4 terrain + logging, 5 non-RL baseline, 6 Phase 0 kill test, 7 training stack,
  8 ORION policy. Key rule: GitHub Actions is the test machine; "pushed" = CI green.
  Second rule: no RL before the kill test.
- **Pushback given:** pushing code that has never run produces a repo that doesn't build;
  from PR 3 on someone must physically run the rover.

### 10:48 Request: start the project and complete the plan
- **Blocker found:** GitHub account `suhasintern2` has no write access to `Suhash-25/ORION-Rover`.
  Asked the user to add it as a collaborator. Work continues locally on stacked branches.
- **Change of plan:** the sandbox turned out to be Ubuntu 24.04 (ROS Jazzy's platform).
  ROS apt was blocked, Docker Hub rate-limited (HTTP 429), downloading pixi from GitHub was
  denied. Installed the full stack from RoboStack via the `py-rattler` Python package instead.
  Result: code is now **run before push**, not only checked by CI.
- **PR 0 (claude/pr0-ci):** merged `origin/sprint-8-debug` into main's history; added `ci.yml`.
  Built main unchanged: `colcon build` fails in orion_description (missing `rviz/` dir). Confirmed.
- **PR 1 (claude/pr1-green):** fixed install dirs, Nav2 stamped cmd_vel + remap, footprint,
  removed `use_stamped_vel`, effort state interface, inline sun/ground (no internet needed),
  headless/teleop launch args, lint on every package, URDF test, `orion_sim_tests`
  (drive + Nav2 goal), `sim-smoke.yml`.
  Evidence: `colcon test --packages-select orion_sim_tests` → 2/2 test files pass
  (drive: 4 tests in 16 s; Nav2 reaches the 1.5 m goal in 12.6 s).
  Negative control: with `enable_stamped_cmd_vel: false` the Nav2 test fails
  ("1.49 m from goal" after 150 s), so the test really catches the original bug.
- Mistakes found on the way: see MISTAKES.md M-001 to M-006.

### 11:05 Question: how is the meta-RL code set up?
- Answered in the thread: RMA-style teacher/student + recoverability critic, trained in a
  custom Bekker/Wong-Reece soil env (not Gazebo), one shared `obs_spec.py`, ONNX export,
  CI only proves the pipeline, real sand runs are the evidence.

### 11:08 Request: complete everything, best possible; Fable 5.1 plans, Opus 5.5 (high) codes, council critique at each step
- **Plan v2:** (1) Fable 5.1 council critiques the plan and writes exact specs (msgs, obs
  spec, safety thresholds, soil model, serial protocol, experiment stats, work packages);
  (2) five Opus 5.5 coding agents in parallel, one per package group, each with unit tests;
  (3) council code review; (4) full verification in the sandbox; (5) push as stacked PRs.
- **Done meanwhile:** IMU link + Gazebo IMU + `/imu/data` bridge (needed by safety and slip).
  Sim test checks z-accel ≈ 9.8 m/s² at rest. 6/6 drive tests pass.

### 11:10 Request: use the harness, log every interaction, plan, change, mistake, lesson
- **Decision:** keep the `hx` harness file conventions (feature_list.json, LESSONS.md,
  progress log) but make every verification a real command (`scripts/verify.sh …`). The old
  `echo`/`print` checks are removed because they reported false passes (see M-001).
- Created this log (JOURNAL, DECISIONS, MISTAKES) and backfilled everything above.

### 11:15 Council review returned (Fable 5.1) → plan v3
- 8 findings adopted (docs/SPEC.md §1). Biggest changes to the plan:
  - "Recoverability margin" renamed **entrapment risk** and only calibrated (not trained) on the
    rover, because nothing on the real rover can label "could have escaped" (F1).
  - Sim randomises model-form error and the gravity-soil coupling itself, otherwise training in
    an Earth-fitted soil model is circular (F2).
  - Kill test pre-registered: 4 cells × 20, Fisher exact + Mann-Whitney, explicit GO/NO-GO (F3).
  - Wheel odometry is never body velocity; sandbox needs external truth (overhead camera +
    AprilTag) or lidar odometry (F4). Open question for the user.
  - Current ≠ sinkage: effort residual vs. a calibrated flat-ground model (F5).
  - RMA kept; privileged inputs are traction descriptors; robust PPO baseline B2 mandatory (F6).
  - Replay harness so the estimator and policy are tested on data with known truth (F7).
  - Safety publishes limits; firmware owns the hard current limit and watchdog (F8).
- Froze the contract myself before parallel coding: orion_msgs (4 msgs, 4 srvs) and
  learning/orion_learning/obs_spec.py (20-float observation, slip, normalisation, layout hash;
  7 tests pass). ONNX artefact contract written for the exporter and the runtime.

### 11:20 Parallel coding started (Opus 5.5, high effort), one git worktree each
- WP1 safety + twist_mux + launches + INTERFACES/SAFETY/BRINGUP docs
- WP2 serial protocol + firmware core + orion_hardware ros2_control plugin + HARDWARE/PROTOCOL docs
- WP3 orion_terrain (Kalman body speed, slip, effort residual, calibration tool)
- WP4 learning/ (soil model, env, PPO/RMA/critic, B0/B2, ONNX export, synthetic bags)
- WP5 orion_policy (runtime, B0, gating, modes) + orion_experiments (runner, stats)
- Each agent logs to docs/engineering-log/agents/WPn.md. Next: council review of each branch,
  integration, full verification.

### 11:28 User approved the overhead camera (D-007)
- Added orion_ground_truth: AprilTag TF -> /orion/ground_truth_pose (PoseStamped, frame
  `sandbox`). 5 geometry tests pass, including pose recovery for two different camera
  placements; negative check: dropping the reference-tag inversion makes that test fail.
- Mistake caught (M-008): ament_python tests did not run at all ("NO TESTS RAN") because this
  setuptools ignores `tests_require`. Fixed with `extras_require={'test': ['pytest']}` and told
  the three agents writing Python packages.

### 11:57 Request (paper thread): full research paper as Markdown in the repo
- Committed docs/RESEARCH.md (about 16k words: agent briefing, full paper, differences from
  SPEC). It resolves a real inconsistency: the kill test size. Adopted a two-stage design (D-008)
  and updated SPEC.md §8 and RESEARCH.md Part 2B to match.

### 11:51 Coordinator FYI: Round 7 candidate (critical slowing down) not adopted
- Keep interfaces able to support it: wheel encoder and IMU logging at >= 100 Hz and a
  perturbation-injection hook in the motor command path. The detector itself is not built
  unless the user approves.

### 12:01 User approved 40 runs in cells C and D (two-stage kill test, D-008)
- SPEC §8, RESEARCH.md and the paper already match. Remaining: the phase-0 protocol YAML and
  analysis.py in orion_experiments get the two-stage design during integration.
