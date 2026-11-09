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
