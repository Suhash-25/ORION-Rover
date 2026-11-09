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
