# Mistakes and what we learnt

Format: what happened → root cause → lesson → guard now in place.

## M-001 Harness reported "passing" for broken features
- **What:** `feature_list.json` marked SLAM, Gazebo and URDF features passing. `hx verify` ran
  checks like `python3 -c "print('World files present')"`, which always exit 0.
- **Root cause:** verification commands that cannot fail.
- **Lesson:** a check is only worth something if it can go red. Prove it by breaking the code once.
- **Guard:** every feature now verifies through `scripts/verify.sh` (real build/tests). The Nav2
  test was run once against the old config and failed as it should.

## M-002 `main` did not build
- **What:** `colcon build` failed: orion_description installed `rviz/ worlds/ meshes/ config/`
  folders that don't exist.
- **Root cause:** copy-pasted CMake install block; no CI, so nobody saw it on a clean checkout.
- **Lesson:** a stale local `install/` hides broken builds (see also the old stale-CMake-cache lesson).
- **Guard:** `ci.yml` builds from a clean checkout on every push.

## M-003 Nav2 could never move the robot
- **What:** Nav2 published `geometry_msgs/Twist` on `/cmd_vel`; Jazzy's `diff_drive_controller`
  only accepts `TwistStamped` on `/diff_drive_controller/cmd_vel`.
- **Root cause:** topic and type contracts lived in nobody's head; `use_stamped_vel: false` was
  copied from Humble-era configs and is ignored in Jazzy.
- **Lesson:** freeze topic names and types in one document and test them end to end.
- **Guard:** `enable_stamped_cmd_vel: true` + remap; `test_nav.py` fails if the rover doesn't
  reach the goal; `docs/INTERFACES.md` (PR 2).

## M-004 Worlds needed the internet
- **What:** both worlds `<include>`d the Sun and Ground Plane from fuel.gazebosim.org. Offline,
  Gazebo had no ground and the robot spawn failed.
- **Lesson:** simulation inputs must be in the repo.
- **Guard:** sun and ground plane are inline SDF.

## M-005 Gazebo servers survived test shutdown
- **What:** the second sim test talked to the first test's still-running Gazebo
  ("Controller already loaded", goal rejected).
- **Root cause:** `gz sim` is a ruby wrapper; launch kills the wrapper, the forked server lives on.
- **Lesson:** isolate tests (own ROS domain + Gazebo partition) and clean up explicitly.
- **Guard:** per-test `ROS_DOMAIN_ID`/`GZ_PARTITION` in CMake; `kill_leftover_gazebo()` in
  the post-shutdown test.

## M-006 Nav2 goal sent before Nav2 was active
- **What:** first version of `test_nav.py` was rejected: "Action server is inactive".
- **Root cause:** the action server exists from `configure`, but only accepts goals once active.
- **Guard:** the test waits on `/lifecycle_manager_navigation/is_active` first.

## M-007 Sandbox-only: xmllint needs the internet
- **What:** `ament_xmllint` fails in the sandbox because it downloads the package.xml schema.
- **Lesson:** not a code bug; it passes in CI. Recorded so nobody "fixes" it by disabling xmllint.

## M-008 Python package tests silently not running
- **What:** colcon reported "NO TESTS RAN" for orion_ground_truth.
- **Root cause:** current setuptools ignores `tests_require`, so colcon used unittest, which
  finds no pytest-style tests.
- **Lesson:** a green run with zero tests is not a pass; always read the test count.
- **Guard:** `extras_require={'test': ['pytest']}` in every ament_python setup.py; CI prints
  `colcon test-result --verbose` with counts.

## M-009 Told the user the wrong GitHub account to give access to
- **What:** we asked the user to add "suhasintern2" as a collaborator. The account Claude
  actually pushes with in the cloud session was aidevops-1, so the access the user granted
  could not help.
- **Root cause:** we used the PR-attribution login as if it were the push identity, without
  checking.
- **Lesson:** check the identity before asking a human to change permissions: run
  `gh api user --jq .login`.
- **Cost:** about 10 minutes and two extra round trips with the user.

## M-010 Temporary test shortcut outlived its reason
- **What:** test_policy remapped /cmd_vel/policy straight to the controller "until twist_mux
  is wired". After the merge, twist_mux was in the bringup too, so two publishers fed the
  controller. The test still passed and hid it.
- **Lesson:** every TEMPORARY gets an entry in the integration todo with its removal
  condition, checked at each merge.

## M-011 Recommended a statistical rule before simulating it
- **What:** we put "drop the effect-size gate" to the user as the recommended option, while
  speed was still a deciding metric. The simulation then showed a false-GO rate of 6.2 %.
- **Root cause:** we reasoned that alpha splitting bounds the error, but that holds for one
  test per look, and the rule runs 4 tests at stage 1.
- **Lesson:** simulate the exact decision rule before recommending it. Never infer error rates
  from the alpha bookkeeping alone.
- **Caught by:** FIX-C's power simulation (`phase0_power`), before any trial was run.
