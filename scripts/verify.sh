#!/usr/bin/env bash
# Single entry point for every verification step. Exit code is the only truth:
# 0 = passed, anything else = failed. Used by CI, by hx (feature_list.json) and by humans.
#
#   scripts/verify.sh build              colcon build of robot_ws
#   scripts/verify.sh unit [pkg...]      colcon test (unit + lint), excluding Gazebo tests
#   scripts/verify.sh sim [-R regex]     headless Gazebo launch tests (orion_sim_tests)
#   scripts/verify.sh learning           pytest + smoke training for learning/
#   scripts/verify.sh all                everything above
#
# ORION_OFFLINE=1 skips ament_xmllint, which downloads the package.xml schema from
# download.ros.org. Only for machines without internet; CI never sets it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$ROOT/robot_ws"

source_ros() {
  set +u  # ROS setup scripts reference unset variables
  if [[ -n "${ORION_ROS_SETUP:-}" ]]; then source "$ORION_ROS_SETUP"
  elif [[ -f /opt/ros/jazzy/setup.bash ]]; then source /opt/ros/jazzy/setup.bash
  else echo "ROS 2 Jazzy not found (set ORION_ROS_SETUP)" >&2; exit 2; fi
  if [[ -f "$WS/install/setup.bash" ]]; then source "$WS/install/setup.bash"; fi
  set -u
}

build() { source_ros; (cd "$WS" && colcon build --event-handlers console_cohesion+); }

unit() {
  build
  local sel=(--packages-skip orion_sim_tests)
  [[ $# -gt 0 ]] && sel=(--packages-select "$@")
  local extra=()
  [[ "${ORION_OFFLINE:-0}" == 1 ]] && extra=(--ctest-args -E xmllint)
  (cd "$WS" && colcon test "${sel[@]}" --event-handlers console_cohesion+ "${extra[@]}" \
     && colcon test-result --verbose)
}

sim() {
  build
  (cd "$WS" && colcon test --packages-select orion_sim_tests --event-handlers console_direct+ \
     ${1:+--ctest-args "$@"} && colcon test-result --verbose)
}

learning() {
  (cd "$ROOT/learning" && python3 -m pytest -q && python3 -m orion_learning.train --config configs/smoke.yaml)
}

cmd="${1:-all}"; shift || true
case "$cmd" in
  build) build ;;
  unit) unit "$@" ;;
  sim) sim "$@" ;;
  learning) learning ;;
  all) unit; sim; [[ -d "$ROOT/learning" ]] && learning ;;
  *) echo "unknown step: $cmd" >&2; exit 2 ;;
esac
