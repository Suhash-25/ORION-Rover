#!/usr/bin/env bash
# Python packages the ROS packages need that have no rosdep key (see each package's
# requirements.txt), installed into the system Python of the ros:jazzy container.
#
#   scripts/ci_pip_deps.sh            # runtime + test deps (CI)
#   ORION_PIP_NO_TORCH=1 scripts/ci_pip_deps.sh   # skip torch (orion_policy ONNX tests skip)
#
# numpy stays < 2: apt's python3-scipy / python3-pandas on Ubuntu 24.04 are built against
# numpy 1.26 and break if pip upgrades it. torch comes from the CPU-only index (~200 MB
# instead of ~2 GB of CUDA wheels).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP=(python3 -m pip install --break-system-packages --no-cache-dir)
CONSTRAINTS="$(mktemp)"
echo 'numpy<2' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

if ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends python3-pip
fi
if [[ -z "${ORION_PIP_NO_TORCH:-}" ]]; then
  "${PIP[@]}" --index-url https://download.pytorch.org/whl/cpu torch
fi
reqs=()
for f in "$ROOT"/robot_ws/src/*/requirements.txt; do
  reqs+=(-r "$f")
done
if [[ -n "${ORION_PIP_NO_TORCH:-}" ]]; then
  grep -hv '^torch' "$ROOT"/robot_ws/src/*/requirements.txt > "$CONSTRAINTS.req"
  reqs=(-r "$CONSTRAINTS.req")
fi
"${PIP[@]}" "${reqs[@]}"
"${PIP[@]}" -e "$ROOT/learning"
python3 - <<'PY'
import numpy, orion_learning, onnxruntime, rosbags  # noqa: F401
print('pip deps ok: numpy', numpy.__version__, '| onnxruntime', onnxruntime.__version__)
PY
