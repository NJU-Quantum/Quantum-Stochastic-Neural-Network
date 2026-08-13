#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KAIWU_WHEEL="${1:-/Users/hronrad/Downloads/kaiwu-1.3.1-cp310-none-any.whl}"
WUYUE_WHEEL="${2:-/Users/hronrad/Downloads/wuyue-1.0-py3-none-any.whl}"

test -f "$KAIWU_WHEEL"
test -f "$WUYUE_WHEEL"

python3.10 -m venv "$ROOT_DIR/hardware/.venv-kaiwu"
"$ROOT_DIR/hardware/.venv-kaiwu/bin/pip" install --upgrade pip
"$ROOT_DIR/hardware/.venv-kaiwu/bin/pip" install "$KAIWU_WHEEL"
"$ROOT_DIR/hardware/.venv-kaiwu/bin/pip" install \
  -r "$ROOT_DIR/hardware/kaiwu_qsnn_photonic/requirements-kaiwu.txt"
"$ROOT_DIR/hardware/.venv-kaiwu/bin/pip" check

python3.11 -m venv "$ROOT_DIR/hardware/.venv-cloud"
"$ROOT_DIR/hardware/.venv-cloud/bin/pip" install --upgrade pip
"$ROOT_DIR/hardware/.venv-cloud/bin/pip" install "$WUYUE_WHEEL"
"$ROOT_DIR/hardware/.venv-cloud/bin/pip" install \
  -r "$ROOT_DIR/hardware/requirements-cloud.txt"
"$ROOT_DIR/hardware/.venv-cloud/bin/pip" check

"$ROOT_DIR/hardware/.venv-kaiwu/bin/python" -c \
  "import inspect, kaiwu as kw; print('Kaiwu:', inspect.signature(kw.cim.CIMOptimizer))"
"$ROOT_DIR/hardware/.venv-cloud/bin/python" -c \
  "from importlib.metadata import version; import wuyue; print('WuYue:', version('wuyue'))"
