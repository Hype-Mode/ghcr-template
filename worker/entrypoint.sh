#!/usr/bin/env bash
# Default container entrypoint. Vast.ai overrides this with an explicit
# `onstart`, but `docker run ghcr.io/.../comfyranch-comfyui` should also just work.
set -euo pipefail
exec python3 "${COMFRANCH_HOME:-/opt/comfyranch}/bootstrap/comfyranch_bootstrap.py" "$@"
