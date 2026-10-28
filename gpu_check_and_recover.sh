#!/usr/bin/env bash
# ------------------------------------------------------------
# GPU / NVML sanity check & quick recovery script (container-side)
# ------------------------------------------------------------
set -euo pipefail

echo "============================================================"
echo "[GPU CHECK] container-side NVML / CUDA sanity check"
echo "============================================================"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FAIL] nvidia-smi not found in this container."
  echo "       This container is not GPU-enabled."
  exit 1
fi

echo "[STEP 1] Running nvidia-smi ..."
if nvidia-smi -L >/dev/null 2>&1; then
  echo "[OK] GPU is visible. No action required."
  exit 0
fi

echo "[WARN] nvidia-smi failed in current environment."

echo
echo "[STEP 2] Retrying without LD_LIBRARY_PATH ..."
if env -u LD_LIBRARY_PATH nvidia-smi -L >/dev/null 2>&1; then
  echo "[RECOVERED] GPU is visible when LD_LIBRARY_PATH is unset."
  echo
  echo "Cause:"
  echo "  LD_LIBRARY_PATH / conda environment is masking NVML (libnvidia-ml)."
  echo
  echo "Recommended action:"
  echo "  - Re-enter via run.sh (it prepends /usr/local/nvidia/lib*)"
  echo "  - Or start a clean shell:"
  echo "      bash --noprofile --norc"
  exit 0
fi

echo "[FAIL] Still cannot access GPU."

echo
echo "[STEP 3] Inspecting NVML linkage (diagnostic info)"
echo "------------------------------------------------------------"
ldd "$(which nvidia-smi)" | grep -i nvidia || true
echo "------------------------------------------------------------"

cat <<'EOM'

[FINAL STATUS]
GPU is NOT accessible inside this container.

Next actions:
  1) Try a clean shell:
       bash --noprofile --norc
       nvidia-smi

  2) If it still fails, exit the container and restart it:
       exit
       docker restart <container_name>
       ./run.sh

  3) If the problem persists, contact the administrator.

EOM

exit 2
