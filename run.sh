#!/usr/bin/env bash
# Auto-generated: re-enter the persistent GPU container
# default: login as invoking host UID/GID
# option : -root => login as root (admin-only dir + RED prompt)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"
HOST_USER="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_HOME="$(getent passwd "${HOST_UID}" | cut -d: -f6)"
[ -n "${HOST_HOME}" ] || { echo "[error] Host HOME could not be resolved"; exit 1; }
CONTAINER_NAME="${REPO_NAME}-${HOST_USER}"

AS_ROOT=0
if [[ "${1:-}" == "-root" ]]; then
  AS_ROOT=1
  shift
fi

# 停止中なら起動
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker start "${CONTAINER_NAME}" >/dev/null
  else
    echo "[error] Container '${CONTAINER_NAME}' が存在しません。再度 setup.sh を実行してください。"
    exit 1
  fi
fi

# /data が見えているか（早期検知）
if ! docker exec "${CONTAINER_NAME}" bash -lc 'test -d /data && ls /data >/dev/null 2>&1'; then
  cat <<'EODATA'
[error] /data is not accessible inside the container.
        /data は共有睡眠データセット置き場です（read-only 提供）。
        この状態では作業を続行しないでください。

[action] ここで停止します。堀江まで連絡してください。
EODATA
  exit 1
fi

# --- NVML / nvidia-smi health check (best-effort) ---
if ! docker exec "${CONTAINER_NAME}" bash -lc 'command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1'; then
  cat <<'EOWARN'
[warn] Container-side nvidia-smi failed.
       Host may still be OK, but inside the container NVML could be mismatched (often due to LD_LIBRARY_PATH/conda).
       This script will enter the container with /usr/local/nvidia/lib* prepended to LD_LIBRARY_PATH to reduce such issues.
EOWARN
fi

NV_LIB="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ADMIN_DIR="/root/_ADMIN_ONLY"

if [[ "${AS_ROOT}" -eq 1 ]]; then
  # root専用の「空作業ディレクトリ」を確保（事故防止）
  docker exec "${CONTAINER_NAME}" bash -lc "mkdir -p '${ADMIN_DIR}' && chmod 700 '${ADMIN_DIR}'"

  cat <<'EOWARN'
============================================================
[警告 / WARNING] You are entering the container as ROOT.
============================================================

[JP]
- このモードは apt など「管理作業」専用です。
- 実験・学習・データ生成は通常モード（./run.sh）で実施してください。
- root のまま /workspace 配下で作業すると、生成物が root 所有になり、
  通常ユーザーで閲覧・編集できなくなる可能性があります。
- 事故防止のため、rootモードでは /workspace ではなく「管理用ディレクトリ」に入ります。

[EN]
- This mode is intended ONLY for admin tasks such as apt installs.
- Run experiments in normal user mode (./run.sh), NOT in -root mode.
- If you work under /workspace as root, outputs may become root-owned and
  may be unreadable/unwritable in user mode.
- For safety, root mode starts in an admin-only directory, not /workspace.

============================================================
EOWARN

  echo "[info] Entering as root (-root). Working dir = ${ADMIN_DIR}"

  # 赤いrootプロンプト + 管理用ディレクトリ開始
  exec docker exec -it -w "${ADMIN_DIR}" \
    -e LD_LIBRARY_PATH="${NV_LIB}:${LD_LIBRARY_PATH:-}" \
    "${CONTAINER_NAME}" \
    bash -lc 'export PS1="\[\e[1;31m\](ROOT)\[\e[0m\] \u@\h:\w\$ "; exec bash'
fi

echo "[info] Entering as UID:GID = ${HOST_UID}:${HOST_GID} (default)."

# --- Ensure user/group entries exist inside container (cheap; creates only on first time) ---
# NOTE: This runs as root inside the container but becomes a no-op after the first setup.
docker exec \
  -e TARGET_UID="${HOST_UID}" \
  -e TARGET_GID="${HOST_GID}" \
  -e TARGET_USER="${HOST_USER}" \
  -e TARGET_HOME="${HOST_HOME}" \
  "${CONTAINER_NAME}" bash -c '
set -e
U="$TARGET_UID"; G="$TARGET_GID"; UN="$TARGET_USER"; HOME_DIR="$TARGET_HOME"

if ! getent group "$G" >/dev/null 2>&1; then
  groupadd -g "$G" "$UN" 2>/dev/null || groupadd -g "$G" "g$G"
fi

if ! getent passwd "$U" >/dev/null 2>&1; then
  useradd -m -d "$HOME_DIR" -u "$U" -g "$G" -s /bin/bash "$UN" 2>/dev/null \
    || useradd -m -d "$HOME_DIR" -u "$U" -g "$G" -s /bin/bash "u$U"
else
  CUR_NAME="$(getent passwd "$U" | cut -d: -f1)"
  CUR_HOME="$(getent passwd "$U" | cut -d: -f6)"
  if [ "$CUR_HOME" != "$HOME_DIR" ]; then
    usermod -d "$HOME_DIR" "$CUR_NAME"
  fi
fi

mkdir -p "$HOME_DIR"
chown "$U:$G" "$HOME_DIR"

# Only replace an existing symlink. Never delete a real SSH directory.
if [ -d /ssh-host ]; then
  if [ -L "$HOME_DIR/.ssh" ]; then
    if [ "$(readlink "$HOME_DIR/.ssh")" != /ssh-host ]; then
      echo "[warn] Existing SSH symlink points elsewhere; leaving it unchanged."
    fi
  elif [ -e "$HOME_DIR/.ssh" ]; then
    echo "[warn] Existing SSH directory/file at $HOME_DIR/.ssh; leaving it unchanged."
  else
    ln -s /ssh-host "$HOME_DIR/.ssh"
    chown -h "$U:$G" "$HOME_DIR/.ssh"
  fi
fi

# Workspace permissions are managed as in the original starter script.
chown -R "$U:$G" /workspace || true
'

# git safe.directory を「入るユーザー側」に入れる
docker exec --user "${HOST_UID}:${HOST_GID}" \
  -e HOME="${HOST_HOME}" \
  "${CONTAINER_NAME}" \
  bash -lc 'git config --global --add safe.directory /workspace >/dev/null 2>&1 || true'

# Optional note for host-ssh presence
if ! docker exec "${CONTAINER_NAME}" bash -lc 'test -d /ssh-host >/dev/null 2>&1'; then
  cat <<'EOSSHWARN'
[warn] /ssh-host is not available in this container.
       Host SSH sharing is enabled by default, but the mount may have failed.
       If you need GitHub SSH:
         - Recreate the container after ensuring host ~/.ssh exists, OR
         - Generate SSH keys inside the container.
EOSSHWARN
fi

# Enter as invoking UID/GID
exec docker exec -it -w /workspace \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME="${HOST_HOME}" \
  -e LD_LIBRARY_PATH="${NV_LIB}:${LD_LIBRARY_PATH:-}" \
  "${CONTAINER_NAME}" bash
