#!/usr/bin/env bash
# KDE Sleep Lab — Research Starter Setup Script (UPDATED)
#
# 方針:
# - image は共有 (research-starter-kit:latest)
# - container は個別（repo名 + username で一意）
# - デフォルトは「実行者UIDでコンテナに入る」(GPU使用者の可視化目的)
# - apt等が必要なときだけ ./run.sh -root で root で入る（同一永続コンテナを更新）
# - rootモードは事故防止のため「別ディレクトリ開始」＋「赤いプロンプト」
#
# 追加要件:
# - ホスト /data をコンテナ /data に read-only bind mount
# - /data が無い/読めない場合は停止し「堀江まで連絡」表示
# - run.sh を生成（再入用・safe.directory 登録、/dataチェック、NVML対策）
# - gpu_check_and_recover.sh を生成（コンテナ内のGPU切り分け）
#
# 追加（SSH共有: 常時ON）:
# - ホスト ~/.ssh をコンテナ /ssh-host に read-only で常時 bind mount
# - run.sh 入室時に <host HOME>/.ssh -> /ssh-host を symlink（鍵はコピーしない）
#
# 重要（今回のバグ修正）:
# - コンテナ内ユーザー（UID=ホストUID）の passwd 上の HOME をホスト HOME と同じパスに設定
#   （ssh の ~ 展開や IdentityFile の絶対パスをホスト側と揃える）
#
# Usage: bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"
IMAGE_NAME="research-starter-kit:latest"

HOST_USER="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_HOME="$(getent passwd "${HOST_UID}" | cut -d: -f6)"
[ -n "${HOST_HOME}" ] || { echo "[error] Host HOME could not be resolved"; exit 1; }

CONTAINER_NAME="${REPO_NAME}-${HOST_USER}"

HOST_DATA_DIR="/data"
CONTAINER_DATA_DIR="/data"

HOST_SSH_DIR="${HOST_HOME}/.ssh"
CONTAINER_SSH_HOST_DIR="/ssh-host"

echo "[info] Repository dir : ${SCRIPT_DIR}"
echo "[info] Image          : ${IMAGE_NAME}"
echo "[info] Host user      : ${HOST_USER}"
echo "[info] Host UID:GID   : ${HOST_UID}:${HOST_GID}"
echo "[info] Host HOME      : ${HOST_HOME}"
echo "[info] Container name : ${CONTAINER_NAME}"
echo "[info] Host /data     : ${HOST_DATA_DIR} (will be mounted read-only to ${CONTAINER_DATA_DIR})"
echo "[info] Host ~/.ssh    : ${HOST_SSH_DIR} (will be mounted read-only to ${CONTAINER_SSH_HOST_DIR})"

# ------------------------------------------------------------
# 0) /data 事前チェック（ホスト側）
# ------------------------------------------------------------
echo "[precheck] Checking host '${HOST_DATA_DIR}' ..."
if [ ! -d "${HOST_DATA_DIR}" ]; then
  cat <<'EOM'
[error] Host '/data' directory is missing.
        /data は共有睡眠データセット置き場です。
        この状態ではコンテナを作成できません。

[action] 管理者対応が必要です。堀江まで連絡してください。
EOM
  exit 1
fi

if ! ls "${HOST_DATA_DIR}" >/dev/null 2>&1; then
  cat <<'EOM'
[error] Host '/data' exists but is NOT readable for your current user.
        /data は読み取り専用で提供されますが、少なくとも read/execute 権限が必要です。

[action] 管理者対応が必要です。堀江まで連絡してください。
EOM
  exit 1
fi
echo "[ok] Host /data precheck passed."

# ------------------------------------------------------------
# 0.5) ~/.ssh 事前チェック（ホスト側）
#     ※常時ON方針だが、無い場合は致命ではないので warn に留める
# ------------------------------------------------------------
echo "[precheck] Checking host '${HOST_SSH_DIR}' ..."
if [ ! -d "${HOST_SSH_DIR}" ]; then
  cat <<EOM
[warn] Host '${HOST_SSH_DIR}' directory is missing.
       Host SSH sharing is enabled by default, but you do not have ~/.ssh.
       GitHub SSH will not work via host-ssh mount unless you create keys on the host
       or generate keys inside the container.

       (You can still continue.)
EOM
else
  if ! ls "${HOST_SSH_DIR}" >/dev/null 2>&1; then
    cat <<EOM
[warn] Host '${HOST_SSH_DIR}' exists but is NOT readable for your current user.
       GitHub SSH via host-ssh mount may fail.

       (You can still continue.)
EOM
  else
    echo "[ok] Host ~/.ssh precheck passed."
  fi
fi

# ------------------------------------------------------------
# 1) 共有イメージの存在確認
# ------------------------------------------------------------
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "[error] Docker image '${IMAGE_NAME}' が見つかりません。"
  echo "        事前に docker load もしくは docker pull を実行してください。"
  exit 1
fi

# ------------------------------------------------------------
# 2) コンテナ作成または起動（GPU必須／永続／rootでsleep）
#    ※ /data を read-only で bind mount
#    ※ ~/.ssh を read-only で bind mount（常時ON）
# ------------------------------------------------------------
if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[1/3] Creating persistent GPU container '${CONTAINER_NAME}' ..."
  docker run -d \
    --init \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    -v "${SCRIPT_DIR}":/workspace \
    -v "${HOST_DATA_DIR}":"${CONTAINER_DATA_DIR}":ro \
    -v "${HOST_SSH_DIR}":"${CONTAINER_SSH_HOST_DIR}":ro \
    -w /workspace \
    "${IMAGE_NAME}" \
    sleep infinity
  echo "[ok] Container created (GPU, root, /data mounted read-only, host ~/.ssh mounted read-only)."
else
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[1/3] Starting existing container '${CONTAINER_NAME}' ..."
    docker start "${CONTAINER_NAME}" >/dev/null
  else
    echo "[1/3] Container '${CONTAINER_NAME}' already running."
  fi
fi

# ------------------------------------------------------------
# 2.5) /data コンテナ内チェック（mount されていて読めるか）
# ------------------------------------------------------------
echo "[check] Verifying /data inside container ..."
if ! docker exec "${CONTAINER_NAME}" bash -lc 'test -d /data && ls /data >/dev/null 2>&1'; then
  cat <<'EOM'
[error] '/data' is not accessible inside the container.
        (missing mount OR permission issue)

[action] ここで停止します。堀江まで連絡してください。
        連絡の際は以下も添付すると切り分けが速いです:
          - docker inspect <container_name> | grep -A3 Mounts
          - host: ls -ld /data ; id
EOM
  exit 1
fi
echo "[ok] Container /data check passed."

# ------------------------------------------------------------
# 2.6) /ssh-host コンテナ内チェック（任意：無ければ warn）
# ------------------------------------------------------------
echo "[check] Verifying host-ssh mount inside container (best-effort) ..."
if ! docker exec "${CONTAINER_NAME}" bash -lc 'test -d /ssh-host >/dev/null 2>&1'; then
  cat <<'EOM'
[warn] '/ssh-host' is not present inside the container.
       Host SSH sharing is enabled by default, but the mount may have failed
       (or host ~/.ssh did not exist at container creation time).

       If you need host SSH keys:
         - Ensure host ~/.ssh exists, then recreate the container, OR
         - Generate SSH keys inside the container.
EOM
else
  echo "[ok] Container host-ssh mount exists."
fi

# ------------------------------------------------------------
# 3) run.sh 生成
#    - default: exec as HOST_UID:HOST_GID (workdir=/workspace)
#    - -root:   exec as root (workdir=/root/_ADMIN_ONLY + RED prompt)
#    - /data check
#    - git safe.directory (for the entering user)
#    - NVML対策 (LD_LIBRARY_PATH prepend)
#    - コンテナ内に UID/GID の user/group を「初回のみ」作成（以降はno-op）
#    - passwd上のHOMEをホスト側のHOMEに統一
#    - HOME 環境変数もホスト側のHOMEに統一
#    - ~/.ssh は /ssh-host への symlink（鍵はコピーしない）
# ------------------------------------------------------------
echo "[2/3] Creating run.sh helper ..."
cat > "${SCRIPT_DIR}/run.sh" <<'EOF'
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
EOF
chmod +x "${SCRIPT_DIR}/run.sh"
echo "[ok] run.sh created. Use './run.sh' (user) or './run.sh -root' (root/admin)."

# ------------------------------------------------------------
# 3.5) gpu_check_and_recover.sh 生成（コンテナ内でのGPU切り分け・復旧用）
# ------------------------------------------------------------
echo "[info] Creating gpu_check_and_recover.sh helper ..."
cat > "${SCRIPT_DIR}/gpu_check_and_recover.sh" <<'EOF'
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
EOF
chmod +x "${SCRIPT_DIR}/gpu_check_and_recover.sh"
echo "[ok] gpu_check_and_recover.sh created. (Run inside container: bash gpu_check_and_recover.sh)"

# ------------------------------------------------------------
# 4) （必要なら）GitHub Actions workflow を保護（事故防止）
# ------------------------------------------------------------
if [ -d "${SCRIPT_DIR}/.github/workflows" ]; then
  count=0
  while IFS= read -r -d '' f; do
    chmod -w "$f" 2>/dev/null || true
    git update-index --skip-worktree "$f" 2>/dev/null || true
    count=$((count+1))
  done < <(find "${SCRIPT_DIR}/.github/workflows" -type f \( -name '*.yml' -o -name '*.yaml' \) -print0 2>/dev/null)
  echo "[ok] Applied read-only & skip-worktree to ${count} workflow YAML(s)."
fi

echo "==============================================================="
echo " Setup complete."
echo " - Persistent GPU container : ${CONTAINER_NAME}"
echo " - Enter container (user)   : ./run.sh"
echo " - Enter container (root)   : ./run.sh -root"
echo " - Workspace mount          : ${SCRIPT_DIR} -> /workspace"
echo " - Shared datasets          : ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR} (read-only)"
echo " - Host SSH mount           : ${HOST_SSH_DIR} -> ${CONTAINER_SSH_HOST_DIR} (read-only)"
echo "   (Container user HOME matches host HOME: ${HOST_HOME})"
echo "==============================================================="

# ------------------------------------------------------------
# 5) setup.sh 自動削除は無効（運用で判断）
# ------------------------------------------------------------
# rm -- "$0"
