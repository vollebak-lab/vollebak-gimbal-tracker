#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_pi.sh" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"

apt-get update
apt-get install -y python3-venv python3-opencv python3-numpy python3-yaml python3-serial v4l-utils

sudo -u "${RUN_USER}" python3 -m venv --system-site-packages "${REPO_DIR}/.venv"
sudo -u "${RUN_USER}" "${REPO_DIR}/.venv/bin/python" -m pip install --upgrade pip
sudo -u "${RUN_USER}" "${REPO_DIR}/.venv/bin/python" -m pip install --no-deps -e "${REPO_DIR}"

usermod -aG dialout,video "${RUN_USER}"

if [[ ! -f "${REPO_DIR}/config/pi.yaml" ]]; then
  install -o "${RUN_USER}" -g "${RUN_USER}" -m 0644 \
    "${REPO_DIR}/config/pi.handoff.yaml" "${REPO_DIR}/config/pi.yaml"
fi

install -m 0644 "${REPO_DIR}/scripts/vollebak-gimbal.service" \
  /etc/systemd/system/vollebak-gimbal.service
systemctl daemon-reload
systemctl enable vollebak-gimbal.service

echo "Installed and enabled vollebak-gimbal.service."
echo "Log out/in so dialout and video group changes take effect, then reboot the Pi."
