#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 BANKON — all rights reserved.
#
# Install WordPress.agent on a VPS as a systemd service under a venv.
# Idempotent: safe to re-run.
#
# Usage (as root):
#   sudo bash scripts/install.sh
#
# After install, edit /etc/wordpress-agent/wordpress-agent.env with real
# credentials, then: sudo systemctl restart wordpress-agent.service

set -euo pipefail

INSTALL_DIR="/opt/wordpress-agent"
ENV_DIR="/etc/wordpress-agent"
ENV_FILE="${ENV_DIR}/wordpress-agent.env"
SERVICE_USER="wpagent"
SERVICE_GROUP="wpagent"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

require_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        echo "ERROR: this script must be run as root" >&2
        exit 1
    fi
}

ensure_user() {
    if ! getent group "${SERVICE_GROUP}" >/dev/null; then
        groupadd --system "${SERVICE_GROUP}"
    fi
    if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
        useradd --system --gid "${SERVICE_GROUP}" \
            --home-dir "${INSTALL_DIR}" \
            --shell /sbin/nologin \
            "${SERVICE_USER}"
    fi
}

ensure_python() {
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.12+ first." >&2
        exit 1
    fi
}

ensure_directories() {
    install -d -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${INSTALL_DIR}"
    install -d -m 0750 -o root -g "${SERVICE_GROUP}" "${ENV_DIR}"
}

copy_source() {
    local src
    src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    rsync -a --delete \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude '.pytest_cache' \
        --exclude '.mypy_cache' \
        --exclude '.ruff_cache' \
        --exclude '.git' \
        --exclude 'tests' \
        --exclude '.env' \
        "${src}/" "${INSTALL_DIR}/"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}"
}

create_venv() {
    if [[ ! -d "${INSTALL_DIR}/.venv" ]]; then
        sudo -u "${SERVICE_USER}" "${PYTHON_BIN}" -m venv "${INSTALL_DIR}/.venv"
    fi
    sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/pip" install --upgrade pip
    sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/pip" install -e "${INSTALL_DIR}"
}

stage_env_file() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        cp "${INSTALL_DIR}/.env.example" "${ENV_FILE}"
        chmod 0640 "${ENV_FILE}"
        chown root:"${SERVICE_GROUP}" "${ENV_FILE}"
        echo
        echo "Staged ${ENV_FILE} from .env.example"
        echo "Edit it with real credentials before starting the service:"
        echo "  sudo \${EDITOR:-nano} ${ENV_FILE}"
    else
        echo "Env file already present at ${ENV_FILE} (left untouched)"
    fi
}

install_systemd_unit() {
    install -m 0644 \
        "${INSTALL_DIR}/deploy/systemd/wordpress-agent.service" \
        /etc/systemd/system/wordpress-agent.service
    systemctl daemon-reload
    systemctl enable wordpress-agent.service
    echo
    echo "Service installed. Start it with:"
    echo "  sudo systemctl start wordpress-agent.service"
    echo "  sudo systemctl status wordpress-agent.service"
    echo "  curl http://127.0.0.1:8765/healthz"
}

main() {
    require_root
    ensure_python
    ensure_user
    ensure_directories
    copy_source
    create_venv
    stage_env_file
    install_systemd_unit
    echo
    echo "WordPress.agent installation complete."
}

main "$@"
