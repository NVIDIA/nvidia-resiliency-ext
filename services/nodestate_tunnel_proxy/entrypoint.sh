#!/usr/bin/env bash
set -euo pipefail

AUTHORIZED_KEYS_SOURCE="${AUTHORIZED_KEYS_SOURCE:-/etc/nvrx-tunnel/authorized_keys}"
AUTHORIZED_KEYS_TARGET="/home/tunnel/.ssh/authorized_keys"

if [[ ! -s "${AUTHORIZED_KEYS_SOURCE}" ]]; then
    echo "missing authorized keys at ${AUTHORIZED_KEYS_SOURCE}" >&2
    exit 1
fi

cp "${AUTHORIZED_KEYS_SOURCE}" "${AUTHORIZED_KEYS_TARGET}"
chown tunnel:tunnel "${AUTHORIZED_KEYS_TARGET}"
chmod 600 "${AUTHORIZED_KEYS_TARGET}"

ssh-keygen -A >/dev/null

nginx -g "daemon off;" &
nginx_pid="$!"

trap 'kill "${nginx_pid}" 2>/dev/null || true' EXIT
exec /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config
