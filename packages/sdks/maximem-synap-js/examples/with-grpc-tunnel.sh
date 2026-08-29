#!/usr/bin/env bash
#
# Run a test script with the gRPC stream enabled, over an SSH tunnel.
#
# The gRPC port is plaintext and is deliberately NOT exposed publicly, so the
# right way to reach it is a tunnel rather than a firewall rule. This opens one,
# runs whatever you pass, and closes it again even if the script fails.
#
#   ./with-grpc-tunnel.sh node test-js-sdk.mjs
#   ./with-grpc-tunnel.sh npx tsx test-ts-sdk.ts
#
# Override the target box with SYNAP_BOX / SYNAP_ZONE.
set -uo pipefail

BOX="${SYNAP_BOX:-temp-synap-cloud}"
ZONE="${SYNAP_ZONE:-us-central1-c}"
PORT="${SYNAP_GRPC_LOCAL_PORT:-50051}"
REMOTE_PORT="${SYNAP_GRPC_REMOTE_PORT:-50051}"

if [[ -z "${SYNAP_API_KEY:-}" ]]; then
  echo "error: SYNAP_API_KEY is not set. Without it the live and gRPC groups skip." >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>   e.g. $0 node test-js-sdk.mjs" >&2
  exit 2
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "note: something already listens on 127.0.0.1:$PORT, reusing it"
  TUNNEL_PID=""
else
  echo "opening tunnel 127.0.0.1:$PORT -> $BOX:$REMOTE_PORT ..."
  gcloud compute ssh "$BOX" --zone "$ZONE" -- \
    -N -L "$PORT:127.0.0.1:$REMOTE_PORT" >/tmp/synap-grpc-tunnel.log 2>&1 &
  TUNNEL_PID=$!

  for _ in $(seq 1 30); do
    nc -z 127.0.0.1 "$PORT" 2>/dev/null && break
    sleep 1
  done
  if ! nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "error: tunnel did not come up. Last lines of /tmp/synap-grpc-tunnel.log:" >&2
    tail -5 /tmp/synap-grpc-tunnel.log >&2
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null
    exit 1
  fi
  echo "tunnel up"
fi

# Closed on any exit path, including a failing test or a Ctrl-C.
cleanup() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null
    echo "tunnel closed"
  fi
}
trap cleanup EXIT INT TERM

export SYNAP_GRPC_HOST=127.0.0.1
export SYNAP_GRPC_PORT="$PORT"
export SYNAP_GRPC_USE_TLS=0

"$@"
