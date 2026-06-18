#!/usr/bin/env bash
#
# Sync the vendored gRPC proto from the canonical source of truth.
#
# synap-vercel-adk ships its own copy of synap_service.proto so the published
# npm package is self-contained. That copy MUST stay byte-identical to the
# canonical proto the server is generated from, or the gRPC client silently
# drifts (missing fields/events = blind to learning-loop telemetry).
#
# Run this whenever synap/proto/synap_service.proto changes. CI enforces parity
# (see src/__tests__/proto-parity.test.ts and the publish workflow guard).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"

CANONICAL="$REPO_ROOT/synap/proto/synap_service.proto"
VENDORED="$PKG_DIR/proto/synap_service.proto"

if [[ ! -f "$CANONICAL" ]]; then
  echo "error: canonical proto not found at $CANONICAL" >&2
  exit 1
fi

cp "$CANONICAL" "$VENDORED"
echo "synced: $VENDORED  ←  $CANONICAL"
