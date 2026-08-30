#!/usr/bin/env bash
#
# Full test of the published Synap JS/TS SDK, from a clean npm install.
#
#   ./run-full-test.sh                      # read-only, safe against production
#   ./run-full-test.sh --write              # also exercises the write paths
#   ./run-full-test.sh --staging            # point at staging instead of prod
#   ./run-full-test.sh --local              # test the working tree, not npm
#
# Needs SYNAP_API_KEY. Nothing else: the SDK defaults to the prod HTTP and gRPC
# endpoints, and initialize() resolves client_id and instance_id from the key.
set -uo pipefail

WRITE=0 STAGING=0 LOCAL=0
for arg in "$@"; do
  case "$arg" in
    --write)   WRITE=1 ;;
    --staging) STAGING=1 ;;
    --local)   LOCAL=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "${SYNAP_API_KEY:-}" ]]; then
  echo "error: SYNAP_API_KEY is not set." >&2
  echo "  export SYNAP_API_KEY=synap_..." >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${SYNAP_TEST_DIR:-$HOME/.synap-sdk-test}"

# Staging needs an explicit base URL; prod is the SDK default. The test script
# derives the gRPC host from the base URL, so setting this one var moves BOTH
# transports and they cannot end up on different deployments.
if [[ "$STAGING" == "1" ]]; then
  export SYNAP_BASE_URL="https://synap-cloud-staging.maximem.ai"
  TARGET="staging"
else
  unset SYNAP_BASE_URL 2>/dev/null || true
  TARGET="PRODUCTION"
fi

MODE="read-only"
[[ "$WRITE" == "1" ]] && MODE="read+write"

echo "────────────────────────────────────────────────────────────────"
echo " Synap JS/TS SDK — full test"
echo "   target   : $TARGET"
echo "   mode     : $MODE"
echo "   source   : $([[ "$LOCAL" == "1" ]] && echo 'local working tree' || echo 'npm @next')"
echo "   workdir  : $WORK"
echo "────────────────────────────────────────────────────────────────"

if [[ "$TARGET" == "PRODUCTION" && "$WRITE" == "1" ]]; then
  echo
  echo "  WARNING: this will write real memories and spend real credits on"
  echo "  PRODUCTION. Writes land under a random throwaway user_id, and the"
  echo "  run deletes the memory it created, but ingestions are billed."
  read -r -p "  Type 'yes' to continue: " confirm
  [[ "$confirm" == "yes" ]] || { echo "  aborted"; exit 1; }
fi

# ── Clean install ──────────────────────────────────────────────────────────
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"

# Written directly rather than via `npm init -y`. The default workdir is
# ~/.synap-sdk-test, and npm refuses a package name starting with a dot, so
# `npm init -y` fails there. It failed silently, npm install then created a
# bare manifest with no "type", and tsx fell back to CJS and choked on
# top-level await. A literal manifest has no such failure mode.
cat > package.json <<'MANIFEST'
{
  "name": "synap-sdk-test-harness",
  "private": true,
  "version": "1.0.0",
  "type": "module"
}
MANIFEST

if [[ "$LOCAL" == "1" ]]; then
  ( cd "$HERE/.." && npm run build >/dev/null 2>&1 ) || { echo "local build failed"; exit 1; }
  npm install "$HERE/.." @grpc/grpc-js @grpc/proto-loader >/dev/null 2>&1
else
  npm install @maximem/synap-js-sdk@next @grpc/grpc-js @grpc/proto-loader >/dev/null 2>&1
fi
npm install -D tsx typescript @types/node >/dev/null 2>&1

INSTALLED=$(node -p "require('@maximem/synap-js-sdk/package.json').version")
echo "  installed @maximem/synap-js-sdk@$INSTALLED"
echo

cp "$HERE/test-js-sdk.mjs" "$HERE/test-ts-sdk.ts" "$HERE/tsconfig.json" .

# ── Run ────────────────────────────────────────────────────────────────────
# The same flag goes to BOTH scripts. Passing it only to the JS one meant a
# "read-only" run still created an ingestion from the TypeScript side.
ARGS=()
[[ "$WRITE" == "1" ]] || ARGS+=(--read-only)

echo "══ JavaScript ══════════════════════════════════════════════════"
node test-js-sdk.mjs ${ARGS[@]+"${ARGS[@]}"}
JS_RC=$?

echo
echo "══ TypeScript: compile-time ════════════════════════════════════"
npx tsc --noEmit && echo "  tsc: clean (every type assertion held)"
TSC_RC=$?

echo
echo "══ TypeScript: runtime ═════════════════════════════════════════"
npx tsx test-ts-sdk.ts ${ARGS[@]+"${ARGS[@]}"}
TS_RC=$?

echo
echo "────────────────────────────────────────────────────────────────"
echo " js=$JS_RC  tsc=$TSC_RC  ts=$TS_RC   (0 = pass)"
echo " workdir kept at $WORK for poking around"
echo "────────────────────────────────────────────────────────────────"
[[ $JS_RC -eq 0 && $TSC_RC -eq 0 && $TS_RC -eq 0 ]] || exit 1
