#!/usr/bin/env bash
# Remote smoke test for the hosted Synap MCP server.
# Usage:  bash smoke_remote.sh <synap_API_KEY> [MCP_URL]
#   MCP_URL defaults to the prod endpoint.
set -u

KEY="${1:?usage: smoke_remote.sh <synap_API_KEY> [MCP_URL]}"
URL="${2:-https://synap-mcp.maximem.ai/mcp}"
PHRASE="SMOKE-$(date +%H%M%S)"           # unique per run so recall is unambiguous

pass=0; fail=0
ok(){ echo "  ✅ $1"; pass=$((pass+1)); }
no(){ echo "  ❌ $1"; fail=$((fail+1)); }

# Streamable-HTTP call: $1=bearer  $2=json-body  -> prints the SSE data line
mcp(){ curl -s -m 25 -X POST "$URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $1" -d "$2" 2>/dev/null | grep '^data:' | sed 's/^data: //'; }

field(){ python3 -c "import sys,json
try: print(json.load(sys.stdin)['result']['content'][0]['text'])
except Exception: print('')"; }

echo "Endpoint: $URL"
echo "== 1. initialize =="
mcp "$KEY" '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  | grep -q '"serverInfo"' && ok "handshake" || no "handshake"

echo "== 2. tools/list =="
R=$(mcp "$KEY" '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
for t in log_exchange recall_context list_recent_memories; do
  echo "$R" | grep -q "\"$t\"" && ok "tool: $t" || no "tool: $t"; done

echo "== 3. bad token rejected =="
mcp synap_bogus '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"log_exchange","arguments":{"user_message":"x"}}}' \
  | grep -qi "status 401" && ok "401 on bad key" || no "bad key not rejected"

echo "== 4. log_exchange (write) =="
R=$(mcp "$KEY" "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"log_exchange\",\"arguments\":{\"user_message\":\"Remember my smoke-test code is $PHRASE.\"}}}")
echo "$R" | field | grep -q "ingestion_id=" && ok "logged ($PHRASE)" || no "log_exchange"

echo "== 5. recall round-trip (waits for async extraction, ~2 min max) =="
found=0
for i in $(seq 1 20); do
  T=$(mcp "$KEY" '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"recall_context","arguments":{"query":"what is my smoke-test code"}}}' | field)
  if echo "$T" | grep -q "$PHRASE"; then echo "  recalled: $T"; found=1; break; fi
  echo "  attempt $i: not yet"; sleep 6
done
[ "$found" = 1 ] && ok "save→extract→recall" || no "round-trip timed out"

echo "== 6. list_recent_memories =="
mcp "$KEY" '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"list_recent_memories","arguments":{"max_results":5}}}' | field | sed 's/^/  /'

echo "================  $pass passed, $fail failed  ================"
