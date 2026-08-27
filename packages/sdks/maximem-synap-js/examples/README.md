# Synap SDK method tests

Two runnable scripts that exercise the native Synap SDK. Both work with no
credentials at all; adding a key turns on the live methods.

| Script | Runs with | Covers |
|---|---|---|
| `test-js-sdk.mjs` | `node`, no build step | Every method at runtime. 74 checks, all 41 members of the Python surface. |
| `test-ts-sdk.ts` | `tsx` + `tsc` | The same client through TypeScript, **plus the shipped `.d.ts` itself**. |

## Setup

```bash
cd synap/sdk/js && npm run build   # the examples depend on dist/
cd examples && npm install
```

`package.json` points at the SDK with `"file:.."`, so you are testing the real
package as a consumer would, not the source tree.

## Run

```bash
# No credentials: local behaviour only. No network, no cost.
node test-js-sdk.mjs
npx tsc --noEmit && npx tsx test-ts-sdk.ts

# With a key: adds every live HTTP method.
export SYNAP_API_KEY=synap_...
export SYNAP_BASE_URL=https://your-synap-host
node test-js-sdk.mjs

# Add the gRPC anticipation stream.
export SYNAP_GRPC_HOST=your-synap-host
node test-js-sdk.mjs
```

Every run uses a **fresh random `user_id` and `customer_id`**, so it never
touches existing data and "did my write land" stays an honest question.

> The live mode **writes memories and spends credits**. It is called out in the
> banner before anything runs.

### Environment

| Variable | Default | Notes |
|---|---|---|
| `SYNAP_API_KEY` | none | Absent means local-only. Nothing else is required. |
| `SYNAP_BASE_URL` | the SDK default | |
| `SYNAP_GRPC_HOST` | none | Absent skips the stream. |
| `SYNAP_GRPC_PORT` | `443` | Use `50051` for a direct or tunnelled connection. |
| `SYNAP_GRPC_USE_TLS` | `1` | `0` for plaintext, which only makes sense over localhost. |
| `SYNAP_INGEST_WAIT` | `120` | Seconds to wait for ingestion. |

Flags: `--offline` forces local-only even with a key; `--no-grpc` skips the stream.

If gRPC is not exposed publicly (it usually is not, and it is plaintext on
50051), tunnel it rather than opening the port:

```bash
gcloud compute ssh <box> --zone <zone> -- -N -L 50051:127.0.0.1:50051
export SYNAP_GRPC_HOST=127.0.0.1 SYNAP_GRPC_PORT=50051 SYNAP_GRPC_USE_TLS=0
```

## What each script is actually for

**`test-js-sdk.mjs`** proves the methods work. It ends by reading
`CONTRACT/conformance/python_surface.json` and reporting how many of Python's 41
members it actually called, so a green run cannot quietly mean half the surface
was never touched.

```
RESULT  74 passed, 0 failed, 0 skipped   (74 checks)
Surface coverage vs Python 0.4.5: 41/41 members exercised
```

**`test-ts-sdk.ts`** is not a retyped copy. Roughly half of it never executes,
because its job is to fail `tsc`:

- return types are what the docs claim (`fetch` really returns `UnifiedContext`,
  `get_compacted` really returns `Json | null`)
- every namespace method returns a Promise, since a synchronous throw from a
  Promise-returning method bypasses `.catch()`
- both `snake_case` and `camelCase` option spellings compile
- required fields stay required, checked with `@ts-expect-error`
- `exactOptionalPropertyTypes` holds, so `{ baseUrl: undefined }` is correctly
  rejected
- errors narrow through the class hierarchy, and `TranscriptConflictError`
  really is a `ConflictError` at the type level
- `noUncheckedIndexedAccess` does not force a non-null assertion at every
  collection access
- the `/grpc` subpath types resolve

`tsconfig.json` here is deliberately stricter than the SDK's own, with
`skipLibCheck: false`. A published SDK can pass every runtime test and still be
unusable in a strict project, and that should fail here rather than in someone's
repo. This is how the scripts caught nine public types that were declared but
never exported.

## Two traps worth knowing

**A context item's `id` is not a `memory_id`.** It addresses the retrieval item,
not the stored memory, so `memories.get(item.id)` returns 404. Real memory ids
come from `memories.status()` / `wait_for_completion()` as `memory_ids`. Both
behaviours are asserted, so if that ever changes the script says so.

**BM25 needs real term overlap on a small corpus.** The per-query threshold is
`max(0.6, min(1.5, 0.3 * tokens))` and a one-item corpus has almost no IDF, so a
query sharing one stem out of three legitimately misses. That is the gate
working. If you adapt the cache examples, keep the overlap realistic or you will
chase a bug that is not there.

## Testing the gRPC group

The gRPC port is plaintext and is not exposed publicly, so reach it through a
tunnel rather than opening a firewall rule. `with-grpc-tunnel.sh` does the whole
thing and closes the tunnel again even if the run fails:

```bash
export SYNAP_API_KEY=synap_...
export SYNAP_BASE_URL=http://35.222.199.195:8000
./with-grpc-tunnel.sh node test-js-sdk.mjs
```

With that, the run is `75 passed, 0 failed, 0 skipped`. Point it elsewhere with
`SYNAP_BOX` and `SYNAP_ZONE`.

## Interpreting a skip

A skip is never a failure, but it should always name its cause.

- `set SYNAP_API_KEY to run` — local-only mode.
- `set SYNAP_GRPC_HOST to run` — use `with-grpc-tunnel.sh`.
- `blocked upstream: <reason>` — an earlier server-side step produced nothing, so
  there was no input for this check. The reason is printed verbatim. This is not
  an SDK gap.
- `server pushes a bundle` reporting `none in 30s` — the client-to-server
  direction is still proven by the sends above it. Anticipation only fires when
  the instance has MACA patterns configured **and** retrieval finds something,
  so an instance with no config produces no bundle no matter what the SDK does.

## When ingestion fails server-side

`memories.wait_for_completion` **fails the check** if the ingestion reaches
`status=failed`, and prints the server's `error_message`. That distinction
matters: the SDK call worked, the ingestion did not, and those are different
claims. An earlier version of this script reported `pass` for a terminal
`failed` status, which hid a broken server and resurfaced three checks later as
a confusing 404 from `user.get_profile` (no memories means no derived profile).

If you see that, read the server log rather than suspecting the SDK. On the
temp box the usual cause is stage-3 extraction exhausting its providers:

```
error=All models failed. Last error: Provider gemini not available
error=All models failed. Last error: Incorrect API key provided: sk-proj-...
```

Two of the three fallback providers there are dead (no `GEMINI_API_KEY`, and an
invalid `OPENAI_API_KEYS` value), so whenever the primary has a bad minute there
is no working fallback and ingestion fails hard. It succeeds most of the time,
which is exactly what makes it confusing.

`memories.get` / `update` / `delete` need a real `memory_id`, so a failed
ingestion blocks them. The script tries the multipart ingestion as a second
source before giving up, and says so either way.
