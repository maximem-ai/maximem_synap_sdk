# synap-harnessx: live verification

`live_verify.py` answers what the unit tests cannot: how `SynapExtension` behaves against a real Synap instance. It does not run in CI because it needs credentials. It needs no model key, because the agent runs on a scripted provider and what is under test is the extension and Synap.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e packages/integrations/synap-harnessx
```

## Run

```bash
export SYNAP_API_KEY=synap_...
export SYNAP_INSTANCE_ID=inst_...     # unless whoami resolves it
python packages/integrations/synap-harnessx/bench/live_verify.py
```

It seeds three memories under a throwaway `user_id`, waits for ingestion, and then checks:

| Check | What it proves |
|---|---|
| recall (3 queries) | A question worded unlike the memory finds it, and the block is in the system prompt the model is sent |
| recall latency | p50 and p95 of the one Synap call that sits inside a turn |
| scope | A different `user_id` recalls nothing of the seeded user's |
| `search_memory` | The tool works through the real engine |
| `store_memory` | The write is accepted, ingests, and is recalled on a later turn |
| conversation id | The default (the agent's session id) is a UUID |
| REST recording | A turn is recorded with no stream open |
| stream | A turn, a tool call, a tool result, the reply and `end_session` go out on `listen()` |

**Every check is paired with "nothing swallowed".** The extension never raises into the agent run, so a server that rejects every call looks the same as one that works. The script captures every error the extension logs, and every stream report that fails, and fails the run on any of them.

On a B2C instance (`user_context_isolation=equals_customer`) no `customer_id` is sent. On B2B a throwaway one is used, or pass `--customer ID`.

The exit code is non-zero if any check fails. Seeded and stored memories are deleted at the end unless you pass `--keep`. `--scope NAME` reuses a previous run's `user_id` instead of seeding. `--out FILE` writes the Markdown report to a file.

## `--dry-run`

```bash
python packages/integrations/synap-harnessx/bench/live_verify.py --dry-run
```

Runs the same script against an in-memory fake, so the script itself can be checked without credentials. It proves the script's logic, not Synap's behaviour.
