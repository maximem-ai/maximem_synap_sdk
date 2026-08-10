# synap-deepagents — verification and conformance

Two scripts that answer questions the unit tests cannot. Neither runs in CI:
both need credentials, and one spends model tokens.

| Script | Answers | Needs |
|---|---|---|
| `live_verify.py` | How does `SynapBackend` behave against a real instance? | `SYNAP_API_KEY` |
| `conformance.py` | Is Synap actually better than the backends deepagents ships? | `SYNAP_API_KEY` + a model key |

Both print a Markdown block meant to be pasted into
`integrations/DEEPAGENTS_HARNESS_INTEGRATION_PLAN.md`. Use `--out FILE` to write
it instead.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e integrations/synap-deepagents
pip install deepagents langchain-anthropic          # or your model provider
```

## `live_verify.py` — Phase 1

```bash
export SYNAP_API_KEY=synap_...
export SYNAP_INSTANCE_ID=inst_...     # unless whoami resolves it
python integrations/synap-deepagents/bench/live_verify.py
```

It seeds five memories under a throwaway `user_id`, waits for ingestion, and
then measures:

- `sdk.fetch` p50/p95 and payload size across `mode` × `precision_level` × with
  and without a query — the two knobs that sit inside an agent turn.
- A `max_results` sweep, to check whether the default of 20 buys anything over 10.
- The backend itself: does `grep` find a memory worded differently from the
  query, is a write readable back in the same turn, and does a miss report
  `file_not_found` rather than a code that makes `MemoryMiddleware` raise.
- What `delete` does. `create` returns an `ingestion_id`, but `status` returns
  `memory_ids` once ingestion finishes. If that bridge is reliable, the
  backend's missing `delete` is worth revisiting.

Seeded memories are deleted at the end unless you pass `--keep`. `--scope NAME`
reuses a previous run's scope instead of seeding again.

## `conformance.py` — Phase 5

deepagents ships no memory benchmark, so there is no scoreboard to enter. This
is the minimum honest substitute: the same four graded fixtures, the same model,
run against three backends mounted the same way at `/memories/`.

```bash
# check the harness without spending anything
python integrations/synap-deepagents/bench/conformance.py --dry-run

# the real thing
export SYNAP_API_KEY=synap_... ANTHROPIC_API_KEY=...
python integrations/synap-deepagents/bench/conformance.py

# subsets
python integrations/synap-deepagents/bench/conformance.py --arms filesystem,store
python integrations/synap-deepagents/bench/conformance.py --fixtures large-scope
```

**Run `--dry-run` first.** It reads the memory file through each arm and checks
the corpus came back. A benchmark that silently serves an empty file on one arm
reports that arm as simply worse, which is the most expensive kind of wrong —
and it is exactly what happened the first time this harness ran.

### The fixtures

| Fixture | Who should win, and why |
|---|---|
| `cross-session-identity` | Nobody. The control — every arm should pass, which is how you know the harness works. |
| `supersession` | Synap. An old fact is contradicted by a newer one; a whole-file paste shows the model both and lets it guess. |
| `large-scope` | Synap. One relevant memory in ~200. The file-shaped arms paste **~25,000 characters** into every prompt to find it. |
| `small-repo-task` | `FilesystemBackend`. Three lines of memory; a local read beats a network round trip and we report that. |

Grading is substring presence, case-insensitive — deliberately dumb. An LLM
judge would add a second source of noise to a comparison whose entire point is
to isolate one variable.

**If Synap does not clearly win `large-scope` and `supersession`, that is a
finding about Synap and it gets written down.** A benchmark that only contains
fixtures you win is a brochure.
