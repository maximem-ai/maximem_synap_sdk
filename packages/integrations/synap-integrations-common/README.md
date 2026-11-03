# synap-integrations-common

Shared utilities used across Synap framework integrations (LangChain, LlamaIndex,
CrewAI, Haystack, AutoGen, Google ADK, OpenAI Agents, Pydantic AI, Semantic
Kernel).

This package exists so that every integration converges on one implementation
of:

- `run_async` — the sync-to-async bridge required because most agent
  frameworks expose **sync** protocol methods while the Synap SDK is async.
- `SynapIntegrationError` + `wrap_sdk_errors` — uniform error handling so
  integrations don't silently swallow SDK failures.
- `default_scope` — consistent scope-path construction
  (`/<customer_id>/<user_id>` vs `/<user_id>`).
- `resolve_address` / `scope_string` — turn a developer's identity (ids, an
  explicit `scope_path`, or level `values`) into the arguments SDK calls take.
- `synap_integrations_common.testing` — shared pytest fixtures and response
  factories, previously duplicated across each integration's `_helpers.py`.

Not intended for direct end-user use. Pin as a runtime dependency of each
integration package.

## Scope ladder

An account may have levels between `customer` and `user` (for example `team`).
`resolve_address` maps what the developer gave to the SDK's arguments. Priority:
an explicit `scope_path` (used as given, no network call), then `values`
(outermost level first, resolved through `sdk.get_scope_ladder()`), then plain
ids. `values` together with `scope_path` or ids is an error, and `values` on an
account whose ladder cannot be read raises instead of guessing.

```python
from synap_integrations_common import resolve_address, record_turn, scope_string

addr = await resolve_address(sdk, values=["acme", "payments", "dana"])
ctx = await sdk.fetch(search_query=["..."], **addr.read_kwargs())      # scope_path=
await sdk.memories.create("note", **addr.write_kwargs())               # scope=
await record_turn(sdk, role="user", content="hi", conversation_id="c1",
                  user_id=addr.user_id, customer_id=addr.customer_id or "",
                  scope_path=addr.scope_path)
label = scope_string(["acme", "payments", "dana"])                     # "/acme/payments/dana"
```

`report_turn` / `record_turn` forward `scope_path` only when it is set. If it
is set and the installed `maximem-synap` cannot carry it, the turn is not sent
(it would be filed at the wrong, wider scope) and one error is logged. Tool,
reasoning and end-session events carry no path: the SDK methods take none and
the server stores the conversation's path from the user/assistant turn.

Needs `maximem-synap` with the scope ladder for `values=`; on an older SDK the
package still imports and ids / explicit paths still resolve.

### Per-call ids with a fixed path

The server lets the path win over `user_id` / `customer_id`. A package that
fixes a path at construction and also takes ids per call must refuse a mismatch,
or one caller reads another's memory:

```python
from synap_integrations_common import ensure_ids_match_path

ensure_ids_match_path(self._scope_path, user_id=user_id, customer_id=customer_id,
                      where="MyStore.search")   # raises SynapIntegrationError on mismatch
```

An empty id is fine (the path supplies it). Errors show short fingerprints of the
ids, not the ids.

### Testing

```python
from synap_integrations_common import resolve_address
from synap_integrations_common.testing import RecordingSDK, make_ladder, recording_sdk


async def my_integration_search(sdk, addr):  # stand-in for the code under test
    return await sdk.fetch(search_query=["q"], **addr.read_kwargs())


async def test_path_reaches_fetch(recording_sdk):
    recording_sdk.ladder = make_ladder("deep")   # or "b2b", "b2c"
    addr = await resolve_address(recording_sdk, values=["acme", "payments", "dana"])
    await my_integration_search(recording_sdk, addr)
    assert recording_sdk.last_kwargs("fetch")["scope_path"]["team"] == "payments"
```

`RecordingSDK(accept_scope_path=False)` behaves like an SDK from before the
ladder (methods reject `scope_path` / `scope`).
