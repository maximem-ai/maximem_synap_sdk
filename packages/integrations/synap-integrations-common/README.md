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
- `synap_integrations_common.testing` — shared pytest fixtures and response
  factories, previously duplicated across each integration's `_helpers.py`.

Not intended for direct end-user use. Pin as a runtime dependency of each
integration package.
