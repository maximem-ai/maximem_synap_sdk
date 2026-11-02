# Changelog

## Unreleased

Needs `maximem-synap>=0.7.1` and `maximem-synap-integrations-common>=0.3.1` (`scope_path`, `ensure_ids_match_path`).

- **Compaction now reaches the model on pydantic-ai 1.107+ and 2.x.** On those versions the history
  processor Synap installed was never called, so the model always got the full history. It is now
  passed to every run (`Agent.iter(capabilities=...)`). What changes for an agent that uses
  `register_synap_tools`, `register_synap_st_system_prompt` or `synap_run` with a `conversation_id`:
  - one extra `get_context_for_prompt` call per run (the first model request only);
  - once Synap has compacted the conversation, older turns are replaced by Synap's compaction block
    and the turns after it; the current turn is untouched.
  On pydantic-ai 1.0.x it works as before. Nothing changes without a `conversation_id`.
- `install_synap_history` overrides `iter` through a subclass of the agent's class, so a
  `copy.copy(agent)` made after the install runs on its own model and tools. A user's own
  `iter(*args, **kwargs)` wrapper on the instance is accepted as a seam; the processor is then wrapped
  around it on the instance (a copy of such an agent still uses the original's `iter`).
- `scope_path` on `SynapDeps` (see its docstring). `user_id` / `customer_id` that disagree with the
  path now raise `SynapIntegrationError` when the deps are built and before each tool call and
  `synap_run`; the system-prompt callback skips and logs at ERROR instead of raising.
