# maximem-synap-openai-agents

Synap memory for the OpenAI Agents SDK: `create_search_tool`, `create_store_tool`, `SynapRunHooks`,
`synap_st_instructions` and `synap_call_model_input_filter`.

## Deeper scope ladders

On an account whose scope ladder has more levels than customer and user, pass the level as `scope_path`
(level key to value). It is fixed when the tool or hooks are built, and sent only when set. The model cannot
see or change it: it is not a parameter of either tool.

```python
path = {"customer": "acme", "team": "payments", "user": "dana"}
search = function_tool(create_search_tool(sdk, user_id="dana", customer_id="acme", scope_path=path))
hooks = SynapRunHooks(sdk, user_id="dana", customer_id="acme", conversation_id=conv_id, scope_path=path)
```

`maximem_synap.ScopeLadder.address(values)` builds `user_id`, `customer_id` and `scope_path` from level values.
Without `scope_path` nothing changes.

**Needs a newer SDK.** A path needs a `maximem-synap` release that has `get_scope_ladder` and `scope_path` on `fetch`, `send_message` and `record_message` (and `scope` on `memories.create`). With an older SDK, a turn that carries a path is dropped and an error is logged: it is never filed at a wider scope. Leave `scope_path` unset on an older SDK.
