# maximem-synap-llamaindex

Synap memory for LlamaIndex: `SynapRetriever`, `SynapChatMemory` and `synap_st_chat_message`.

## Deeper scope ladders

On an account whose scope ladder has more levels than customer and user, pass the level as `scope_path`
(level key to value). It is sent with every fetch and every turn the object makes, and only when set.

```python
path = {"customer": "acme", "team": "payments", "user": "dana"}
retriever = SynapRetriever(sdk, user_id="dana", customer_id="acme", scope_path=path)
memory = SynapChatMemory(sdk, conversation_id=conv_id, user_id="dana", customer_id="acme", scope_path=path)
```

`maximem_synap.ScopeLadder.address(values)` builds `user_id`, `customer_id` and `scope_path` from level values.
Without `scope_path` nothing changes.

**Needs a newer SDK.** A path needs a `maximem-synap` release that has `get_scope_ladder` and `scope_path` on `fetch`, `send_message` and `record_message` (and `scope` on `memories.create`). With an older SDK, a turn that carries a path is dropped and an error is logged: it is never filed at a wider scope. Leave `scope_path` unset on an older SDK.
