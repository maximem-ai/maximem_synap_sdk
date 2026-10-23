"""Synap memory integration for Smolagents.

Hugging Face's Smolagents keeps its own fixed step log (`AgentMemory`) with no
pluggable backend, so Synap plugs in through the two extension points Smolagents
*does* expose:

- **Tools** (`@tool`): `create_synap_tools` returns `search_memory` / `store_memory`
  functions the model can call — idiomatic for `CodeAgent`, which invokes tools as
  plain Python.
- **Step callbacks**: `synap_step_callbacks` returns the `step_callbacks` mapping
  that reports every step on Synap's live stream (user turn, assistant turn, tool
  calls, tool results, reasoning) and records each completed action into Synap for
  future recall. `create_synap_recorder` returns the bare callback behind it.

Plus `attach_synap_history`, which the recorder applies for you: it wraps the agent's
`write_memory_to_messages` so that once Synap has compacted the conversation the model
sees Synap's compaction in place of the agent's earlier steps; and
`synap_st_instructions` to fold that compaction into the agent's static `instructions`
for an agent with no recorder.

Smolagents is synchronous and its tools do not support `async`, so every surface
drives the async Synap SDK through the shared `run_async` bridge.

Typical wiring::

    from smolagents import CodeAgent, InferenceClientModel
    from maximem_synap import MaximemSynapSDK
    from synap_smolagents import create_synap_tools, synap_step_callbacks

    sdk = MaximemSynapSDK(api_key="sk-...")
    agent = CodeAgent(
        model=InferenceClientModel(),
        tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
        step_callbacks=synap_step_callbacks(
            sdk, user_id="alice", conversation_id="conv_abc"),
    )
    agent.run("What did we decide about the rollout?")

Tools/reads degrade (a Synap blip returns a not-found message); the ``store_memory``
tool raises ``SynapIntegrationError``; the recorder logs and never raises (it runs
inside the agent loop, where a raising callback would abort the run). The first
report opens the gRPC stream (``sdk.instance.listen()``) when none is open;
``SYNAP_AUTO_LISTEN=false`` keeps everything on REST.
"""

from synap_smolagents.callbacks import create_synap_recorder, synap_step_callbacks
from synap_smolagents.history import attach_synap_history
from synap_smolagents.short_term import synap_st_instructions
from synap_smolagents.tools import create_synap_tools

__all__ = [
    "create_synap_tools",
    "create_synap_recorder",
    "synap_step_callbacks",
    "synap_st_instructions",
    "attach_synap_history",
]
