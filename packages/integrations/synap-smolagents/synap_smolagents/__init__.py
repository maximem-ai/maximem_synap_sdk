"""Synap memory integration for Smolagents.

Hugging Face's Smolagents keeps its own fixed step log (`AgentMemory`) with no
pluggable backend, so Synap plugs in through the two extension points Smolagents
*does* expose:

- **Tools** (`@tool`): `create_synap_tools` returns `search_memory` / `store_memory`
  functions the model can call — idiomatic for `CodeAgent`, which invokes tools as
  plain Python.
- **Step callbacks**: `create_synap_recorder` returns a callback for
  `step_callbacks={ActionStep: ...}` that records each completed action into Synap
  for future recall.

Plus `synap_st_instructions` to fold Synap short-term context into the agent's
static `instructions`.

Smolagents is synchronous and its tools do not support `async`, so every surface
drives the async Synap SDK through the shared `run_async` bridge.

Typical wiring::

    from smolagents import CodeAgent, InferenceClientModel
    from smolagents.memory import ActionStep
    from maximem_synap import MaximemSynapSDK
    from synap_smolagents import create_synap_tools, create_synap_recorder

    sdk = MaximemSynapSDK(api_key="sk-...")
    agent = CodeAgent(
        model=InferenceClientModel(),
        tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
        step_callbacks={ActionStep: create_synap_recorder(
            sdk, user_id="alice", conversation_id="conv_abc")},
    )
    agent.run("What did we decide about the rollout?")

Tools/reads degrade (a Synap blip returns a not-found message); the ``store_memory``
tool raises ``SynapIntegrationError``; the recorder logs and never raises (it runs
inside the agent loop, where a raising callback would abort the run).
"""

from synap_smolagents.callbacks import create_synap_recorder
from synap_smolagents.short_term import synap_st_instructions
from synap_smolagents.tools import create_synap_tools

__all__ = [
    "create_synap_tools",
    "create_synap_recorder",
    "synap_st_instructions",
]
