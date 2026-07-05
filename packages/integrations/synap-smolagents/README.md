# Synap + Smolagents

Synap memory tools and short-term context helpers for [Smolagents](https://pypi.org/).

## Install

```bash
pip install maximem-synap maximem-synap-smolagents
```

## Usage

```python
from maximem_synap import MaximemSynapSDK
from synap_smolagents import create_search_tool, create_store_tool, synap_st_instructions

sdk = MaximemSynapSDK(api_key="your-api-key")
await sdk.initialize()

search = create_search_tool(sdk, user_id="alice", conversation_id="conv-1")
store = create_store_tool(sdk, user_id="alice")

# CodeAgent(system_prompt=synap_st_instructions(...))
instructions = synap_st_instructions(
    sdk,
    conversation_id="conv-1",
    system="You are a helpful assistant.",
)
```

See the root repo README for scoping (`user_id`, `customer_id`, `conversation_id`) and benchmark context.
