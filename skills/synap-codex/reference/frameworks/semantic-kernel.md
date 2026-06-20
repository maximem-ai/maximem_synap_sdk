# Semantic Kernel

`pip install synap-semantic-kernel`

For Microsoft Semantic Kernel.

| Class | Purpose |
| --- | --- |
| `SynapPlugin` | Kernel plugin with `search_memory` and `store_memory` kernel functions |

## Quick start

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from synap_semantic_kernel import SynapPlugin

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="default"))

kernel.add_plugin(
    SynapPlugin(sdk=sdk, user_id="alice", customer_id="acme"),
    plugin_name="synap",
)

result = await kernel.invoke_prompt(
    "{{synap.search_memory query='project priorities'}} What are my top priorities?"
)
```

## Plugin functions

```
search_memory(query: str, max_results: int = 5) -> str
# returns formatted string of results — drop directly into prompt templates

store_memory(content: str, memory_type: str = "fact") -> str
# returns "Memory stored successfully."
```

The string-returning shape makes these easy to embed in prompt templates with `{{synap.search_memory ...}}`.

## Auto function calling

Let the kernel decide when to invoke Synap functions:

```python
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory

settings = kernel.get_prompt_execution_settings_from_service_id("default")
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

kernel.add_plugin(SynapPlugin(sdk=sdk, user_id="alice"), plugin_name="synap")

chat_history = ChatHistory()
chat_history.add_user_message("What do you remember about my travel preferences?")

response = await kernel.invoke_stream(
    function=kernel.get_function("chat", "chat"),
    settings=settings,
    chat_history=chat_history,
)
```

## Live doc

`https://docs.maximem.ai/integrations/semantic-kernel`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
