# Pipecat (voice)

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/pipecat.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-pipecat`

For Pipecat's frame-processor voice pipeline. Two processors slot into the pipeline, before LLM and after response.

| Class | Purpose |
| --- | --- |
| `SynapMemoryProcessor` | Injects context before LLM |
| `SynapRecorder` | Records turns after response |

## Quick start

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.openai import OpenAILLMService
from synap_pipecat import SynapMemoryProcessor, SynapRecorder

memory = SynapMemoryProcessor(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    max_results=6,
)

recorder = SynapRecorder(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",           # B2B only; omit on B2C
    conversation_id="call-001",   # optional; auto-generated if omitted
)

pipeline = Pipeline([
    transport.input(),
    stt,
    memory,           # ← inject memory before LLM
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
    recorder,         # ← record turn after response
])
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## SynapMemoryProcessor

Intercepts `LLMMessagesFrame` events and prepends a system message with the user's relevant memories before the frame reaches the LLM:

```python
memory = SynapMemoryProcessor(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    max_results=6,
    mode="fast",   # "fast" or "accurate"
)
```

Failures degrade gracefully — the frame passes through unmodified.

## SynapRecorder

Intercepts `TranscriptionFrame` (user) and `LLMFullResponseEndFrame` (assistant) events and ingests the completed turn:

```python
recorder = SynapRecorder(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    conversation_id="call-001",
)
```

Write failures surface as `SynapIntegrationError` — Pipecat's frame error handling catches and logs.

## Pipeline placement (visual)

```
transport.input() → STT → SynapMemoryProcessor → UserAggregator → LLM → TTS → transport.output()
                                                                                        ↓
                                                                     AssistantAggregator
                                                                                        ↓
                                                                            SynapRecorder
```

The order matters: memory **before** the user aggregator/LLM so it lands in the prompt; recorder **after** the assistant aggregator so it captures the completed turn.

## Live doc

`https://docs.maximem.ai/integrations/pipecat`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
