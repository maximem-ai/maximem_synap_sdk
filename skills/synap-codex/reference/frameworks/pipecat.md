# Pipecat (voice)

`pip install synap-pipecat`

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
    customer_id="acme",   # optional
    max_results=6,
)

recorder = SynapRecorder(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",
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

## SynapMemoryProcessor

Intercepts `LLMMessagesFrame` events and prepends a system message with the user's relevant memories before the frame reaches the LLM:

```python
memory = SynapMemoryProcessor(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",
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
    customer_id="acme",
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
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
