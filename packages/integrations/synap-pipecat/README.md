# synap-pipecat

Synap integration for [Pipecat](https://docs.pipecat.ai) — preload long-term memory into voice pipelines and record every turn back to Synap.

## Install

```bash
pip install synap-pipecat
```

Requires `pipecat-ai>=0.0.80`, `maximem-synap>=0.2.0`.

## Quickstart

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMAssistantAggregator,
)
from maximem_synap import MaximemSynapSDK
from synap_pipecat import SynapMemoryProcessor, SynapRecorder

sdk = MaximemSynapSDK(api_key="sk-...")
context = LLMContext(messages=[{"role": "system", "content": "You are a helpful assistant."}])

user_agg = LLMUserAggregator(context=context)
asst_agg = LLMAssistantAggregator(context=context)

pipeline = Pipeline([
    transport.input(),
    stt,
    SynapMemoryProcessor(sdk, user_id="alice", customer_id="acme"),
    user_agg,
    llm,
    tts,
    transport.output(),
    asst_agg,
    SynapRecorder(sdk, user_id="alice", customer_id="acme"),
])
```

## Scope

- **`SynapMemoryProcessor`** — sits BEFORE the user context aggregator. On each `TranscriptionFrame`, fetches relevant context from Synap via `sdk.fetch()` and appends it as a system message to the shared `LLMContext`. The transcription frame itself flows through unchanged so the user aggregator can record it as the user turn.
- **`SynapRecorder`** — sits AFTER the assistant context aggregator. Buffers the latest user transcription and the streamed assistant response; on `LLMFullResponseEndFrame`, calls `sdk.conversation.record_message()` for both turns.

## Error policy

- **Reads** (`SynapMemoryProcessor`) degrade gracefully — SDK failures log at `ERROR` and the frame continues downstream with no Synap context injected. A Synap blip must never break a live voice call.
- **Writes** (`SynapRecorder`) surface `SynapIntegrationError` via `wrap_sdk_errors_async`. The processor logs the failure and pushes an `ErrorFrame` upstream; the pipeline itself stays alive (errors propagate as frames, not exceptions).
- Non-matching frames pass through untouched in both processors — standard Pipecat `push_frame` contract.

## Constructor args (both processors)

- `sdk`: a configured `MaximemSynapSDK`
- `user_id`: required — Synap memory is user-scoped
- `customer_id`: optional customer/org scope; empty string means customer-less
- `conversation_id`: optional explicit conversation id (auto-generated per process lifetime when absent, shared across both processors if you pass the same instance)
