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
    SynapMemoryProcessor(sdk, user_id="alice", customer_id="acme", context=context),
    user_agg,
    llm,
    tts,
    transport.output(),
    asst_agg,
    SynapRecorder(sdk, user_id="alice", customer_id="acme", context=context),
])
```

> **Pass `context=` to both processors.** `SynapMemoryProcessor` without a
> context is a read-only tap; it fetches nothing into the prompt. On
> Pipecat ≥1.x the universal aggregators consume `TranscriptionFrame`, so a
> `SynapRecorder` at the end of the pipeline recovers the user turn from the
> shared context, and it is also the context Synap's compaction is spliced
> into. Without `context=` it records only the assistant half.

## Scope

- **`SynapMemoryProcessor`** — sits BEFORE the user context aggregator. On each `TranscriptionFrame`, fetches relevant context from Synap via `sdk.fetch()` and appends it as a system message to the shared `LLMContext`. The transcription frame itself flows through unchanged so the user aggregator can record it as the user turn.
- **`SynapRecorder`**: sits AFTER the assistant context aggregator. Records both turns of each exchange. The user turn comes from the latest `TranscriptionFrame` to reach the processor, or (on universal-aggregator pipelines where that frame never arrives) from the shared `context`. The assistant turn comes from streamed `LLMTextFrame`s or the `TTSTextFrame` stream, or, behind the universal `LLMAssistantAggregator`, from `LLMContextAssistantTurnFrame` (see below).
- **`SynapShortTermProcessor`**: sits BEFORE the user aggregator, with the `conversation_id` of the recorder. Keeps one tagged system block at the head of the context, and never repeats a turn the context already holds.

## What the recorder sees depends on where it sits

Pipecat 1.x's `LLMAssistantAggregator` consumes `LLMFullResponseStartFrame`/`EndFrame`, every `TextFrame` (so `TTSTextFrame` too), `LLMThoughtTextFrame` and the function-call frames, and pushes none of them on (`llm_response_universal.py`, `LLMAssistantAggregator.process_frame`). Behind it, as in the quickstart, the recorder gets one end-of-turn frame, `LLMContextAssistantTurnFrame` (pipecat-ai 1.7+), and records the turn from that. Tool calls and results are read back out of the shared `context` after the turn, since no frame carries them; reasoning is not recoverable there.

To get tool calls at the moment they are dispatched, and reasoning, put the recorder BEFORE the assistant aggregator (for example `llm, SynapRecorder(...), tts, ..., asst_agg`). It then ends the turn on `LLMFullResponseEndFrame`, and the aggregator's `LLMContextAssistantTurnFrame` is ignored so a turn is never recorded twice.

## The gRPC stream is the default; REST only as a fallback, never both

The first turn, tool call or tool result opens the stream (`sdk.instance.listen()`) if you did not, and everything after rides it. The server persists streamed turns itself, so `sdk.conversation.record_message` is called only when the stream cannot carry the turn. Set `SYNAP_AUTO_LISTEN=false` (or `set_auto_listen(sdk, False)` from `synap_integrations_common`) to stay on REST.

## Synap's compaction replaces the earlier turns

With `context=` given, after each turn it records the recorder checks whether Synap has compacted the conversation. If so, the shared `LLMContext` is rewritten in place (`set_messages`, the same call pipecat's own `LLMContextSummarizer` ends in): the earlier turns become one `<synap_conversation_history>` system message plus the turns after the compaction, and the current exchange is untouched. Before the first compaction, or if Synap does not answer within 1.5s, the context is left alone and `enable_auto_context_summarization` (if you turned it on) stays the fallback. `SynapShortTermProcessor` does the same before each user turn.

## Error policy

- **Reads** (`SynapMemoryProcessor`) degrade gracefully — SDK failures log at `ERROR` and the frame continues downstream with no Synap context injected. A Synap blip must never break a live voice call.
- **Writes** (`SynapRecorder`) that fail on both the stream and REST are logged and pushed upstream as an `ErrorFrame` carrying a `SynapIntegrationError`; the pipeline itself stays alive (errors propagate as frames, not exceptions).
- Non-matching frames pass through untouched in both processors — standard Pipecat `push_frame` contract.

## Constructor args (both processors)

- `sdk`: a configured `MaximemSynapSDK`
- `user_id`: required — Synap memory is user-scoped
- `customer_id`: B2B instances only, where it is required. Not accepted on a B2C instance, which rejects it with HTTP 400.
- `conversation_id` *(recorder, short-term processor)*: optional explicit id for the recorder (auto-generated when absent), required for the short-term processor: pass the same value to both
