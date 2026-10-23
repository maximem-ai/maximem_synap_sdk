"""Synap integration for Pipecat voice pipelines.

Pipecat's pipeline is a chain of :class:`FrameProcessor` nodes that exchange
``Frame`` objects. Long-term memory integrations plug in at two points:

1. **Before the user context aggregator** — inject relevant memory as an
   extra system message so the LLM sees it before generating its reply.
   Provided here as :class:`SynapMemoryProcessor`.
2. **After the assistant context aggregator** — record each completed turn
   to Synap: on the gRPC stream, which the first turn opens if you did not
   (``SYNAP_AUTO_LISTEN=false`` turns that off), and over REST only when the
   stream cannot carry it. Once Synap has compacted the conversation, the
   shared context's earlier turns are replaced by the compaction. Provided
   here as :class:`SynapRecorder`; pass it the shared ``context=``.

Typical wiring::

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator, LLMAssistantAggregator,
    )
    from synap_pipecat import SynapMemoryProcessor, SynapRecorder

    context = LLMContext(messages=[...])
    user_agg = LLMUserAggregator(context=context)
    asst_agg = LLMAssistantAggregator(context=context)

    pipeline = Pipeline([
        transport.input(), stt,
        SynapMemoryProcessor(sdk, user_id="alice", customer_id="acme"),
        user_agg, llm, tts, transport.output(), asst_agg,
        SynapRecorder(sdk, user_id="alice", customer_id="acme"),
    ])

See :class:`SynapMemoryProcessor` and :class:`SynapRecorder` for the full
frame-by-frame contract and error policy (reads degrade silently — a
Synap blip must never break a live call; writes surface as
``SynapIntegrationError`` via ``ErrorFrame``).
"""

from synap_pipecat.compaction import splice_synap_history
from synap_pipecat.processors import SynapMemoryProcessor, SynapRecorder
from synap_pipecat.short_term import SynapShortTermProcessor

__all__ = [
    "SynapMemoryProcessor",
    "SynapRecorder",
    "SynapShortTermProcessor",
    "splice_synap_history",
]
