"""Synap retriever for LlamaIndex RAG pipelines.

Maps Synap's typed memory items to LlamaIndex NodeWithScore objects,
enabling memory-augmented retrieval in any LlamaIndex pipeline.
"""

import logging
from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class SynapRetriever(BaseRetriever):
    """LlamaIndex retriever backed by Synap memory."""

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 20,
        types: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if sdk is None:
            raise ValueError("SynapRetriever requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapRetriever requires a non-empty user_id")

        super().__init__(**kwargs)
        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id
        self._mode = mode
        self._max_results = max_results
        self._types = types

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return run_async(self._aretrieve(query_bundle))

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async with wrap_sdk_errors_async(
            "llamaindex.SynapRetriever", logger,
            user_id=self._user_id, query_len=len(query_bundle.query_str),
        ):
            response = await self._sdk.fetch(
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id,
                search_query=[query_bundle.query_str],
                max_results=self._max_results,
                types=self._types,
                mode=self._mode,
                include_conversation_context=False,
            )

        nodes: List[NodeWithScore] = []

        for fact in response.facts:
            nodes.append(NodeWithScore(node=TextNode(
                text=fact.content, id_=fact.id,
                metadata={
                    "type": "fact", "confidence": fact.confidence,
                    "source": fact.source,
                    "scope": response.scope_map.get(fact.id, ""),
                    "temporal_category": fact.temporal_category,
                },
            ), score=fact.confidence))

        for pref in response.preferences:
            nodes.append(NodeWithScore(node=TextNode(
                text=pref.content, id_=pref.id,
                metadata={
                    "type": "preference", "strength": pref.strength,
                    "category": pref.category,
                    "scope": response.scope_map.get(pref.id, ""),
                },
            ), score=pref.strength))

        for ep in response.episodes:
            nodes.append(NodeWithScore(node=TextNode(
                text=ep.summary, id_=ep.id,
                metadata={
                    "type": "episode", "significance": ep.significance,
                    "occurred_at": str(ep.occurred_at),
                    "scope": response.scope_map.get(ep.id, ""),
                },
            ), score=ep.significance))

        for em in response.emotions:
            nodes.append(NodeWithScore(node=TextNode(
                text=f"{em.emotion_type}: {em.context}", id_=em.id,
                metadata={
                    "type": "emotion", "emotion_type": em.emotion_type,
                    "intensity": em.intensity,
                    "scope": response.scope_map.get(em.id, ""),
                },
            ), score=em.intensity))

        for te in response.temporal_events:
            nodes.append(NodeWithScore(node=TextNode(
                text=te.content, id_=te.id,
                metadata={
                    "type": "temporal_event",
                    "event_date": str(te.event_date),
                    "valid_until": str(te.valid_until) if te.valid_until else None,
                    "scope": response.scope_map.get(te.id, ""),
                },
            ), score=te.temporal_confidence))

        nodes.sort(key=lambda n: n.score or 0, reverse=True)
        return nodes
