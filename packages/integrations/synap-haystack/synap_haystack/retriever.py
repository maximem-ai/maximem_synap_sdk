"""Synap retriever component for Haystack pipelines.

A Haystack @component that retrieves memory context from Synap
and returns it as a list of Haystack Documents.
"""

import logging
from typing import Dict, List, Optional

from haystack import Document, component

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


@component
class SynapRetriever:
    """Haystack component that retrieves memory from Synap.

    Example::

        retriever = SynapRetriever(sdk=sdk, user_id="u1")
        pipeline.add_component("memory", retriever)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 20,
    ):
        if sdk is None:
            raise ValueError("SynapRetriever requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapRetriever requires a non-empty user_id")

        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_results = max_results

    @component.output_types(documents=List[Document])
    def run(self, query: str) -> Dict[str, List[Document]]:
        return run_async(self._arun(query))

    async def _arun(self, query: str) -> Dict[str, List[Document]]:
        async with wrap_sdk_errors_async(
            "haystack.SynapRetriever.run",
            logger,
            user_id=self.user_id,
            query_len=len(query),
        ):
            response = await self.sdk.fetch(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[query],
                max_results=self.max_results,
                mode=self.mode,
                include_conversation_context=False,
            )

        docs: List[Document] = []

        for fact in response.facts:
            docs.append(Document(
                content=fact.content,
                meta={"type": "fact", "id": fact.id, "confidence": fact.confidence,
                      "scope": response.scope_map.get(fact.id, "")},
            ))

        for pref in response.preferences:
            docs.append(Document(
                content=pref.content,
                meta={"type": "preference", "id": pref.id, "strength": pref.strength,
                      "scope": response.scope_map.get(pref.id, "")},
            ))

        for ep in response.episodes:
            docs.append(Document(
                content=ep.summary,
                meta={"type": "episode", "id": ep.id, "significance": ep.significance,
                      "scope": response.scope_map.get(ep.id, "")},
            ))

        for em in response.emotions:
            docs.append(Document(
                content=f"{em.emotion_type}: {em.context}",
                meta={"type": "emotion", "id": em.id, "intensity": em.intensity,
                      "scope": response.scope_map.get(em.id, "")},
            ))

        for te in response.temporal_events:
            docs.append(Document(
                content=te.content,
                meta={"type": "temporal_event", "id": te.id,
                      "scope": response.scope_map.get(te.id, "")},
            ))

        return {"documents": docs}
