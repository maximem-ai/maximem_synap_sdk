"""Synap retriever for LangChain RAG pipelines.

Maps Synap's typed memory items (facts, preferences, episodes, etc.)
to LangChain Document objects with rich metadata including type,
confidence, scope, and temporal info.
"""

import logging
from typing import List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class SynapRetriever(BaseRetriever):
    """Retriever that queries Synap memory and returns Documents.

    Example::

        retriever = SynapRetriever(sdk=sdk, user_id="user-456")
        docs = retriever.invoke("What are the user's preferences?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sdk: MaximemSynapSDK
    user_id: str
    customer_id: Optional[str] = None
    conversation_id: Optional[str] = None
    mode: str = "accurate"
    max_results: int = 20
    types: Optional[List[str]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return run_async(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        async with wrap_sdk_errors_async(
            "langchain.SynapRetriever", logger,
            user_id=self.user_id, query_len=len(query),
        ):
            response = await self.sdk.fetch(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
                search_query=[query],
                max_results=self.max_results,
                types=self.types,
                mode=self.mode,
                include_conversation_context=False,
            )

        docs: List[Document] = []

        for fact in response.facts:
            docs.append(Document(
                page_content=fact.content,
                metadata={
                    "type": "fact", "id": fact.id,
                    "confidence": fact.confidence, "source": fact.source,
                    "scope": response.scope_map.get(fact.id, ""),
                    "valid_until": str(fact.valid_until) if fact.valid_until else None,
                    "temporal_category": fact.temporal_category,
                },
            ))

        for pref in response.preferences:
            docs.append(Document(
                page_content=pref.content,
                metadata={
                    "type": "preference", "id": pref.id,
                    "strength": pref.strength, "category": pref.category,
                    "scope": response.scope_map.get(pref.id, ""),
                },
            ))

        for ep in response.episodes:
            docs.append(Document(
                page_content=ep.summary,
                metadata={
                    "type": "episode", "id": ep.id,
                    "significance": ep.significance,
                    "occurred_at": str(ep.occurred_at),
                    "scope": response.scope_map.get(ep.id, ""),
                },
            ))

        for em in response.emotions:
            docs.append(Document(
                page_content=f"{em.emotion_type}: {em.context}",
                metadata={
                    "type": "emotion", "id": em.id,
                    "intensity": em.intensity, "emotion_type": em.emotion_type,
                    "detected_at": str(em.detected_at),
                    "scope": response.scope_map.get(em.id, ""),
                },
            ))

        for te in response.temporal_events:
            docs.append(Document(
                page_content=te.content,
                metadata={
                    "type": "temporal_event", "id": te.id,
                    "event_date": str(te.event_date),
                    "valid_until": str(te.valid_until) if te.valid_until else None,
                    "temporal_category": te.temporal_category,
                    "scope": response.scope_map.get(te.id, ""),
                },
            ))

        return docs
