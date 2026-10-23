"""``SynapMemory``: CrewAI's ``Memory`` wired so recall works against Synap.

CrewAI's ``Memory.recall`` turns the question into an embedding and hands the storage
backend only that embedding. Synap embeds on the server and needs the question text, so
with a plain ``Memory(storage=SynapStorageBackend(...))`` recall can never work.

``SynapMemory`` closes the gap with an embedder that remembers, for the duration of a
``recall``, which text produced which vector. The backend gets the vector from CrewAI,
looks the text up, and sends that to Synap. No threads or context variables are involved,
so it behaves the same on every CrewAI release that has the unified ``Memory`` (1.10+),
and no embedding provider (or OpenAI key) is needed for storage.

    from synap_crewai import SynapMemory, SynapStorageBackend

    backend = SynapStorageBackend(sdk=sdk, user_id="alice")
    memory = SynapMemory(storage=backend, llm=my_llm)
    memory.remember("Isha is vegetarian", scope="/diet", categories=["preference"], importance=0.8)
    memory.recall("What does Isha eat?")
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Optional, Sequence

from crewai.memory import Memory
from pydantic import PrivateAttr

from synap_crewai.storage import SynapStorageBackend

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Deterministic, local, dependency-free text embedder.

    Hashes words and word pairs into a fixed number of buckets and L2-normalises, so
    similar texts get similar vectors and word order matters. CrewAI uses embeddings
    itself to drop near-duplicates inside a batch and to score results; Synap ignores
    them for retrieval. Pass your own ``embedder=`` to ``SynapMemory`` if you rely on
    CrewAI's own similarity behaviour.
    """

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 (CrewAI calls it positionally)
        return [self._one(text) for text in input]

    def _bucket(self, feature: str) -> tuple:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], "big") % self.dims, (1.0 if digest[4] & 1 else -1.0)

    def _one(self, text: str) -> List[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dims
        for feature in features:
            bucket, sign = self._bucket(feature)
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            # Text with no word characters: still return a non-empty, stable vector.
            vec[self._bucket(text)[0]] = 1.0
            return vec
        return [x / norm for x in vec]


def _key(vector: Sequence[float]) -> tuple:
    return tuple(round(float(x), 8) for x in vector)


class QueryRegistry:
    """Bounded, thread-safe map from an embedding to the question text that produced it."""

    def __init__(self, max_entries: int = 512) -> None:
        self._max = max_entries
        self._items: "OrderedDict[tuple, str]" = OrderedDict()
        self._lock = threading.Lock()

    def register(self, vector: Sequence[float], text: str) -> None:
        key = _key(vector)
        with self._lock:
            self._items[key] = text
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def lookup(self, vector: Sequence[float]) -> Optional[str]:
        if not vector:
            return None
        with self._lock:
            return self._items.get(_key(vector))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class RecordingEmbedder:
    """Wraps an embedder; while a ``recall`` is in flight, remembers text -> vector.

    Outside a recall (saving) nothing is recorded, so CrewAI's save-time duplicate search
    finds no matching question and the backend answers it without calling Synap.
    """

    def __init__(self, inner: Callable[[List[str]], Any], registry: QueryRegistry) -> None:
        self.inner = inner
        self.registry = registry
        self._depth = 0
        self._lock = threading.Lock()

    def __call__(self, input: List[str]) -> Any:  # noqa: A002
        vectors = self.inner(input)
        if self._depth > 0:
            for text, vector in zip(input, vectors):
                vec = vector.tolist() if hasattr(vector, "tolist") else vector
                if vec is not None and len(vec):
                    self.registry.register(vec, text)
        return vectors

    def recording(self) -> "Iterator[None]":
        recorder = self

        class _Scope:
            def __enter__(self_inner) -> None:
                with recorder._lock:
                    recorder._depth += 1

            def __exit__(self_inner, *exc: Any) -> None:
                with recorder._lock:
                    recorder._depth -= 1

        return _Scope()  # type: ignore[return-value]


class SynapMemory(Memory):
    """CrewAI ``Memory`` that hands the question text to :class:`SynapStorageBackend`.

    Accepts everything ``Memory`` accepts. ``embedder`` must be a callable (or omitted);
    a provider-config dict is not supported because the embedder has to be wrapped.
    """

    _synap_recorder: Any = PrivateAttr(default=None)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        embedder = kwargs.get("embedder")
        if len(args) >= 3:  # CrewAI 1.10 takes (llm, storage, embedder, ...) positionally
            embedder = args[2]
            args = args[:2]
        if isinstance(embedder, dict):
            raise ValueError(
                "SynapMemory needs a callable embedder (or none). Synap embeds on the "
                "server, so the default local HashingEmbedder is used when omitted."
            )
        recorder = RecordingEmbedder(embedder or HashingEmbedder(), QueryRegistry())
        kwargs["embedder"] = recorder
        super().__init__(*args, **kwargs)
        self._synap_recorder = recorder
        if isinstance(self._storage, SynapStorageBackend):
            self._storage.attach_query_registry(recorder.registry)

    def recall(self, query: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        with self._synap_recorder.recording():
            return super().recall(query, *args, **kwargs)


__all__ = ["SynapMemory", "HashingEmbedder", "QueryRegistry", "RecordingEmbedder"]
