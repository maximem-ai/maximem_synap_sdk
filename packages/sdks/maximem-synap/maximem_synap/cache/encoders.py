"""Local embedding models for history search (COMPACTION_HYBRID_PLAN.md P6).

Optional: installed with ``pip install "maximem-synap[local-search]"``.
The same model matches fetch queries to prefetched items in the anticipation
cache (anticipation_cache.py).
Without the extra, history search stops at keywords (literal values and
BM25), which is where exact values are found anyway; embeddings add the
turns that say the same thing in other words, or in another language.

The model runs on this machine's CPU. It is downloaded once, into
``~/.synap/models``, and loaded in a background thread, so the first prompt
never waits for it: searches before it is ready use keywords only.

Each encoder's ``model`` names it; vectors stored under another name are
never compared with it.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, List, Optional

from ..behavior import semantic_match_params

logger = logging.getLogger(__name__)

# Chosen on the labelled probe-to-turn set (harnesses/history_search/retrieval_bench.py).
# 19 Oct 2026, 100 probes in English and 50 each in Portuguese and Hindi,
# found in the top 4 (keywords only: 66 / 35 / 34): potion-multilingual-128M
# 72 / 37 / 36 at 0.18 ms per turn; paraphrase-multilingual-MiniLM-L12-v2
# 73 / 36 / 33 at 18 ms; English-only all-MiniLM-L6-v2 78 / 33 / 25.
#
# 20 Oct 2026, with passage search (history_search.split_passages): the answer
# text shown for 148 probes over 15K/20K and two large-turn conversations,
# potion 93 by whole turns / 99 by passages; multilingual MiniLM 117 by
# passages (about 17 ms per passage and 11 ms per query on an M4);
# EmbeddingGemma 2 116 at about 15x MiniLM's CPU. End to end, records a third
# to half the size answered as well as or better than today's with it.
#
# 21 Oct 2026: the same model as an int8 ONNX export, run by OnnxEncoder, the
# file the JS SDK runs too: the probe set above unchanged (SDK search 92/100,
# pt 41/50, hi 41/50; embeddings alone 74 vs 72), about half the CPU per turn
# (4.1 vs 16.5 ms, one text per run), 118 MB instead of 224. Anticipation matching
# (anticipation_cache.py) uses it as well, and its thresholds were tuned on it
# (harnesses/anticipation_match), so the name lives in the behavior contract.
DEFAULT_MODEL = semantic_match_params()["model"]
# The module each kind needs, to tell "extra not installed" from a failure.
REQUIRES = {"onnx": "onnxruntime", "fastembed": "fastembed", "model2vec": "model2vec"}
MODELS_DIR = Path.home() / ".synap" / "models"


def models_dir() -> Path:
    """Where models are downloaded: SYNAP_MODELS_DIR, else ~/.synap/models.

    Read at call time, as the JS SDK does, so an air-gapped or read-only-home
    deployment can point it at a prepared folder.
    """
    override = os.environ.get("SYNAP_MODELS_DIR", "").strip()
    return Path(override) if override else MODELS_DIR


class Model2VecEncoder:
    """A static embedding model: a lookup table, sub-millisecond on CPU."""

    def __init__(self, model_id: str, cache_dir: Optional[Path] = None):
        cache_dir = cache_dir or models_dir()
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel

        self.model = f"model2vec:{model_id}"
        local = snapshot_download(model_id, cache_dir=str(cache_dir))
        self._model = StaticModel.from_pretrained(local)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts))]


class FastEmbedEncoder:
    """A small transformer, ONNX on CPU (fastembed)."""

    # e5 models are trained with these prefixes.
    _PREFIXES = {"e5": ("query: ", "passage: ")}

    def __init__(self, model_id: str, cache_dir: Optional[Path] = None):
        cache_dir = cache_dir or models_dir()
        from fastembed import TextEmbedding

        self.model = f"fastembed:{model_id}"
        self._model = TextEmbedding(model_name=model_id, cache_dir=str(cache_dir))
        self._query, self._doc = next(
            (p for key, p in self._PREFIXES.items() if key in model_id.lower()), ("", ""))

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.embed([self._doc + t for t in texts])]

    def encode_query(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.embed([self._query + t for t in texts])]


class OnnxEncoder:
    """A sentence-transformers model exported to ONNX, run with onnxruntime:
    the tokenizer, the model, mean pooling over the attention mask, L2 norm.

    The JS SDK runs the same file the same way (js/src/cache/encoder.ts), so
    both SDKs produce the same vectors and one set of thresholds serves both.
    ``model_id`` is ``<hf repo>@<revision>/<onnx file>``: pinned, because a
    threshold tuned on one export does not hold for another.
    """

    MAX_TOKENS = 128  # the model's max_seq_length

    def __init__(self, model_id: str, cache_dir: Optional[Path] = None):
        cache_dir = cache_dir or models_dir()
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        repo, _, rest = model_id.partition("@")
        revision, _, onnx_file = rest.partition("/")
        self.model = f"onnx:{model_id}"
        get = lambda f: hf_hub_download(repo, f, revision=revision, cache_dir=str(cache_dir))  # noqa: E731
        self._tok = Tokenizer.from_file(get("tokenizer.json"))
        self._tok.enable_truncation(self.MAX_TOKENS)
        self._tok.enable_padding(pad_id=self._tok.token_to_id("<pad>") or 0, pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, (__import__("os").cpu_count() or 2) // 2))
        self._session = ort.InferenceSession(get(onnx_file), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._np = np
        # e5 models are trained with these prefixes (as FastEmbedEncoder).
        self._query, self._doc = ("query: ", "passage: ") if "e5" in repo.lower() else ("", "")

    def encode(self, texts: List[str], batch: int = 1) -> List[List[float]]:
        return self._run([self._doc + t for t in texts], batch)

    def encode_query(self, texts: List[str], batch: int = 1) -> List[List[float]]:
        return self._run([self._query + t for t in texts], batch)

    def encode_tokens(self, text: str):
        """The text's token vectors before pooling, each L2-normalised, with
        the tokens: (tokens, matrix). Special tokens are left out."""
        np = self._np
        enc = self._tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0][0]
        keep = [k for k, sm in enumerate(enc.special_tokens_mask) if not sm]
        m = hidden[keep]
        m = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
        return [enc.tokens[k] for k in keep], m

    def _run(self, texts: List[str], batch: int = 1) -> List[List[float]]:
        # One text per run by default. The int8 export quantizes activations
        # per run, over the whole batch, padding included, so a text's vector
        # would depend on the texts batched with it; one at a time it is the
        # same in every run and in both SDKs (21 Oct 2026: batches of 32 moved
        # pairwise cosines by up to 0.03 against the JS SDK's vectors).
        np = self._np
        out: List[List[float]] = []
        for start in range(0, len(texts), batch):
            enc = self._tok.encode_batch(list(texts[start:start + batch]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.extend(pooled.astype(float).tolist())
        return out


def load_encoder(name: str, cache_dir: Optional[Path] = None):
    """``"model2vec:<hf id>"``, ``"fastembed:<model name>"`` or
    ``"onnx:<hf repo>@<revision>/<onnx file>"``; None when the extra is not
    installed or the model cannot be loaded."""
    kind, _, model_id = name.partition(":")
    cache_dir = cache_dir or models_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if kind == "model2vec":
            return Model2VecEncoder(model_id, cache_dir)
        if kind == "fastembed":
            return FastEmbedEncoder(model_id, cache_dir)
        if kind == "onnx":
            return OnnxEncoder(model_id, cache_dir)
        logger.warning("Unknown history search model %r; using keywords only", name)
    except ImportError:
        logger.info("History search model %s needs maximem-synap[local-search]; using keywords only", name)
    except Exception as e:
        logger.warning("History search model %s failed to load (%s); using keywords only", name, e)
    return None


def load_in_background(name: str, ready: Callable[[object], None], cache_dir: Optional[Path] = None) -> threading.Thread:
    """Load ``name`` off the prompt path and hand it to ``ready`` (not called
    when it cannot be loaded)."""
    def run():
        encoder = load_encoder(name, cache_dir)
        if encoder is not None:
            ready(encoder)

    thread = threading.Thread(target=run, name="synap-history-encoder", daemon=True)
    thread.start()
    return thread
