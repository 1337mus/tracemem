"""Local retrieval helpers and explicitly weak lexical vector baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import EvidenceRecord
from .storage import EvidenceStore

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class Embedder(Protocol):
    @property
    def metadata(self) -> dict[str, str | int]: ...

    def encode(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class SearchHit:
    record: EvidenceRecord
    score: float
    citation: str


@dataclass(frozen=True, slots=True)
class PackedContext:
    """Packed text and the exact whole hits that contributed to it."""

    text: str
    selected_hits: tuple[SearchHit, ...]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _unit_vector(vector: list[float]) -> list[float]:
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embeddings must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embeddings must contain finite numbers")
        values.append(number)
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return []
    return [value / norm for value in values]


class SignedHashEmbedder:
    """Create deterministic signed token-hash vectors for offline diagnostics."""

    def __init__(self, dimensions: int = 384) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "kind": "signed-token-hash",
            "version": 1,
            "dimensions": self.dimensions,
            "normalization": "l2",
            "label": "weak lexical feature vectors; not semantic embeddings",
        }

    def encode(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class SentenceTransformerEmbedder:
    """Wrap a SentenceTransformer loaded from an existing local directory only."""

    def __init__(self, model_directory: str | Path) -> None:
        directory = Path(model_directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError("model_directory must be an existing local directory")
        resolved = directory.resolve()

        # Keep optional model resolution, telemetry, and later hub calls offline.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "install the optional semantic dependencies to use a local model"
            ) from exc

        self._model = SentenceTransformer(
            str(resolved), local_files_only=True, trust_remote_code=False
        )
        self._fingerprint = self._fingerprint_directory(resolved)

    @staticmethod
    def _fingerprint_directory(directory: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                relative = path.relative_to(directory).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                with path.open("rb") as model_file:
                    for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
        return digest.hexdigest()

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "kind": "sentence-transformers-local",
            "version": 1,
            "fingerprint": self._fingerprint,
            "normalization": "l2",
        }

    def encode(self, text: str) -> list[float]:
        encoded = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [float(value) for value in encoded]


class Retriever:
    """Rank exact evidence revisions with BM25, cosine similarity, or RRF."""

    def __init__(
        self,
        store: EvidenceStore,
        embedder: Embedder | None = None,
        *,
        rrf_k: int = 60,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        self.store = store
        self.embedder = embedder if embedder is not None else SignedHashEmbedder()
        self.rrf_k = rrf_k
        try:
            metadata = json.dumps(
                self.embedder.metadata,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("embedder metadata must be JSON serializable") from exc
        self._fingerprint = hashlib.sha256(metadata.encode("utf-8")).hexdigest()

    @property
    def embedder_fingerprint(self) -> str:
        return self._fingerprint

    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> list[SearchHit]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query) > 4096:
            raise ValueError("query is too long")
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("mode must be bm25, vector, or hybrid")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
            raise ValueError("k must be an integer between 1 and 100")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        if mode == "bm25":
            lexical_rows = self.store.lexical_search(query, limit=k)
            return [
                SearchHit(
                    record=record,
                    score=score,
                    citation=f"{record.record_id}@{record.revision}:0:{len(record.text)}",
                )
                for record, score in lexical_rows
                if score > 0
            ]

        records = self.store.records(current_only=True)
        if not records:
            return []
        record_by_key = {(record.record_id, record.revision): record for record in records}
        ranked: dict[str, list[tuple[EvidenceRecord, float]]] = {}
        if mode == "hybrid":
            lexical_rows = self.store.lexical_search(query, limit=min(1000, len(records)))
            ranked["bm25"] = [
                (record, score)
                for record, score in lexical_rows
                if score > 0 and (record.record_id, record.revision) in record_by_key
            ]
        if mode in {"vector", "hybrid"}:
            ranked["vector"] = self._vector_search(query, query_tokens, records)

        if mode != "hybrid":
            results = ranked[mode][:k]
        else:
            fused: dict[tuple[str, str], float] = {}
            for source_results in ranked.values():
                for rank, (record, _) in enumerate(source_results, start=1):
                    key = (record.record_id, record.revision)
                    fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            results = [
                (record_by_key[key], score) for key, score in fused.items() if key in record_by_key
            ]
            results.sort(key=lambda item: (-item[1], item[0].record_id, item[0].revision))
            results = results[:k]

        return [
            SearchHit(
                record=record,
                score=score,
                citation=f"{record.record_id}@{record.revision}:0:{len(record.text)}",
            )
            for record, score in results
        ]

    def _vector_search(
        self,
        query: str,
        query_tokens: set[str],
        records: list[EvidenceRecord],
    ) -> list[tuple[EvidenceRecord, float]]:
        query_vector = _unit_vector(self.embedder.encode(query))
        if not query_vector:
            return []
        hash_only = isinstance(self.embedder, SignedHashEmbedder)
        scored: list[tuple[EvidenceRecord, float]] = []
        for record in records:
            if hash_only and not query_tokens.intersection(_tokens(record.text)):
                continue
            vector = self.store.vector_cache(record.record_id, record.revision, self._fingerprint)
            if vector is None:
                vector = _unit_vector(self.embedder.encode(record.text))
                self.store.cache_vector(
                    record.record_id, record.revision, self._fingerprint, vector
                )
            else:
                vector = _unit_vector(vector)
            if not vector:
                continue
            if len(vector) != len(query_vector):
                raise ValueError("cached embedding dimensions do not match the query")
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0].record_id, item[0].revision))
        return scored


def _context_line(hit: SearchHit) -> str:
    citation = hit.citation
    text = hit.record.text
    role = hit.record.role
    recorded_at = hit.record.recorded_at
    content_kind = hit.record.content_kind
    if not all(isinstance(value, str) for value in (citation, text, role, content_kind)):
        raise TypeError("hits must provide string citations and evidence metadata")
    timestamp = "unknown" if recorded_at is None else str(recorded_at)
    timestamp = timestamp.replace("\r", " ").replace("\n", " ")
    return f"{citation} [role={role} recorded_at={timestamp} content_kind={content_kind}] {text}"


def _pack_context(hits: Sequence[SearchHit], max_bytes: int) -> PackedContext:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")

    lines: list[str] = []
    selected_hits: list[SearchHit] = []
    used_bytes = 0
    for hit in hits:
        line = _context_line(hit)
        line_bytes = len(line.encode("utf-8"))
        separator_bytes = 1 if lines else 0
        if used_bytes + separator_bytes + line_bytes <= max_bytes:
            lines.append(line)
            selected_hits.append(hit)
            used_bytes += separator_bytes + line_bytes
    return PackedContext("\n".join(lines), tuple(selected_hits))


def pack_context_with_manifest(hits: Sequence[SearchHit], max_bytes: int) -> PackedContext:
    """Pack whole citation/evidence lines and preserve their exact hit manifest."""
    return _pack_context(hits, max_bytes)


def pack_context(hits: Sequence[SearchHit], max_bytes: int) -> str:
    """Pack complete citation/evidence lines under a UTF-8 byte ceiling."""
    return _pack_context(hits, max_bytes).text
