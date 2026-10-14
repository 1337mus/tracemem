"""Validated embedding vectors returned by an explicitly remote inference run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def text_sha256(text: str) -> str:
    """Return the lowercase SHA-256 key used for exact-text vector lookup."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrecomputedEmbedder:
    """Serve validated vectors without importing or running a local model."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dimensions: int,
        vectors: Mapping[str, Sequence[int | float]],
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        if not isinstance(vectors, Mapping):
            raise TypeError("vectors must be a mapping of text SHA-256 keys to vectors")

        copied_vectors: dict[str, tuple[float, ...]] = {}
        for key, vector in vectors.items():
            if not isinstance(key, str) or _SHA256_PATTERN.fullmatch(key) is None:
                raise ValueError("vector keys must be lowercase SHA-256 hashes")
            if isinstance(vector, (str, bytes, bytearray)) or not isinstance(vector, Sequence):
                raise TypeError("each precomputed vector must be a numeric sequence")
            if len(vector) != dimensions:
                raise ValueError("vector dimensions do not match the configured dimensions")

            values: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("vectors must contain finite numbers")
                try:
                    number = float(value)
                except (OverflowError, ValueError):
                    raise ValueError("vectors must contain finite numbers") from None
                if not math.isfinite(number):
                    raise ValueError("vectors must contain finite numbers")
                values.append(number)
            copied_vectors[key] = tuple(values)

        self._model_id = model_id
        self._revision = revision
        self._dimensions = dimensions
        self._vectors = copied_vectors
        fingerprint_payload = {
            "model_id": model_id,
            "revision": revision,
            "dimensions": dimensions,
            "vectors": {key: list(copied_vectors[key]) for key in sorted(copied_vectors)},
        }
        encoded_payload = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._fingerprint = hashlib.sha256(encoded_payload).hexdigest()

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "kind": "precomputed-remote-vectors",
            "model_id": self._model_id,
            "revision": self._revision,
            "dimensions": self._dimensions,
            "fingerprint": self._fingerprint,
            "normalization": "remote-model-output",
        }

    def encode(self, text: str) -> list[float]:
        """Return the exact vector for text, or fail without local inference."""
        key = text_sha256(text)
        try:
            vector = self._vectors[key]
        except KeyError:
            raise ValueError("no precomputed vector exists for this text SHA-256") from None
        return list(vector)


__all__ = ["PrecomputedEmbedder", "text_sha256"]
