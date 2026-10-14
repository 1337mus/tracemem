"""Offline validation for remote vector results and retrieval integration."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from tracemem import EvidenceStore
from tracemem.benchmark import load_synthetic_data
from tracemem.retrieval import Retriever


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrecomputedEmbedderTests(unittest.TestCase):
    def test_exact_hash_lookup_returns_an_isolated_vector_copy(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        embedder = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=3,
            vectors={_text_hash("exact text"): [0.6, 0.8, 0.0]},
        )

        vector = embedder.encode("exact text")
        vector[0] = 99.0

        self.assertEqual(embedder.encode("exact text"), [0.6, 0.8, 0.0])
        self.assertEqual(embedder.metadata["dimensions"], 3)
        self.assertEqual(embedder.metadata["model_id"], "synthetic/model")
        self.assertEqual(embedder.metadata["revision"], "revision-1")

    def test_unknown_exact_text_fails_without_fallback(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        embedder = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=2,
            vectors={_text_hash("known text"): [1.0, 0.0]},
        )

        with self.assertRaisesRegex(ValueError, "precomputed vector"):
            embedder.encode("unknown text")

    def test_fingerprint_is_stable_and_binds_model_revision_and_vectors(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        first_hash = _text_hash("first")
        second_hash = _text_hash("second")
        first = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=2,
            vectors={first_hash: [1.0, 0.0], second_hash: [0.0, 1.0]},
        )
        reordered = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=2,
            vectors={second_hash: [0.0, 1.0], first_hash: [1.0, 0.0]},
        )
        changed_model = PrecomputedEmbedder(
            model_id="other/model",
            revision="revision-1",
            dimensions=2,
            vectors={first_hash: [1.0, 0.0], second_hash: [0.0, 1.0]},
        )
        changed_vector = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=2,
            vectors={first_hash: [0.5, 0.5], second_hash: [0.0, 1.0]},
        )

        self.assertEqual(first.metadata["fingerprint"], reordered.metadata["fingerprint"])
        self.assertNotEqual(first.metadata["fingerprint"], changed_model.metadata["fingerprint"])
        self.assertNotEqual(first.metadata["fingerprint"], changed_vector.metadata["fingerprint"])

    def test_rejects_non_sha256_keys_and_malformed_vectors(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            PrecomputedEmbedder(
                model_id="synthetic/model",
                revision="revision-1",
                dimensions=2,
                vectors={"not-a-hash": [1.0, 0.0]},
            )

        with self.assertRaisesRegex(ValueError, "dimensions"):
            PrecomputedEmbedder(
                model_id="synthetic/model",
                revision="revision-1",
                dimensions=2,
                vectors={_text_hash("text"): [1.0]},
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            PrecomputedEmbedder(
                model_id="synthetic/model",
                revision="revision-1",
                dimensions=2,
                vectors={_text_hash("text"): [float("nan"), 0.0]},
            )

        with self.assertRaisesRegex(TypeError, "finite numbers"):
            PrecomputedEmbedder(
                model_id="synthetic/model",
                revision="revision-1",
                dimensions=2,
                vectors={_text_hash("text"): [True, 0.0]},
            )

    def test_precomputed_vectors_plug_into_retriever_without_local_encoder(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        records, queries, _ = load_synthetic_data()
        vectors = {
            _text_hash(text): [1.0, 0.0, 0.0]
            for text in [*(record.text for record in records), queries[0]["query"]]
        }
        embedder = PrecomputedEmbedder(
            model_id="synthetic/model",
            revision="revision-1",
            dimensions=3,
            vectors=vectors,
        )

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(records)
                hits = Retriever(store, embedder=embedder).search(
                    queries[0]["query"], mode="vector", k=1
                )

        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
