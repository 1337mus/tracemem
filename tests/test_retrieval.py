import importlib.util
import json
import math
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SignedHashEmbedderTests(unittest.TestCase):
    def test_hash_vectors_are_stable_normalized_and_labeled_lexical(self) -> None:
        spec = importlib.util.find_spec("tracemem.retrieval")
        self.assertIsNotNone(spec, "retrieval module is not implemented")
        from tracemem.retrieval import SignedHashEmbedder

        embedder = SignedHashEmbedder(dimensions=96)
        first = embedder.encode("Memory ticket 42")
        repeated = embedder.encode("Memory ticket 42")
        changed_case = embedder.encode("memory ticket 42")

        self.assertEqual(first, repeated)
        self.assertEqual(first, changed_case)
        self.assertEqual(len(first), 96)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)
        self.assertEqual(embedder.metadata["kind"], "signed-token-hash")
        self.assertIn("weak lexical", str(embedder.metadata["label"]))
        self.assertIn("not semantic", str(embedder.metadata["label"]))


class LocalSentenceTransformerTests(unittest.TestCase):
    def test_model_load_is_explicitly_local_only(self) -> None:
        import tracemem.retrieval as retrieval

        self.assertTrue(
            hasattr(retrieval, "SentenceTransformerEmbedder"),
            "local-only SentenceTransformer support is not implemented",
        )
        SentenceTransformerEmbedder = retrieval.SentenceTransformerEmbedder

        with TemporaryDirectory() as model_directory:
            model = Mock()
            model.encode.return_value = [0.6, 0.8]
            constructor = Mock(return_value=model)
            fake_library = SimpleNamespace(SentenceTransformer=constructor)
            offline_variables = {
                "HF_HUB_OFFLINE": "0",
                "TRANSFORMERS_OFFLINE": "0",
                "HF_HUB_DISABLE_TELEMETRY": "0",
            }
            with patch.dict(os.environ, offline_variables):
                with patch.dict("sys.modules", {"sentence_transformers": fake_library}):
                    embedder = SentenceTransformerEmbedder(model_directory)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")

            constructor.assert_called_once_with(
                str(Path(model_directory).resolve()),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.assertEqual(embedder.encode("local words"), [0.6, 0.8])
            model.encode.assert_called_once_with(
                "local words", convert_to_numpy=True, normalize_embeddings=True
            )

    def test_fingerprint_tracks_model_contents_not_path_or_mtime(self) -> None:
        import tracemem.retrieval as retrieval

        self.assertTrue(hasattr(retrieval, "SentenceTransformerEmbedder"))
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_directory = root / "first-model"
            second_directory = root / "copied-model"
            first_directory.mkdir()
            second_directory.mkdir()
            first_config = first_directory / "config.json"
            second_config = second_directory / "config.json"
            first_config.write_bytes(b"alpha")
            second_config.write_bytes(b"alpha")
            fixed_time_ns = 1_700_000_000_000_000_000
            os.utime(first_config, ns=(fixed_time_ns, fixed_time_ns))
            os.utime(second_config, ns=(fixed_time_ns, fixed_time_ns))

            constructor = Mock(return_value=Mock())
            fake_library = SimpleNamespace(SentenceTransformer=constructor)
            with patch.dict("sys.modules", {"sentence_transformers": fake_library}):
                first = retrieval.SentenceTransformerEmbedder(first_directory)
                same_contents = retrieval.SentenceTransformerEmbedder(second_directory)
                first_config.write_bytes(b"bravo")
                os.utime(first_config, ns=(fixed_time_ns, fixed_time_ns))
                changed_contents = retrieval.SentenceTransformerEmbedder(first_directory)

            self.assertEqual(first.metadata["fingerprint"], same_contents.metadata["fingerprint"])
            self.assertNotEqual(
                first.metadata["fingerprint"], changed_contents.metadata["fingerprint"]
            )


class ContextPackingTests(unittest.TestCase):
    def test_utf8_byte_limit_includes_citation_metadata_and_unicode_evidence(self) -> None:
        import tracemem.retrieval as retrieval

        self.assertTrue(
            hasattr(retrieval, "pack_context"),
            "citation-aware context packing is not implemented",
        )
        hits = [
            SimpleNamespace(
                citation="user@rev:0:5",
                record=SimpleNamespace(
                    role="user",
                    recorded_at="2026-09-01T10:00:00Z",
                    content_kind="mixed_untrusted",
                    text="café",
                ),
            ),
            SimpleNamespace(
                citation="assistant@rev:0:8",
                record=SimpleNamespace(
                    role="assistant",
                    recorded_at="2026-09-01T10:01:00Z",
                    content_kind="conversation",
                    text="I infer this.",
                ),
            ),
            SimpleNamespace(
                citation="tool@rev:0:9",
                record=SimpleNamespace(
                    role="tool",
                    recorded_at="2026-09-01T10:02:00Z",
                    content_kind="tool_output",
                    text="status: ready",
                ),
            ),
        ]
        expected_lines = [
            "user@rev:0:5 [role=user recorded_at=2026-09-01T10:00:00Z "
            "content_kind=mixed_untrusted] café",
            "assistant@rev:0:8 [role=assistant recorded_at=2026-09-01T10:01:00Z "
            "content_kind=conversation] I infer this.",
            "tool@rev:0:9 [role=tool recorded_at=2026-09-01T10:02:00Z "
            "content_kind=tool_output] status: ready",
        ]
        expected = "\n".join(expected_lines)
        exact_budget = len(expected.encode("utf-8"))
        first_line_budget = len(expected_lines[0].encode("utf-8"))

        packed = retrieval.pack_context(hits, max_bytes=exact_budget)
        one_line = retrieval.pack_context(hits[:1], max_bytes=first_line_budget)
        too_small = retrieval.pack_context(hits[:1], max_bytes=first_line_budget - 1)

        self.assertEqual(packed, expected)
        self.assertEqual(one_line, expected_lines[0])
        self.assertLessEqual(len(packed.encode("utf-8")), exact_budget)
        self.assertEqual(too_small, "")


class RetrieverTests(unittest.TestCase):
    def test_bm25_uses_bounded_current_heads_without_loading_records_or_encoding(self) -> None:
        import tracemem
        import tracemem.retrieval as retrieval

        class NoEncodeEmbedder:
            @property
            def metadata(self) -> dict[str, str | int]:
                return {"kind": "test-no-encode", "version": 1}

            def encode(self, text: str) -> list[float]:
                raise AssertionError("BM25 must not encode text")

        records = [
            tracemem.EvidenceRecord(
                source="scale-test",
                session_id="session",
                message_id="message-00",
                part_index=0,
                role="user",
                text="proton relay retired archive entry",
            ),
            tracemem.EvidenceRecord(
                source="scale-test",
                session_id="session",
                message_id="message-00",
                part_index=0,
                role="user",
                text="proton relay current archive replacement",
            ),
        ]
        records.extend(
            tracemem.EvidenceRecord(
                source="scale-test",
                session_id="session",
                message_id=f"message-{index:02d}",
                part_index=0,
                role="user",
                text=f"proton relay component {index} release note",
            )
            for index in range(1, 32)
        )
        query, top_k = "proton relay", 7

        with TemporaryDirectory() as directory:
            with tracemem.EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(records)
                expected = store.lexical_search(query, limit=top_k)
                retriever = retrieval.Retriever(store, embedder=NoEncodeEmbedder())

                with patch.object(store, "records", wraps=store.records) as records_call:
                    actual = retriever.search(query, mode="bm25", k=top_k)
                    records_call.assert_not_called()

                self.assertEqual(len(expected), top_k)
                self.assertEqual(
                    [
                        (
                            hit.record.record_id,
                            hit.record.revision,
                            hit.score,
                            hit.citation,
                        )
                        for hit in actual
                    ],
                    [
                        (
                            record.record_id,
                            record.revision,
                            score,
                            f"{record.record_id}@{record.revision}:0:{len(record.text)}",
                        )
                        for record, score in expected
                    ],
                )
                self.assertNotEqual(actual[0].record.revision, records[0].revision)

    def test_modes_cite_the_exact_current_revision_with_unicode_offsets(self) -> None:
        import tracemem
        import tracemem.retrieval as retrieval

        self.assertTrue(hasattr(retrieval, "Retriever"), "retrieval search is not implemented")
        original = tracemem.EvidenceRecord(
            source="test",
            session_id="session",
            message_id="message-1",
            part_index=0,
            role="user",
            text="The project launch date was April 7.",
        )
        current = tracemem.EvidenceRecord(
            source="test",
            session_id="session",
            message_id="message-1",
            part_index=0,
            role="user",
            text="Deploy the café project on April 9.",
        )
        unrelated = tracemem.EvidenceRecord(
            source="test",
            session_id="session",
            message_id="message-2",
            part_index=0,
            role="user",
            text="The garden needs water before sunrise.",
        )

        with TemporaryDirectory() as directory:
            with tracemem.EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                self.assertEqual(store.ingest([original, current, unrelated]), 3)
                retriever = retrieval.Retriever(
                    store, embedder=retrieval.SignedHashEmbedder(dimensions=128)
                )

                for mode in ("bm25", "vector", "hybrid"):
                    hits = retriever.search("project launch date", mode=mode, k=5)
                    self.assertTrue(hits, mode)
                    self.assertEqual(hits[0].record.revision, current.revision, mode)
                    self.assertEqual(
                        hits[0].citation,
                        f"{current.record_id}@{current.revision}:0:{len(current.text)}",
                        mode,
                    )

                first_fingerprint = retriever.embedder_fingerprint
                first_cache = store.vector_cache(
                    current.record_id, current.revision, first_fingerprint
                )
                self.assertIsNotNone(first_cache)
                self.assertEqual(len(first_cache), 128)

                changed_embedder = retrieval.Retriever(
                    store, embedder=retrieval.SignedHashEmbedder(dimensions=256)
                )
                changed_embedder.search("project launch date", mode="vector", k=5)
                second_fingerprint = changed_embedder.embedder_fingerprint
                second_cache = store.vector_cache(
                    current.record_id, current.revision, second_fingerprint
                )
                self.assertNotEqual(first_fingerprint, second_fingerprint)
                self.assertIsNotNone(second_cache)
                self.assertEqual(len(second_cache), 256)


class LexicalScaleBenchmarkTests(unittest.TestCase):
    def test_small_scale_report_is_synthetic_bounded_and_ranking_equivalent(self) -> None:
        from experiments import benchmark_lexical_scale

        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                benchmark_lexical_scale.main(["--corpus-size", "32", "--repeats", "2"]),
                0,
            )
        report = json.loads(stdout.getvalue())

        self.assertEqual(report["corpus"]["requested_size"], 32)
        self.assertEqual(report["corpus"]["revision_count"], 33)
        self.assertEqual(report["corpus"]["current_head_count"], 32)
        self.assertEqual(report["repeats"], 2)
        self.assertIn("python", report["environment"])
        self.assertIn("sqlite", report["environment"])
        self.assertEqual(report["measurements"]["rankings_identical"], True)
        legacy = report["measurements"]["legacy_full_head_materialization"]
        optimized = report["measurements"]["optimized_bm25"]
        self.assertEqual(legacy["result_count"], 10)
        self.assertEqual(legacy["ranking_sha256"], optimized["ranking_sha256"])
        self.assertEqual(
            report["measurement_policy"],
            {
                "latency": "separate pass with tracemalloc disabled",
                "python_allocations": "separate pass with tracemalloc enabled",
                "ranking_equivalence": "checked on every repeat in both passes",
            },
        )
        for measurements in (legacy, optimized):
            latency = measurements["latency_ms"]
            self.assertLessEqual(latency["min"], latency["median"])
            self.assertLessEqual(latency["median"], latency["max"])
            peak = measurements["tracemalloc_peak_python_bytes"]
            self.assertLessEqual(peak["min"], peak["median"])
            self.assertLessEqual(peak["median"], peak["max"])
        for name in (
            "source_sha256",
            "retrieval_source_sha256",
            "storage_source_sha256",
            "legacy_function_source_sha256",
            "generator_sha256",
            "corpus_query_sha256",
        ):
            self.assertEqual(len(report[name]), 64)
        self.assertEqual(
            report["source_sha256"],
            __import__("hashlib")
            .sha256(Path(benchmark_lexical_scale.__file__).read_bytes())
            .hexdigest(),
        )
        import tracemem.retrieval as retrieval_module
        import tracemem.storage as storage_module

        self.assertEqual(
            report["retrieval_source_sha256"],
            __import__("hashlib").sha256(Path(retrieval_module.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["storage_source_sha256"],
            __import__("hashlib").sha256(Path(storage_module.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["legacy_function_source_sha256"],
            __import__("hashlib")
            .sha256(
                __import__("inspect")
                .getsource(benchmark_lexical_scale._legacy_search)
                .encode("utf-8")
            )
            .hexdigest(),
        )
        self.assertIn("not process RSS", " ".join(report["measurement_limits"]))
        self.assertNotIn("synthetic scale item", stdout.getvalue())

        with self.assertRaisesRegex(ValueError, "corpus size"):
            benchmark_lexical_scale.run_scale_benchmark(
                benchmark_lexical_scale.MAX_CORPUS_SIZE + 1, 1
            )

    def test_measure_separates_latency_from_tracemalloc_and_checks_each_repeat(self) -> None:
        import tracemalloc

        from experiments import benchmark_lexical_scale

        expected = [("record", "revision", 1.0)]
        hit = SimpleNamespace(
            record=SimpleNamespace(record_id="record", revision="revision"), score=1.0
        )
        observed_tracing: list[bool] = []

        def action() -> list[SimpleNamespace]:
            observed_tracing.append(tracemalloc.is_tracing())
            return [hit]

        benchmark_lexical_scale._measure(action, repeats=2, expected_rows=expected)

        self.assertEqual(observed_tracing, [False, False, True, True])

    def test_measure_rejects_an_intermediate_ranking_mismatch(self) -> None:
        from experiments import benchmark_lexical_scale

        expected = [("record", "revision", 1.0)]
        expected_hit = SimpleNamespace(
            record=SimpleNamespace(record_id="record", revision="revision"), score=1.0
        )
        different_hit = SimpleNamespace(
            record=SimpleNamespace(record_id="other", revision="revision"), score=1.0
        )
        rankings = iter(
            [
                [expected_hit],
                [expected_hit],
                [expected_hit],
                [expected_hit],
                [different_hit],
                [expected_hit],
                [expected_hit],
            ]
        )

        with self.assertRaisesRegex(AssertionError, "allocation repeat 2"):
            benchmark_lexical_scale._measure(
                lambda: next(rankings), repeats=3, expected_rows=expected
            )


if __name__ == "__main__":
    unittest.main()
