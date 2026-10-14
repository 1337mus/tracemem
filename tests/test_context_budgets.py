"""Tests for the synthetic whole-passage context-budget diagnostic."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class ContextBudgetBenchmarkTests(unittest.TestCase):
    def test_real_fixture_packs_one_ranking_per_mode_and_keeps_unknowns_separate(self) -> None:
        from tracemem import EvidenceStore
        from tracemem.benchmark import load_synthetic_data
        from tracemem.budget_benchmark import run_context_budget_benchmark
        from tracemem.retrieval import Retriever

        records, queries, fixture_sha256 = load_synthetic_data()
        with TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(records)
                original_search = Retriever.search
                with patch.object(
                    Retriever, "search", autospec=True, side_effect=original_search
                ) as search:
                    result = run_context_budget_benchmark(
                        store,
                        queries,
                        input_sha256=fixture_sha256,
                        budgets_bytes=(0, 4096),
                    )

        self.assertEqual(search.call_count, len(queries) * 3)
        self.assertEqual(result["fixture_sha256"], fixture_sha256)
        self.assertEqual(result["answerable_queries"], 14)
        self.assertEqual(result["unknown_queries"], 3)
        self.assertEqual(set(result["modes"]), {"bm25", "vector", "hybrid"})

        for mode in ("bm25", "vector", "hybrid"):
            with self.subTest(mode=mode):
                report = result["modes"][mode]
                self.assertEqual(set(report["by_budget"]), {"0", "4096"})
                self.assertEqual(len(report["per_query"]), len(queries))
                zero = report["by_budget"]["0"]
                self.assertEqual(zero["answerable_queries"], 14)
                self.assertEqual(zero["unknown_queries"], 3)
                self.assertEqual(zero["sufficient_evidence_queries"], 0)
                self.assertEqual(zero["sufficient_evidence_rate"], 0.0)

                rows = {row["query_id"]: row for row in report["per_query"]}
                answerable = rows["q01"]
                self.assertEqual(answerable["budgets"]["0"]["packed_message_ids"], [])
                self.assertEqual(answerable["budgets"]["0"]["packed_bytes"], 0)
                self.assertTrue(answerable["budgets"]["0"]["empty_pack"])
                self.assertEqual(answerable["budgets"]["0"]["relevant_message_recall"], 0.0)
                self.assertFalse(answerable["budgets"]["0"]["sufficient_evidence_survives"])
                self.assertTrue(answerable["budgets"]["4096"]["sufficient_evidence_survives"])
                self.assertTrue(answerable["retrieved_citations"])
                self.assertEqual(
                    answerable["budgets"]["4096"]["packed_citations"],
                    answerable["retrieved_citations"],
                )
                self.assertLessEqual(answerable["budgets"]["4096"]["packed_bytes"], 4096)

                unknown = rows["q14"]
                self.assertIsNone(unknown["retrieved_relevant_message_recall"])
                self.assertIsNone(unknown["budgets"]["0"]["relevant_message_recall"])
                self.assertIsNone(unknown["budgets"]["0"]["sufficient_evidence_survives"])
                self.assertTrue(unknown["budgets"]["0"]["empty_pack"])

    def test_zero_answerable_denominator_keeps_answerable_metrics_null(self) -> None:
        from tracemem import EvidenceStore
        from tracemem.benchmark import load_synthetic_data
        from tracemem.budget_benchmark import run_context_budget_benchmark

        records, queries, fixture_sha256 = load_synthetic_data()
        unknown = next(query for query in queries if not query["relevant_message_ids"])
        with TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(records)
                result = run_context_budget_benchmark(
                    store,
                    [unknown],
                    input_sha256=fixture_sha256,
                    budgets_bytes=(0,),
                )

        summary = result["modes"]["bm25"]["by_budget"]["0"]
        self.assertEqual(summary["answerable_queries"], 0)
        self.assertEqual(summary["unknown_queries"], 1)
        self.assertIsNone(summary["mean_relevant_message_recall"])
        self.assertIsNone(summary["sufficient_evidence_rate"])
        self.assertEqual(summary["unknown_empty_packs"], 1)

    def test_budget_and_depth_options_reject_ambiguous_or_unbounded_values(self) -> None:
        from tracemem import EvidenceStore
        from tracemem.budget_benchmark import run_context_budget_benchmark

        invalid_budgets = (
            (),
            (True,),
            (-1,),
            (0, 0),
            tuple(range(33)),
            "256",
        )
        with TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                maximum = run_context_budget_benchmark(
                    store,
                    [],
                    input_sha256="0" * 64,
                    budgets_bytes=tuple(range(32)),
                )
                self.assertEqual(maximum["budgets_bytes"], list(range(32)))
                for budgets in invalid_budgets:
                    with self.subTest(budgets=budgets):
                        with self.assertRaises(ValueError):
                            run_context_budget_benchmark(
                                store,
                                [],
                                input_sha256="0" * 64,
                                budgets_bytes=budgets,
                            )
                for k in (True, 0, 101):
                    with self.subTest(k=k):
                        with self.assertRaisesRegex(ValueError, "k must be an integer"):
                            run_context_budget_benchmark(
                                store,
                                [],
                                input_sha256="0" * 64,
                                budgets_bytes=(0,),
                                k=k,
                            )

    def test_retrieval_config_names_the_selected_embedder(self) -> None:
        from tracemem import EvidenceStore
        from tracemem.budget_benchmark import run_context_budget_benchmark

        class CustomEmbedder:
            @property
            def metadata(self) -> dict[str, str | int]:
                return {"kind": "test-custom-vector", "version": 7}

            def encode(self, text: str) -> list[float]:
                return [1.0, 0.0]

        with TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                result = run_context_budget_benchmark(
                    store,
                    [],
                    input_sha256="0" * 64,
                    budgets_bytes=(0,),
                    embedder=CustomEmbedder(),
                )

        self.assertEqual(result["embedder"]["kind"], "test-custom-vector")
        self.assertEqual(result["retrieval_config"]["vector_embedder"], "test-custom-vector")

    def test_cli_writes_a_new_fixture_only_report_exclusively(self) -> None:
        from experiments import evaluate_context_budgets

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "context-budget.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(evaluate_context_budgets.main(["--output", str(output_path)]), 0)
            original = output_path.read_bytes()

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as failure:
                    evaluate_context_budgets.main(["--output", str(output_path)])
            self.assertEqual(failure.exception.code, 2)
            self.assertEqual(output_path.read_bytes(), original)

        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("nightly database maintenance", original.decode("utf-8"))
        report = json.loads(original)
        self.assertEqual(report["benchmark"], "synthetic-context-budget-diagnostic")
        self.assertEqual(report["budgets_bytes"], [0, 256, 512, 1024, 2048, 4096])
        self.assertNotIn("query", report["modes"]["bm25"]["per_query"][0])


if __name__ == "__main__":
    unittest.main()
