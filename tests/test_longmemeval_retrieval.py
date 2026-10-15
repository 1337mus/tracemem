"""Protocol tests for the fixture-only LongMemEval turn-retrieval diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import evaluate_longmemeval_retrieval as evaluator

_MODES = ("bm25", "vector", "hybrid")
_NAMESPACE = "private-fictional-namespace"


def _question(
    question_id: str,
    query: str,
    session_id: str | None,
    turns: list[dict[str, Any]] | None,
    *,
    answer_session_ids: list[str] | None = None,
    answer: str = "PRIVATE_ANSWER_SENTINEL",
) -> dict[str, Any]:
    if session_id is None:
        session_ids: list[str] = []
        dates: list[str] = []
        sessions: list[list[dict[str, Any]]] = []
    else:
        session_ids = [session_id]
        dates = ["2035-08-09"]
        sessions = [turns or []]
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": query,
        "answer": answer,
        "question_date": "2035-08-10",
        "haystack_session_ids": session_ids,
        "haystack_dates": dates,
        "haystack_sessions": sessions,
        "answer_session_ids": answer_session_ids or [],
    }


def _fictional_items() -> list[dict[str, Any]]:
    return [
        _question(
            "fictional-multi-label",
            "PRIVATE_QUERY_SENTINEL: Which color was the blue bird, red kite?",
            "fictional-shared-session",
            [
                {
                    "role": "user",
                    "content": "The blue bird and red kite were cobalt.",
                    "has_answer": True,
                },
                {
                    "role": "assistant",
                    "content": "A red kite beside the blue bird was amber.",
                    "has_answer": True,
                },
            ],
            answer_session_ids=["fictional-shared-session"],
        ),
        _question(
            "fictional-question-two",
            "Ceramic compass, velvet observatory?",
            "fictional-shared-session",
            [
                {
                    "role": "user",
                    "content": "Ceramic compass: velvet observatory.",
                    "has_answer": True,
                }
            ],
            answer_session_ids=["fictional-shared-session"],
        ),
        _question(
            "fictional-unlabeled",
            "Find the violet folder in the quiet cabinet?",
            "fictional-unlabeled-session",
            [{"role": "user", "content": "The violet folder is in the quiet cabinet."}],
            answer_session_ids=["fictional-unlabeled-session"],
        ),
        _question("fictional-empty-history", "Where is the empty archive?", None, None),
        _question(
            "fictional-abstention_abs",
            "Does the imaginary silver tunnel exist?",
            "fictional-abstention-session",
            [
                {
                    "role": "user",
                    "content": "A fictional silver tunnel exists behind the mural.",
                    "has_answer": True,
                }
            ],
            answer="",
        ),
        _question(
            "fictional-unicode",
            "Where is café 東京?",
            "fictional-unicode-session",
            [
                {
                    "role": "user",
                    "content": "The café 東京 sign glows beside the 🧪 window.",
                    "has_answer": True,
                }
            ],
            answer_session_ids=["fictional-unicode-session"],
        ),
    ]


def _raw(items: object | None = None) -> bytes:
    return json.dumps(
        _fictional_items() if items is None else items, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _question_report(report: dict[str, Any], question_id: str) -> dict[str, Any]:
    return next(item for item in report["questions"] if item["question_id"] == question_id)


def _cell(report: dict[str, Any], question_id: str, mode: str, cutoff: int) -> dict[str, Any]:
    question = _question_report(report, question_id)
    return next(item for item in question["retrieval"][mode] if item["cutoff"] == cutoff)


def _aggregate(report: dict[str, Any], mode: str, cutoff: int) -> dict[str, Any]:
    return next(item for item in report["aggregates"][mode] if item["cutoff"] == cutoff)


class LongMemEvalRetrievalTests(unittest.TestCase):
    def test_injected_embedder_controls_vector_ranking_without_changing_bm25(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.remote_vectors import PrecomputedEmbedder, text_sha256

        items = [
            _question(
                "fictional-controlled-embedding",
                "Which fictional object was selected?",
                "fictional-controlled-session",
                [
                    {"role": "user", "content": "Fictional evidence answer.", "has_answer": True},
                    {"role": "user", "content": "Fictional distractor text."},
                ],
            )
        ]
        raw = _raw(items)
        question = parse_longmemeval_bytes(raw, namespace=_NAMESPACE, variant="s")[0]
        embedder = PrecomputedEmbedder(
            model_id="synthetic/test-only",
            revision="test-revision",
            dimensions=2,
            vectors={
                text_sha256(question.question): [1.0, 0.0],
                text_sha256(question.records[0].text): [1.0, 0.0],
                text_sha256(question.records[1].text): [0.0, 1.0],
            },
        )

        default = evaluator.evaluate_retrieval_bytes(
            raw, namespace=_NAMESPACE, variant="s", cutoffs=(1, 2)
        )
        measured = evaluator.evaluate_retrieval_bytes(
            raw,
            namespace=_NAMESPACE,
            variant="s",
            cutoffs=(1, 2),
            embedder=embedder,
        )

        default_question, measured_question = default["questions"][0], measured["questions"][0]
        self.assertEqual(
            measured_question["retrieval"]["bm25"], default_question["retrieval"]["bm25"]
        )
        vector_at_one = measured_question["retrieval"]["vector"][0]
        self.assertEqual(vector_at_one["returned"][0]["record_id"], question.records[0].record_id)
        self.assertEqual(vector_at_one["label_recall"], 1.0)
        self.assertEqual(measured["embedding"]["metadata"], embedder.metadata)

    def test_injected_embedder_missing_vector_fails_without_fallback(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        raw = _raw(
            [
                _question(
                    "fictional-missing-vector",
                    "Which fictional color was selected?",
                    "fictional-missing-vector-session",
                    [{"role": "user", "content": "A fictional color was violet."}],
                )
            ]
        )
        embedder = PrecomputedEmbedder(
            model_id="synthetic/test-only",
            revision="test-revision",
            dimensions=2,
            vectors={},
        )

        with self.assertRaisesRegex(ValueError, "no precomputed vector"):
            evaluator.evaluate_retrieval_bytes(
                raw,
                namespace=_NAMESPACE,
                variant="s",
                cutoffs=(1,),
                embedder=embedder,
            )

    def test_empty_question_evaluation_keeps_injected_embedder_metadata(self) -> None:
        from tracemem.remote_vectors import PrecomputedEmbedder

        embedder = PrecomputedEmbedder(
            model_id="synthetic/test-only",
            revision="test-revision",
            dimensions=2,
            vectors={},
        )

        report = evaluator._evaluate_questions(
            (),
            input_sha256="0" * 64,
            variant="s",
            cutoffs=(1,),
            embedder=embedder,
        )

        self.assertEqual(report["embedding"]["metadata"], embedder.metadata)
        self.assertEqual(report["questions"], [])
        self.assertEqual(report["counts"]["question_count"], 0)

    def test_file_api_passes_injected_embedder_through(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.remote_vectors import PrecomputedEmbedder, text_sha256

        raw = _raw(
            [
                _question(
                    "fictional-file-embedding",
                    "Which fictional object was selected?",
                    "fictional-file-embedding-session",
                    [
                        {
                            "role": "user",
                            "content": "Fictional evidence answer.",
                            "has_answer": True,
                        },
                        {"role": "user", "content": "Fictional distractor text."},
                    ],
                )
            ]
        )
        question = parse_longmemeval_bytes(raw, namespace=_NAMESPACE, variant="s")[0]
        embedder = PrecomputedEmbedder(
            model_id="synthetic/test-only",
            revision="test-revision",
            dimensions=2,
            vectors={
                text_sha256(question.question): [1.0, 0.0],
                text_sha256(question.records[0].text): [1.0, 0.0],
                text_sha256(question.records[1].text): [0.0, 1.0],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_bytes(raw)

            report = evaluator.evaluate_retrieval_file(
                path,
                namespace=_NAMESPACE,
                variant="s",
                question_ids=("fictional-file-embedding",),
                cutoffs=(1, 2),
                embedder=embedder,
            )

        self.assertEqual(report["embedding"]["metadata"], embedder.metadata)
        self.assertEqual(
            report["questions"][0]["retrieval"]["vector"][0]["returned"][0]["record_id"],
            question.records[0].record_id,
        )

    def test_partial_and_complete_multilabel_turn_retrieval(self) -> None:
        raw = _raw()

        report = evaluator.evaluate_retrieval_bytes(
            raw, namespace=_NAMESPACE, variant="s", cutoffs=(2, 1)
        )

        self.assertEqual(report["cutoffs"], [2, 1])
        self.assertEqual(report["modes"], list(_MODES))
        self.assertEqual(report["input_sha256"], hashlib.sha256(raw).hexdigest())
        for mode in _MODES:
            complete = _cell(report, "fictional-multi-label", mode, 2)
            partial = _cell(report, "fictional-multi-label", mode, 1)
            self.assertEqual(complete["explicit_label_count"], 2)
            self.assertEqual(complete["found_label_count"], 2)
            self.assertEqual(complete["label_recall"], 1.0)
            self.assertTrue(complete["all_labeled_turns_found"])
            self.assertEqual(len(complete["returned"]), 2)
            self.assertEqual(partial["explicit_label_count"], 2)
            self.assertEqual(partial["found_label_count"], 1)
            self.assertEqual(partial["label_recall"], 0.5)
            self.assertFalse(partial["all_labeled_turns_found"])
            self.assertEqual(len(partial["returned"]), 1)
            for hit in complete["returned"]:
                self.assertEqual(set(hit), {"record_id", "revision", "citation"})

    def test_counts_exclusions_aggregates_and_private_content_boundary(self) -> None:
        raw = _raw()

        report = evaluator.evaluate_retrieval_bytes(
            raw, namespace=_NAMESPACE, variant="s", cutoffs=(1, 2, 5)
        )

        self.assertEqual(
            report["counts"],
            {
                "question_count": 6,
                "abstention_count": 1,
                "unlabeled_count": 2,
                "scored_count": 3,
            },
        )
        for mode in _MODES:
            abstention = _cell(report, "fictional-abstention_abs", mode, 5)
            unlabeled = _cell(report, "fictional-unlabeled", mode, 5)
            empty = _cell(report, "fictional-empty-history", mode, 5)
            self.assertEqual(abstention["exclusion_reason"], "abstention")
            self.assertEqual(abstention["explicit_label_count"], 1)
            self.assertIsNone(abstention["label_recall"])
            self.assertIsNone(abstention["all_labeled_turns_found"])
            self.assertEqual(unlabeled["exclusion_reason"], "missing_turn_labels")
            self.assertEqual(unlabeled["explicit_label_count"], 0)
            self.assertGreater(len(unlabeled["returned"]), 0)
            self.assertIsNone(unlabeled["label_recall"])
            self.assertIsNone(unlabeled["all_labeled_turns_found"])
            self.assertEqual(empty["exclusion_reason"], "missing_turn_labels")
            self.assertEqual(empty["returned"], [])
            aggregate = _aggregate(report, mode, 5)
            self.assertEqual(aggregate["question_count"], 6)
            self.assertEqual(aggregate["scored_question_count"], 3)
            self.assertEqual(aggregate["nonempty_retrieval_question_count"], 5)
            self.assertIsInstance(aggregate["macro_label_recall"], float)
            self.assertIsInstance(aggregate["all_labels_found_fraction"], float)
            cutoff_one = _aggregate(report, mode, 1)
            self.assertEqual(cutoff_one["scored_question_count"], 3)
            self.assertAlmostEqual(cutoff_one["macro_label_recall"], 5 / 6)
            self.assertAlmostEqual(cutoff_one["all_labels_found_fraction"], 2 / 3)
            cutoff_two = _aggregate(report, mode, 2)
            self.assertEqual(cutoff_two["scored_question_count"], 3)
            self.assertEqual(cutoff_two["macro_label_recall"], 1.0)
            self.assertEqual(cutoff_two["all_labels_found_fraction"], 1.0)

        serialized = json.dumps(report, ensure_ascii=False)
        for private_text in (
            "PRIVATE_QUERY_SENTINEL",
            "PRIVATE_ANSWER_SENTINEL",
            "The blue bird and red kite were cobalt.",
            "2035-08-09",
            _NAMESPACE,
        ):
            self.assertNotIn(private_text, serialized)

    def test_zero_scored_denominator_is_null(self) -> None:
        only_unlabeled = [
            _question(
                "fictional-no-labels",
                "Which unused token?",
                None,
                None,
            )
        ]
        report = evaluator.evaluate_retrieval_bytes(
            _raw(only_unlabeled), namespace=_NAMESPACE, variant="m", cutoffs=(3,)
        )

        for mode in _MODES:
            aggregate = _aggregate(report, mode, 3)
            self.assertEqual(aggregate["scored_question_count"], 0)
            self.assertIsNone(aggregate["macro_label_recall"])
            self.assertIsNone(aggregate["all_labels_found_fraction"])

    def test_punctuation_only_query_is_a_valid_empty_retrieval(self) -> None:
        punctuation_question = _question(
            "fictional-punctuation-query",
            "???",
            "fictional-punctuation-session",
            [
                {
                    "role": "user",
                    "content": "A fictional phrase belongs in this record.",
                    "has_answer": True,
                }
            ],
        )
        report = evaluator.evaluate_retrieval_bytes(
            _raw([punctuation_question]), namespace=_NAMESPACE, variant="s", cutoffs=(3,)
        )

        for mode in _MODES:
            cell = _cell(report, "fictional-punctuation-query", mode, 3)
            self.assertEqual(cell["returned"], [])
            self.assertEqual(cell["found_label_count"], 0)
            self.assertEqual(cell["label_recall"], 0.0)
            aggregate = _aggregate(report, mode, 3)
            self.assertEqual(aggregate["scored_question_count"], 1)
            self.assertEqual(aggregate["macro_label_recall"], 0.0)
            self.assertEqual(aggregate["all_labels_found_fraction"], 0.0)

    def test_empty_input_reports_zero_counts_and_null_scores(self) -> None:
        report = evaluator.evaluate_retrieval_bytes(b"[]", namespace=_NAMESPACE, variant="oracle")

        self.assertEqual(
            report["counts"],
            {
                "question_count": 0,
                "abstention_count": 0,
                "unlabeled_count": 0,
                "scored_count": 0,
            },
        )
        for mode in _MODES:
            aggregate = _aggregate(report, mode, 5)
            self.assertIsNone(aggregate["macro_label_recall"])
            self.assertIsNone(aggregate["all_labels_found_fraction"])
            self.assertEqual(aggregate["nonempty_retrieval_question_count"], 0)

    def test_evaluator_isolates_questions_even_when_prior_corpus_matches_query(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.retrieval import Retriever
        from tracemem.storage import EvidenceStore

        items = [
            _question(
                "fictional-matching-source-first",
                "Ceramic compass, velvet observatory?",
                "fictional-matching-source-session",
                [
                    {
                        "role": "user",
                        "content": "Ceramic compass: velvet observatory.",
                        "has_answer": True,
                    }
                ],
            ),
            _question(
                "fictional-later-isolated-question",
                "Ceramic compass, velvet observatory?",
                "fictional-unrelated-session",
                [
                    {
                        "role": "user",
                        "content": "A garnet snowflake rests below the mirror.",
                        "has_answer": True,
                    }
                ],
            ),
        ]
        questions = parse_longmemeval_bytes(_raw(items), namespace=_NAMESPACE, variant="s")
        source_first, isolated_question = questions
        source_ids = {record.record_id for record in source_first.records}

        report = evaluator.evaluate_retrieval_bytes(
            _raw(items), namespace=_NAMESPACE, variant="s", cutoffs=(1,)
        )
        for mode in _MODES:
            source_cell = _cell(report, source_first.question_id, mode, 1)
            self.assertEqual({hit["record_id"] for hit in source_cell["returned"]}, source_ids)
            isolated_cell = _cell(report, isolated_question.question_id, mode, 1)
            self.assertEqual(isolated_cell["returned"], [])
            self.assertEqual(isolated_cell["found_label_count"], 0)
            self.assertEqual(isolated_cell["label_recall"], 0.0)

        # A shared corpus really would contaminate the later question for these queries.
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "combined.sqlite") as combined_store:
                combined_store.ingest(source_first.records + isolated_question.records)
                combined_retriever = Retriever(combined_store)
                for mode in _MODES:
                    combined_hits = combined_retriever.search(
                        isolated_question.question, mode=mode, k=5
                    )
                    self.assertEqual({hit.record.record_id for hit in combined_hits}, source_ids)

    def test_reference_only_changes_do_not_change_rankings(self) -> None:
        original = _fictional_items()
        edited = deepcopy(original)
        edited[0]["answer"] = "A different fictional answer."
        edited[0]["question_type"] = "knowledge-update"
        edited[0]["question_date"] = "2036-01-02"
        edited[0]["answer_session_ids"] = []
        for turn in edited[0]["haystack_sessions"][0]:
            turn.pop("has_answer", None)

        before = evaluator.evaluate_retrieval_bytes(
            _raw(original), namespace=_NAMESPACE, variant="s", cutoffs=(1, 2, 5)
        )
        after = evaluator.evaluate_retrieval_bytes(
            _raw(edited), namespace=_NAMESPACE, variant="s", cutoffs=(1, 2, 5)
        )

        for question_id in ("fictional-multi-label", "fictional-question-two"):
            for mode in _MODES:
                for cutoff in (1, 2, 5):
                    self.assertEqual(
                        _cell(before, question_id, mode, cutoff)["returned"],
                        _cell(after, question_id, mode, cutoff)["returned"],
                    )

    def test_report_citation_resolves_exact_unicode_record_and_revision(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.citations import fetch_citation
        from tracemem.storage import EvidenceStore

        questions = parse_longmemeval_bytes(_raw(), namespace=_NAMESPACE, variant="s")
        question = questions[-1]
        report = evaluator.evaluate_retrieval_bytes(
            _raw(), namespace=_NAMESPACE, variant="s", cutoffs=(5,)
        )
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "unicode.sqlite") as store:
                store.ingest(question.records)
                expected_record = question.records[0]
                for mode in _MODES:
                    returned = _cell(report, question.question_id, mode, 5)["returned"]
                    self.assertEqual(len(returned), 1)
                    hit = returned[0]
                    resolved = fetch_citation(store, hit["citation"])
                    self.assertEqual(resolved.record.record_id, expected_record.record_id)
                    self.assertEqual(resolved.record.revision, expected_record.revision)
                    self.assertEqual(hit["record_id"], expected_record.record_id)
                    self.assertEqual(hit["revision"], expected_record.revision)
                    self.assertEqual(
                        resolved.quote,
                        "The café 東京 sign glows beside the 🧪 window.",
                    )

    def test_empty_and_nonempty_reports_match_configured_retriever_metadata(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.retrieval import Retriever
        from tracemem.storage import EvidenceStore

        raw = _raw([_fictional_items()[0]])
        question = parse_longmemeval_bytes(raw, namespace=_NAMESPACE, variant="s")[0]
        empty_report = evaluator.evaluate_retrieval_bytes(b"[]", namespace=_NAMESPACE, variant="s")
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "metadata.sqlite") as store:
                store.ingest(question.records)
                configured_retriever = Retriever(store, rrf_k=evaluator._DEFAULT_RRF_K)
                expected_embedding = {
                    "metadata": configured_retriever.embedder.metadata,
                    "fingerprint": configured_retriever.embedder_fingerprint,
                }

        with patch.object(evaluator, "Retriever", wraps=Retriever) as retriever_factory:
            nonempty_report = evaluator.evaluate_retrieval_bytes(
                raw, namespace=_NAMESPACE, variant="s"
            )

        self.assertEqual(retriever_factory.call_count, 1)
        self.assertEqual(
            retriever_factory.call_args.kwargs,
            {"rrf_k": evaluator._DEFAULT_RRF_K},
        )
        self.assertEqual(empty_report["embedding"], expected_embedding)
        self.assertEqual(nonempty_report["embedding"], expected_embedding)
        self.assertEqual(empty_report["rrf_k"], configured_retriever.rrf_k)
        self.assertEqual(nonempty_report["rrf_k"], configured_retriever.rrf_k)

    def test_report_replay_is_deterministic_except_environment(self) -> None:
        raw = _raw()
        first = evaluator.evaluate_retrieval_bytes(raw, namespace=_NAMESPACE, variant="s")
        second = evaluator.evaluate_retrieval_bytes(raw, namespace=_NAMESPACE, variant="s")
        first.pop("environment")
        second.pop("environment")
        self.assertEqual(first, second)

    def test_cutoffs_are_validated_before_database_creation(self) -> None:
        invalid_cutoffs: tuple[object, ...] = (
            (),
            (1, 1),
            (True,),
            (0,),
            (101,),
            (1.0,),
        )
        for cutoffs in invalid_cutoffs:
            with self.subTest(cutoffs=cutoffs):
                with patch.object(
                    evaluator.tempfile,
                    "TemporaryDirectory",
                    side_effect=AssertionError("SQLite temp directory created"),
                ):
                    with self.assertRaisesRegex(ValueError, "cutoffs"):
                        evaluator.evaluate_retrieval_bytes(
                            b"not JSON",
                            namespace=_NAMESPACE,
                            variant="s",
                            cutoffs=cutoffs,  # type: ignore[arg-type]
                        )

    def test_overlong_query_is_rejected_before_database_creation(self) -> None:
        items = [
            _question(
                "fictional-long-query",
                "q" * 4097,
                "long-query-session",
                [{"role": "user", "content": "a fictional answer label", "has_answer": True}],
                answer_session_ids=["long-query-session"],
            )
        ]
        with patch.object(
            evaluator.tempfile,
            "TemporaryDirectory",
            side_effect=AssertionError("SQLite temp directory created"),
        ):
            with self.assertRaisesRegex(ValueError, "4096-character limit") as failure:
                evaluator.evaluate_retrieval_bytes(_raw(items), namespace=_NAMESPACE, variant="s")
        self.assertNotIn("q" * 4097, str(failure.exception))

    def test_cli_writes_private_exclusive_report_and_preserves_input(self) -> None:
        raw = _raw()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fictional.json"
            output_path = Path(directory) / "report.json"
            input_path.write_bytes(raw)

            result = evaluator.main(
                [
                    "--input",
                    str(input_path),
                    "--namespace",
                    _NAMESPACE,
                    "--variant",
                    "s",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(input_path.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["input_sha256"], hashlib.sha256(raw).hexdigest())

            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                evaluator.main(
                    [
                        "--input",
                        str(input_path),
                        "--namespace",
                        _NAMESPACE,
                        "--variant",
                        "s",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )

    def test_cli_rejects_existing_symlink_and_invalid_input_without_output(self) -> None:
        raw = _raw()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fictional.json"
            input_path.write_bytes(raw)
            target = root / "target.json"
            output_link = root / "output-link.json"
            output_link.symlink_to(target)

            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                evaluator.main(
                    [
                        "--input",
                        str(input_path),
                        "--namespace",
                        _NAMESPACE,
                        "--variant",
                        "s",
                        "--output",
                        str(output_link),
                    ]
                )
            self.assertTrue(output_link.is_symlink())
            self.assertFalse(target.exists())

            invalid_input = root / "invalid.json"
            invalid_input.write_bytes(b"not json")
            no_output = root / "must-not-exist.json"
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                evaluator.main(
                    [
                        "--input",
                        str(invalid_input),
                        "--namespace",
                        _NAMESPACE,
                        "--variant",
                        "s",
                        "--output",
                        str(no_output),
                    ]
                )
            self.assertFalse(no_output.exists())
            self.assertNotIn("not json", stderr.getvalue())
            self.assertNotIn("PRIVATE_QUERY_SENTINEL", stderr.getvalue())

    @unittest.skipUnless(hasattr(os, "mkfifo") and os.name == "posix", "requires POSIX FIFO")
    def test_cli_fifo_is_rejected_without_blocking_and_regular_file_is_accepted(self) -> None:
        raw = _raw()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "input.fifo"
            os.mkfifo(fifo)
            output = root / "fifo-report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.evaluate_longmemeval_retrieval",
                    "--input",
                    str(fifo),
                    "--namespace",
                    _NAMESPACE,
                    "--variant",
                    "s",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                check=False,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(b"regular file", completed.stderr)
            self.assertFalse(output.exists())

            regular = root / "input.json"
            regular.write_bytes(raw)
            completed_regular = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.evaluate_longmemeval_retrieval",
                    "--input",
                    str(regular),
                    "--namespace",
                    _NAMESPACE,
                    "--variant",
                    "s",
                    "--output",
                    str(root / "regular-report.json"),
                ],
                capture_output=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed_regular.returncode, 0, completed_regular.stderr.decode())

    def test_cli_rejects_input_above_adapter_limit_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "oversized.json"
            with input_path.open("wb") as stream:
                stream.truncate(64 * 1024 * 1024 + 1)
            output_path = Path(directory) / "report.json"
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                evaluator.main(
                    [
                        "--input",
                        str(input_path),
                        "--namespace",
                        _NAMESPACE,
                        "--variant",
                        "s",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertFalse(output_path.exists())
            self.assertIn("64 MiB", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
