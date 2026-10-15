"""Offline controls for replaying the pinned authored QA contrast receipt."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import (
    modal_qa_contrast_token_count,
    prepare_qa_contrast,
    replay_qa_contrast_tokens,
)
from tracemem.retrieval import Retriever, SignedHashEmbedder, _context_line
from tracemem.storage import EvidenceStore


class QAContrastTokenReplayTests(unittest.TestCase):
    def test_default_macro_recall_table_uses_four_answerable_cells_per_group(self) -> None:
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()
        expected = {
            "bm25": (0.0, 0.25, 0.75, 1.0),
            "vector": (0.0, 0.25, 0.875, 0.875),
            "hybrid": (0.0, 0.125, 0.75, 1.0),
        }

        aggregates = replay_qa_contrast_tokens.replay_from_counts(receipt_raw)["aggregates"]

        self.assertEqual(len(aggregates), 12)
        for mode, values in expected.items():
            rows = [row for row in aggregates if row["mode"] == mode]
            self.assertEqual([row["macro_label_recall"] for row in rows], list(values))
            self.assertTrue(all(row["macro_denominator"] == 4 for row in rows))
            self.assertTrue(all(row["unknown_exclusion_count"] == 2 for row in rows))
            self.assertTrue(all(row["infeasible_exclusion_count"] == 0 for row in rows))

    def test_reference_and_scoring_fields_stay_outside_prompt_messages(self) -> None:
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()

        result = replay_qa_contrast_tokens.replay_from_counts(receipt_raw)

        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("answer_record_ids", serialized)
        self.assertNotIn("authored_answer_record_ids", serialized)
        self.assertNotIn('"answer":', serialized)
        feasible_cells = [cell for cell in result["cells"] if cell["status"] == "feasible"]
        self.assertTrue(feasible_cells)
        for cell in feasible_cells:
            self.assertEqual(set(cell["final_prompt"]), {"prompt_id", "messages"})
            self.assertEqual(
                [set(message) for message in cell["final_prompt"]["messages"]],
                [{"role", "content"}, {"role", "content"}],
            )
            self.assertIn("scoring", cell)
            self.assertNotIn("scoring", cell["final_prompt"])

    def test_budget_validation_rejects_unbounded_or_ambiguous_inputs(self) -> None:
        invalid_budgets = (
            ((), "nonempty tuple"),
            ([], "nonempty tuple"),
            ((True,), "non-negative integers"),
            ((1, 1), "duplicates"),
            ((-1,), "non-negative integers"),
            ((1.5,), "non-negative integers"),
            ((1_000_001,), "no greater than 1,000,000"),
            (tuple(range(17)), "at most 16 values"),
        )
        for budgets, reason in invalid_budgets:
            with self.subTest(budgets=budgets), self.assertRaisesRegex(ValueError, reason):
                replay_qa_contrast_tokens.replay_from_counts(b"", budgets=budgets)  # type: ignore[arg-type]

    def test_missing_exact_prompt_lookup_fails_without_estimation(self) -> None:
        make_counter = getattr(replay_qa_contrast_tokens, "_make_exact_counter", None)
        self.assertTrue(callable(make_counter), "the exact prompt counter is missing")
        fixture_raw = Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        _, questions = prepare_qa_contrast._load_fixture(fixture_raw)

        count_prompt_tokens = make_counter(questions[0], {}, {})

        with self.assertRaisesRegex(ValueError, "missing from the frozen count catalog"):
            count_prompt_tokens("")

    def test_receipt_tampering_and_runner_provenance_changes_are_rejected(self) -> None:
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()
        tampered = receipt_raw.replace(b'"schema_version": 1', b'"schema_version": 2', 1)

        with self.assertRaisesRegex(ValueError, "exact committed fictional receipt"):
            replay_qa_contrast_tokens.replay_from_counts(tampered)

        with tempfile.TemporaryDirectory() as directory:
            changed_runner = Path(directory) / "changed_runner.py"
            changed_runner.write_text("# changed source\n", encoding="utf-8")
            with patch.object(modal_qa_contrast_token_count, "__file__", str(changed_runner)):
                with self.assertRaisesRegex(ValueError, "payload or runner provenance"):
                    replay_qa_contrast_tokens.replay_from_counts(receipt_raw)

    def test_infeasible_and_unknown_cells_are_separate_from_recall_denominators(self) -> None:
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()

        result = replay_qa_contrast_tokens.replay_from_counts(receipt_raw, budgets=(0, 128))

        for mode in ("bm25", "vector", "hybrid"):
            zero_budget_cells = [
                cell for cell in result["cells"] if cell["mode"] == mode and cell["ceiling"] == 0
            ]
            self.assertEqual(len(zero_budget_cells), 6)
            self.assertTrue(
                all(cell["status"] == "infeasible_empty_prompt" for cell in zero_budget_cells)
            )
            self.assertTrue(
                all(cell["scoring"]["label_recall"] is None for cell in zero_budget_cells)
            )
            aggregate = next(
                row for row in result["aggregates"] if row["mode"] == mode and row["ceiling"] == 0
            )
            self.assertEqual(aggregate["macro_denominator"], 0)
            self.assertIsNone(aggregate["macro_label_recall"])
            self.assertEqual(aggregate["unknown_exclusion_count"], 2)
            self.assertEqual(aggregate["infeasible_exclusion_count"], 6)
            self.assertEqual(aggregate["infeasible_answerable_exclusion_count"], 4)

        unknown_cells = [
            cell
            for cell in result["cells"]
            if cell["ceiling"] == 128 and cell["scoring"]["explicit_label_count"] == 0
        ]
        self.assertEqual(len(unknown_cells), 6)
        self.assertTrue(
            all(
                cell["scoring"]["exclusion_reason"] == "abstention_has_no_positive_labels"
                and cell["scoring"]["label_recall"] is None
                for cell in unknown_cells
            )
        )

    def test_cells_match_independent_greedy_packing_and_exact_receipt_counts(self) -> None:
        fixture_raw = Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()
        result = replay_qa_contrast_tokens.replay_from_counts(receipt_raw)
        payload = modal_qa_contrast_token_count.build_payload()
        messages_by_id = {row["prompt_id"]: row["messages"] for row in payload["items"]}
        receipt = json.loads(receipt_raw)
        counts = {
            row["prompt_id"]: row["prompt_tokens"]
            for row in receipt["result"]["count_receipt"]["counts"]
        }
        _, questions = prepare_qa_contrast._load_fixture(fixture_raw)
        result_cells = {cell["cell_id"]: cell for cell in result["cells"]}
        skip_then_accept = 0

        for question in questions:
            with tempfile.TemporaryDirectory() as directory:
                with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                    store.ingest(question.records)
                    retriever = Retriever(store, SignedHashEmbedder(), rrf_k=60)
                    for mode in ("bm25", "vector", "hybrid"):
                        hits = retriever.search(question.question, mode=mode, k=5)

                        def exact_count(context: str, current_question=question) -> int:
                            messages = prepare_qa_contrast._messages(current_question, context)
                            prompt_id = hashlib.sha256(
                                json.dumps(
                                    messages,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ).encode()
                            ).hexdigest()
                            self.assertEqual(messages_by_id[prompt_id], messages)
                            return counts[prompt_id]

                        for ceiling in (128, 256, 512, 1024):
                            cell = result_cells[f"{question.question_id}:{mode}:{ceiling}"]
                            empty_tokens = exact_count("")
                            self.assertEqual(cell["empty_prompt_tokens"], empty_tokens)
                            if empty_tokens > ceiling:
                                self.assertEqual(cell["status"], "infeasible_empty_prompt")
                                continue

                            selected = []
                            lines: list[str] = []
                            skipped = False
                            for hit in hits:
                                candidate = "\n".join((*lines, _context_line(hit)))
                                if exact_count(candidate) <= ceiling:
                                    lines.append(_context_line(hit))
                                    selected.append(hit)
                                    if skipped:
                                        skip_then_accept += 1
                                else:
                                    skipped = True
                            context = "\n".join(lines)
                            messages = prepare_qa_contrast._messages(question, context)
                            prompt_id = hashlib.sha256(
                                json.dumps(
                                    messages,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ).encode()
                            ).hexdigest()
                            self.assertEqual(cell["status"], "feasible")
                            self.assertEqual(cell["used_tokens"], exact_count(context))
                            self.assertEqual(
                                cell["selected_citation_ids"], [h.citation for h in selected]
                            )
                            selected_ids = [hit.record.record_id for hit in selected]
                            self.assertEqual(cell["selected_record_ids"], selected_ids)
                            labels = set(question.answer_record_ids)
                            found = sum(record_id in labels for record_id in selected_ids)
                            scoring = cell["scoring"]
                            self.assertEqual(scoring["explicit_label_count"], len(labels))
                            self.assertEqual(scoring["found_label_count"], found)
                            self.assertEqual(
                                scoring["label_recall"],
                                None if question.is_abstention else found / len(labels),
                            )
                            self.assertEqual(
                                cell["final_prompt"], {"prompt_id": prompt_id, "messages": messages}
                            )

        self.assertEqual(skip_then_accept, 3)

    def test_provenance_labels_results_as_dev_authored_and_provisional(self) -> None:
        replay = getattr(replay_qa_contrast_tokens, "replay_from_counts", None)
        self.assertTrue(callable(replay), "the offline replay API is missing")
        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()

        provenance = replay(receipt_raw)["provenance"]

        self.assertTrue(provenance["dev_only"])
        self.assertIn("authored_agent_only", provenance)
        self.assertTrue(provenance["authored_agent_only"])
        self.assertTrue(provenance["provisional"])

    def test_replay_api_returns_the_full_72_cell_grid(self) -> None:
        replay = getattr(replay_qa_contrast_tokens, "replay_from_counts", None)
        self.assertTrue(callable(replay), "the offline replay API is missing")

        receipt_raw = Path(
            "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
        ).read_bytes()
        result = replay(receipt_raw)

        self.assertEqual(len(result["cells"]), 72)
        self.assertEqual(
            result["provenance"]["receipt_sha256"], hashlib.sha256(receipt_raw).hexdigest()
        )
        self.assertEqual(
            {(cell["mode"], cell["ceiling"]) for cell in result["cells"]},
            {
                (mode, ceiling)
                for mode in ("bm25", "vector", "hybrid")
                for ceiling in (128, 256, 512, 1024)
            },
        )


if __name__ == "__main__":
    unittest.main()
