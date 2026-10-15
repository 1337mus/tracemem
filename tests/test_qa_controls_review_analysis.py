from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import analyze_qa_controls_reviews as _analysis
from experiments import prepare_qa_controls_reviews as _reviews
from experiments import qa_interpretation_answer_receipt as _receipt_contract
from tests.test_qa_interpretation_answer_analysis import _fake_receipt


def _complete_document(
    receipt: dict[str, object],
    *,
    seed: int = 17,
    reviewer_id: str = "reviewer-a",
    reviewer_kind: str = "human",
) -> dict[str, object]:
    _bundle, key, template = _reviews.prepare_reviews(receipt, seed=seed)
    document = copy.deepcopy(template)
    task_keys = sorted(key["task_keys"], key=lambda row: row["task_id"])
    positions: dict[str, int] = {}
    label_pairs = [
        (source, support)
        for source, counts in zip(
            ("correct", "incorrect", "uncertain"),
            ((1, 1, 1), (1, 1, 2), (2, 1, 2)),
            strict=True,
        )
        for support, count in zip(("supported", "unsupported", "uncertain"), counts, strict=True)
        for _ in range(count)
    ]
    for task_key in task_keys:
        variant_id = task_key["variant_id"]
        position = positions.get(variant_id, 0)
        positions[variant_id] = position + 1
        source_label, support_label = label_pairs[position]
        row = next(row for row in document["responses"] if row["task_id"] == task_key["task_id"])
        row.update(
            {
                "source_answer": source_label,
                "prompt_support": support_label,
                "rationale": f"synthetic rationale {position}",
                "reviewer_id": reviewer_id,
                "reviewer_kind": reviewer_kind,
                "timestamp": "2026-01-02T03:04:05Z",
            }
        )
    return document


class QAControlsReviewAnalysisTests(unittest.TestCase):
    def test_valid_response_returns_detached_rows_in_bundle_order(self) -> None:
        receipt = _fake_receipt()
        bundle, _key, _template = _reviews.prepare_reviews(receipt, seed=17)
        document = _complete_document(receipt)
        before = copy.deepcopy(document)

        rows = _analysis.validate_responses(receipt, document, seed=17)

        self.assertEqual(
            [row["task_id"] for row in rows], [task["task_id"] for task in bundle["tasks"]]
        )
        self.assertEqual(document, before)
        rows[0]["rationale"] = "changed result"
        self.assertEqual(document["responses"][0]["rationale"], before["responses"][0]["rationale"])
        document["responses"][1]["rationale"] = "changed input"
        self.assertNotEqual(rows[1]["rationale"], document["responses"][1]["rationale"])

    def test_preparer_source_hash_is_checked_before_packet_generation(self) -> None:
        self.assertEqual(
            _analysis._PREPARER_SHA256,
            "e4300d44839ff816721215b6b8aa0f02a38082a1b2f23e6205d361a35bc04924",
        )
        with tempfile.TemporaryDirectory() as directory:
            changed_source = Path(directory) / "changed-preparer.py"
            changed_source.write_text("changed source", encoding="utf-8")
            with patch.object(_reviews, "__file__", str(changed_source)):
                with self.assertRaisesRegex(
                    ValueError, "pinned QA controls review preparer source hash"
                ):
                    _analysis.validate_responses({}, {}, seed=17)

    def test_rejects_malformed_response_documents_with_specific_reasons(self) -> None:
        receipt = _fake_receipt()
        base = _complete_document(receipt)
        _bundle, _key, blank_template = _reviews.prepare_reviews(receipt, seed=17)
        cases = [
            (
                "blank template",
                lambda document: document.update(responses=blank_template["responses"]),
                "source_answer label is invalid",
            ),
            (
                "changed allowed choices",
                lambda document: document["allowed_choices"].update(reviewer_kind=["agent"]),
                "allowed_choices differs from response template",
            ),
            (
                "boolean version",
                lambda document: document.update(version=True),
                "unsupported response document kind or version",
            ),
            (
                "changed kind",
                lambda document: document.update(kind="other-kind"),
                "unsupported response document kind or version",
            ),
            (
                "bundle hash mismatch",
                lambda document: document.update(bundle_sha256="0" * 64),
                "response bundle hash does not match",
            ),
            (
                "unexpected document field",
                lambda document: document.update(extra=True),
                "response document has missing or unexpected fields",
            ),
            (
                "unexpected row field",
                lambda document: document["responses"][0].update(extra=True),
                "response row has missing or unexpected fields",
            ),
            (
                "incomplete rows",
                lambda document: document["responses"].pop(),
                "missing response task ID",
            ),
            (
                "duplicate task ID",
                lambda document: document["responses"][1].update(
                    task_id=document["responses"][0]["task_id"]
                ),
                "duplicate response task ID",
            ),
            (
                "foreign task ID",
                lambda document: document["responses"][0].update(task_id="foreign-task"),
                "unknown response task ID",
            ),
            (
                "unknown source label",
                lambda document: document["responses"][0].update(source_answer="maybe"),
                "source_answer label is invalid",
            ),
            (
                "non-string support label",
                lambda document: document["responses"][0].update(prompt_support=None),
                "prompt_support label is invalid",
            ),
            (
                "blank rationale",
                lambda document: document["responses"][0].update(rationale="  \n"),
                "rationale must be non-blank text",
            ),
            (
                "oversized rationale",
                lambda document: document["responses"][0].update(rationale="x" * 4001),
                "rationale must be non-blank text of at most 4000",
            ),
            (
                "blank reviewer ID",
                lambda document: document["responses"][0].update(reviewer_id=" \t "),
                "reviewer_id must be non-blank text",
            ),
            (
                "oversized reviewer ID",
                lambda document: document["responses"][0].update(reviewer_id="r" * 257),
                "reviewer_id must be non-blank text of at most 256",
            ),
            (
                "invalid reviewer kind",
                lambda document: document["responses"][0].update(reviewer_kind="human-ish"),
                "reviewer_kind must be human or agent",
            ),
            (
                "mixed reviewer IDs",
                lambda document: document["responses"][1].update(reviewer_id="reviewer-b"),
                "single reviewer ID and kind",
            ),
            (
                "mixed reviewer kinds",
                lambda document: document["responses"][1].update(reviewer_kind="agent"),
                "single reviewer ID and kind",
            ),
            (
                "non-UTC timestamp",
                lambda document: document["responses"][0].update(
                    timestamp="2026-01-02T03:04:05+00:00"
                ),
                "timestamp must use canonical UTC format",
            ),
            (
                "invalid UTC date",
                lambda document: document["responses"][0].update(timestamp="2026-02-30T03:04:05Z"),
                "timestamp is not a valid UTC datetime",
            ),
        ]
        for name, mutate, reason in cases:
            with self.subTest(case=name):
                document = copy.deepcopy(base)
                mutate(document)
                with self.assertRaisesRegex(ValueError, reason):
                    _analysis.validate_responses(receipt, document, seed=17)

    def test_reports_descriptive_counts_by_reviewer_and_frozen_variant_mapping(self) -> None:
        receipt = _fake_receipt()
        document = _complete_document(receipt)
        _bundle, key, _template = _reviews.prepare_reviews(receipt, seed=17)

        report = _analysis.analyze_reviews(receipt, [document], seed=17)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["label"], "provisional descriptive QA controls review accounting")
        self.assertTrue(report["dev_only"])
        self.assertTrue(report["provisional"])
        self.assertEqual(report["test"], [])
        self.assertFalse(report["independent_trials"])
        self.assertEqual(report["authored_reference_human_label_count"], 0)
        self.assertEqual(report["independent_human_reviewers_verified"], 0)
        self.assertEqual(
            report["scope"],
            {
                "correlated_groups": 6,
                "questions": 12,
                "tasks": 36,
                "cells": 36,
                "unique_prompts": 36,
            },
        )
        self.assertEqual(report["reviewer_count"], 1)
        reviewer = report["reviewers"][0]
        self.assertEqual(reviewer["reviewer_id"], "reviewer-a")
        self.assertEqual(reviewer["reviewer_kind_self_attested"], "human")
        self.assertEqual(
            reviewer["unique_task_summary"],
            {
                "denominator": 36,
                "source_answer_counts": {"correct": 9, "incorrect": 12, "uncertain": 15},
                "prompt_support_counts": {"supported": 12, "unsupported": 9, "uncertain": 15},
                "joint_counts": {
                    "correct": {"supported": 3, "unsupported": 3, "uncertain": 3},
                    "incorrect": {"supported": 3, "unsupported": 3, "uncertain": 6},
                    "uncertain": {"supported": 6, "unsupported": 3, "uncertain": 6},
                },
            },
        )
        variants = reviewer["variants"]
        expected_variant_ids = sorted({row["variant_id"] for row in key["task_keys"]})
        self.assertEqual([group["variant_id"] for group in variants], expected_variant_ids)
        response_by_task = {row["task_id"]: row for row in document["responses"]}
        key_by_variant: dict[str, list[dict[str, object]]] = {}
        for task_key in key["task_keys"]:
            key_by_variant.setdefault(task_key["variant_id"], []).append(task_key)
        for group in variants:
            variant_id = group["variant_id"]
            self.assertEqual(group["denominator"], 12)
            self.assertEqual(
                group["source_answer_counts"],
                {"correct": 3, "incorrect": 4, "uncertain": 5},
            )
            self.assertEqual(
                group["prompt_support_counts"],
                {"supported": 4, "unsupported": 3, "uncertain": 5},
            )
            self.assertEqual(
                group["joint_counts"],
                {
                    "correct": {"supported": 1, "unsupported": 1, "uncertain": 1},
                    "incorrect": {"supported": 1, "unsupported": 1, "uncertain": 2},
                    "uncertain": {"supported": 2, "unsupported": 1, "uncertain": 2},
                },
            )
            expected_rows = []
            for task_key in key_by_variant[variant_id]:
                response = response_by_task[task_key["task_id"]]
                expected_rows.append(
                    {
                        field: task_key[field]
                        for field in (
                            "task_id",
                            "cell_id",
                            "scenario_id",
                            "question_id",
                            "variant_id",
                            "prompt_id",
                        )
                    }
                    | {
                        "source_answer": response["source_answer"],
                        "prompt_support": response["prompt_support"],
                    }
                )
            expected_rows.sort(key=lambda row: row["cell_id"])
            self.assertEqual(group["cells"], expected_rows)
            self.assertTrue(all("rationale" not in row for row in group["cells"]))
        pins = key["source_pins"]
        provenance = report["provenance"]
        self.assertEqual(provenance["bundle_sha256"], key["bundle_sha256"])
        self.assertEqual(provenance["source_pins"], pins)
        self.assertEqual(provenance["receipt_sha256"], pins["receipt_sha256"])
        self.assertEqual(provenance["result_sha256"], pins["result_sha256"])
        self.assertEqual(provenance["report_sha256"], pins["report_sha256"])
        self.assertEqual(
            provenance["review_preparer_source_sha256"],
            hashlib.sha256(Path(_reviews.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["accounting_tool_source_sha256"],
            hashlib.sha256(Path(_analysis.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["response_documents"],
            [
                {
                    "reviewer_id": "reviewer-a",
                    "canonical_sha256": hashlib.sha256(
                        _receipt_contract._canonical_bytes(document)
                    ).hexdigest(),
                }
            ],
        )

    def test_reversed_rows_and_document_order_keep_counts_sorted_and_unpooled(self) -> None:
        receipt = _fake_receipt()
        reviewer_z = _complete_document(receipt, reviewer_id="reviewer-z")
        reviewer_a_original = _complete_document(
            receipt, reviewer_id="reviewer-a", reviewer_kind="agent"
        )
        reviewer_a = _complete_document(receipt, reviewer_id="reviewer-a", reviewer_kind="agent")
        reviewer_a["responses"].reverse()
        bundle, _key, _template = _reviews.prepare_reviews(receipt, seed=17)

        validated = _analysis.validate_responses(receipt, reviewer_a, seed=17)
        first = _analysis.analyze_reviews(receipt, [reviewer_z, reviewer_a], seed=17)
        second = _analysis.analyze_reviews(receipt, [reviewer_a, reviewer_z], seed=17)
        original_order = _analysis.analyze_reviews(receipt, [reviewer_a_original], seed=17)

        self.assertEqual(
            [row["task_id"] for row in validated],
            [task["task_id"] for task in bundle["tasks"]],
        )
        self.assertEqual(first["reviewers"], second["reviewers"])
        self.assertEqual(
            [row["reviewer_id"] for row in first["reviewers"]], ["reviewer-a", "reviewer-z"]
        )
        self.assertEqual(
            [row["reviewer_kind_self_attested"] for row in first["reviewers"]], ["agent", "human"]
        )
        self.assertEqual(first["independent_human_reviewers_verified"], 0)
        self.assertEqual(first["reviewers"][0], original_order["reviewers"][0])
        self.assertNotEqual(
            first["provenance"]["response_documents"][0]["canonical_sha256"],
            original_order["provenance"]["response_documents"][0]["canonical_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "response_documents must be a nonempty list"):
            _analysis.analyze_reviews(receipt, [], seed=17)
        duplicate = _complete_document(receipt, reviewer_id="reviewer-a")
        with self.assertRaisesRegex(ValueError, "duplicate reviewer ID across response documents"):
            _analysis.analyze_reviews(receipt, [reviewer_a, duplicate], seed=17)

    def test_cli_writes_private_report_exclusively_and_preserves_inputs(self) -> None:
        receipt = _fake_receipt()
        document = _complete_document(receipt)
        receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
        response_bytes = json.dumps(document, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "fake-receipt.json"
            response_path = root / "fake-response.json"
            output_path = root / "analysis.json"
            receipt_path.write_bytes(receipt_bytes)
            response_path.write_bytes(response_bytes)
            args = [
                "--receipt",
                str(receipt_path),
                "--responses",
                str(response_path),
                "--seed",
                "17",
                "--output",
                str(output_path),
            ]

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_analysis.main(args), 0)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["reviewer_count"], 1)
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(response_path.read_bytes(), response_bytes)

            output_before = output_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(_analysis.main(args), 2)
            self.assertEqual(output_path.read_bytes(), output_before)

            target = root / "preserved.json"
            target.write_text("keep", encoding="utf-8")
            output_symlink = root / "analysis-link.json"
            output_symlink.symlink_to(target)
            linked_args = args.copy()
            linked_args[linked_args.index(str(output_path))] = str(output_symlink)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(_analysis.main(linked_args), 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(response_path.read_bytes(), response_bytes)

    def test_cli_rejects_invalid_json_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "fake-receipt.json"
            response_path = root / "duplicate-key-response.json"
            output_path = root / "must-not-exist.json"
            receipt = _fake_receipt()
            receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
            response_bytes = b'{"version":1,"version":1}'
            receipt_path.write_bytes(receipt_bytes)
            response_path.write_bytes(response_bytes)
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                result = _analysis.main(
                    [
                        "--receipt",
                        str(receipt_path),
                        "--responses",
                        str(response_path),
                        "--seed",
                        "17",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("receipt is not strict JSON", error.getvalue())
            self.assertFalse(output_path.exists())
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(response_path.read_bytes(), response_bytes)


if __name__ == "__main__":
    unittest.main()
