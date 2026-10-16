from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any
from unittest.mock import patch

from training import audit_verifier_data
from training.audit_verifier_data import summarize_dataset
from training.verifier_data import render_training_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_DATASET_PATH = REPO_ROOT / "examples" / "synthetic" / "verifier-pairs-v1.json"
VERIFIER_AUDIT_PATH = REPO_ROOT / "docs" / "verifier-pairs-audit.json"
PHENOMENA = ("attribution", "condition", "actor", "negation", "time", "causality")
SPLITS = ("train", "dev", "test")


def invented_dataset() -> dict[str, Any]:
    support_labels = (
        ("supported", "unsupported"),
        ("supported", "uncertain"),
        ("unsupported", "uncertain"),
        ("supported", "unsupported"),
        ("uncertain", "unsupported"),
        ("supported", "uncertain"),
    )
    eligibility_labels = (
        ("eligible", "ineligible"),
        ("eligible", "uncertain"),
        ("eligible", "ineligible"),
        ("uncertain", "eligible"),
        ("eligible", "ineligible"),
        ("eligible", "uncertain"),
    )
    cases: list[dict[str, Any]] = []
    for index, phenomenon in enumerate(PHENOMENA):
        split = SPLITS[index // 2]
        message_id = f"synthetic-message-{index}"
        quote = (
            f"In this fictional example, the user might place a made-up token {index} "
            "in imaginary room Juniper."
        )
        citation = {
            "message_id": message_id,
            "quote": quote,
            "valid_from": None,
            "valid_to": None,
        }
        candidates = []
        for candidate_index, (support, eligibility) in enumerate(
            zip(support_labels[index], eligibility_labels[index], strict=True)
        ):
            candidates.append(
                {
                    "candidate_id": f"synthetic-candidate-{index}-{candidate_index}",
                    "claim": (
                        f"Fictional candidate {candidate_index} for synthetic case {index} "
                        "about imaginary room Juniper."
                    ),
                    **citation,
                    "support": support,
                    "eligibility": eligibility,
                    "support_reason": f"Fictional support rationale {index}-{candidate_index}.",
                    "eligibility_reason": (
                        f"Fictional eligibility rationale {index}-{candidate_index}."
                    ),
                }
            )
        cases.append(
            {
                "case_id": f"synthetic-case-{index}",
                "split": split,
                "scenario_group": f"synthetic-scenario-{index}",
                "template_group": f"synthetic-template-{index}",
                "entity_ids": [f"synthetic-entity-{index}"],
                "phenomenon": phenomenon,
                "messages": [
                    {
                        "session_id": f"synthetic-session-{index}",
                        "message_id": message_id,
                        "role": "user",
                        "text": quote,
                        "recorded_at": "2026-02-03T10:00:00Z",
                        "content_kind": "conversation",
                    }
                ],
                "candidates": candidates,
            }
        )
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "training.audit_verifier_data", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_dataset(path: Path, payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _assert_corpus_audit_receipt(
    test_case: unittest.TestCase,
    dataset_path: Path,
    expected_report_bytes: bytes,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        report_path = root / "report.json"
        train_path = root / "train.jsonl"
        result = _run_cli(
            "--dataset",
            str(dataset_path),
            "--output",
            str(report_path),
            "--train-output",
            str(train_path),
        )
        test_case.assertEqual(result.returncode, 0, result.stderr)
        actual_report_bytes = report_path.read_bytes()
        if actual_report_bytes != expected_report_bytes:
            test_case.fail("generated audit report bytes differ from the committed receipt")
        report = json.loads(actual_report_bytes)
        train_rows = [
            json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
        ]
        return report, train_rows


class VerifierDatasetAuditTests(unittest.TestCase):
    def test_summary_counts_each_label_axis_and_no_filter_baseline_separately(self) -> None:
        summary = summarize_dataset(invented_dataset())

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["annotation_status"], "agent_authored")
        self.assertEqual(summary["human_reviewers"], 0)
        self.assertEqual(summary["case_count"], 6)
        self.assertEqual(summary["scenario_count"], 6)
        self.assertEqual(summary["candidate_count"], 12)
        self.assertEqual(summary["paired_support_cases"], 2)
        self.assertEqual(summary["uncertain_support_cases"], 4)
        self.assertEqual(summary["split_group_overlap"], 0)

        expected_counts = {
            "train": {
                "support": {"supported": 2, "unsupported": 1, "uncertain": 1},
                "eligibility": {"eligible": 2, "ineligible": 1, "uncertain": 1},
                "phenomena": {"attribution": 1, "condition": 1},
                "no_filter": {
                    "candidate_count": 4,
                    "all_candidates_retained": 4,
                    "supported_retained": 2,
                    "supported_total": 2,
                    "unsupported_retained": 1,
                    "unsupported_total": 1,
                    "unsupported_rejected": 0,
                    "uncertain_support_unscored": 1,
                    "eligible_retained": 2,
                    "eligible_total": 2,
                    "ineligible_retained": 1,
                    "ineligible_total": 1,
                    "uncertain_eligibility_unscored": 1,
                },
            },
            "dev": {
                "support": {"supported": 1, "unsupported": 2, "uncertain": 1},
                "eligibility": {"eligible": 2, "ineligible": 1, "uncertain": 1},
                "phenomena": {"actor": 1, "negation": 1},
                "no_filter": {
                    "candidate_count": 4,
                    "all_candidates_retained": 4,
                    "supported_retained": 1,
                    "supported_total": 1,
                    "unsupported_retained": 2,
                    "unsupported_total": 2,
                    "unsupported_rejected": 0,
                    "uncertain_support_unscored": 1,
                    "eligible_retained": 2,
                    "eligible_total": 2,
                    "ineligible_retained": 1,
                    "ineligible_total": 1,
                    "uncertain_eligibility_unscored": 1,
                },
            },
            "test": {
                "support": {"supported": 1, "unsupported": 1, "uncertain": 2},
                "eligibility": {"eligible": 2, "ineligible": 1, "uncertain": 1},
                "phenomena": {"time": 1, "causality": 1},
                "no_filter": {
                    "candidate_count": 4,
                    "all_candidates_retained": 4,
                    "supported_retained": 1,
                    "supported_total": 1,
                    "unsupported_retained": 1,
                    "unsupported_total": 1,
                    "unsupported_rejected": 0,
                    "uncertain_support_unscored": 2,
                    "eligible_retained": 2,
                    "eligible_total": 2,
                    "ineligible_retained": 1,
                    "ineligible_total": 1,
                    "uncertain_eligibility_unscored": 1,
                },
            },
        }
        self.assertEqual(set(summary["per_split"]), set(SPLITS))
        for split, expected in expected_counts.items():
            with self.subTest(split=split):
                split_summary = summary["per_split"][split]
                self.assertEqual(split_summary["case_count"], 2)
                self.assertEqual(split_summary["scenario_count"], 2)
                self.assertEqual(split_summary["candidate_count"], 4)
                self.assertEqual(split_summary["support_counts"], expected["support"])
                self.assertEqual(split_summary["eligibility_counts"], expected["eligibility"])
                self.assertEqual(set(split_summary["phenomenon_counts"]), set(PHENOMENA))
                self.assertEqual(
                    {
                        key: count
                        for key, count in split_summary["phenomenon_counts"].items()
                        if count
                    },
                    expected["phenomena"],
                )
                self.assertEqual(split_summary["no_filter"], expected["no_filter"])

    def test_scenario_count_uses_distinct_groups_and_keeps_case_count(self) -> None:
        payload = invented_dataset()
        payload["cases"][1]["scenario_group"] = payload["cases"][0]["scenario_group"]
        payload["cases"][1]["template_group"] = payload["cases"][0]["template_group"]

        summary = summarize_dataset(payload)

        self.assertEqual(summary["case_count"], 6)
        self.assertEqual(summary["scenario_count"], 5)
        self.assertEqual(summary["per_split"]["train"]["case_count"], 2)
        self.assertEqual(summary["per_split"]["train"]["scenario_count"], 1)

        serialized = json.dumps(summary, sort_keys=True).lower()
        self.assertNotIn("accuracy", serialized)
        self.assertNotIn("quality", serialized)

    def test_summary_validates_dataset_before_computing_counts(self) -> None:
        payload = invented_dataset()
        payload["scope_policy"] = "unrecognized-policy"

        with self.assertRaisesRegex(ValueError, "scope_policy"):
            summarize_dataset(payload)

    def test_cli_reports_source_hash_and_exports_only_renderer_train_rows(self) -> None:
        payload = invented_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "source.json"
            report_path = root / "report.json"
            train_path = root / "train.jsonl"
            raw = _write_dataset(dataset_path, payload)

            result = _run_cli(
                "--dataset",
                str(dataset_path),
                "--output",
                str(report_path),
                "--train-output",
                str(train_path),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(report["summary"], summarize_dataset(payload))

            actual_rows = [
                json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            expected_rows = render_training_rows(payload, "train")
            self.assertEqual(actual_rows, expected_rows)
            self.assertEqual(len(actual_rows), 4)
            serialized_rows = json.dumps(actual_rows, ensure_ascii=False)
            for index in range(2, 6):
                message = payload["cases"][index]["messages"][0]
                self.assertNotIn(message["message_id"], serialized_rows)
                self.assertNotIn(message["text"], serialized_rows)

    def test_cli_preflights_all_destinations_before_creating_report(self) -> None:
        payload = invented_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "source.json"
            report_path = root / "report.json"
            train_path = root / "existing-train.jsonl"
            _write_dataset(dataset_path, payload)
            train_path.write_text("keep this receipt", encoding="utf-8")

            result = _run_cli(
                "--dataset",
                str(dataset_path),
                "--output",
                str(report_path),
                "--train-output",
                str(train_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("already exists", result.stderr)
            self.assertFalse(report_path.exists())
            self.assertEqual(train_path.read_text(encoding="utf-8"), "keep this receipt")

    def test_cli_keeps_report_if_train_output_creation_loses_a_race(self) -> None:
        payload = invented_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "source.json"
            report_path = root / "report.json"
            train_path = root / "train.jsonl"
            _write_dataset(dataset_path, payload)
            create_exclusive = audit_verifier_data._create_exclusive

            def race_on_train_output(path: Path, content: str) -> None:
                if path == train_path:
                    path.write_text("created by competing writer", encoding="utf-8")
                create_exclusive(path, content)

            stderr = io.StringIO()
            with patch.object(
                audit_verifier_data,
                "_create_exclusive",
                side_effect=race_on_train_output,
            ):
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    audit_verifier_data.main(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--output",
                            str(report_path),
                            "--train-output",
                            str(train_path),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertTrue(report_path.exists())
            self.assertEqual(
                train_path.read_text(encoding="utf-8"),
                "created by competing writer",
            )

    def test_cli_rejects_path_collisions_and_existing_report(self) -> None:
        payload = invented_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "source.json"
            report_path = root / "report.json"
            _write_dataset(dataset_path, payload)

            collisions = (
                (str(dataset_path), None, "must be different"),
                (str(report_path), str(report_path), "must be different"),
                (str(report_path), str(dataset_path), "must be different"),
            )
            for output_path, train_output_path, reason in collisions:
                with self.subTest(output=output_path, train_output=train_output_path):
                    arguments = [
                        "--dataset",
                        str(dataset_path),
                        "--output",
                        output_path,
                    ]
                    if train_output_path is not None:
                        arguments.extend(("--train-output", train_output_path))
                    result = _run_cli(*arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(reason, result.stderr)
                    self.assertFalse(report_path.exists())

            report_path.write_text("existing report", encoding="utf-8")
            result = _run_cli(
                "--dataset",
                str(dataset_path),
                "--output",
                str(report_path),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "existing report")

    def test_cli_rejects_duplicate_keys_nonfinite_json_and_invalid_dataset_without_outputs(
        self,
    ) -> None:
        valid_json = json.dumps(invented_dataset(), ensure_ascii=False)
        invalid_inputs = (
            (
                valid_json.replace('"version": 1', '"version": 1, "version": 1', 1).encode("utf-8"),
                "duplicate JSON object key",
            ),
            (
                valid_json.replace('"version": 1', '"version": NaN', 1).encode("utf-8"),
                "non-finite JSON constant",
            ),
            (
                valid_json.replace('"version": 1', '"version": Infinity', 1).encode("utf-8"),
                "non-finite JSON constant",
            ),
            (
                valid_json.replace('"version": 1', '"version": 1e400', 1).encode("utf-8"),
                "non-finite JSON number",
            ),
            (
                valid_json.replace(
                    '"scope_policy": "personal-memory-research-v1"',
                    '"scope_policy": "bad"',
                    1,
                ).encode("utf-8"),
                "scope_policy",
            ),
            (b"\xff", "UTF-8"),
        )
        for raw_bytes, reason in invalid_inputs:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                dataset_path = root / "invalid.json"
                report_path = root / "report.json"
                train_path = root / "train.jsonl"
                dataset_path.write_bytes(raw_bytes)

                result = _run_cli(
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(report_path),
                    "--train-output",
                    str(train_path),
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(reason, result.stderr)
                self.assertFalse(report_path.exists())
                self.assertFalse(train_path.exists())

    def test_committed_corpus_matches_audit_receipt_and_exports_train_only(self) -> None:
        source_bytes = VERIFIER_DATASET_PATH.read_bytes()
        payload = json.loads(source_bytes)
        expected_report_bytes = VERIFIER_AUDIT_PATH.read_bytes()

        report, train_rows = _assert_corpus_audit_receipt(
            self,
            VERIFIER_DATASET_PATH,
            expected_report_bytes,
        )

        self.assertEqual(report["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
        summary = report["summary"]
        self.assertEqual(summary["case_count"], 36)
        self.assertEqual(summary["scenario_count"], 36)
        self.assertEqual(summary["candidate_count"], 72)
        self.assertEqual(summary["paired_support_cases"], 36)
        self.assertEqual(len(train_rows), 48)

        expected_per_split = {
            "train": {
                "case_count": 24,
                "scenario_count": 24,
                "candidate_count": 48,
                "support_counts": {"supported": 24, "unsupported": 24, "uncertain": 0},
                "eligibility_counts": {"eligible": 24, "ineligible": 12, "uncertain": 12},
            },
            "dev": {
                "case_count": 6,
                "scenario_count": 6,
                "candidate_count": 12,
                "support_counts": {"supported": 6, "unsupported": 6, "uncertain": 0},
                "eligibility_counts": {"eligible": 4, "ineligible": 4, "uncertain": 4},
            },
            "test": {
                "case_count": 6,
                "scenario_count": 6,
                "candidate_count": 12,
                "support_counts": {"supported": 6, "unsupported": 6, "uncertain": 0},
                "eligibility_counts": {"eligible": 4, "ineligible": 4, "uncertain": 4},
            },
        }
        self.assertEqual(set(summary["per_split"]), set(expected_per_split))
        for split, expected in expected_per_split.items():
            with self.subTest(split=split):
                for field, value in expected.items():
                    self.assertEqual(summary["per_split"][split][field], value)

        serialized_train_rows = json.dumps(train_rows, ensure_ascii=False)
        for case in payload["cases"]:
            if case["split"] == "train":
                continue
            for message in case["messages"]:
                self.assertNotIn(message["message_id"], serialized_train_rows)
                self.assertNotIn(message["text"], serialized_train_rows)


if __name__ == "__main__":
    unittest.main()
