"""Command-line dataset audit coverage for external local fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _case(case_id: str, scenario_id: str, split: str, text: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "split": split,
        "expected": [],
        "messages": [
            {
                "session_id": f"session-{case_id}",
                "message_id": f"message-{case_id}",
                "role": "user",
                "text": text,
            }
        ],
    }


class AuditDatasetCliTests(unittest.TestCase):
    def test_default_output_bytes_remain_unchanged(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"

        result = subprocess.run(
            [sys.executable, str(cli_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(
            hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            "a2d6fc4857a694feb6c7a9be3d2caa4c11c0f5402d58f7e7538107774957f580",
        )

    def test_combines_fixture_cases_and_reports_ordered_input_digests(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"
        metadata_row = {
            "scenario_group": "same declared scenario",
            "template_group": "same declared template",
            "entity_ids": ["same declared entity"],
        }
        payloads = [
            {
                "cases": [_case("dev-case", "dev-scenario", "dev", "unrelated words one")],
                "case_metadata": {"dev-case": metadata_row},
            },
            {
                "cases": [_case("test-case", "test-scenario", "test", "different words two")],
                "case_metadata": {"test-case": metadata_row},
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_paths = [
                Path(temporary_directory) / f"fixture-{index}.json" for index in range(2)
            ]
            raw_inputs = [
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                for payload in payloads
            ]
            for path, raw in zip(input_paths, raw_inputs, strict=True):
                path.write_bytes(raw)

            result = subprocess.run(
                [sys.executable, str(cli_path), *(str(path) for path in input_paths)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertNotIn(str(input_paths[0]), result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["split_case_counts"]["dev"], 1)
        self.assertEqual(report["split_case_counts"]["test"], 1)
        self.assertEqual(
            report["input_sha256"],
            [hashlib.sha256(raw).hexdigest() for raw in raw_inputs],
        )
        self.assertEqual(
            report["declared_metadata"]["overlaps"]["scenario_group"],
            [
                {
                    "value": "same declared scenario",
                    "splits": ["dev", "test"],
                    "case_ids": ["dev-case", "test-case"],
                }
            ],
        )

    def test_rejects_duplicate_case_ids_and_metadata_collisions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"
        metadata_row = {
            "scenario_group": "scenario-group",
            "template_group": "template-group",
            "entity_ids": ["entity"],
        }
        input_sets = (
            (
                [
                    {"cases": [_case("same", "first", "dev", "first text")]},
                    {"cases": [_case("same", "second", "test", "second text")]},
                ],
                "duplicate case_id",
            ),
            (
                [
                    {
                        "cases": [_case("case-a", "a", "dev", "alpha")],
                        "case_metadata": {"case-b": metadata_row},
                    },
                    {
                        "cases": [_case("case-b", "b", "test", "beta")],
                        "case_metadata": {"case-b": metadata_row},
                    },
                ],
                "duplicate case_metadata",
            ),
        )

        for payloads, expected_error in input_sets:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    input_paths = [
                        Path(temporary_directory) / f"fixture-{index}.json"
                        for index in range(len(payloads))
                    ]
                    for path, payload in zip(input_paths, payloads, strict=True):
                        path.write_text(json.dumps(payload), encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, str(cli_path), *(str(path) for path in input_paths)],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected_error, result.stderr)

    def test_rejects_duplicate_case_metadata_keys_and_nonfinite_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"
        case_json = json.dumps(_case("case", "scenario", "dev", "private text"))
        metadata_row = json.dumps(
            {
                "scenario_group": "scenario-group",
                "template_group": "template-group",
                "entity_ids": ["entity"],
            }
        )
        invalid_inputs = (
            (
                f'{{"cases":[{case_json}],"case_metadata":{{"case":{metadata_row},'
                f'"case":{metadata_row}}}}}',
                "duplicate JSON object key",
            ),
            (
                f'{{"cases":[{case_json[:-1]},"unused":NaN}}]}}',
                "non-finite JSON constant",
            ),
        )

        for raw_json, expected_error in invalid_inputs:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    fixture_path = Path(temporary_directory) / "fixture.json"
                    fixture_path.write_text(raw_json, encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, str(cli_path), str(fixture_path)],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected_error, result.stderr)

    def test_rejects_invalid_fixture_shapes_without_partial_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"
        valid_payload = json.dumps({"cases": [_case("valid", "s", "dev", "text")]})
        invalid_inputs = (
            ("[]", "cases list"),
            ("{}", "cases list"),
            ('{"cases":"not-a-list"}', "cases list"),
            ('{"cases":[],"case_metadata":[]}', "case_metadata must be an object"),
            (
                json.dumps(
                    {
                        "cases": [_case("bad", "scenario", "dev", "text")],
                        "case_metadata": {"bad": {}},
                    }
                ),
                "exactly scenario_group",
            ),
        )

        for invalid_json, expected_error in invalid_inputs:
            with self.subTest(expected_error=expected_error, invalid_json=invalid_json):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    valid_path = Path(temporary_directory) / "valid.json"
                    invalid_path = Path(temporary_directory) / "invalid.json"
                    valid_path.write_text(valid_payload, encoding="utf-8")
                    invalid_path.write_text(invalid_json, encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, str(cli_path), str(valid_path), str(invalid_path)],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected_error, result.stderr)

    def test_absent_and_explicitly_empty_metadata_remain_distinct(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_path = repo_root / "experiments" / "audit_dataset.py"
        payloads = (
            {"cases": [_case("case", "scenario", "dev", "text")]},
            {"cases": [_case("case", "scenario", "dev", "text")], "case_metadata": {}},
        )
        reports = []

        for payload in payloads:
            with tempfile.TemporaryDirectory() as temporary_directory:
                fixture_path = Path(temporary_directory) / "fixture.json"
                fixture_path.write_text(json.dumps(payload), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(cli_path), str(fixture_path)],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            reports.append(json.loads(result.stdout))

        self.assertNotIn("declared_metadata", reports[0])
        self.assertEqual(reports[1]["declared_metadata"]["covered_case_count"], 0)
        self.assertFalse(reports[1]["declared_metadata"]["complete"])


if __name__ == "__main__":
    unittest.main()
