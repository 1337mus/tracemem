"""Regression tests for APIs exercised by the installed-package smoke."""

from __future__ import annotations

import copy
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _copy_smoke_inputs(source_root: Path) -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    report_path = source_root / "docs/baseline-context-budgets.json"
    bundle_path = source_root / "examples/synthetic/reviews/scenario-review-blind.json"
    report_path.parent.mkdir(parents=True)
    bundle_path.parent.mkdir(parents=True)
    shutil.copyfile(repository / "docs/baseline-context-budgets.json", report_path)
    shutil.copyfile(
        repository / "examples/synthetic/reviews/scenario-review-blind.json", bundle_path
    )
    return report_path, bundle_path


class InstalledNewApiSmokeTests(unittest.TestCase):
    def test_budget_comparison_allows_only_python_and_sqlite_value_differences(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        recorded = {
            "benchmark": "synthetic-context-budget-diagnostic",
            "environment": {"python": "3.12.14", "sqlite": "3.53.1"},
            "metrics": {"relevant_message_recall": 0.5},
        }
        actual = copy.deepcopy(recorded)
        actual["environment"] = {"python": "3.13.0", "sqlite": "3.54.0"}

        comparison = _compare_context_budget_reports(actual, recorded)

        self.assertTrue(comparison["stable_fields_match"])
        self.assertEqual(comparison["environment"]["actual"], actual["environment"])
        self.assertEqual(comparison["environment"]["recorded"], recorded["environment"])

    def test_budget_comparison_rejects_environment_schema_changes(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        recorded = {"environment": {"python": "3.12.14", "sqlite": "3.53.1"}}
        for environment in (
            {"python": "3.12.14"},
            {"python": "3.12.14", "sqlite": "3.53.1", "platform": "Darwin"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(
                    SystemExit, "context-budget environment schema changed"
                ):
                    _compare_context_budget_reports({"environment": environment}, recorded)

    def test_budget_comparison_rejects_environment_field_type_changes(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        recorded = {"environment": {"python": "3.12.14", "sqlite": "3.53.1"}}
        actual = {"environment": {"python": 3, "sqlite": "3.53.1"}}

        with self.assertRaisesRegex(
            SystemExit, "context-budget environment field type changed: environment.python"
        ):
            _compare_context_budget_reports(actual, recorded)

    def test_real_api_execution_matches_frozen_outputs_in_an_empty_work_dir(self) -> None:
        from experiments.smoke_installed import _exercise_new_apis

        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_root = temporary_root / "source"
            work_dir = temporary_root / "work"
            report_path, bundle_path = _copy_smoke_inputs(source_root)
            work_dir.mkdir()
            report_before = report_path.read_bytes()
            bundle_before = bundle_path.read_bytes()

            result = _exercise_new_apis(source_root, work_dir)

            self.assertTrue(result["context_budget"]["stable_fields_match"])
            self.assertEqual(
                result["context_budget"]["report_sha256"],
                "9e14de94e09893f400e17226fa2cebceeec377c1493f4f263fa325d49a055c01",
            )
            self.assertEqual(
                result["context_budget"]["fixture_sha256"],
                "76862babda5e6e26be1ed14f5265c64e43627a54b1499dba47585605396637f3",
            )
            self.assertEqual(
                set(result["context_budget"]["environment"]["actual"]), {"python", "sqlite"}
            )
            self.assertEqual(
                result["review_html"]["bundle_sha256"],
                "65ed4a2e0bbd227c3a521899676c4db98085f855083ec2d2a16aecbdda55bda5",
            )
            self.assertEqual(
                result["review_html"]["html_sha256"],
                "07521f971f5554e5476d15a1299eb75cf0b5b20fbc6854c79b6053a1aecc474e",
            )
            self.assertEqual(result["review_html"]["task_count"], 84)
            self.assertEqual(report_path.read_bytes(), report_before)
            self.assertEqual(bundle_path.read_bytes(), bundle_before)
            self.assertEqual(
                {path.name for path in work_dir.iterdir()}, {"context-budget-evidence.sqlite3"}
            )

            recorded = json.loads(report_before)
            self.assertEqual(
                result["context_budget"]["environment"]["recorded"], recorded["environment"]
            )

    def test_new_api_smoke_rejects_a_changed_blind_bundle_copy(self) -> None:
        from experiments.smoke_installed import _exercise_new_apis

        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_root = temporary_root / "source"
            _, bundle_path = _copy_smoke_inputs(source_root)
            bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
            work_dir = temporary_root / "work"
            work_dir.mkdir()

            with self.assertRaisesRegex(SystemExit, "blind review bundle SHA-256 changed"):
                _exercise_new_apis(source_root, work_dir)

    def test_new_api_smoke_rejects_a_changed_baseline_report_copy(self) -> None:
        from experiments.smoke_installed import _exercise_new_apis

        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_root = temporary_root / "source"
            report_path, _ = _copy_smoke_inputs(source_root)
            report_path.write_bytes(report_path.read_bytes() + b"\n")
            work_dir = temporary_root / "work"
            work_dir.mkdir()

            with self.assertRaisesRegex(
                SystemExit, "context-budget baseline report SHA-256 changed"
            ):
                _exercise_new_apis(source_root, work_dir)

    def test_rendered_html_hash_check_rejects_a_changed_temporary_copy(self) -> None:
        from experiments.smoke_installed import (
            REVIEW_HTML_SHA256,
            _require_frozen_sha256,
        )
        from tracemem.review_html import render_review_html

        with TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory) / "source"
            _, bundle_path = _copy_smoke_inputs(source_root)
            bundle = json.loads(bundle_path.read_bytes())
            html_path = Path(temporary_directory) / "review.html"
            html_path.write_text(render_review_html(bundle), encoding="utf-8")
            canonical_html = html_path.read_bytes()

            self.assertEqual(
                _require_frozen_sha256(
                    "rendered blind review HTML", canonical_html, REVIEW_HTML_SHA256
                ),
                REVIEW_HTML_SHA256,
            )
            html_path.write_bytes(canonical_html + b" ")
            with self.assertRaisesRegex(SystemExit, "rendered blind review HTML SHA-256 changed"):
                _require_frozen_sha256(
                    "rendered blind review HTML", html_path.read_bytes(), REVIEW_HTML_SHA256
                )

    def test_budget_comparison_rejects_a_changed_metric(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        repository = Path(__file__).resolve().parents[1]
        report = json.loads((repository / "docs/baseline-context-budgets.json").read_bytes())
        actual = copy.deepcopy(report)
        actual["modes"]["bm25"]["by_budget"]["512"]["mean_relevant_message_recall"] = 0.0

        with self.assertRaisesRegex(
            SystemExit,
            r"report\.modes\.bm25\.by_budget\.512\.mean_relevant_message_recall",
        ):
            _compare_context_budget_reports(actual, report)

    def test_budget_comparison_rejects_a_changed_packed_citation(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        repository = Path(__file__).resolve().parents[1]
        report = json.loads((repository / "docs/baseline-context-budgets.json").read_bytes())
        actual = copy.deepcopy(report)
        actual["modes"]["bm25"]["per_query"][0]["budgets"]["512"]["packed_citations"][0] = (
            "changed-citation"
        )

        with self.assertRaisesRegex(
            SystemExit,
            r"report\.modes\.bm25\.per_query\[0\]\.budgets\.512\.packed_citations\[0\]",
        ):
            _compare_context_budget_reports(actual, report)

    def test_budget_comparison_rejects_changed_configuration(self) -> None:
        from experiments.smoke_installed import _compare_context_budget_reports

        repository = Path(__file__).resolve().parents[1]
        report = json.loads((repository / "docs/baseline-context-budgets.json").read_bytes())
        actual = copy.deepcopy(report)
        actual["retrieval_config"]["rrf_k"] = 61

        with self.assertRaisesRegex(
            SystemExit, r"context-budget report changed at report\.retrieval_config\.rrf_k"
        ):
            _compare_context_budget_reports(actual, report)


if __name__ == "__main__":
    unittest.main()
