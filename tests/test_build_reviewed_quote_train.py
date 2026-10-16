from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.build_reviewed_quote_train import (
    _require_non_target_cases_match,
    build_reviewed_train,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATHS = {
    "v2_source": ROOT / "datasets/sft-quote-v2/train-seeds.json",
    "v2_review": ROOT / "datasets/sft-target-review/v2-train-review.json",
    "v3_source": ROOT / "datasets/sft-quote-v3/target-train-seeds.json",
    "v3_review": ROOT / "datasets/sft-target-review/v3-train-review.json",
    "historical_train": ROOT / "datasets/sft-quote-v3/train.json",
}


def _inputs() -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in INPUT_PATHS.items()}


def _cli_args(paths: dict[str, Path], output: Path) -> list[str]:
    args: list[str] = []
    for name, option in (
        ("v2_source", "--v2-source"),
        ("v2_review", "--v2-review"),
        ("v3_source", "--v3-source"),
        ("v3_review", "--v3-review"),
        ("historical_train", "--historical-train"),
    ):
        args.extend([option, str(paths[name])])
    args.extend(["--output-dir", str(output)])
    return args


class BuildReviewedQuoteTrainTests(unittest.TestCase):
    def test_non_target_comparison_rejects_reordered_rows(self) -> None:
        original = [
            {"case_id": "case-a", "expected": [], "messages": [{"text": "first"}]},
            {"case_id": "case-b", "expected": [], "messages": [{"text": "second"}]},
        ]

        with self.assertRaisesRegex(ValueError, "changed case order"):
            _require_non_target_cases_match(original, list(reversed(original)), "test source")

    def test_rejects_malformed_review_bytes(self) -> None:
        inputs = _inputs()
        with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
            build_reviewed_train(
                v2_source=inputs["v2_source"],
                v2_review=b"{",
                v3_source=inputs["v3_source"],
                v3_review=inputs["v3_review"],
                historical_train=inputs["historical_train"],
            )

    def test_builds_reproducible_corrected_training_corpus(self) -> None:
        inputs = _inputs()
        first = build_reviewed_train(**inputs)
        second = build_reviewed_train(**inputs)

        self.assertEqual(first, second)
        dataset = first["dataset"]
        report = first["report"]
        cases = dataset["cases"]
        self.assertEqual(len(cases), 768)
        self.assertEqual(len({case["scenario_group"] for case in cases}), 72)
        self.assertEqual(sum(not case["expected"] for case in cases), 140)
        self.assertEqual(sum(len(case["expected"]) for case in cases), 696)
        self.assertEqual(report["old_claim_count"], 680)
        self.assertEqual(report["new_claim_count"], 696)
        self.assertEqual(report["changed_row_count"], 64)
        self.assertEqual(report["changed_family_count"], 7)
        self.assertEqual(report["reviewer_status"], "agent-only;zero human reviewers")
        self.assertEqual(report["v2_review_gate"]["family_count"], 48)
        self.assertEqual(report["v2_review_gate"]["human_reviewers"], 0)
        self.assertEqual(report["v3_review_gate"]["family_count"], 24)
        self.assertEqual(report["v3_review_gate"]["human_reviewers"], 0)
        self.assertEqual(
            report["input_sha256"],
            {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        )
        self.assertTrue(report["source_preservation"]["historical_rebuild_matches"])
        self.assertTrue(report["source_preservation"]["non_target_fields_match"])
        historical_cases = json.loads(inputs["historical_train"])["cases"]
        self.assertEqual(len(historical_cases), len(cases))
        for old_case, new_case in zip(historical_cases, cases, strict=True):
            self.assertEqual(new_case["case_id"], f"V4T-{old_case['case_id']}")
            unnamespaced = _strip_v4t_namespace(new_case)
            self.assertEqual(_without_expected(unnamespaced), _without_expected(old_case))
        for case in cases:
            for value in (
                case["case_id"],
                case["scenario_group"],
                case["template_group"],
                *case["entity_ids"],
            ):
                self.assertTrue(value.startswith("V4T-"))
            message_ids = set()
            for message in case["messages"]:
                self.assertTrue(message["session_id"].startswith("V4T-"))
                self.assertTrue(message["message_id"].startswith("V4T-"))
                message_ids.add(message["message_id"])
            for claim in case["expected"]:
                self.assertTrue(claim["message_id"].startswith("V4T-"))
                self.assertIn(claim["message_id"], message_ids)

    def test_rejects_historical_train_that_does_not_rebuild_exactly(self) -> None:
        inputs = _inputs()
        historical = json.loads(inputs["historical_train"])
        historical["cases"][0]["tags"].append("unexpected")
        inputs["historical_train"] = json.dumps(historical).encode()

        with self.assertRaisesRegex(ValueError, "historical train does not match"):
            build_reviewed_train(**inputs)

    def test_cli_requires_all_input_paths_and_validates_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            missing = subprocess.run(
                [sys.executable, "-m", "training.build_reviewed_quote_train"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("--v2-source", missing.stderr)
            self.assertFalse(output.exists())

            paths = {name: path for name, path in INPUT_PATHS.items()}
            invalid_review = Path(temporary) / "invalid-review.json"
            invalid_review.write_bytes(b"{")
            paths["v2_review"] = invalid_review
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.build_reviewed_quote_train",
                    *_cli_args(paths, output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("v2 review is not valid UTF-8 JSON", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_writes_only_two_files_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            paths = {name: path for name, path in INPUT_PATHS.items()}
            command = [
                sys.executable,
                "-m",
                "training.build_reviewed_quote_train",
                *_cli_args(paths, output),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"train.json", "report.json"}
            )
            written_train = json.loads((output / "train.json").read_bytes())
            written_report = json.loads((output / "report.json").read_bytes())
            self.assertEqual(len(written_train["cases"]), 768)
            self.assertEqual(written_report["new_claim_count"], 696)

            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertIn("output directory already exists", second.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"train.json", "report.json"}
            )

    def test_cli_rejects_existing_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "real-output"
            destination.mkdir()
            symlink = root / "output-link"
            symlink.symlink_to(destination, target_is_directory=True)
            paths = {name: path for name, path in INPUT_PATHS.items()}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.build_reviewed_quote_train",
                    *_cli_args(paths, symlink),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("output directory already exists", result.stderr)
            self.assertEqual(list(destination.iterdir()), [])


def _without_expected(case: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in case.items() if key != "expected"}


def _strip_v4t_namespace(case: dict[str, object]) -> dict[str, object]:
    copied = json.loads(json.dumps(case))
    for field in ("case_id", "scenario_group", "template_group"):
        copied[field] = copied[field].removeprefix("V4T-")
    copied["entity_ids"] = [value.removeprefix("V4T-") for value in copied["entity_ids"]]
    message_ids: dict[str, str] = {}
    for message in copied["messages"]:
        message["session_id"] = message["session_id"].removeprefix("V4T-")
        prior_id = message["message_id"]
        message_ids[prior_id] = prior_id.removeprefix("V4T-")
        message["message_id"] = message_ids[prior_id]
    for claim in copied["expected"]:
        claim["message_id"] = message_ids[claim["message_id"]]
    return copied


if __name__ == "__main__":
    unittest.main()
