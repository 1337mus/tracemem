"""Offline contract tests for the frozen verifier-pair receipt evaluator."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from experiments import evaluate_verifier_pairs
from experiments.verifier_pairs import score_outputs

_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _ROOT / "examples/synthetic/verifier-pairs-v1.json"


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _inputs(mode: str = "baseline") -> tuple[bytes, dict[str, object], dict[str, object]]:
    from training.verifier_data import render_candidate_prompt, render_training_rows

    dataset_raw = _DATASET.read_bytes()
    dataset = json.loads(dataset_raw)
    cases = dataset["cases"]
    items = [
        {
            "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
            "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
        }
        for case in cases
        if case["split"] == "dev"
        for candidate in case["candidates"]
    ]
    model = {"model_id": "offline-test-model"}
    settings = {"max_new_tokens": 128}
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "split": "dev",
        "source_sha256": {
            "examples/synthetic/verifier-pairs-v1.json": hashlib.sha256(dataset_raw).hexdigest(),
            "training/verifier_data.py": hashlib.sha256(
                (_ROOT / "training/verifier_data.py").read_bytes()
            ).hexdigest(),
        },
        "runner_sha256": "a" * 64,
        "model": model,
        "settings": settings,
        "items": items,
        "train_rows": render_training_rows(dataset, "train") if mode == "train" else [],
    }
    labels = {
        f"{case['case_id']}::{candidate['candidate_id']}": candidate
        for case in cases
        if case["split"] == "dev"
        for candidate in case["candidates"]
    }
    outputs = [
        {
            "item_id": item["item_id"],
            "raw_output": json.dumps(
                {
                    "support": labels[item["item_id"]]["support"],
                    "eligibility": labels[item["item_id"]]["eligibility"],
                }
            ),
            "finish_reason": "eos",
            "input_tokens": 1,
            "input_token_ids": [1],
            "output_tokens": 2,
            "output_token_ids": [3, 2],
            "ended_with_eos": True,
            "hit_max_new_tokens": False,
        }
        for item in items
    ]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "success",
        "mode": payload["mode"],
        "split": payload["split"],
        "payload_sha256": _canonical_sha(payload),
        "runner_sha256": payload["runner_sha256"],
        "model": model,
        "settings": settings,
        "tokenizer": {"eos_token_id": 2},
        "baseline_outputs": outputs,
    }
    if mode == "train":
        receipt["tuned_outputs"] = [dict(row) for row in outputs]
    return dataset_raw, payload, receipt


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


class EvaluateVerifierPairsTests(unittest.TestCase):
    def test_rejects_receipt_bound_to_different_payload(self) -> None:
        dataset_raw, payload, receipt = _inputs()
        receipt["payload_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "payload SHA-256"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_valid_receipt_scores_identically_to_direct_scorer(self) -> None:
        dataset_raw, payload, receipt = _inputs()

        report = evaluate_verifier_pairs.evaluate_report(
            dataset_raw, _json_bytes(payload), _json_bytes(receipt)
        )
        direct = score_outputs(json.loads(dataset_raw), "dev", receipt["baseline_outputs"])

        self.assertEqual(report["arms"]["baseline"]["score"], direct)
        self.assertEqual(
            report["arms"]["baseline"]["receipt_sha256"],
            hashlib.sha256(_json_bytes(receipt)).hexdigest(),
        )

    def test_train_receipt_scores_baseline_and_tuned_arms(self) -> None:
        dataset_raw, payload, receipt = _inputs("train")

        report = evaluate_verifier_pairs.evaluate_report(
            dataset_raw, _json_bytes(payload), _json_bytes(receipt)
        )

        self.assertEqual(set(report["arms"]), {"baseline", "tuned"})
        self.assertEqual(report["arms"]["baseline"]["score"], report["arms"]["tuned"]["score"])

    def test_rejects_payload_prompts_that_differ_from_the_renderer(self) -> None:
        dataset_raw, payload, receipt = _inputs()
        payload["items"][0]["prompt"] += " changed"
        receipt["payload_sha256"] = _canonical_sha(payload)

        with self.assertRaisesRegex(ValueError, "prompts"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_rejects_raw_dataset_bytes_that_do_not_match_payload_source_hash(self) -> None:
        dataset_raw, payload, receipt = _inputs()

        with self.assertRaisesRegex(ValueError, "source SHA-256"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw + b"\n", _json_bytes(payload), _json_bytes(receipt)
            )

    def test_rejects_receipt_metadata_that_does_not_match_payload(self) -> None:
        mismatches = (
            ("model", {"model_id": "other"}),
            ("settings", {"max_new_tokens": 64}),
            ("runner_sha256", "b" * 64),
            ("mode", "train"),
            ("split", "test"),
            ("status", "failed"),
        )
        for key, value in mismatches:
            with self.subTest(key=key):
                dataset_raw, payload, receipt = _inputs()
                receipt[key] = value
                with self.assertRaises(ValueError):
                    evaluate_verifier_pairs.evaluate_report(
                        dataset_raw, _json_bytes(payload), _json_bytes(receipt)
                    )

    def test_rejects_duplicate_json_keys(self) -> None:
        dataset_raw, payload, _ = _inputs()
        duplicate_receipt = b'{"status":"success","status":"failed"}'

        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), duplicate_receipt
            )

    def test_rejects_train_rows_that_differ_from_the_train_renderer(self) -> None:
        dataset_raw, payload, receipt = _inputs("train")
        payload["train_rows"][0]["completion"] = '{"support":"supported"}'
        receipt["payload_sha256"] = _canonical_sha(payload)

        with self.assertRaisesRegex(ValueError, "train_rows"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_rejects_inconsistent_token_counts_and_finish_flags(self) -> None:
        dataset_raw, payload, receipt = _inputs()
        receipt["baseline_outputs"][0]["input_tokens"] = 2

        with self.assertRaisesRegex(ValueError, "token counts"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_rejects_generation_limit_flag_for_short_output(self) -> None:
        dataset_raw, payload, receipt = _inputs()
        receipt["baseline_outputs"][0]["hit_max_new_tokens"] = True

        with self.assertRaisesRegex(ValueError, "generation-limit flag"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_rejects_different_input_tokens_between_train_arms(self) -> None:
        dataset_raw, payload, receipt = _inputs("train")
        receipt["tuned_outputs"][0]["input_token_ids"] = [9]

        with self.assertRaisesRegex(ValueError, "input token IDs must match"):
            evaluate_verifier_pairs.evaluate_report(
                dataset_raw, _json_bytes(payload), _json_bytes(receipt)
            )

    def test_cli_output_is_exclusive_private_and_preserves_existing_file(self) -> None:
        dataset_raw, payload, receipt = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            dataset_path, payload_path, receipt_path = (
                directory / "dataset.json",
                directory / "payload.json",
                directory / "receipt.json",
            )
            output_path = directory / "report.json"
            dataset_path.write_bytes(dataset_raw)
            payload_path.write_bytes(_json_bytes(payload))
            receipt_path.write_bytes(_json_bytes(receipt))

            self.assertEqual(
                evaluate_verifier_pairs.main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--payload",
                        str(payload_path),
                        "--receipt",
                        str(receipt_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            original = output_path.read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate_verifier_pairs.main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--payload",
                        str(payload_path),
                        "--receipt",
                        str(receipt_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(output_path.read_bytes(), original)
