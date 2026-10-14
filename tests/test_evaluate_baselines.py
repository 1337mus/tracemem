"""Offline receipt-evaluator tests."""

import hashlib
import json
import unittest

from experiments import evaluate_baselines, modal_baseline
from tracemem.challenge import load_challenge


def _receipt(mode: str, prompt_version: str = "v1") -> dict[str, object]:
    payload = modal_baseline.build_payload(mode, prompt_version=prompt_version)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "prompt_version": prompt_version,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": modal_baseline.payload_sha256(payload),
        "model": payload["model"],
        "package_versions": {"torch": "2.7.1", "transformers": "4.51.3"},
        "gpu_compute_seconds": 2.5,
        "model_load_seconds": 1.25,
        "remote_function_elapsed_seconds": 8.0,
        "peak_gpu_memory_bytes": 123_456,
        "modal_function_call_id": "fc-synthetic-test",
        "modal_app_id": "ap-synthetic-test",
        "python_version": "3.12.0",
        "torch_cuda_version": "12.8",
        "gpu_name": "NVIDIA T4",
    }
    if mode == "extraction":
        cases, _ = load_challenge()
        outputs = [
            {
                "case_id": case["case_id"],
                "raw_output": json.dumps({"claims": case["expected"]}),
                "input_tokens": 100,
                "generated_tokens": 20,
                "hit_max_new_tokens": False,
            }
            for case in cases
        ]
        receipt["case_count"] = len(outputs)
        receipt["generated_outputs"] = outputs
    else:
        vectors = {item["text_sha256"]: [1.0, *([0.0] * 383)] for item in payload["items"]}
        receipt["vectors"] = vectors
        receipt["vector_count"] = len(vectors)
    return receipt


def _encode(receipt: dict[str, object]) -> bytes:
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")


class EvaluateReceiptTests(unittest.TestCase):
    def test_all_gold_extraction_receipt_is_evaluated_verbatim(self) -> None:
        raw = _encode(_receipt("extraction"))

        report = evaluate_baselines.evaluate_receipt(raw)

        self.assertEqual(report["mode"], "extraction")
        self.assertEqual(report["prompt_version"], "v1")
        self.assertIn("not an independent evaluation", report["evaluation_scope"])
        self.assertEqual(report["raw_receipt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["token_totals"], {"input": 2_400, "generated": 480})
        self.assertEqual(report["truncated_cases"], 0)
        self.assertEqual(report["result"]["exact_match"]["f1"], 1.0)
        self.assertEqual(report["result"]["evidence_integrity"]["invalid_claims"], 0)
        self.assertIn("diagnostic", report["diagnostic_note"])
        self.assertEqual(report["provenance"]["modal_app_id"], "ap-synthetic-test")

    def test_prompt_versions_are_explicit_and_cannot_be_auto_detected(self) -> None:
        for version, opposite in (("v1", "explicit-v2"), ("explicit-v2", "v1")):
            with self.subTest(version=version):
                raw = _encode(_receipt("extraction", prompt_version=version))
                report = evaluate_baselines.evaluate_receipt(raw, prompt_version=version)
                self.assertEqual(report["prompt_version"], version)
                with self.assertRaisesRegex(ValueError, "prompt version|payload hash"):
                    evaluate_baselines.evaluate_receipt(raw, prompt_version=opposite)

    def test_legacy_v1_receipts_without_new_metadata_remain_valid(self) -> None:
        receipt = _receipt("extraction")
        receipt.pop("prompt_version")

        report = evaluate_baselines.evaluate_receipt(_encode(receipt))

        self.assertEqual(report["prompt_version"], "v1")
        self.assertNotIn("runner_sha256", report["provenance"])
        with self.assertRaisesRegex(ValueError, "payload hash"):
            evaluate_baselines.evaluate_receipt(_encode(receipt), prompt_version="explicit-v2")

        receipt["runner_sha256"] = "a" * 64
        report = evaluate_baselines.evaluate_receipt(_encode(receipt))
        self.assertEqual(report["provenance"]["runner_sha256"], "a" * 64)

        receipt["runner_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "runner SHA-256"):
            evaluate_baselines.evaluate_receipt(_encode(receipt))

    def test_tampered_input_hash_payload_hash_and_model_are_rejected(self) -> None:
        for key in ("input_sha256", "payload_sha256"):
            with self.subTest(key=key):
                receipt = _receipt("extraction")
                value = receipt[key]
                assert isinstance(value, str)
                receipt[key] = ("0" if value[0] != "0" else "1") + value[1:]
                with self.assertRaisesRegex(ValueError, "hash"):
                    evaluate_baselines.evaluate_receipt(_encode(receipt))

        receipt = _receipt("extraction")
        model = dict(receipt["model"])
        model["revision"] = "0" * 40
        receipt["model"] = model
        with self.assertRaisesRegex(ValueError, "model"):
            evaluate_baselines.evaluate_receipt(_encode(receipt))

    def test_duplicate_missing_and_extra_extraction_outputs_are_rejected(self) -> None:
        duplicate = _receipt("extraction")
        outputs = list(duplicate["generated_outputs"])
        outputs[-1] = dict(outputs[0])
        duplicate["generated_outputs"] = outputs
        with self.assertRaisesRegex(ValueError, "duplicate extraction output"):
            evaluate_baselines.evaluate_receipt(_encode(duplicate))

        missing = _receipt("extraction")
        missing["generated_outputs"] = missing["generated_outputs"][:-1]
        with self.assertRaisesRegex(ValueError, "missing extraction outputs"):
            evaluate_baselines.evaluate_receipt(_encode(missing))

        extra = _receipt("extraction")
        extra_outputs = list(extra["generated_outputs"])
        unexpected = dict(extra_outputs[0])
        unexpected["case_id"] = "not-a-fixture-case"
        extra_outputs.append(unexpected)
        extra["generated_outputs"] = extra_outputs
        with self.assertRaisesRegex(ValueError, "unexpected extraction output IDs"):
            evaluate_baselines.evaluate_receipt(_encode(extra))

    def test_extraction_reports_generation_limit_hits(self) -> None:
        receipt = _receipt("extraction")
        outputs = list(receipt["generated_outputs"])
        first = dict(outputs[0])
        first["generated_tokens"] = 512
        first["hit_max_new_tokens"] = True
        outputs[0] = first
        receipt["generated_outputs"] = outputs

        report = evaluate_baselines.evaluate_receipt(_encode(receipt))

        self.assertEqual(report["truncated_cases"], 1)
        self.assertEqual(report["token_totals"]["generated"], 972)

    def test_valid_finite_embedding_receipt_runs_local_retrieval_benchmark(self) -> None:
        report = evaluate_baselines.evaluate_receipt(_encode(_receipt("embeddings")))

        self.assertEqual(report["mode"], "embeddings")
        self.assertEqual(set(report["result"]["modes"]), {"bm25", "vector", "hybrid"})
        self.assertEqual(report["result"]["input_sha256"], report["input_sha256"])

    def test_embedding_receipt_rejects_nan_vectors(self) -> None:
        receipt = _receipt("embeddings")
        vectors = dict(receipt["vectors"])
        first_hash = next(iter(vectors))
        vector = list(vectors[first_hash])
        vector[0] = float("nan")
        vectors[first_hash] = vector
        receipt["vectors"] = vectors
        raw = json.dumps(receipt, separators=(",", ":")).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
            evaluate_baselines.evaluate_receipt(raw)

    def test_prompt_version_cli_defaults_to_v1_and_accepts_explicit_v2(self) -> None:
        parser = evaluate_baselines.build_parser()
        self.assertEqual(parser.parse_args(["--input", "receipt.json"]).prompt_version, "v1")
        self.assertEqual(
            parser.parse_args(
                ["--input", "receipt.json", "--prompt-version", "explicit-v2"]
            ).prompt_version,
            "explicit-v2",
        )


if __name__ == "__main__":
    unittest.main()
