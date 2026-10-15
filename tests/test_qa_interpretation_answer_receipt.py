from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_qa_interpretation_controls import _independent_reconstruction

_REQUEST_SHA256 = "df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871"
_RUNNER_SHA256 = "a" * 64
_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
_TOKENIZER_SHA256 = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _fake_receipt(request: dict[str, object]) -> dict[str, object]:
    import hashlib

    from experiments.qa_interpretation_answer_receipt import request_sha256

    tokenizer = {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "template_sha256": _TEMPLATE_SHA256,
        "backend_tokenizer_sha256": _TOKENIZER_SHA256,
        "use_fast": True,
        "trust_remote_code": False,
        "package_versions": {
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "jinja2": "3.1.6",
        },
        "python_version": "3.12.10",
        "template_options": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "truncation": False,
        },
        "eos_token_id": 151645,
    }
    return {
        "schema_version": 1,
        "request_sha256": request_sha256(request),
        "runner_sha256": _RUNNER_SHA256,
        "model": {
            "model_id": request["model"]["model_id"],
            "revision": request["model"]["revision"],
            "generation_protocol_sha256": hashlib.sha256(
                _canonical_bytes(request["model"])
            ).hexdigest(),
        },
        "tokenizer": tokenizer,
        "runtime": {
            "python_version": "3.12.10",
            "torch_version": "2.7.1",
            "accelerate_version": "1.6.0",
            "cuda_version": "12.6",
            "gpu_name": "Tesla T4",
            "gpu_capability": "7.5",
            "model_load_seconds": 1.0,
            "gpu_compute_seconds": 2.0,
            "elapsed_seconds": 3.0,
            "gpu_peak_memory_bytes": 0,
            "request_count": 36,
            "modal_call_id": "fake-modal-call-36",
        },
        "outputs": [
            {
                "prompt_id": item["prompt_id"],
                "raw_output": "FAKE RECEIPT ONLY; no model was run",
                "input_tokens": 1,
                "input_token_ids": [100],
                "generated_tokens": 2,
                "generated_token_ids": [101, 151645],
                "stop_reason": "eos_token",
                "hit_max_new_tokens": False,
                "truncated": False,
            }
            for item in request["items"]
        ],
    }


class QAInterpretationAnswerReceiptTests(unittest.TestCase):
    def test_build_request_matches_independent_frozen_request(self) -> None:
        module_name = "experiments.qa_interpretation_answer_receipt"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name),
            "the offline answer request contract module must exist",
        )
        receipt = importlib.import_module(module_name)
        items, _ = _independent_reconstruction()
        expected = {
            "schema_version": 1,
            "mode": "qa-interpretation-controls-answer-request-v1",
            "prepared_payload_sha256": (
                "36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1"
            ),
            "model": {
                "model_id": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "precision": "float16",
                "attn_implementation": "eager",
                "trust_remote_code": False,
                "use_safetensors": True,
                "use_fast": True,
                "enable_thinking": False,
                "do_sample": False,
                "num_beams": 1,
                "seed": 0,
                "max_new_tokens": 128,
                "max_prompt_tokens": 1024,
            },
            "tokenizer": {
                "template_sha256": _TEMPLATE_SHA256,
                "backend_tokenizer_sha256": _TOKENIZER_SHA256,
                "package_versions": {
                    "transformers": "4.51.3",
                    "tokenizers": "0.21.1",
                    "jinja2": "3.1.6",
                },
                "template_options": {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "truncation": False,
                },
                "eos_token_id": 151645,
            },
            "items": items,
        }

        request = receipt.build_request()
        self.assertEqual(request, expected)
        self.assertEqual(hashlib.sha256(_canonical_bytes(request)).hexdigest(), _REQUEST_SHA256)
        self.assertEqual(receipt.request_sha256(request), _REQUEST_SHA256)

    def test_build_request_deep_copies_pinned_model_settings(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        legacy = importlib.import_module("experiments.qa_contrast_answer_receipt")
        request = receipt.build_request()
        self.assertIsNot(request["model"], legacy._MODEL)
        request["model"]["seed"] = 7
        self.assertEqual(receipt.build_request()["model"]["seed"], 0)

    def test_validate_request_accepts_only_the_exact_pinned_request(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        validate = getattr(receipt, "validate_request", None)
        self.assertTrue(callable(validate), "the pinned request validator is missing")
        request = receipt.build_request()
        self.assertIs(validate(request), request)
        mutations = (
            lambda value: value.__setitem__("mode", "other-mode"),
            lambda value: value["model"].__setitem__("seed", 1),
            lambda value: value["items"][0]["messages"][1].__setitem__("content", "changed"),
            lambda value: value["items"].reverse(),
            lambda value: value.__setitem__("references_sha256", "f" * 64),
            lambda value: value.__setitem__("items", tuple(value["items"])),
            lambda value: value["model"].__setitem__("use_fast", 1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(request)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "request"):
                    validate(changed)

    def test_validate_result_accepts_a_complete_fake_36_output_receipt(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        validate = getattr(receipt, "validate_result", None)
        self.assertTrue(callable(validate), "the offline receipt validator is missing")
        request = receipt.build_request()
        fake = _fake_receipt(request)
        self.assertIs(validate(fake, request, _RUNNER_SHA256), fake)
        self.assertEqual(len(fake["outputs"]), 36)
        self.assertEqual(fake["runtime"]["request_count"], 36)
        self.assertTrue(all(output["raw_output"].startswith("FAKE") for output in fake["outputs"]))

    def test_validate_result_rejects_incomplete_python_and_blank_cuda_versions(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        for target, field, value in (
            ("tokenizer", "python_version", "3.12."),
            ("runtime", "python_version", "3.12."),
            ("runtime", "cuda_version", "   "),
        ):
            with self.subTest(target=target, field=field, value=value):
                fake = _fake_receipt(request)
                fake[target][field] = value
                with self.assertRaises(ValueError):
                    receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_build_request_rejects_source_and_prepared_payload_drift(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        with tempfile.TemporaryDirectory() as directory:
            changed_source = Path(directory) / "changed-source.py"
            changed_source.write_bytes(b"# drifted source\n")
            with patch.object(receipt, "_PREPARER_PATH", changed_source):
                with self.assertRaisesRegex(ValueError, "preparer source"):
                    receipt.build_request()
            with patch.object(receipt, "_LEGACY_RECEIPT_PATH", changed_source):
                with self.assertRaisesRegex(ValueError, "legacy QA contrast"):
                    receipt.build_request()
        with patch.object(
            receipt.prepare_qa_interpretation_controls,
            "prepare_controls",
            return_value=({"schema_version": 1}, {}, {}),
        ):
            with self.assertRaisesRegex(ValueError, "prepared QA interpretation payload"):
                receipt.build_request()

    def test_validate_result_rejects_provenance_and_pin_drift(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        cases = (
            (
                "request hash",
                lambda fake: fake.__setitem__("request_sha256", "b" * 64),
                "provenance",
            ),
            ("runner hash", lambda fake: fake.__setitem__("runner_sha256", "b" * 64), "provenance"),
            (
                "protocol hash",
                lambda fake: fake["model"].__setitem__("generation_protocol_sha256", "b" * 64),
                "protocol hash",
            ),
            (
                "fast boolean",
                lambda fake: fake["tokenizer"].__setitem__("use_fast", 1),
                "tokenizer provenance",
            ),
            (
                "template boolean",
                lambda fake: fake["tokenizer"]["template_options"].__setitem__("tokenize", 1),
                "tokenizer provenance",
            ),
            (
                "package version",
                lambda fake: fake["tokenizer"]["package_versions"].__setitem__(
                    "tokenizers", "0.21.2"
                ),
                "tokenizer provenance",
            ),
            (
                "runtime request count boolean",
                lambda fake: fake["runtime"].__setitem__("request_count", True),
                "runtime provenance",
            ),
            (
                "GPU identity",
                lambda fake: fake["runtime"].__setitem__("gpu_name", "CPU"),
                "runtime provenance",
            ),
            (
                "NaN timing",
                lambda fake: fake["runtime"].__setitem__("elapsed_seconds", float("nan")),
                "finite and nonnegative",
            ),
            (
                "infinite timing",
                lambda fake: fake["runtime"].__setitem__("model_load_seconds", float("inf")),
                "finite and nonnegative",
            ),
            (
                "negative timing",
                lambda fake: fake["runtime"].__setitem__("gpu_compute_seconds", -0.1),
                "finite and nonnegative",
            ),
            (
                "negative peak memory",
                lambda fake: fake["runtime"].__setitem__("gpu_peak_memory_bytes", -1),
                "runtime provenance",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(case=name):
                fake = _fake_receipt(request)
                mutate(fake)
                with self.assertRaisesRegex(ValueError, reason):
                    receipt.validate_result(fake, request, _RUNNER_SHA256)
        with self.assertRaisesRegex(ValueError, "expected runner SHA-256"):
            receipt.validate_result(_fake_receipt(request), request, "A" * 64)

    def test_validate_result_rejects_missing_reordered_and_duplicate_outputs(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        cases = (
            ("missing", lambda fake: fake["outputs"].pop(), "exactly 36"),
            ("reordered", lambda fake: fake["outputs"].reverse(), "output identity"),
            (
                "duplicate",
                lambda fake: fake["outputs"].__setitem__(1, copy.deepcopy(fake["outputs"][0])),
                "output identity",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(case=name):
                fake = _fake_receipt(request)
                mutate(fake)
                with self.assertRaisesRegex(ValueError, reason):
                    receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_validate_result_rejects_token_vector_mismatches_and_overlimits(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        cases = (
            (
                "boolean input count",
                lambda output: output.__setitem__("input_tokens", True),
            ),
            (
                "input count mismatch",
                lambda output: output.__setitem__("input_token_ids", []),
            ),
            (
                "boolean input ID",
                lambda output: output.__setitem__("input_token_ids", [True]),
            ),
            (
                "input token over limit",
                lambda output: (
                    output.__setitem__("input_tokens", 1025),
                    output.__setitem__("input_token_ids", [1] * 1025),
                ),
            ),
            (
                "boolean generated count",
                lambda output: output.__setitem__("generated_tokens", True),
            ),
            (
                "generated count mismatch",
                lambda output: output.__setitem__("generated_token_ids", [101]),
            ),
            (
                "boolean generated ID",
                lambda output: output.__setitem__("generated_token_ids", [True, 151645]),
            ),
            (
                "generated token over limit",
                lambda output: (
                    output.__setitem__("generated_tokens", 129),
                    output.__setitem__("generated_token_ids", [11] * 129),
                ),
            ),
            ("oversized text", lambda output: output.__setitem__("raw_output", "x" * 65_537)),
        )
        for name, mutate in cases:
            with self.subTest(case=name):
                fake = _fake_receipt(request)
                mutate(fake["outputs"][0])
                with self.assertRaisesRegex(ValueError, "output identity, text, or token counts"):
                    receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_validate_result_accepts_exact_token_caps_and_eos_at_cap(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        for stop_reason, token_ids, truncated in (
            ("eos_token", [11] * 127 + [151645], False),
            ("max_new_tokens", [11] * 128, True),
        ):
            with self.subTest(stop_reason=stop_reason):
                fake = _fake_receipt(request)
                output = fake["outputs"][0]
                output["generated_tokens"] = 128
                output["generated_token_ids"] = token_ids
                output["stop_reason"] = stop_reason
                output["hit_max_new_tokens"] = True
                output["truncated"] = truncated
                receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_validate_result_rejects_early_eos_and_inconsistent_stop_flags(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        cases = (
            (
                "early EOS",
                lambda output: output.__setitem__("generated_token_ids", [151645, 101]),
                "EOS stop",
            ),
            (
                "repeated EOS",
                lambda output: output.__setitem__("generated_token_ids", [151645, 151645]),
                "EOS stop",
            ),
            (
                "truncated EOS",
                lambda output: output.__setitem__("truncated", True),
                "output identity",
            ),
            (
                "max reason with EOS",
                lambda output: (
                    output.__setitem__("stop_reason", "max_new_tokens"),
                    output.__setitem__("truncated", True),
                ),
                "token-cap stop",
            ),
            (
                "max reason below cap",
                lambda output: (
                    output.__setitem__("stop_reason", "max_new_tokens"),
                    output.__setitem__("truncated", True),
                ),
                "token-cap stop",
            ),
            (
                "invalid reason",
                lambda output: output.__setitem__("stop_reason", "other"),
                "stop reason",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(case=name):
                fake = _fake_receipt(request)
                mutate(fake["outputs"][0])
                if name in {"max reason with EOS", "max reason below cap"}:
                    fake["outputs"][0]["generated_tokens"] = (
                        128 if name == "max reason with EOS" else 2
                    )
                    fake["outputs"][0]["generated_token_ids"] = (
                        [11] * 127 + [151645] if name == "max reason with EOS" else [11, 11]
                    )
                    fake["outputs"][0]["hit_max_new_tokens"] = name == "max reason with EOS"
                with self.assertRaisesRegex(ValueError, reason):
                    receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_validate_result_accepts_exact_aggregate_token_boundaries(self) -> None:
        receipt = importlib.import_module("experiments.qa_interpretation_answer_receipt")
        request = receipt.build_request()
        fake = _fake_receipt(request)
        for output in fake["outputs"]:
            output["input_tokens"] = 1024
            output["input_token_ids"] = list(range(1024))
            output["generated_tokens"] = 128
            output["generated_token_ids"] = [11] * 128
            output["stop_reason"] = "max_new_tokens"
            output["hit_max_new_tokens"] = True
            output["truncated"] = True
        receipt.validate_result(fake, request, _RUNNER_SHA256)

    def test_contract_imports_and_builds_without_gpu_runtime_packages(self) -> None:
        script = """
import builtins
blocked = {"modal", "torch", "transformers", "tokenizers"}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("unexpected runtime import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments.qa_interpretation_answer_receipt import build_request, validate_request
validate_request(build_request())
print("offline-contract-ok")
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{_ROOT / 'src'}:{_ROOT}"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "offline-contract-ok")


if __name__ == "__main__":
    unittest.main()
