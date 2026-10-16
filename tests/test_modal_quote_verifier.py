from __future__ import annotations

import builtins
import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import experiments.modal_quote_verifier as modal_quote_verifier
from experiments import quote_verifier_payload


def _fake_runtime(
    *, long_prompt: bool = False
) -> tuple[dict[str, types.ModuleType], dict[str, Any]]:
    state: dict[str, Any] = {"tokenize_calls": 0, "generate_calls": 0, "long_prompt": long_prompt}

    class FakeTensor:
        ndim = 2

        def __init__(self, tokens: int) -> None:
            self.shape = (1, tokens)

        def to(self, device: object) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[list[int]]:
            return [[11] * self.shape[-1]]

    class FakeToken:
        def item(self) -> int:
            return 7

    class FakeCompletion:
        shape = (1,)

        def detach(self) -> FakeCompletion:
            return self

        def cpu(self) -> FakeCompletion:
            return self

        def tolist(self) -> list[int]:
            return [7]

        def __getitem__(self, index: int) -> FakeToken:
            return FakeToken()

    class FakeGenerated:
        def __getitem__(self, index: tuple[int, slice]) -> FakeCompletion:
            return FakeCompletion()

    class FakeEvent:
        def __init__(self, enable_timing: bool) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, end: FakeEvent) -> float:
            return 1.0

    class FakeTokenizer:
        chat_template = "fake chat template"
        is_fast = True

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeTokenizer:
            return cls()

        def apply_chat_template(
            self, messages: list[dict[str, str]], **kwargs: object
        ) -> FakeTensor:
            state["tokenize_calls"] += 1
            return FakeTensor(4097 if state["long_prompt"] and state["tokenize_calls"] == 1 else 1)

        def decode(self, token_ids: FakeCompletion, *, skip_special_tokens: bool) -> str:
            return "```malformed output```"

    class FakeModel:
        device = "cuda:0"
        generation_config = types.SimpleNamespace(eos_token_id=7, pad_token_id=None)

        def eval(self) -> FakeModel:
            return self

        def generate(self, *args: object, **kwargs: object) -> FakeGenerated:
            state["generate_calls"] += 1
            state.setdefault("tokenize_calls_at_first_generation", state["tokenize_calls"])
            return FakeGenerated()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeModel:
            return FakeModel()

    torch_module = types.ModuleType("torch")
    for name, value in {
        "Tensor": FakeTensor,
        "bfloat16": object(),
        "inference_mode": nullcontext,
        "manual_seed": lambda seed: None,
        "version": types.SimpleNamespace(cuda="fake-cuda"),
        "cuda": types.SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: None,
            current_device=lambda: 0,
            get_device_name=lambda device: "fake-A10G",
            reset_peak_memory_stats=lambda device: None,
            Event=FakeEvent,
            synchronize=lambda: None,
            max_memory_allocated=lambda device: 1234,
        ),
    }.items():
        setattr(torch_module, name, value)
    modal_module = types.ModuleType("modal")
    for name, value in {"current_function_call_id": lambda: "fc-fake"}.items():
        setattr(modal_module, name, value)
    transformers_module = types.ModuleType("transformers")
    for name, value in {
        "AutoModelForCausalLM": FakeAutoModel,
        "AutoTokenizer": FakeTokenizer,
    }.items():
        setattr(transformers_module, name, value)
    return {
        "torch": torch_module,
        "modal": modal_module,
        "transformers": transformers_module,
    }, state


class ModalQuoteVerifierTests(unittest.TestCase):
    def test_modal_function_options_are_single_use_bounded_and_source_free(self) -> None:
        options = modal_quote_verifier.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["timeout"], 2400)
        self.assertTrue(options["single_use_containers"])
        self.assertFalse(options["include_source"])

    def test_malformed_remote_request_is_rejected_before_model_imports(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"model runtime imported before request validation: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "request keys"):
                modal_quote_verifier._remote_compute({})

    def test_tampered_canonical_payload_is_rejected_before_model_imports(self) -> None:
        payload = quote_verifier_payload.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": quote_verifier_payload.payload_sha256(payload),
            "runner_sha256": payload["runner_source_sha256"],
        }
        payload["items"][0]["prompt"] += " TAMPERED"
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"model runtime imported before hash rejection: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "payload hash"):
                modal_quote_verifier._remote_compute(request)

    def test_runner_hash_must_match_approved_payload_before_model_imports(self) -> None:
        payload = quote_verifier_payload.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": quote_verifier_payload.payload_sha256(payload),
            "runner_sha256": "0" * 64,
        }
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"model runtime imported before runner hash rejection: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "runner hash"):
                modal_quote_verifier._remote_compute(request)

    def test_rehashed_but_unallowlisted_items_are_rejected_before_model_imports(self) -> None:
        payload = quote_verifier_payload.build_payload()
        payload["items"][0]["candidate_count"] += 1
        item_bytes = json.dumps(
            payload["items"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        payload["items_sha256"] = hashlib.sha256(item_bytes).hexdigest()
        request = {
            "payload": payload,
            "payload_sha256": quote_verifier_payload.payload_sha256(payload),
            "runner_sha256": payload["runner_source_sha256"],
        }
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"model runtime imported before item hash rejection: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "item hash"):
                modal_quote_verifier._remote_compute(request)

    def test_runner_source_hash_changes_the_approved_payload_hash(self) -> None:
        payload = quote_verifier_payload.build_payload()
        digest = quote_verifier_payload.payload_sha256(payload)
        original_source_hash = quote_verifier_payload._source_sha256
        runner_path = Path(modal_quote_verifier.__file__)

        def changed_runner_hash(path: Path) -> str:
            return "f" * 64 if path == runner_path else original_source_hash(path)

        with patch.object(
            quote_verifier_payload, "_source_sha256", side_effect=changed_runner_hash
        ):
            changed = quote_verifier_payload.build_payload()

        self.assertNotEqual(digest, quote_verifier_payload.payload_sha256(changed))

    def test_remote_tokenizes_all_items_before_generation_and_preserves_raw_outputs(self) -> None:
        payload = quote_verifier_payload.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": quote_verifier_payload.payload_sha256(payload),
            "runner_sha256": payload["runner_source_sha256"],
        }
        modules, state = _fake_runtime()
        distributions = [
            types.SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in (
                ("modal", "1.6.1"),
                ("torch", "2.7.1"),
                ("transformers", "4.51.3"),
                ("accelerate", "1.6.0"),
                ("tokenizers", "0.21.4"),
                ("huggingface-hub", "0.36.2"),
            )
        ]

        with (
            patch("importlib.metadata.distributions", return_value=distributions),
            patch.dict(sys.modules, modules),
        ):
            receipt = modal_quote_verifier._remote_compute(request)

        self.assertEqual(state["tokenize_calls"], 10)
        self.assertEqual(state["tokenize_calls_at_first_generation"], 10)
        self.assertEqual(state["generate_calls"], 10)
        self.assertEqual(receipt["modal_function_call_id"], "fc-fake")
        self.assertEqual(receipt["case_count"], 10)
        self.assertEqual(receipt["item_count"], 10)
        self.assertEqual(len(receipt["generated_outputs"]), 10)
        first = receipt["generated_outputs"][0]
        self.assertEqual(first["raw_output"], "```malformed output```")
        self.assertEqual(first["input_token_ids"], [11])
        self.assertEqual(first["output_token_ids"], [7])
        self.assertTrue(first["ended_with_eos"])
        self.assertFalse(first["hit_max_new_tokens"])
        self.assertEqual(first["finish_reason"], "eos")
        self.assertEqual(
            [row["item_id"] for row in receipt["generated_outputs"]],
            [item["item_id"] for item in payload["items"]],
        )
        self.assertEqual(
            receipt["package_versions"],
            {
                "modal": "1.6.1",
                "torch": "2.7.1",
                "transformers": "4.51.3",
                "accelerate": "1.6.0",
                "tokenizers": "0.21.4",
                "huggingface-hub": "0.36.2",
            },
        )
        self.assertEqual(receipt["gpu_name"], "fake-A10G")
        self.assertGreaterEqual(receipt["remote_function_elapsed_seconds"], 0)
        self.assertGreaterEqual(receipt["model_load_seconds"], 0)
        self.assertEqual(receipt["peak_gpu_memory_bytes"], 1234)

    def test_remote_rejects_overlong_input_before_any_generation(self) -> None:
        payload = quote_verifier_payload.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": quote_verifier_payload.payload_sha256(payload),
            "runner_sha256": payload["runner_source_sha256"],
        }
        modules, state = _fake_runtime(long_prompt=True)

        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "truncation is disabled"):
                modal_quote_verifier._remote_compute(request)

        self.assertEqual(state["tokenize_calls"], 1)
        self.assertEqual(state["generate_calls"], 0)

    def test_default_cli_prepares_payload_and_saves_audit_without_modal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "selection-audit.json"
            output = StringIO()
            with (
                patch.object(
                    modal_quote_verifier,
                    "_build_modal_function",
                    create=True,
                    side_effect=AssertionError("preparation must not build Modal"),
                ),
                patch.object(
                    modal_quote_verifier,
                    "verify_personal_workspace",
                    side_effect=AssertionError("preparation must not verify launch access"),
                ),
                redirect_stdout(output),
            ):
                result = modal_quote_verifier.main(
                    ["--show-payload", "--audit-output", str(audit_path)]
                )

            self.assertEqual(result, 0)
            prepared = json.loads(output.getvalue())
            self.assertFalse(prepared["remote_started"])
            self.assertEqual(prepared["item_count"], 10)
            self.assertTrue(audit_path.exists())
            self.assertEqual(len(json.loads(audit_path.read_text())["cases"]), 12)

    def test_invalid_cli_approval_refuses_before_modal_or_receipt_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            audit_path = Path(directory) / "audit.json"
            error = StringIO()
            with (
                patch.object(modal_quote_verifier, "_build_modal_function") as build_modal,
                redirect_stderr(error),
            ):
                result = modal_quote_verifier.main(
                    [
                        "--launch",
                        "--approve-payload-sha256",
                        "0" * 64,
                        "--approve-runner-sha256",
                        "0" * 64,
                        "--output",
                        str(receipt_path),
                        "--audit-output",
                        str(audit_path),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("exact prepared payload SHA-256", error.getvalue())
            self.assertFalse(receipt_path.exists())
            build_modal.assert_not_called()

    def test_receipt_collision_refuses_before_modal_and_preserves_existing_bytes(self) -> None:
        payload = quote_verifier_payload.build_payload()
        digest = quote_verifier_payload.payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(modal_quote_verifier.__file__).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            audit_path = Path(directory) / "audit.json"
            receipt_path.write_text("keep this receipt\n")
            error = StringIO()
            with (
                patch.object(modal_quote_verifier, "_build_modal_function") as build_modal,
                redirect_stderr(error),
            ):
                result = modal_quote_verifier.main(
                    [
                        "--launch",
                        "--approve-payload-sha256",
                        digest,
                        "--approve-runner-sha256",
                        runner_hash,
                        "--output",
                        str(receipt_path),
                        "--audit-output",
                        str(audit_path),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("FileExistsError", error.getvalue())
            self.assertEqual(receipt_path.read_text(), "keep this receipt\n")
            build_modal.assert_not_called()

    def test_workspace_verification_failure_writes_safe_exclusive_failure_receipt(self) -> None:
        payload = quote_verifier_payload.build_payload()
        digest = quote_verifier_payload.payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(modal_quote_verifier.__file__).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            audit_path = Path(directory) / "audit.json"
            error = StringIO()
            with (
                patch.object(
                    modal_quote_verifier,
                    "verify_personal_workspace",
                    side_effect=RuntimeError("fresh workspace verification failed"),
                ) as verify,
                patch.object(modal_quote_verifier, "_build_modal_function") as build_modal,
                redirect_stderr(error),
            ):
                result = modal_quote_verifier.main(
                    [
                        "--launch",
                        "--approve-payload-sha256",
                        digest,
                        "--approve-runner-sha256",
                        runner_hash,
                        "--output",
                        str(receipt_path),
                        "--audit-output",
                        str(audit_path),
                    ]
                )

            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(result, 2)
            verify.assert_called_once_with()
            build_modal.assert_not_called()
            self.assertEqual(receipt["status"], "launch_failed")
            self.assertFalse(receipt["remote_started"])
            self.assertIn("fresh workspace verification failed", receipt["error"])


if __name__ == "__main__":
    unittest.main()
