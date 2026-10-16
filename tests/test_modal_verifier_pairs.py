"""Offline contracts for the bounded paired verifier runner."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from experiments import modal_verifier_pairs
from training.verifier_data import render_candidate_prompt, validate_verifier_dataset

_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _ROOT / "examples/synthetic/verifier-pairs-v1.json"


class ModalVerifierPairsTests(unittest.TestCase):
    def test_modal_envelope_is_single_use_source_free_and_bounded(self) -> None:
        options = modal_verifier_pairs.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["buffer_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["startup_timeout"], 600)
        self.assertEqual(options["timeout"], 3600)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_prepare_cli_stays_offline_and_reports_no_remote_start(self) -> None:
        original_import = builtins.__import__
        stdout = io.StringIO()

        def reject_remote_imports(name: str, *args: object, **kwargs: object) -> object:
            if name == "modal" or name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"prepare imported remote runtime {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_remote_imports):
            with contextlib.redirect_stdout(stdout):
                exit_code = modal_verifier_pairs.main(["prepare", "--mode", "baseline"])

        prepared = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIs(prepared["remote_started"], False)
        self.assertEqual(prepared["item_count"], 12)
        self.assertEqual(len(prepared["payload_sha256"]), 64)

    def test_baseline_payload_contains_only_label_free_dev_candidate_prompts(self) -> None:
        dataset = json.loads(_DATASET.read_text(encoding="utf-8"))
        cases = validate_verifier_dataset(dataset)
        expected = {
            f"{case['case_id']}::{candidate['candidate_id']}": render_candidate_prompt(
                case, candidate["candidate_id"]
            )
            for case in cases
            if case["split"] == "dev"
            for candidate in case["candidates"]
        }

        payload = modal_verifier_pairs.build_payload("baseline")

        self.assertEqual(payload["mode"], "baseline")
        self.assertEqual(payload["split"], "dev")
        self.assertEqual(len(expected), 12)
        self.assertEqual(
            payload["items"],
            [{"item_id": item_id, "prompt": prompt} for item_id, prompt in expected.items()],
        )
        self.assertEqual(payload["train_rows"], [])
        self.assertTrue(all(set(item) == {"item_id", "prompt"} for item in payload["items"]))

    def test_train_payload_contains_exactly_48_reviewed_rows_and_no_test_items(self) -> None:
        payload = modal_verifier_pairs.build_payload("train")

        self.assertEqual(len(payload["items"]), 12)
        self.assertEqual(len(payload["train_rows"]), 48)
        self.assertTrue(all(set(row) == {"prompt", "completion"} for row in payload["train_rows"]))
        self.assertTrue(all(set(item) == {"item_id", "prompt"} for item in payload["items"]))
        case_ids = {item["item_id"].split("::", maxsplit=1)[0] for item in payload["items"]}
        dataset_cases = validate_verifier_dataset(json.loads(_DATASET.read_text()))
        expected_case_ids = {case["case_id"] for case in dataset_cases if case["split"] == "dev"}
        self.assertEqual(
            case_ids,
            expected_case_ids,
        )

    def test_dataset_and_renderer_changes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(_DATASET.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with patch.object(modal_verifier_pairs, "_DATASET_PATH", changed):
                with self.assertRaisesRegex(ValueError, "dataset SHA-256"):
                    modal_verifier_pairs.build_payload("baseline")
            changed.write_text("changed prompt renderer", encoding="utf-8")
            with patch.object(modal_verifier_pairs, "_RENDERER_PATH", changed):
                with self.assertRaisesRegex(ValueError, "renderer SHA-256"):
                    modal_verifier_pairs.build_payload("baseline")

    def test_remote_rejects_tampered_payload_before_model_runtime_imports(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        digest = modal_verifier_pairs.payload_sha256(payload)
        request = {
            "payload": payload,
            "payload_sha256": digest,
            "runner_sha256": payload["runner_sha256"],
        }
        payload["items"][0]["prompt"] += " changed after approval"
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"model runtime imported before hash guard: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            with self.assertRaisesRegex(ValueError, "payload SHA-256"):
                modal_verifier_pairs._remote_compute(request)

    def test_remote_rejects_rehashed_invalid_item_shape_before_model_imports(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        payload["items"][0]["labels"] = {"support": "unsupported"}
        request = {
            "payload": payload,
            "payload_sha256": modal_verifier_pairs.payload_sha256(payload),
            "runner_sha256": payload["runner_sha256"],
        }
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"model runtime imported before shape guard: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            with self.assertRaisesRegex(ValueError, "item shape"):
                modal_verifier_pairs._remote_compute(request)

    def test_remote_rejects_runner_hash_mismatch_before_model_imports(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        request = {
            "payload": payload,
            "payload_sha256": modal_verifier_pairs.payload_sha256(payload),
            "runner_sha256": "0" * 64,
        }
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"model runtime imported before runner guard: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            with self.assertRaisesRegex(ValueError, "runner SHA-256"):
                modal_verifier_pairs._remote_compute(request)

    def test_remote_does_not_require_injected_modal_distribution_metadata(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        request = {
            "payload": payload,
            "payload_sha256": modal_verifier_pairs.payload_sha256(payload),
            "runner_sha256": payload["runner_sha256"],
        }
        modal_module = ModuleType("modal")
        torch_module = ModuleType("torch")
        torch_module.cuda = SimpleNamespace(
            is_available=lambda: False,
            is_bf16_supported=lambda: False,
        )
        peft_module = ModuleType("peft")
        for name in ("LoraConfig", "get_peft_model", "prepare_model_for_kbit_training"):
            setattr(peft_module, name, object())
        transformers_module = ModuleType("transformers")
        for name in ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"):
            setattr(transformers_module, name, object())
        pinned = {
            "torch": "2.7.1",
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "accelerate": "1.6.0",
            "peft": "0.15.2",
            "bitsandbytes": "0.45.5",
            "safetensors": "0.5.3",
        }

        def package_version(package: str) -> str:
            if package == "modal":
                from importlib.metadata import PackageNotFoundError

                raise PackageNotFoundError(package)
            return pinned[package]

        with (
            patch.dict(
                sys.modules,
                {
                    "modal": modal_module,
                    "torch": torch_module,
                    "peft": peft_module,
                    "transformers": transformers_module,
                },
            ),
            patch("importlib.metadata.version", side_effect=package_version),
            patch("platform.python_version_tuple", return_value=("3", "12", "1")),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires an A10G CUDA device"):
                modal_verifier_pairs._remote_compute(request)

    def test_stale_cli_approval_is_rejected_before_modal_or_receipt_creation(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(
                modal_verifier_pairs,
                "_build_modal_function",
                side_effect=AssertionError("stale approval reached Modal construction"),
            ):
                with contextlib.redirect_stderr(stderr):
                    exit_code = modal_verifier_pairs.main(
                        [
                            "launch",
                            "--mode",
                            "baseline",
                            "--approve-payload-sha256",
                            "0" * 64,
                            "--approve-runner-sha256",
                            payload["runner_sha256"],
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(exit_code, 2)
            self.assertIn("exact current payload", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_rehashed_changed_prompt_is_rejected_before_receipt_reservation(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        payload["items"][0]["prompt"] += " private text supplied by caller"
        from experiments import sft_launch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(
                sft_launch,
                "_reserve_output",
                side_effect=AssertionError("changed payload reached receipt reservation"),
            ):
                with self.assertRaisesRegex(ValueError, "differs from the current fixed payload"):
                    modal_verifier_pairs.run_once(
                        payload,
                        approved_payload_sha256=modal_verifier_pairs.payload_sha256(payload),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )
            self.assertFalse(output.exists())

    def test_receipt_reservation_is_exclusive_and_preserves_existing_file(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_text("owned", encoding="utf-8")
            from experiments import sft_launch

            with patch.object(
                sft_launch,
                "_acquire_launch_lock",
                side_effect=AssertionError("existing receipt should fail before lock acquisition"),
            ):
                with self.assertRaisesRegex(FileExistsError, "must be new"):
                    modal_verifier_pairs.run_once(
                        payload,
                        approved_payload_sha256=modal_verifier_pairs.payload_sha256(payload),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )
            self.assertEqual(output.read_text(encoding="utf-8"), "owned")

    def test_launch_failure_writes_a_raw_receipt_before_reraising(self) -> None:
        payload = modal_verifier_pairs.build_payload("baseline")
        from experiments import sft_launch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            with (
                patch.object(sft_launch, "_LAUNCH_LOCK", Path(directory) / "launch.lock"),
                patch.object(
                    sft_launch,
                    "verify_personal_workspace",
                    side_effect=RuntimeError("workspace stale"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "workspace stale"):
                    modal_verifier_pairs.run_once(
                        payload,
                        approved_payload_sha256=modal_verifier_pairs.payload_sha256(payload),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )
            failure = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "launch_failed")
            self.assertIs(failure["remote_started"], False)
            self.assertEqual(failure["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
