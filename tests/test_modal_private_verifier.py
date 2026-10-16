"""Offline contract tests for the bounded private verifier runner."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import stat
import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments import modal_private_verifier, sft_launch


def _candidate_manifest() -> dict[str, object]:
    return {
        "training_payload_sha256": (
            "384f9a02c6db633c90421d436db877898c02f0d9389f902fd6f881efff754861"
        ),
        "training_receipt_sha256": (
            "c5847c80375506c5cd143608141eec39d030052c3fa6ed39a6bf2c5244048832"
        ),
        "adapter_relative_path": (
            "runs/384f9a02c6db633c90421d436db877898c02f0d9389f902fd6f881efff754861/adapter"
        ),
        "adapter_artifact_sha256": {
            "adapter_config.json": (
                "e681a3435ab4a39f801ac4336d3dbb5339d2cdc3ed3a2a3b84d2ad5fa8aa1c77"
            ),
            "adapter_model.safetensors": (
                "24be4cef389ffb1a249a010a90afd42876b8aabc6611ce47a7ffb5b8b6d3e0ce"
            ),
        },
    }


def _one_item() -> list[dict[str, str]]:
    return [{"item_id": "opaque-candidate-01", "prompt": "A minimized private prompt."}]


class ModalPrivateVerifierTests(unittest.TestCase):
    def test_modal_envelope_is_bounded_source_free_single_container_inference(self) -> None:
        options = modal_private_verifier.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["buffer_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["startup_timeout"], 600)
        self.assertEqual(options["timeout"], 900)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_remote_function_serializes_source_free_under_64_kib(self) -> None:
        try:
            from modal._utils.function_utils import FunctionSourceInfo
            from modal._vendor import cloudpickle
        except ImportError as error:
            self.skipTest(f"Modal serialization dependencies unavailable: {error}")

        sentinel = "PRIVATE_SERIALIZATION_SENTINEL"
        payload = modal_private_verifier.build_payload(
            [{"item_id": "opaque-private-01", "prompt": sentinel + "x" * 11_000}],
            _candidate_manifest(),
        )
        remote = modal_private_verifier._make_remote_function(payload)
        serialized = FunctionSourceInfo(remote, serialized=True).serialized_function()

        cloudpickle.loads(serialized)
        self.assertNotIn(sentinel.encode(), serialized)
        self.assertLess(len(serialized), 64 * 1024)

    def test_build_payload_keeps_only_minimized_prompts_and_exact_private_provenance(self) -> None:
        items = _one_item()
        candidate_manifest = _candidate_manifest()

        payload = modal_private_verifier.build_payload(items, candidate_manifest)

        self.assertEqual(payload["items"], items)
        self.assertEqual(payload["candidate_manifest"], candidate_manifest)
        self.assertEqual(payload["mode"], "private_inference_only")
        self.assertFalse(payload["training_performed"])
        self.assertEqual(
            payload["provenance"]["permitted_use"],
            "explicit_user_authorized_private_evaluation_only",
        )
        self.assertIn("no_redistribution_license_asserted", payload["provenance"]["license"])
        self.assertEqual(set(payload["items"][0]), {"item_id", "prompt"})
        self.assertLessEqual(
            len(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()), 180_000
        )

    def test_run_once_rejects_changed_payload_before_launch(self) -> None:
        payload = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())
        approved_payload_sha256 = modal_private_verifier.payload_sha256(payload)
        approved_runner_sha256 = hashlib.sha256(
            Path(modal_private_verifier.__file__).read_bytes()
        ).hexdigest()
        payload["items"][0]["prompt"] += " changed after approval"
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "private-result.json"
            with patch.object(
                modal_private_verifier,
                "_build_modal_function",
                side_effect=AssertionError("payload approval failure reached Modal construction"),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "payload SHA-256"):
                    modal_private_verifier.run_once(
                        payload,
                        approved_payload_sha256=approved_payload_sha256,
                        approved_runner_sha256=approved_runner_sha256,
                        output_path=output_path,
                    )
            self.assertFalse(output_path.exists())

    def test_run_once_rejects_output_path_outside_local_private_directory(self) -> None:
        payload = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "private-result.json"
            with patch.object(
                modal_private_verifier,
                "_build_modal_function",
                side_effect=AssertionError("unsafe output path reached Modal construction"),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "under .local"):
                    modal_private_verifier.run_once(
                        payload,
                        approved_payload_sha256=modal_private_verifier.payload_sha256(payload),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output_path,
                    )
            self.assertFalse(output_path.exists())

    def test_build_payload_rejects_labels_and_unapproved_provenance_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "only item_id and prompt"):
            modal_private_verifier.build_payload(
                [{**_one_item()[0], "labels": {"verdict": "private"}}], _candidate_manifest()
            )

        manifest_with_source_path = {
            **_candidate_manifest(),
            "source_transcript_path": "private/transcript.json",
        }
        with self.assertRaisesRegex(ValueError, "candidate manifest"):
            modal_private_verifier.build_payload(_one_item(), manifest_with_source_path)

    def test_remote_rejects_altered_approved_payload_before_model_imports(self) -> None:
        payload = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())
        remote = modal_private_verifier._make_remote_function(payload)
        changed = deepcopy(payload)
        changed["items"][0]["prompt"] += " altered after approval"
        request = {
            "payload": changed,
            "payload_sha256": modal_private_verifier.payload_sha256(changed),
            "runner_sha256": payload["runner_sha256"],
        }
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError("model runtime imported before private approval guard")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            result = remote(request)
        self.assertEqual(result["status"], "remote_failed")
        self.assertEqual(
            result["error"], {"type": "ValueError", "message": "private verifier failed"}
        )

    def test_run_once_preserves_sanitized_remote_failure_and_approved_hashes(self) -> None:
        payload = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())
        payload_sha = modal_private_verifier.payload_sha256(payload)
        remote = modal_private_verifier._make_remote_function(payload)
        remote.__globals__["_REMOTE_COMPUTE"] = Mock(
            side_effect=ValueError("private prompt and output must not escape")
        )
        fake_function = SimpleNamespace(remote=remote)
        fake_app = SimpleNamespace(app_id="ap-offline-failure", run=lambda **_: nullcontext())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_dir = root / ".local"
            local_dir.mkdir(mode=0o700)
            private_dir = local_dir / "private-modal-verifier"
            output_path = private_dir / "failure.json"
            with (
                patch.object(modal_private_verifier, "_ROOT", root),
                patch.object(modal_private_verifier, "_LOCAL_DIR", local_dir),
                patch.object(modal_private_verifier, "_PRIVATE_OUTPUT_DIR", private_dir),
                patch.object(sft_launch, "_require_supported_python"),
                patch.object(
                    sft_launch,
                    "_acquire_launch_lock",
                    side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
                ),
                patch.object(sft_launch, "verify_personal_workspace"),
                patch.object(sft_launch, "verify_modal_inventory", return_value=[]),
                patch.object(
                    modal_private_verifier,
                    "_build_modal_function",
                    return_value=(fake_app, fake_function),
                ),
            ):
                receipt = modal_private_verifier.run_once(
                    payload,
                    approved_payload_sha256=payload_sha,
                    approved_runner_sha256=payload["runner_sha256"],
                    output_path=output_path,
                )

            self.assertEqual(receipt["status"], "remote_failed")
            self.assertEqual(
                receipt["error"], {"type": "ValueError", "message": "private verifier failed"}
            )
            self.assertEqual(receipt["payload_sha256"], payload_sha)
            self.assertEqual(receipt["runner_sha256"], payload["runner_sha256"])
            saved = output_path.read_text(encoding="utf-8")
            self.assertNotIn("private prompt", saved)
            self.assertNotIn("output must not escape", saved)

    def test_run_once_writes_private_fake_call_receipt_once_without_prompt_echo(self) -> None:
        items = _one_item()
        payload = modal_private_verifier.build_payload(items, _candidate_manifest())
        output_row = {
            "item_id": items[0]["item_id"],
            "raw_output": "A private answer.",
            "input_tokens": 8,
            "output_tokens": 3,
            "input_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "output_token_ids": [9, 10, 11],
            "finish_reason": "eos",
            "ended_with_eos": True,
            "hit_max_new_tokens": False,
        }
        payload_sha = modal_private_verifier.payload_sha256(payload)
        remote_result = {
            "schema_version": 1,
            "status": "success",
            "mode": "private_inference_only",
            "payload_sha256": payload_sha,
            "runner_sha256": payload["runner_sha256"],
            "provenance": payload["provenance"],
            "candidate_manifest": payload["candidate_manifest"],
            "model": payload["model"],
            "settings": payload["settings"],
            "prior_adapter": payload["prior_adapter"],
            "continued_adapter": payload["continued_adapter"],
            "adapter_artifact_sha256": {
                "prior": {
                    "adapter_config.json": payload["prior_adapter"]["config_sha256"],
                    "adapter_model.safetensors": payload["prior_adapter"]["safetensors_sha256"],
                },
                "continued": payload["continued_adapter"]["artifact_sha256"],
            },
            "training_performed": False,
            "versions": {"modal": "1.6.1", "torch": "2.7.1", "python": "3.12.1"},
            "tokenizer": {"class": "FixtureTokenizer"},
            "modal_function_call_id": "fc-offline-fixture",
            "base_outputs": [output_row],
            "prior_outputs": [output_row],
            "continued_outputs": [output_row],
            "timings": {"total_seconds": 1.0},
            "hardware": {"gpu_name": "A10G"},
        }
        fake_function = SimpleNamespace(remote=lambda request: remote_result)
        fake_app = SimpleNamespace(app_id="ap-offline-fixture", run=lambda **_: nullcontext())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_dir = root / ".local"
            local_dir.mkdir(mode=0o700)
            private_dir = local_dir / "private-modal-verifier"
            output_path = private_dir / "result.json"
            with (
                patch.object(modal_private_verifier, "_ROOT", root),
                patch.object(modal_private_verifier, "_LOCAL_DIR", local_dir),
                patch.object(modal_private_verifier, "_PRIVATE_OUTPUT_DIR", private_dir),
                patch.object(sft_launch, "_require_supported_python"),
                patch.object(
                    sft_launch,
                    "_acquire_launch_lock",
                    side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
                ),
                patch.object(sft_launch, "verify_personal_workspace"),
                patch.object(sft_launch, "verify_modal_inventory", return_value=[]),
                patch.object(
                    modal_private_verifier,
                    "_build_modal_function",
                    return_value=(fake_app, fake_function),
                ),
            ):
                receipt = modal_private_verifier.run_once(
                    payload,
                    approved_payload_sha256=payload_sha,
                    approved_runner_sha256=payload["runner_sha256"],
                    output_path=output_path,
                )

            self.assertEqual(receipt["status"], "success")
            self.assertEqual(receipt["base_outputs"][0]["raw_output"], "A private answer.")
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(private_dir.stat().st_mode), 0o700)
            saved = output_path.read_text(encoding="utf-8")
            self.assertNotIn(items[0]["prompt"], saved)
            self.assertEqual(json.loads(saved)["payload_sha256"], payload_sha)
            with self.assertRaisesRegex(FileExistsError, "must be new"):
                sft_launch._reserve_output(output_path)


if __name__ == "__main__":
    unittest.main()
