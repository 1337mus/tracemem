"""Offline contract tests for the bounded private verifier runner."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import stat
import sys
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


def _private_sft_candidate_manifest() -> dict[str, object]:
    return {
        "training_payload_sha256": "a" * 64,
        "training_receipt_sha256": "b" * 64,
        "adapter_relative_path": f"runs/{'a' * 64}/adapter",
        "adapter_artifact_sha256": {
            "adapter_config.json": "c" * 64,
            "adapter_model.safetensors": "d" * 64,
        },
    }


def _one_item() -> list[dict[str, str]]:
    return [{"item_id": "opaque-candidate-01", "prompt": "A minimized private prompt."}]


def _batch_items(count: int, prompt: str = "Fictional validation prompt.") -> list[dict[str, str]]:
    return [{"item_id": f"item-{index:03d}", "prompt": prompt} for index in range(count)]


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
            _private_sft_candidate_manifest(),
            private_sft=True,
        )
        remote = modal_private_verifier._make_remote_function(payload)
        serialized = FunctionSourceInfo(remote, serialized=True).serialized_function()

        cloudpickle.loads(serialized)
        self.assertNotIn(sentinel.encode(), serialized)
        self.assertLess(len(serialized), 64 * 1024)

    def test_validation_batch_remote_function_stays_source_free_under_64_kib(self) -> None:
        try:
            from modal._utils.function_utils import FunctionSourceInfo
            from modal._vendor import cloudpickle
        except ImportError as error:
            self.skipTest(f"Modal serialization dependencies unavailable: {error}")

        sentinel = "PRIVATE_BATCH_SERIALIZATION_SENTINEL"
        payload = modal_private_verifier.build_payload(
            _batch_items(128, sentinel + "x" * 50),
            _private_sft_candidate_manifest(),
            private_sft=True,
            validation_batch=True,
        )
        remote = modal_private_verifier._make_remote_function(payload)
        serialized = FunctionSourceInfo(remote, serialized=True).serialized_function()

        cloudpickle.loads(serialized)
        self.assertNotIn(sentinel.encode(), serialized)
        self.assertLess(len(serialized), 64 * 1024)

    def test_default_item_cap_stays_32_and_validation_batch_flag_requires_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 to 32"):
            modal_private_verifier.build_payload(_batch_items(33), _candidate_manifest())
        with self.assertRaisesRegex(ValueError, "validation_batch must be an explicit boolean"):
            modal_private_verifier.build_payload(
                _one_item(), _candidate_manifest(), validation_batch=1
            )

    def test_validation_batch_accepts_100_and_128_items_but_rejects_129(self) -> None:
        for count in (100, 128):
            with self.subTest(count=count):
                payload = modal_private_verifier.build_payload(
                    _batch_items(count), _candidate_manifest(), validation_batch=True
                )
                self.assertEqual(len(payload["items"]), count)
                self.assertEqual(
                    payload["provenance"]["validation_batch"],
                    modal_private_verifier._VALIDATION_BATCH_PROVENANCE_TAG,
                )
                self.assertEqual(payload["prior_adapter"], modal_private_verifier._PRIOR_ADAPTER)
                self.assertEqual(
                    payload["continued_adapter"]["relative_path"],
                    payload["candidate_manifest"]["adapter_relative_path"],
                )
        with self.assertRaisesRegex(ValueError, "1 to 128"):
            modal_private_verifier.build_payload(
                _batch_items(129), _candidate_manifest(), validation_batch=True
            )

    def test_default_and_validation_batch_payload_byte_caps_are_independent(self) -> None:
        large_prompt = "x" * 6_000
        with self.assertRaisesRegex(ValueError, "byte limit"):
            modal_private_verifier.build_payload(
                _batch_items(32, large_prompt), _candidate_manifest()
            )
        with self.assertRaisesRegex(ValueError, "byte limit"):
            modal_private_verifier.build_payload(
                _batch_items(128, large_prompt), _candidate_manifest(), validation_batch=True
            )

    def test_validation_batch_exact_envelope_reconstructs_only_fixed_marker(self) -> None:
        payload = modal_private_verifier.build_payload(
            _batch_items(100), _candidate_manifest(), validation_batch=True
        )
        expected, _, _ = modal_private_verifier._validate_approved_payload(
            payload, modal_private_verifier.payload_sha256(payload), payload["runner_sha256"]
        )
        self.assertEqual(expected, payload)

        for mutation in ("marker", "extra provenance"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(payload)
                if mutation == "marker":
                    changed["provenance"]["validation_batch"] = "unapproved_batch"
                else:
                    changed["provenance"]["extra"] = "not-allowlisted"
                with self.assertRaisesRegex(ValueError, "fixed envelope"):
                    modal_private_verifier._validate_approved_payload(
                        changed,
                        modal_private_verifier.payload_sha256(changed),
                        changed["runner_sha256"],
                    )

    def test_remote_enforces_approved_item_limits_before_model_imports(self) -> None:
        cases = (
            (False, 33),
            (True, 129),
        )
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError("model runtime imported before prompt limit guard")
            return original_import(name, *args, **kwargs)

        for validation_batch, count in cases:
            with self.subTest(validation_batch=validation_batch, count=count):
                accepted_count = count - 1
                payload = modal_private_verifier.build_payload(
                    _batch_items(accepted_count),
                    _candidate_manifest(),
                    validation_batch=validation_batch,
                )
                remote = modal_private_verifier._make_remote_function(payload)
                compute = remote.__globals__["_REMOTE_COMPUTE"]
                payload["items"] = _batch_items(count)
                digest = modal_private_verifier.payload_sha256(payload)
                compute.__globals__["_EXPECTED_PAYLOAD_SHA256"] = digest
                request = {
                    "payload": payload,
                    "payload_sha256": digest,
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(
                        ValueError, "private verifier prompt count is invalid"
                    ):
                        compute(request)

    def test_remote_enforces_approved_byte_limits_before_model_imports(self) -> None:
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError("model runtime imported before payload byte-limit guard")
            return original_import(name, *args, **kwargs)

        for validation_batch, count in ((False, 32), (True, 128)):
            with self.subTest(validation_batch=validation_batch, count=count):
                payload = modal_private_verifier.build_payload(
                    _batch_items(count), _candidate_manifest(), validation_batch=validation_batch
                )
                remote = modal_private_verifier._make_remote_function(payload)
                compute = remote.__globals__["_REMOTE_COMPUTE"]
                for item in payload["items"]:
                    item["prompt"] = "x" * 6_000
                digest = modal_private_verifier.payload_sha256(payload)
                compute.__globals__["_EXPECTED_PAYLOAD_SHA256"] = digest
                request = {
                    "payload": payload,
                    "payload_sha256": digest,
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(
                        ValueError, "private verifier payload exceeds its byte limit"
                    ):
                        compute(request)

    def test_remote_function_freezes_validated_private_adapter_descriptors(self) -> None:
        candidate_manifest = _private_sft_candidate_manifest()
        payload = modal_private_verifier.build_payload(
            _one_item(), candidate_manifest, private_sft=True
        )

        remote = modal_private_verifier._make_remote_function(payload)
        compute = remote.__globals__["_REMOTE_COMPUTE"]

        self.assertEqual(compute.__globals__["_EXPECTED_CANDIDATE_MANIFEST"], candidate_manifest)
        self.assertEqual(compute.__globals__["_EXPECTED_PRIOR_ADAPTER"], payload["prior_adapter"])
        self.assertEqual(
            compute.__globals__["_EXPECTED_CONTINUED_ADAPTER"], payload["continued_adapter"]
        )

    def test_remote_function_injects_only_the_validated_batch_limits(self) -> None:
        for validation_batch, expected_items, expected_bytes in (
            (False, 32, 180_000),
            (True, 128, 720_000),
        ):
            with self.subTest(validation_batch=validation_batch):
                payload = modal_private_verifier.build_payload(
                    _one_item(), _candidate_manifest(), validation_batch=validation_batch
                )
                remote = modal_private_verifier._make_remote_function(payload)
                compute = remote.__globals__["_REMOTE_COMPUTE"]
                self.assertEqual(compute.__globals__["_APPROVED_MAX_ITEMS"], expected_items)
                self.assertEqual(compute.__globals__["_APPROVED_MAX_PAYLOAD_BYTES"], expected_bytes)
                self.assertNotIn("payload", compute.__globals__)
                self.assertNotIn(payload["items"][0]["prompt"], repr(compute.__globals__))

    def test_modal_function_mounts_approved_volumes_read_only_without_creation(self) -> None:
        payload = modal_private_verifier.build_payload(
            _one_item(), _private_sft_candidate_manifest(), private_sft=True
        )
        calls: list[tuple[str, bool]] = []
        registration: dict[str, object] = {}

        class FakeImage:
            def pip_install(self, *_packages: str) -> FakeImage:
                return self

        class FakeApp:
            def function(self, *, image: object, **options: object) -> object:
                registration["options"] = options
                return lambda remote: remote

        class FakeVolume:
            @classmethod
            def from_name(cls, name: str, *, create_if_missing: bool) -> object:
                calls.append((name, create_if_missing))
                return SimpleNamespace(
                    name=name,
                    with_mount_options=lambda *, read_only: SimpleNamespace(
                        name=name, read_only=read_only
                    ),
                )

        fake_modal = SimpleNamespace(
            Image=SimpleNamespace(debian_slim=lambda **_options: FakeImage()),
            App=lambda _name: FakeApp(),
            Volume=FakeVolume,
        )
        with patch.dict(sys.modules, {"modal": fake_modal}):
            _app, remote = modal_private_verifier._build_modal_function(payload)

        self.assertTrue(callable(remote))
        self.assertEqual(
            calls,
            [("tracemem-verifier-v2", False), ("tracemem-private-sft", False)],
        )
        volumes = registration["options"]["volumes"]
        self.assertEqual(set(volumes), {"/mnt/tracemem-verifier-v2", "/mnt/tracemem-private-sft"})
        self.assertTrue(all(volume.read_only for volume in volumes.values()))

    def test_build_payload_keeps_only_minimized_prompts_and_exact_private_provenance(self) -> None:
        items = _one_item()
        candidate_manifest = _candidate_manifest()

        payload = modal_private_verifier.build_payload(items, candidate_manifest)

        self.assertEqual(payload["items"], items)
        self.assertEqual(payload["candidate_manifest"], candidate_manifest)
        self.assertEqual(payload["mode"], "private_inference_only")
        self.assertFalse(payload["training_performed"])
        self.assertNotIn("validation_batch", payload["provenance"])
        self.assertEqual(
            payload["provenance"]["permitted_use"],
            "explicit_user_authorized_private_evaluation_only",
        )
        self.assertIn("no_redistribution_license_asserted", payload["provenance"]["license"])
        self.assertEqual(set(payload["items"][0]), {"item_id", "prompt"})
        self.assertLessEqual(
            len(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()), 180_000
        )

    def test_private_sft_payload_compares_base_synthetic_v2_and_private_candidate(self) -> None:
        candidate_manifest = _private_sft_candidate_manifest()

        payload = modal_private_verifier.build_payload(
            _one_item(), candidate_manifest, private_sft=True
        )

        self.assertEqual(payload["candidate_manifest"], candidate_manifest)
        self.assertEqual(payload["prior_adapter"]["volume_name"], "tracemem-verifier-v2")
        self.assertEqual(payload["prior_adapter"]["volume_mount"], "/mnt/tracemem-verifier-v2")
        self.assertEqual(
            payload["prior_adapter"]["relative_path"],
            modal_private_verifier._EXPECTED_CANDIDATE_MANIFEST["adapter_relative_path"],
        )
        self.assertEqual(
            payload["prior_adapter"]["config_sha256"],
            modal_private_verifier._EXPECTED_CANDIDATE_MANIFEST["adapter_artifact_sha256"][
                "adapter_config.json"
            ],
        )
        self.assertEqual(
            payload["prior_adapter"]["safetensors_sha256"],
            modal_private_verifier._EXPECTED_CANDIDATE_MANIFEST["adapter_artifact_sha256"][
                "adapter_model.safetensors"
            ],
        )
        self.assertEqual(payload["continued_adapter"]["volume_name"], "tracemem-private-sft")
        self.assertEqual(payload["continued_adapter"]["volume_mount"], "/mnt/tracemem-private-sft")
        self.assertEqual(
            payload["continued_adapter"]["relative_path"],
            candidate_manifest["adapter_relative_path"],
        )
        self.assertEqual(
            payload["continued_adapter"]["artifact_sha256"],
            candidate_manifest["adapter_artifact_sha256"],
        )
        self.assertEqual(payload["mode"], "private_inference_only")
        self.assertFalse(payload["training_performed"])
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "mode",
                "provenance",
                "runner_sha256",
                "model",
                "settings",
                "candidate_manifest",
                "prior_adapter",
                "continued_adapter",
                "items",
                "training_performed",
            },
        )

    def test_private_sft_manifest_rejects_paths_hashes_and_unapproved_fields(self) -> None:
        invalid_manifests: list[tuple[dict[str, object], str]] = []

        path_manifest = _private_sft_candidate_manifest()
        path_manifest["adapter_relative_path"] = "runs/../outside/adapter"
        invalid_manifests.append((path_manifest, "hashes or adapter path are invalid"))

        payload_hash_manifest = _private_sft_candidate_manifest()
        payload_hash_manifest["training_payload_sha256"] = "A" * 64
        invalid_manifests.append((payload_hash_manifest, "hashes or adapter path are invalid"))

        artifact_hash_manifest = _private_sft_candidate_manifest()
        artifact_hash_manifest["adapter_artifact_sha256"]["adapter_model.safetensors"] = "bad"
        invalid_manifests.append((artifact_hash_manifest, "hashes or adapter path are invalid"))

        extra_manifest = _private_sft_candidate_manifest()
        extra_manifest["source_transcript_path"] = "private/transcript.json"
        invalid_manifests.append((extra_manifest, "fields are not allowlisted"))

        extra_artifact_manifest = _private_sft_candidate_manifest()
        extra_artifact_manifest["adapter_artifact_sha256"]["weights.bin"] = "e" * 64
        invalid_manifests.append((extra_artifact_manifest, "hashes or adapter path are invalid"))

        for manifest, message in invalid_manifests:
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, message):
                modal_private_verifier.build_payload(_one_item(), manifest, private_sft=True)

    def test_private_sft_mode_requires_bool_and_default_call_stays_identical(self) -> None:
        legacy = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())
        explicit_legacy = modal_private_verifier.build_payload(
            _one_item(), _candidate_manifest(), private_sft=False
        )
        self.assertEqual(explicit_legacy, legacy)
        self.assertEqual(legacy["provenance"]["data_origin"], "user_provided_private_conversations")
        with self.assertRaisesRegex(ValueError, "private_sft must be an explicit boolean"):
            modal_private_verifier.build_payload(_one_item(), _candidate_manifest(), private_sft=1)

    def test_private_sft_provenance_covers_authorized_evaluation_prompts(self) -> None:
        private = modal_private_verifier.build_payload(
            _one_item(), _private_sft_candidate_manifest(), private_sft=True
        )
        legacy = modal_private_verifier.build_payload(_one_item(), _candidate_manifest())

        self.assertEqual(
            private["provenance"]["data_origin"],
            "authorized_private_checkpoint_evaluation_prompts",
        )
        self.assertEqual(
            {key: value for key, value in private["provenance"].items() if key != "data_origin"},
            {key: value for key, value in legacy["provenance"].items() if key != "data_origin"},
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

    def test_approved_payload_reconstructs_private_sft_mode_from_fixed_volume(self) -> None:
        payload = modal_private_verifier.build_payload(
            _one_item(), _private_sft_candidate_manifest(), private_sft=True
        )

        expected, digest, runner_hash = modal_private_verifier._validate_approved_payload(
            payload,
            modal_private_verifier.payload_sha256(payload),
            payload["runner_sha256"],
        )

        self.assertEqual(digest, modal_private_verifier.payload_sha256(payload))
        self.assertEqual(runner_hash, payload["runner_sha256"])
        self.assertEqual(expected, payload)

    def test_run_once_rejects_mutated_private_sft_envelope_before_launch(self) -> None:
        payload = modal_private_verifier.build_payload(
            _one_item(), _private_sft_candidate_manifest(), private_sft=True
        )
        payload["continued_adapter"]["volume_mount"] = "/tmp/unapproved-adapter"
        approved_digest = modal_private_verifier.payload_sha256(payload)

        with patch.object(
            modal_private_verifier,
            "_build_modal_function",
            side_effect=AssertionError("mutated envelope reached Modal construction"),
        ):
            with self.assertRaisesRegex(ValueError, "fixed envelope"):
                modal_private_verifier.run_once(
                    payload,
                    approved_payload_sha256=approved_digest,
                    approved_runner_sha256=payload["runner_sha256"],
                    output_path=Path("/tmp/not-used.json"),
                )

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

    def test_remote_rejects_mutated_private_metadata_before_model_imports(self) -> None:
        payload = modal_private_verifier.build_payload(
            _one_item(), _private_sft_candidate_manifest(), private_sft=True
        )
        remote = modal_private_verifier._make_remote_function(payload)
        compute = remote.__globals__["_REMOTE_COMPUTE"]
        changed = deepcopy(payload)
        changed["candidate_manifest"]["adapter_artifact_sha256"]["adapter_model.safetensors"] = (
            "e" * 64
        )
        changed["continued_adapter"]["artifact_sha256"]["adapter_model.safetensors"] = "e" * 64
        changed_digest = modal_private_verifier.payload_sha256(changed)
        compute.__globals__["_EXPECTED_PAYLOAD_SHA256"] = changed_digest
        request = {
            "payload": changed,
            "payload_sha256": changed_digest,
            "runner_sha256": payload["runner_sha256"],
        }
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError("model runtime imported before metadata guard")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            with self.assertRaisesRegex(ValueError, "metadata is not allowlisted"):
                compute(request)

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
