"""Offline contract tests for frozen-adapter transfer inference."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from experiments import modal_verifier_transfer
from training.verifier_data import render_candidate_prompt

_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _ROOT / "examples/synthetic/verifier-transfer-v1.json"


class ModalVerifierTransferTests(unittest.TestCase):
    def test_payload_binds_allowlisted_sources_and_exact_label_free_dev_prompts(self) -> None:
        self.assertTrue(
            callable(getattr(modal_verifier_transfer, "build_payload", None)),
            "transfer runner must expose its offline payload builder",
        )
        payload = modal_verifier_transfer.build_payload()
        dataset_raw = _DATASET.read_bytes()
        renderer_raw = (_ROOT / "training/verifier_data.py").read_bytes()
        dataset = json.loads(dataset_raw)
        expected_items = [
            {
                "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
                "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
            }
            for case in dataset["cases"]
            if case["split"] == "dev"
            for candidate in case["candidates"]
        ]

        self.assertEqual(payload["mode"], "transfer")
        self.assertEqual(payload["split"], "dev")
        self.assertEqual(payload["items"], expected_items)
        self.assertEqual(len(payload["items"]), 48)
        self.assertEqual(
            payload["source_sha256"],
            {
                "examples/synthetic/verifier-transfer-v1.json": hashlib.sha256(
                    dataset_raw
                ).hexdigest(),
                "training/verifier_data.py": hashlib.sha256(renderer_raw).hexdigest(),
            },
        )
        self.assertEqual(payload["train_rows"], [])
        self.assertEqual(
            payload["adapter"]["safetensors_sha256"],
            "014a21986d0b2fb6fe884ab8346a1708012558632b5dbbe4325c701394e8f9b2",
        )
        self.assertEqual(
            payload["adapter"]["config_sha256"],
            "9c79cb0e2879e658f96bbad4cc41aa17c603fe03405f70100b3adc86b8692153",
        )

    def test_payload_mutation_cannot_change_pinned_model_or_adapter_contract(self) -> None:
        payload = modal_verifier_transfer.build_payload()
        payload["model"]["quantization"]["type"] = "changed"
        payload["settings"]["seed"] = 999
        payload["adapter"]["config_sha256"] = "0" * 64

        fresh_payload = modal_verifier_transfer.build_payload()

        self.assertEqual(fresh_payload["model"]["quantization"]["type"], "nf4")
        self.assertEqual(fresh_payload["settings"]["seed"], 17)
        self.assertEqual(
            fresh_payload["adapter"]["config_sha256"],
            "9c79cb0e2879e658f96bbad4cc41aa17c603fe03405f70100b3adc86b8692153",
        )

    def test_modal_envelope_is_one_source_free_a10g_call_without_retries(self) -> None:
        self.assertTrue(
            callable(getattr(modal_verifier_transfer, "modal_function_options", None)),
            "transfer runner must expose its bounded Modal envelope",
        )
        options = modal_verifier_transfer.modal_function_options()
        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["timeout"], 1200)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_remote_rejects_adapter_hash_mismatch_before_model_imports(self) -> None:
        self.assertTrue(
            callable(getattr(modal_verifier_transfer, "_make_remote_function", None)),
            "remote transfer function must exist before adapter loading",
        )
        payload = modal_verifier_transfer.build_payload()
        payload["adapter"]["config_sha256"] = "0" * 64
        request = {
            "payload": payload,
            "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
            "runner_sha256": payload["runner_sha256"],
        }
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"model runtime imported before adapter guard: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_model_imports):
            with self.assertRaisesRegex(ValueError, "adapter"):
                modal_verifier_transfer._make_remote_function(None)(request)

    def test_remote_checks_mounted_adapter_files_before_model_imports(self) -> None:
        original_import = builtins.__import__
        config_bytes = b"adapter config fixture"
        weights_bytes = b"safetensors fixture"

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {
                "modal",
                "torch",
                "peft",
                "transformers",
            }:
                raise AssertionError(f"model runtime imported before adapter file guard: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            (adapter_dir / "adapter_config.json").write_bytes(config_bytes)
            (adapter_dir / "adapter_model.safetensors").write_bytes(weights_bytes)
            adapter = {
                **modal_verifier_transfer._ADAPTER,
                "volume_mount": str(adapter_dir.resolve()),
                "relative_path": ".",
                "config_sha256": "0" * 64,
                "safetensors_sha256": hashlib.sha256(weights_bytes).hexdigest(),
            }
            with patch.object(modal_verifier_transfer, "_ADAPTER", adapter):
                payload = modal_verifier_transfer.build_payload()
                request = {
                    "payload": payload,
                    "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(ValueError, "adapter_config.json SHA-256 mismatch"):
                        modal_verifier_transfer._make_remote_function(None)(request)

    def test_remote_allows_volume_mount_symlink_before_model_imports(self) -> None:
        marker = "ML_IMPORT_SENTINEL: modal"
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(marker)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume_tree = root / "volume-tree"
            relative_path = Path("runs/approved/adapter")
            adapter_dir = volume_tree / relative_path
            adapter_dir.mkdir(parents=True)
            config_bytes = b"approved adapter config"
            weights_bytes = b"approved adapter weights"
            (adapter_dir / "adapter_config.json").write_bytes(config_bytes)
            (adapter_dir / "adapter_model.safetensors").write_bytes(weights_bytes)
            mount_path = root / "mounted-volume"
            mount_path.symlink_to(volume_tree, target_is_directory=True)
            adapter = {
                **modal_verifier_transfer._ADAPTER,
                "volume_mount": str(mount_path),
                "relative_path": relative_path.as_posix(),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "safetensors_sha256": hashlib.sha256(weights_bytes).hexdigest(),
            }

            with patch.object(modal_verifier_transfer, "_ADAPTER", adapter):
                payload = modal_verifier_transfer.build_payload()
                request = {
                    "payload": payload,
                    "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(AssertionError, marker) as raised:
                        modal_verifier_transfer._make_remote_function(None)(request)

            self.assertEqual(str(raised.exception), marker)

    def test_remote_rejects_nested_volume_symlink_escape_before_model_imports(self) -> None:
        marker = "ML_IMPORT_SENTINEL: modal"
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(marker)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume_tree = root / "volume-tree"
            escaped_tree = root / "escaped-tree"
            relative_path = Path("runs/approved/adapter")
            adapter_dir = escaped_tree / relative_path
            adapter_dir.mkdir(parents=True)
            config_bytes = b"approved adapter config"
            weights_bytes = b"approved adapter weights"
            (adapter_dir / "adapter_config.json").write_bytes(config_bytes)
            (adapter_dir / "adapter_model.safetensors").write_bytes(weights_bytes)
            volume_tree.mkdir()
            (volume_tree / "runs").symlink_to(escaped_tree / "runs", target_is_directory=True)
            mount_path = root / "mounted-volume"
            mount_path.symlink_to(volume_tree, target_is_directory=True)
            adapter = {
                **modal_verifier_transfer._ADAPTER,
                "volume_mount": str(mount_path),
                "relative_path": relative_path.as_posix(),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "safetensors_sha256": hashlib.sha256(weights_bytes).hexdigest(),
            }

            with patch.object(modal_verifier_transfer, "_ADAPTER", adapter):
                payload = modal_verifier_transfer.build_payload()
                request = {
                    "payload": payload,
                    "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(
                        ValueError, "remote transfer adapter directory is invalid"
                    ):
                        modal_verifier_transfer._make_remote_function(None)(request)

    def test_remote_rejects_symlinked_adapter_artifact_before_model_imports(self) -> None:
        marker = "ML_IMPORT_SENTINEL: modal"
        original_import = builtins.__import__

        def reject_model_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(marker)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume_tree = root / "volume-tree"
            relative_path = Path("runs/approved/adapter")
            adapter_dir = volume_tree / relative_path
            adapter_dir.mkdir(parents=True)
            config_bytes = b"approved adapter config"
            weights_bytes = b"approved adapter weights"
            escaped_config = root / "adapter_config.json"
            escaped_config.write_bytes(config_bytes)
            (adapter_dir / "adapter_config.json").symlink_to(escaped_config)
            (adapter_dir / "adapter_model.safetensors").write_bytes(weights_bytes)
            mount_path = root / "mounted-volume"
            mount_path.symlink_to(volume_tree, target_is_directory=True)
            adapter = {
                **modal_verifier_transfer._ADAPTER,
                "volume_mount": str(mount_path),
                "relative_path": relative_path.as_posix(),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "safetensors_sha256": hashlib.sha256(weights_bytes).hexdigest(),
            }

            with patch.object(modal_verifier_transfer, "_ADAPTER", adapter):
                payload = modal_verifier_transfer.build_payload()
                request = {
                    "payload": payload,
                    "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
                    "runner_sha256": payload["runner_sha256"],
                }
                with patch("builtins.__import__", side_effect=reject_model_imports):
                    with self.assertRaisesRegex(
                        ValueError, "remote transfer adapter artifact is not a regular file"
                    ):
                        modal_verifier_transfer._make_remote_function(None)(request)

    def test_modal_builder_mounts_the_adapter_volume_read_only(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeImage:
            def pip_install(self, *packages: str) -> FakeImage:
                calls.append(("packages", packages))
                return self

        class FakeVolume:
            def with_mount_options(self, *, read_only: bool) -> FakeVolume:
                calls.append(("read_only", read_only))
                return self

        class FakeApp:
            def __init__(self, name: str) -> None:
                calls.append(("app", name))

            def function(self, *, image: object, **options: object):
                calls.append(("options", options))
                return lambda function: function

        class FakeImageFactory:
            @staticmethod
            def debian_slim(**kwargs: object) -> FakeImage:
                calls.append(("image", kwargs))
                return FakeImage()

        class FakeVolumeFactory:
            @staticmethod
            def from_name(name: str, **kwargs: object) -> FakeVolume:
                calls.append(("volume", name))
                return FakeVolume()

        modal_module = ModuleType("modal")
        modal_module.Image = FakeImageFactory  # type: ignore[attr-defined]
        modal_module.App = FakeApp  # type: ignore[attr-defined]
        modal_module.Volume = FakeVolumeFactory  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"modal": modal_module}):
            _app, _function = modal_verifier_transfer._build_modal_function(
                modal_verifier_transfer.build_payload()
            )

        self.assertIn(("read_only", True), calls)
        self.assertEqual(
            [value for key, value in calls if key == "volume"],
            [modal_verifier_transfer._ADAPTER["volume_name"]],
        )
        options = next(value for key, value in calls if key == "options")
        self.assertFalse(options["include_source"])
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)

    def test_kbit_model_preparation_precedes_base_generation(self) -> None:
        source = (_ROOT / "experiments/modal_verifier_transfer.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        remote = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_remote_compute"
        )
        preparation_index = next(
            index
            for index, node in enumerate(remote.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "prepare_model_for_kbit_training"
        )
        preparation = remote.body[preparation_index]
        assert isinstance(preparation, ast.Assign)
        assert isinstance(preparation.value, ast.Call)
        self.assertTrue(
            any(
                isinstance(target, ast.Name) and target.id == "base_model"
                for target in preparation.targets
            )
        )
        self.assertTrue(
            any(
                keyword.arg == "use_gradient_checkpointing"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in preparation.value.keywords
            )
        )
        baseline_index = next(
            index
            for index, node in enumerate(remote.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "baseline_outputs"
                for target in node.targets
            )
        )
        self.assertLess(preparation_index, baseline_index)

    def test_run_once_rejects_changed_payload_and_stale_approval_before_side_effects(self) -> None:
        from experiments import sft_launch

        payload = modal_verifier_transfer.build_payload()
        changed_payload = json.loads(json.dumps(payload))
        changed_payload["items"][0]["prompt"] += " changed after review"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created" / "receipt.json"
            with (
                patch.object(
                    sft_launch,
                    "_reserve_output",
                    side_effect=AssertionError("invalid approval reached receipt creation"),
                ),
                patch.object(
                    sft_launch,
                    "verify_personal_workspace",
                    side_effect=AssertionError("invalid approval reached workspace check"),
                ),
                patch.object(
                    sft_launch,
                    "verify_modal_inventory",
                    side_effect=AssertionError("invalid approval reached Modal inventory"),
                ),
                patch.object(
                    modal_verifier_transfer,
                    "_build_modal_function",
                    side_effect=AssertionError("invalid approval reached Modal construction"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "current fixed transfer payload"):
                    modal_verifier_transfer.run_once(
                        changed_payload,
                        approved_payload_sha256=modal_verifier_transfer.payload_sha256(
                            changed_payload
                        ),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )
                with self.assertRaisesRegex(ValueError, "exact current payload and runner"):
                    modal_verifier_transfer.run_once(
                        payload,
                        approved_payload_sha256="0" * 64,
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )

            self.assertFalse(output.parent.exists())

    def test_run_once_checks_workspace_and_inventory_then_writes_bound_receipt(self) -> None:
        from experiments import sft_launch

        payload = modal_verifier_transfer.build_payload()
        digest = modal_verifier_transfer.payload_sha256(payload)
        calls: list[object] = []
        fake_result = {
            "schema_version": 1,
            "status": "success",
            "mode": "transfer",
            "split": "dev",
            "payload_sha256": digest,
            "runner_sha256": payload["runner_sha256"],
            "model": payload["model"],
            "settings": payload["settings"],
            "adapter": payload["adapter"],
            "adapter_artifact_sha256": {
                "adapter_config.json": payload["adapter"]["config_sha256"],
                "adapter_model.safetensors": payload["adapter"]["safetensors_sha256"],
            },
            "tokenizer": {"eos_token_id": 151645},
            "baseline_outputs": [{"item_id": "case-a::candidate-a"}],
            "tuned_outputs": [{"item_id": "case-a::candidate-a"}],
        }

        class FakeAppContext:
            def __enter__(self) -> FakeAppContext:
                calls.append("app_enter")
                return self

            def __exit__(self, *_args: object) -> bool:
                calls.append("app_exit")
                return False

        class FakeFunction:
            def remote(self, request: dict[str, object]) -> dict[str, object]:
                calls.append(("remote", request))
                return fake_result

        class FakeApp:
            app_id = "ap-transfer-test"

            def run(self, *, detach: bool) -> FakeAppContext:
                calls.append(("app_run", detach))
                return FakeAppContext()

        def build_fake_modal(supplied_payload: dict[str, object]) -> tuple[FakeApp, FakeFunction]:
            calls.append(("build", supplied_payload))
            self.assertEqual(calls[:2], ["workspace", "inventory"])
            return FakeApp(), FakeFunction()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            shared_lock = Path(directory) / "sft-pilot-launch.lock"
            with (
                patch.object(sft_launch, "_LAUNCH_LOCK", shared_lock),
                patch.object(
                    sft_launch,
                    "verify_personal_workspace",
                    side_effect=lambda: calls.append("workspace"),
                ),
                patch.object(
                    sft_launch,
                    "verify_modal_inventory",
                    side_effect=lambda: calls.append("inventory") or [{"state": "stopped"}],
                ),
                patch.object(
                    modal_verifier_transfer,
                    "_build_modal_function",
                    side_effect=build_fake_modal,
                ),
            ):
                receipt = modal_verifier_transfer.run_once(
                    payload,
                    approved_payload_sha256=digest,
                    approved_runner_sha256=payload["runner_sha256"],
                    output_path=output,
                )

            expected_request = {
                "payload": payload,
                "payload_sha256": digest,
                "runner_sha256": payload["runner_sha256"],
            }
            self.assertEqual(
                calls,
                [
                    "workspace",
                    "inventory",
                    ("build", payload),
                    ("app_run", False),
                    "app_enter",
                    ("remote", expected_request),
                    "app_exit",
                ],
            )
            self.assertEqual(receipt["status"], "success")
            self.assertEqual(receipt["payload_sha256"], digest)
            self.assertEqual(receipt["runner_sha256"], payload["runner_sha256"])
            self.assertEqual(receipt["modal_app_id"], "ap-transfer-test")
            self.assertEqual(receipt["launch_checks"]["modal_app_count"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(shared_lock.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "must be new"):
                sft_launch._reserve_output(output)

    def test_run_once_records_remote_failure_once_then_reraises(self) -> None:
        from experiments import sft_launch

        payload = modal_verifier_transfer.build_payload()
        digest = modal_verifier_transfer.payload_sha256(payload)
        calls: list[str] = []

        class FakeAppContext:
            def __enter__(self) -> FakeAppContext:
                calls.append("app_enter")
                return self

            def __exit__(self, *_args: object) -> bool:
                calls.append("app_exit")
                return False

        class FakeFunction:
            def remote(self, _request: dict[str, object]) -> dict[str, object]:
                calls.append("remote")
                raise RuntimeError("remote adapter failure")

        class FakeApp:
            app_id = "ap-transfer-failure-test"

            def run(self, *, detach: bool) -> FakeAppContext:
                self_test.assertFalse(detach)
                return FakeAppContext()

        self_test = self

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            shared_lock = Path(directory) / "sft-pilot-launch.lock"
            with (
                patch.object(sft_launch, "_LAUNCH_LOCK", shared_lock),
                patch.object(sft_launch, "verify_personal_workspace"),
                patch.object(sft_launch, "verify_modal_inventory", return_value=[]),
                patch.object(
                    modal_verifier_transfer,
                    "_build_modal_function",
                    return_value=(FakeApp(), FakeFunction()),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote adapter failure"):
                    modal_verifier_transfer.run_once(
                        payload,
                        approved_payload_sha256=digest,
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )

            failure = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(calls, ["app_enter", "remote", "app_exit"])
            self.assertEqual(failure["status"], "launch_failed")
            self.assertEqual(failure["payload_sha256"], digest)
            self.assertTrue(failure["remote_started"])
            self.assertEqual(failure["error"]["type"], "RuntimeError")
            self.assertEqual(failure["error"]["message"], "remote adapter failure")


if __name__ == "__main__":
    unittest.main()
