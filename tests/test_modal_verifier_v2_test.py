"""Offline contract tests for verifier-v2 blind-test inference."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from experiments import modal_verifier_v2, modal_verifier_v2_test
from training.verifier_data import render_candidate_prompt


def _fixture_dataset() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index in range(24):
        suffix = f"{index:03d}"
        source = f"For fixture {suffix}, the operator says the east ramp is closed until noon."
        message_id = f"message-{suffix}-user"
        candidates = []
        for candidate_index, (claim, support) in enumerate(
            (
                (f"The operator says the east ramp closes for fixture {suffix}.", "supported"),
                (f"The east ramp is confirmed closed for fixture {suffix}.", "unsupported"),
            )
        ):
            candidates.append(
                {
                    "candidate_id": f"candidate-{suffix}-{candidate_index}",
                    "claim": claim,
                    "message_id": message_id,
                    "quote": source,
                    "valid_from": None,
                    "valid_to": None,
                    "support": support,
                    "eligibility": "eligible",
                    "support_reason": "Fixture annotation.",
                    "eligibility_reason": "Fixture policy label.",
                }
            )
        cases.append(
            {
                "case_id": f"case-{suffix}",
                "split": "test",
                "scenario_group": f"scenario-{suffix}",
                "template_group": f"template-{suffix}",
                "entity_ids": [f"entity-{suffix}"],
                "phenomenon": "attribution",
                "messages": [
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": f"message-{suffix}-assistant",
                        "role": "assistant",
                        "text": f"What does the update say in fixture {suffix}?",
                        "recorded_at": "2026-01-01T10:00:00Z",
                        "content_kind": "conversation",
                    },
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": message_id,
                        "role": "user",
                        "text": source,
                        "recorded_at": "2026-01-01T10:01:00Z",
                        "content_kind": "conversation",
                    },
                ],
                "candidates": candidates,
            }
        )
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }


@contextmanager
def _fixed_inputs() -> Iterator[tuple[Path, dict[str, Any], dict[str, Any]]]:
    dataset = _fixture_dataset()
    dataset_raw = json.dumps(dataset, ensure_ascii=False, separators=(",", ":")).encode()
    receipt_raw = b'{"receipt":"fixture"}'
    training_payload = {
        "schema_version": 1,
        "mode": "continue_train",
        "model": modal_verifier_v2._MODEL,
        "settings": modal_verifier_v2._SETTINGS,
    }
    training_payload_sha256 = modal_verifier_v2.payload_sha256(training_payload)
    candidate = {
        "model": training_payload["model"],
        "settings": training_payload["settings"],
        "training_payload_sha256": training_payload_sha256,
        "training_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "adapter_relative_path": f"runs/{training_payload_sha256}/adapter",
        "adapter_artifact_sha256": {
            "adapter_config.json": "c" * 64,
            "adapter_model.safetensors": "d" * 64,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        dataset_path = Path(directory) / "blind-test.json"
        receipt_path = Path(directory) / "training-receipt.json"
        dataset_path.write_bytes(dataset_raw)
        receipt_path.write_bytes(receipt_raw)
        with (
            patch.object(modal_verifier_v2_test, "_DATASET_PATH", dataset_path),
            patch.object(
                modal_verifier_v2_test,
                "_DATASET_SHA256",
                hashlib.sha256(dataset_raw).hexdigest(),
            ),
            patch.object(modal_verifier_v2, "build_payload", return_value=training_payload),
            patch(
                "experiments.evaluate_verifier_v2.evaluate_report",
                return_value={"candidate": candidate},
            ),
        ):
            yield receipt_path, dataset, candidate


class ModalVerifierV2BlindTestTests(unittest.TestCase):
    def test_build_payload_contains_only_label_free_test_prompts_and_candidate_provenance(
        self,
    ) -> None:
        with _fixed_inputs() as (receipt_path, dataset, candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)

        self.assertEqual(payload["mode"], "blind_test_inference")
        self.assertEqual(payload["split"], "test")
        self.assertEqual(payload["counts"], {"test_cases": 24, "test_candidates": 48})
        self.assertEqual(len(payload["items"]), 48)
        self.assertTrue(all(set(item) == {"item_id", "prompt"} for item in payload["items"]))
        self.assertTrue(all("completion" not in item for item in payload["items"]))
        self.assertEqual(
            payload["items"][0]["prompt"],
            render_candidate_prompt(
                dataset["cases"][0], dataset["cases"][0]["candidates"][0]["candidate_id"]
            ),
        )
        self.assertEqual(
            payload["candidate_provenance"]["training_receipt_sha256"],
            candidate["training_receipt_sha256"],
        )
        self.assertEqual(
            payload["candidate_provenance"]["adapter_artifact_sha256"],
            candidate["adapter_artifact_sha256"],
        )
        self.assertFalse(payload["training_performed"])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("support_reason", serialized)
        self.assertNotIn("eligibility_reason", serialized)
        self.assertNotIn('"support":', serialized)
        self.assertNotIn('"eligibility":', serialized)

    def test_modal_function_options_are_one_source_free_a10g_call_without_retries(self) -> None:
        options = modal_verifier_v2_test.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["timeout"], 900)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_modal_builder_mounts_both_adapters_read_only_and_pins_remote_helper(self) -> None:
        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            image_calls: list[tuple[str, Any]] = []
            volume_calls: list[tuple[str, dict[str, Any]]] = []
            app_calls: list[tuple[Any, dict[str, Any]]] = []

            class Image:
                def pip_install(self, *packages: str) -> Image:
                    image_calls.append(("packages", packages))
                    return self

                def add_local_file(self, path: Path, **options: Any) -> Image:
                    image_calls.append((str(path), options))
                    return self

            class Volume:
                def __init__(self, name: str) -> None:
                    self.name = name
                    self.mount_options: dict[str, Any] = {}

                def with_mount_options(self, **options: Any) -> Volume:
                    self.mount_options = options
                    return self

            class App:
                app_id = "app-fixture"

                def __init__(self, name: str) -> None:
                    self.name = name

                def function(self, **options: Any) -> Any:
                    app_calls.append((self, options))
                    return lambda function: function

                @contextmanager
                def run(self, detach: bool = False) -> Iterator[None]:
                    yield

            volumes: dict[str, Volume] = {}

            def from_name(name: str, **options: Any) -> Volume:
                volume_calls.append((name, options))
                volume = Volume(name)
                volumes[name] = volume
                return volume

            fake_modal = SimpleNamespace(
                Image=SimpleNamespace(debian_slim=lambda **_options: Image()),
                App=App,
                Volume=SimpleNamespace(from_name=from_name),
            )
            with patch.dict("sys.modules", {"modal": fake_modal}):
                app, function = modal_verifier_v2_test._build_modal_function(payload)

        self.assertEqual(app.name, "tracemem-verifier-v2-blind-test")
        self.assertTrue(callable(function))
        self.assertEqual(len(volume_calls), 2)
        self.assertTrue(all(options == {"create_if_missing": False} for _, options in volume_calls))
        self.assertTrue(
            all(volume.mount_options == {"read_only": True} for volume in volumes.values())
        )
        self.assertEqual(
            {mount: volume.name for mount, volume in app_calls[0][1]["volumes"].items()},
            {
                payload["prior_adapter"]["volume_mount"]: payload["prior_adapter"]["volume_name"],
                payload["continued_adapter"]["volume_mount"]: payload["continued_adapter"][
                    "volume_name"
                ],
            },
        )
        self.assertEqual(app_calls[0][1]["timeout"], 900)
        self.assertFalse(app_calls[0][1]["include_source"])
        self.assertIn(
            (
                str(modal_verifier_v2_test._REMOTE_HELPER_PATH),
                {
                    "remote_path": modal_verifier_v2_test._REMOTE_HELPER_CONTAINER_PATH,
                    "copy": True,
                },
            ),
            image_calls,
        )

    def test_modal_sdk_serializes_compact_function_without_capturing_prompts(self) -> None:
        from modal._utils.function_utils import FunctionSourceInfo
        from modal._vendor import cloudpickle

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
        sentinel = "SERIALIZATION_SENTINEL_DO_NOT_CAPTURE"
        for index, item in enumerate(payload["items"]):
            item["prompt"] = f"{sentinel}_{index}-" + "x" * 4096
        self.assertGreater(len(modal_verifier_v2_test._canonical_bytes(payload)), 150_000)
        approved_hash = modal_verifier_v2_test.payload_sha256(payload)
        remote = modal_verifier_v2_test._make_remote_function(
            None, None, payload, payload["runner_sha256"]
        )
        payload["items"][0]["prompt"] = "changed after remote function construction"

        source = FunctionSourceInfo(remote, serialized=True)
        serialized = source.serialized_function()
        cloudpickle.loads(serialized)

        self.assertEqual(remote.__globals__["_EXPECTED_PAYLOAD_SHA256"], approved_hash)
        self.assertNotIn("items", remote.__globals__["_EXPECTED_PAYLOAD_COMPACT"])
        self.assertNotIn(sentinel.encode(), serialized)
        self.assertLess(len(serialized), 60 * 1024)

    def test_run_once_rejects_changed_payload_before_receipt_or_launch_guards(self) -> None:
        from experiments import sft_launch

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            changed = deepcopy(payload)
            changed["items"][0]["prompt"] += " changed after approval"
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "not-created" / "receipt.json"
                with (
                    patch.object(
                        sft_launch,
                        "_reserve_output",
                        side_effect=AssertionError("invalid approval reached receipt reservation"),
                    ),
                    patch.object(
                        sft_launch,
                        "verify_personal_workspace",
                        side_effect=AssertionError("invalid approval reached workspace check"),
                    ),
                    patch.object(
                        sft_launch,
                        "verify_modal_inventory",
                        side_effect=AssertionError("invalid approval reached inventory check"),
                    ),
                    patch.object(
                        modal_verifier_v2_test,
                        "_build_modal_function",
                        side_effect=AssertionError("invalid approval reached Modal construction"),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "current fixed blind-test payload"):
                        modal_verifier_v2_test.run_once(
                            changed,
                            training_receipt_path=receipt_path,
                            approved_payload_sha256=modal_verifier_v2_test.payload_sha256(changed),
                            approved_runner_sha256=payload["runner_sha256"],
                            output_path=output,
                        )
                self.assertFalse(output.parent.exists())

    def test_remote_rejects_changed_payload_before_ml_imports(self) -> None:
        import builtins

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            changed = deepcopy(payload)
            changed["items"][0]["prompt"] += " changed after approval"
            request = {
                "payload": changed,
                "payload_sha256": modal_verifier_v2_test.payload_sha256(changed),
                "runner_sha256": payload["runner_sha256"],
            }
            remote = modal_verifier_v2_test._make_remote_function(
                None, None, payload, payload["runner_sha256"]
            )
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_ml_imports):
            with self.assertRaisesRegex(ValueError, "not allowlisted") as raised:
                remote(request)
        self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))

    def test_remote_recomputes_payload_hash_instead_of_trusting_request_hash(self) -> None:
        import builtins

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            changed = deepcopy(payload)
            changed["items"][0]["prompt"] += " changed after approval"
            request = {
                "payload": changed,
                "payload_sha256": modal_verifier_v2_test.payload_sha256(payload),
                "runner_sha256": payload["runner_sha256"],
            }
            remote = modal_verifier_v2_test._make_remote_function(
                None, None, payload, payload["runner_sha256"]
            )
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_ml_imports):
            with self.assertRaisesRegex(ValueError, "does not match approval") as raised:
                remote(request)
        self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))

    def test_remote_rejects_label_leak_before_ml_imports(self) -> None:
        import builtins

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            changed = deepcopy(payload)
            changed["items"][0]["support"] = "supported"
            request = {
                "payload": changed,
                "payload_sha256": modal_verifier_v2_test.payload_sha256(changed),
                "runner_sha256": payload["runner_sha256"],
            }
            remote = modal_verifier_v2_test._make_remote_function(
                None, None, payload, payload["runner_sha256"]
            )
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_ml_imports):
            with self.assertRaisesRegex(ValueError, "not allowlisted") as raised:
                remote(request)
        self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))

    def test_remote_rejects_changed_runner_approval_before_ml_imports(self) -> None:
        import builtins

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            request = {
                "payload": payload,
                "payload_sha256": modal_verifier_v2_test.payload_sha256(payload),
                "runner_sha256": "0" * 64,
            }
            remote = modal_verifier_v2_test._make_remote_function(
                None, None, payload, payload["runner_sha256"]
            )
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_ml_imports):
            with self.assertRaisesRegex(ValueError, "runner SHA-256") as raised:
                remote(request)
        self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))

    def test_build_payload_rejects_changed_remote_helper(self) -> None:
        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            with tempfile.TemporaryDirectory() as directory:
                helper_path = Path(directory) / "verifier_v2_test_remote.py"
                helper_path.write_text("# changed helper\n", encoding="utf-8")
                with patch.object(modal_verifier_v2_test, "_REMOTE_HELPER_PATH", helper_path):
                    with self.assertRaisesRegex(ValueError, "helper SHA-256"):
                        modal_verifier_v2_test.build_payload(receipt_path)

    def test_remote_rejects_changed_helper_before_ml_imports(self) -> None:
        import builtins

        with _fixed_inputs() as (receipt_path, _dataset, _candidate):
            payload = modal_verifier_v2_test.build_payload(receipt_path)
            with tempfile.TemporaryDirectory() as directory:
                helper_path = Path(directory) / "verifier_v2_test_remote.py"
                helper_path.write_text("# changed helper\n", encoding="utf-8")
                with patch.object(modal_verifier_v2_test, "_REMOTE_HELPER_PATH", helper_path):
                    remote = modal_verifier_v2_test._make_remote_function(
                        None, None, payload, payload["runner_sha256"]
                    )
                request = {
                    "payload": payload,
                    "payload_sha256": modal_verifier_v2_test.payload_sha256(payload),
                    "runner_sha256": payload["runner_sha256"],
                }
                original_import = builtins.__import__

                def reject_ml_imports(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name.split(".", maxsplit=1)[0] in {
                        "modal",
                        "torch",
                        "peft",
                        "transformers",
                    }:
                        raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=reject_ml_imports):
                    with self.assertRaisesRegex(ValueError, "helper SHA-256") as raised:
                        remote(request)
                self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))

    def test_adapter_mount_allows_mount_symlink_and_rejects_nested_symlinks(self) -> None:
        from experiments.verifier_v2_test_remote import _verify_adapter_mount

        expected: dict[str, str]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "volume" / "runs" / "run-1" / "adapter"
            adapter.mkdir(parents=True)
            for filename, content in (
                ("adapter_config.json", b"{}"),
                ("adapter_model.safetensors", b"weights"),
            ):
                (adapter / filename).write_bytes(content)
            expected = {
                filename: hashlib.sha256(content).hexdigest()
                for filename, content in (
                    ("adapter_config.json", b"{}"),
                    ("adapter_model.safetensors", b"weights"),
                )
            }
            mount_alias = root / "mounted-volume"
            mount_alias.symlink_to(root / "volume", target_is_directory=True)

            self.assertEqual(
                _verify_adapter_mount(str(mount_alias), "runs/run-1/adapter", expected),
                expected,
            )

            external = root / "external"
            external.mkdir()
            symlink_cases = (
                ("nested", "runs", root / "volume" / "runs"),
                ("final-directory", "adapter", adapter),
                ("artifact", "adapter_config.json", adapter / "adapter_config.json"),
            )
            for name, target_name, destination in symlink_cases:
                with self.subTest(name=name):
                    trial = root / name / "volume"
                    trial.mkdir(parents=True)
                    run_root = trial / "runs" / "run-1"
                    run_root.mkdir(parents=True)
                    if name == "nested":
                        run_root.rmdir()
                        run_root.parent.rmdir()
                        run_root.parent.symlink_to(destination, target_is_directory=True)
                    elif name == "final-directory":
                        for filename in expected:
                            (run_root / filename).write_bytes(b"fixture")
                        (run_root / "adapter").symlink_to(destination, target_is_directory=True)
                    else:
                        artifact_dir = run_root / "adapter"
                        artifact_dir.mkdir()
                        (artifact_dir / "adapter_model.safetensors").write_bytes(b"weights")
                        (artifact_dir / target_name).symlink_to(destination)

                    with self.assertRaisesRegex(ValueError, "unsafe"):
                        _verify_adapter_mount(str(trial), "runs/run-1/adapter", expected)


if __name__ == "__main__":
    unittest.main()
