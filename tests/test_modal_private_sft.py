"""Offline contracts for the private SFT Modal wrapper."""

from __future__ import annotations

import builtins
import hashlib
import json
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from experiments import modal_private_sft, private_sft, private_sft_remote, sft_launch
from experiments.private_sft import payload_sha256
from experiments.verifier_v2_test_remote import _verify_adapter_mount


class ModalPrivateSftTests(unittest.TestCase):
    @staticmethod
    def _minimal_payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": "private_sft",
            "provenance": deepcopy(private_sft._PROVENANCE),
            "source_sha256": {
                "private_dataset": "1" * 64,
                "source_records": "2" * 64,
                "synthetic_replay_dataset": None,
            },
            "model": deepcopy(private_sft._MODEL),
            "settings": deepcopy(private_sft._SETTINGS),
            "prior_adapter": private_sft._prior_adapter(),
            "output_volume": {
                "name": private_sft._OUTPUT_VOLUME_NAME,
                "mount": private_sft._OUTPUT_VOLUME_MOUNT,
            },
            "train_rows": [
                {
                    "prompt": "fictional prompt",
                    "completion": '{"support":"supported","eligibility":"eligible"}',
                }
            ],
            "dev_prompts": ["fictional dev prompt"],
        }

    def test_modal_function_options_are_one_source_free_a10g_call(self) -> None:
        options = modal_private_sft.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertLessEqual(options["timeout"], 1800)
        self.assertFalse(options["include_source"])

    def test_remote_rejects_changed_payload_before_model_imports(self) -> None:
        payload = {"private_marker": "FICTIONAL_PRIVATE_SENTINEL"}
        remote = modal_private_sft._make_remote_function(payload, "0" * 64)
        changed = deepcopy(payload)
        changed["private_marker"] = "CHANGED_SENTINEL"
        request = {
            "payload": changed,
            "payload_sha256": payload_sha256(changed),
            "runner_sha256": "0" * 64,
        }
        original_import = builtins.__import__

        def block_model_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"torch", "peft", "transformers"}:
                raise AssertionError("model runtime imported before approval validation")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_model_import):
            result = remote(request)

        self.assertEqual(result["status"], "remote_failed")
        self.assertNotIn("CHANGED_SENTINEL", str(result))
        self.assertNotIn("FICTIONAL_PRIVATE_SENTINEL", str(result))

    def test_completion_loss_masks_prompt_and_includes_eos(self) -> None:
        labels = private_sft_remote._completion_labels([10, 11], [20, 21], 2)

        self.assertEqual(labels, [-100, -100, 20, 21, 2])

    def test_partial_accumulation_window_uses_its_actual_size(self) -> None:
        self.assertEqual(private_sft_remote._accumulation_window_size(4, 6, 4), 2)

    def test_token_id_hash_uses_canonical_json(self) -> None:
        self.assertEqual(
            private_sft_remote._token_ids_sha256([1, 2]),
            hashlib.sha256(b"[1,2]").hexdigest(),
        )

    def test_generation_decodes_content_only_and_keeps_eos_token_ids(self) -> None:
        import ast
        from typing import Any

        source_tree = ast.parse(Path(private_sft_remote.__file__).read_text())
        remote_compute = next(
            node
            for node in source_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_remote_compute"
        )
        generate_node = next(
            node
            for node in ast.walk(remote_compute)
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                generate_node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        class FakeRow:
            def __init__(self, values: list[int]) -> None:
                self.values = values

            def detach(self) -> FakeRow:
                return self

            def cpu(self) -> FakeRow:
                return self

            def tolist(self) -> list[int]:
                return self.values

        class FakeTensor:
            ndim = 2

            def __init__(self, rows: list[list[int]]) -> None:
                self.rows = rows
                self.shape = (len(rows), len(rows[0]))

            def __getitem__(self, key: tuple[int, slice]) -> FakeRow:
                row, token_slice = key
                return FakeRow(self.rows[row][token_slice])

        class FakeTorch:
            Tensor = FakeTensor
            long = "long"
            cuda = type("FakeCuda", (), {"synchronize": staticmethod(lambda: None)})()

            @staticmethod
            def tensor(rows: list[list[int]], **_kwargs: object) -> FakeTensor:
                return FakeTensor(rows)

            @staticmethod
            def ones_like(tensor: FakeTensor) -> FakeTensor:
                return FakeTensor([[1] * tensor.shape[1]])

            @staticmethod
            def inference_mode() -> object:
                return nullcontext()

        class FakeTokenizer:
            def __init__(self) -> None:
                self.decode_options: list[bool] = []

            def decode(self, _ids: list[int], *, skip_special_tokens: bool) -> str:
                self.decode_options.append(skip_special_tokens)
                suffix = "" if skip_special_tokens else "<|im_end|>"
                return '{"support":"supported","eligibility":"eligible"}' + suffix

        class FakeModel:
            def eval(self) -> None:
                return None

            def generate(self, *, input_ids: FakeTensor, **_kwargs: object) -> FakeTensor:
                return FakeTensor([input_ids.rows[0] + [21, 2]])

        tokenizer = FakeTokenizer()
        namespace: dict[str, Any] = {
            "Any": Any,
            "_TOKEN_IDS_SHA256": private_sft_remote._token_ids_sha256,
        }
        exec(compile(module, private_sft_remote.__file__, "exec"), namespace)
        generate = namespace["generate"]
        generate.__globals__.update(
            {
                "torch": FakeTorch,
                "tokenizer": tokenizer,
                "prepared": [("dev_000", [10, 11])],
                "input_device": "cpu",
                "settings": {"do_sample": False, "max_new_tokens": 4},
                "eos_id": 2,
                "pad_id": 2,
            }
        )

        output = generate(FakeModel())[0]

        self.assertEqual(tokenizer.decode_options, [True])
        self.assertEqual(output["raw_output"], '{"support":"supported","eligibility":"eligible"}')
        self.assertEqual(output["finish_reason"], "eos")
        self.assertEqual(output["output_token_ids"], [21, 2])
        self.assertEqual(
            output["output_token_ids_sha256"], private_sft_remote._token_ids_sha256([21, 2])
        )

    def test_payload_shape_rejects_unrecognized_completion_labels(self) -> None:
        payload = {
            "schema_version": 1,
            "mode": "private_sft",
            "provenance": deepcopy(private_sft._PROVENANCE),
            "source_sha256": {
                "private_dataset": "1" * 64,
                "source_records": "2" * 64,
                "synthetic_replay_dataset": None,
            },
            "model": deepcopy(private_sft._MODEL),
            "settings": deepcopy(private_sft._SETTINGS),
            "prior_adapter": private_sft._prior_adapter(),
            "output_volume": {"name": "tracemem-private-sft", "mount": "/mnt/tracemem-private-sft"},
            "train_rows": [{"prompt": "fictional prompt", "completion": '{"support":"maybe"}'}],
            "dev_prompts": ["fictional dev prompt"],
        }

        with self.assertRaisesRegex(ValueError, "payload is invalid"):
            private_sft_remote._validate_payload_shape(
                payload,
                private_sft._PROVENANCE,
                private_sft._MODEL,
                private_sft._SETTINGS,
                private_sft._prior_adapter(),
                {"name": "tracemem-private-sft", "mount": "/mnt/tracemem-private-sft"},
            )

    @staticmethod
    def _run_fake_launcher(
        root: Path,
        payload: dict[str, object],
        payload_hash: str,
        runner_hash: str,
        result: dict[str, object],
        *,
        workspace_error: Exception | None = None,
    ) -> tuple[dict[str, object], list[str], Path]:
        local_dir = root / ".local"
        output_dir = local_dir / "private-modal-sft"
        output_dir.mkdir(parents=True, mode=0o700)
        lock_path = local_dir / "launch.lock"
        receipt_path = output_dir / "receipt.json"
        events: list[str] = []
        app = Mock(app_id="fake-app", run=Mock(return_value=nullcontext()))
        function = Mock()

        def remote(_request: dict[str, object]) -> dict[str, object]:
            events.append("remote")
            return result

        def verify_workspace() -> None:
            events.append("workspace")
            if workspace_error is not None:
                raise workspace_error

        function.remote.side_effect = remote
        stopped_apps = [{"app_id": "historical", "state": "stopped"}]
        real_acquire = sft_launch._acquire_launch_lock
        real_reserve = sft_launch._reserve_output
        real_write = sft_launch._write_receipt

        def acquire_lock() -> int:
            events.append("lock")
            return real_acquire()

        def reserve_output(path: Path) -> int:
            events.append("reserve")
            return real_reserve(path)

        def write_receipt(descriptor: int, value: dict[str, object]) -> None:
            events.append("receipt")
            real_write(descriptor, value)

        with (
            patch.object(modal_private_sft, "_LOCAL_DIR", local_dir),
            patch.object(modal_private_sft, "_PRIVATE_OUTPUT_DIR", output_dir),
            patch.object(
                modal_private_sft,
                "_build_modal_function",
                side_effect=lambda *_: (events.append("build"), (app, function))[1],
            ),
            patch.object(sft_launch, "_LAUNCH_LOCK", lock_path),
            patch.object(sft_launch, "_acquire_launch_lock", side_effect=acquire_lock),
            patch.object(sft_launch, "_reserve_output", side_effect=reserve_output),
            patch.object(sft_launch, "_write_receipt", side_effect=write_receipt),
            patch.object(
                sft_launch, "_require_supported_python", side_effect=lambda: events.append("python")
            ),
            patch.object(sft_launch, "verify_personal_workspace", side_effect=verify_workspace),
            patch.object(
                sft_launch,
                "verify_modal_inventory",
                side_effect=lambda: (events.append("inventory"), stopped_apps)[1],
            ),
        ):
            receipt = modal_private_sft.run_once(
                payload,  # type: ignore[arg-type]
                approved_payload_sha256=payload_hash,
                approved_runner_sha256=runner_hash,
                output_path=receipt_path,
            )
        return receipt, events, receipt_path

    def test_launcher_uses_shared_helpers_and_persists_private_success_receipt(self) -> None:
        payload = self._minimal_payload()
        payload_hash = payload_sha256(payload)  # type: ignore[arg-type]
        runner_hash = modal_private_sft.runner_sha256()
        arms = [{"item_id": "dev_000", "raw_output": "fictional output"}]
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "success",
            "mode": "private_sft",
            "training_performed": True,
            "payload_sha256": payload_hash,
            "runner_sha256": runner_hash,
            "runtime_metadata": {"worker_id": "fictional"},
            "base_outputs": arms,
            "prior_outputs": arms,
            "trained_outputs": arms,
        }

        with TemporaryDirectory() as tmp:
            receipt, events, path = self._run_fake_launcher(
                Path(tmp).resolve(), payload, payload_hash, runner_hash, result
            )

            self.assertEqual(receipt["status"], "success")
            self.assertEqual(receipt["runtime_metadata"], result["runtime_metadata"])
            self.assertEqual(receipt["launch_checks"]["modal_app_count"], 1)  # type: ignore[index]
            self.assertEqual(
                events,
                [
                    "lock",
                    "reserve",
                    "python",
                    "workspace",
                    "inventory",
                    "build",
                    "remote",
                    "receipt",
                ],
            )
            self.assertEqual(json.loads(path.read_text()), receipt)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)

    def test_launcher_persists_sanitized_failure_receipt_with_completed_arms(self) -> None:
        payload = self._minimal_payload()
        payload_hash = payload_sha256(payload)  # type: ignore[arg-type]
        runner_hash = modal_private_sft.runner_sha256()
        arms = [{"item_id": "dev_000", "raw_output": "fictional prior output"}]
        partial: dict[str, object] = {
            "schema_version": 1,
            "status": "remote_failed",
            "mode": "private_sft",
            "training_performed": None,
            "payload_sha256": payload_hash,
            "runner_sha256": runner_hash,
            "stage": "training",
            "base_outputs": arms,
            "prior_outputs": arms,
            "error": {"type": "RuntimeError", "message": "private SFT failed"},
        }

        with TemporaryDirectory() as tmp:
            receipt, _events, path = self._run_fake_launcher(
                Path(tmp).resolve(), payload, payload_hash, runner_hash, partial
            )

            self.assertEqual(receipt["status"], "remote_failed")
            self.assertEqual(receipt["base_outputs"], arms)
            self.assertEqual(receipt["prior_outputs"], arms)
            self.assertEqual(json.loads(path.read_text()), receipt)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)

    def test_launcher_persists_sanitized_preflight_failure_receipt(self) -> None:
        payload = self._minimal_payload()
        payload_hash = payload_sha256(payload)  # type: ignore[arg-type]
        runner_hash = modal_private_sft.runner_sha256()

        with TemporaryDirectory() as tmp:
            receipt, events, path = self._run_fake_launcher(
                Path(tmp).resolve(),
                payload,
                payload_hash,
                runner_hash,
                {},
                workspace_error=RuntimeError("PRIVATE_SENTINEL must not reach the receipt"),
            )

            self.assertEqual(receipt["status"], "launch_failed")
            self.assertEqual(receipt["stage"], "preflight")
            self.assertEqual(receipt["error"]["message"], "private SFT launch failed")  # type: ignore[index]
            self.assertEqual(events, ["lock", "reserve", "python", "workspace", "receipt"])
            self.assertNotIn("PRIVATE_SENTINEL", path.read_text())
            self.assertEqual(json.loads(path.read_text()), receipt)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)

    def test_runner_digest_covers_injected_sources_and_callable_stays_bounded(self) -> None:
        required_sources = {
            "experiments/verifier_v2_test_remote.py",
            "experiments/sft_launch.py",
            "experiments/modal_verifier_pairs.py",
        }
        self.assertTrue(required_sources <= set(modal_private_sft._RUNNER_MODULES))
        expected = hashlib.sha256()
        for relative in modal_private_sft._RUNNER_MODULES:
            expected.update(relative.encode("utf-8"))
            expected.update(b"\0")
            expected.update((modal_private_sft._ROOT / relative).read_bytes())
            expected.update(b"\0")
        self.assertEqual(modal_private_sft.runner_sha256(), expected.hexdigest())

        entry = modal_private_sft._make_remote_function({}, "0" * 64)
        compute = entry.__globals__["_REMOTE_COMPUTE"]
        required_globals = {
            name
            for name in compute.__code__.co_names
            if name.startswith("_") and not name.startswith("__")
        }
        self.assertLessEqual(required_globals, set(compute.__globals__))
        try:
            from modal._serialization import serialize
        except ImportError:
            self.skipTest("Modal SDK is an optional development dependency")
        self.assertLessEqual(len(serialize(entry)), 65_536)

    def test_remote_private_json_writer_is_exclusive_and_mode_0600(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            modal_private_sft._write_private_json_exclusive(path, {"status": "success"})
            before = path.read_bytes()
            self.assertEqual(json.loads(before), {"status": "success"})
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                modal_private_sft._write_private_json_exclusive(path, {"status": "overwrite"})
            self.assertEqual(path.read_bytes(), before)

    def test_trained_model_reload_is_prepared_for_kbit_training(self) -> None:
        import ast

        tree = ast.parse(Path(private_sft_remote.__file__).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_model_for_kbit_training"
        ]
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "trained_base"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "prepare_model_for_kbit_training"
                for node in ast.walk(tree)
            )
        )

    def test_source_free_mount_guard_accepts_only_mount_root_symlink(self) -> None:
        entry = modal_private_sft._make_remote_function({}, "0" * 64)
        compute = entry.__globals__["_REMOTE_COMPUTE"]
        verify_mount = compute.__globals__["_VERIFY_ADAPTER_MOUNT"]
        self.assertEqual(verify_mount.__code__, _verify_adapter_mount.__code__)

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            volume = base / "volume"
            adapter = volume / "runs" / "payload" / "adapter"
            adapter.mkdir(parents=True)
            config = adapter / "adapter_config.json"
            weights = adapter / "adapter_model.safetensors"
            config.write_bytes(b"fictional config")
            weights.write_bytes(b"fictional weights")
            mount_alias = base / "mount"
            mount_alias.symlink_to(volume, target_is_directory=True)
            relative = "runs/payload/adapter"
            expected = {
                "adapter_config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
                "adapter_model.safetensors": hashlib.sha256(weights.read_bytes()).hexdigest(),
            }

            self.assertEqual(verify_mount(str(mount_alias), relative, expected), expected)
            unsafe = base / "unsafe"
            unsafe.mkdir()
            adapter_real = volume / "real-adapter"
            adapter.rename(adapter_real)
            adapter.symlink_to(unsafe, target_is_directory=True)
            with self.assertRaises(ValueError):
                verify_mount(str(mount_alias), relative, expected)


if __name__ == "__main__":
    unittest.main()
