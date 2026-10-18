"""Offline contracts for the private SFT Modal wrapper."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import sys
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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

    @classmethod
    def _balanced_payload(cls) -> dict[str, object]:
        payload = cls._minimal_payload()
        labels = ["supported"] * 117 + ["unsupported"] * 116 + ["uncertain"] * 23
        rows = [
            {
                "prompt": f"fictional prompt {index}",
                "completion": json.dumps(
                    {"support": label, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, label in enumerate(labels)
        ]
        payload["train_rows"] = rows
        payload["sampling_plan"] = private_sft.build_sampling_plan(rows, seed=17)
        return payload

    @classmethod
    def _explicit_exposure_payload(cls) -> dict[str, object]:
        payload = cls._minimal_payload()
        labels = ("supported", "unsupported", "uncertain")
        rows = [
            {
                "prompt": f"fictional compact-pool prompt {index}",
                "completion": json.dumps(
                    {"support": label, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, label in enumerate(labels)
        ]
        indices = [index % len(rows) for index in range(256)]
        payload["train_rows"] = rows
        payload["sampling_plan"] = {
            "strategy": "explicit-exposures-v1",
            "seed": 17,
            "row_indices": indices,
        }
        return payload

    @staticmethod
    def _validate_source_free_payload(payload: dict[str, object]) -> dict[str, object]:
        entry = modal_private_sft._make_remote_function(payload, "0" * 64)
        compute = entry.__globals__["_REMOTE_COMPUTE"]
        validator = compute.__globals__["_VALIDATE_PAYLOAD_SHAPE"]
        output_volume = {
            "name": private_sft._OUTPUT_VOLUME_NAME,
            "mount": private_sft._OUTPUT_VOLUME_MOUNT,
        }
        return validator(
            payload,
            private_sft._PROVENANCE,
            private_sft._MODEL,
            private_sft._SETTINGS,
            private_sft._prior_adapter(),
            output_volume,
        )

    def test_modal_function_options_are_one_source_free_a10g_call(self) -> None:
        options = modal_private_sft.modal_function_options()

        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertLessEqual(options["timeout"], 1800)
        self.assertFalse(options["include_source"])

    def test_modal_builder_mounts_public_cache_read_only_and_offline(self) -> None:
        class FakeVolume:
            def __init__(self, name: str) -> None:
                self.name = name
                self.read_only = False

            def with_mount_options(self, *, read_only: bool) -> FakeVolume:
                self.read_only = read_only
                return self

        class FakeImage:
            def __init__(self) -> None:
                self.environment: dict[str, str] = {}

            def pip_install(self, *_packages: str) -> FakeImage:
                return self

            def env(self, environment: dict[str, str]) -> FakeImage:
                self.environment.update(environment)
                return self

        class FakeApp:
            def __init__(self, _name: str) -> None:
                self.options: dict[str, object] = {}

            def function(self, **options: object) -> object:
                self.options = options
                return lambda function: function

        calls: list[tuple[str, dict[str, object]]] = []
        volumes: dict[str, FakeVolume] = {}
        image = FakeImage()

        def from_name(name: str, **options: object) -> FakeVolume:
            calls.append((name, options))
            volume = FakeVolume(name)
            volumes[name] = volume
            return volume

        modal_module = SimpleNamespace(
            Image=SimpleNamespace(debian_slim=lambda **_options: image),
            App=FakeApp,
            Volume=SimpleNamespace(from_name=from_name),
        )
        with (
            patch.object(modal_private_sft, "_REMOTE_OUTPUT_VOLUME", None),
            patch.dict(sys.modules, {"modal": modal_module}),
        ):
            built_app, _function = modal_private_sft._build_modal_function({}, "0" * 64)

        cache_name = "tracemem-public-model-cache-v1"
        cache_mount = "/mnt/tracemem-public-model-cache"
        self.assertEqual(
            dict(calls),
            {
                cache_name: {"create_if_missing": False},
                private_sft._PRIOR_VOLUME_NAME: {"create_if_missing": False},
                private_sft._OUTPUT_VOLUME_NAME: {"create_if_missing": True},
            },
        )
        self.assertTrue(volumes[cache_name].read_only)
        self.assertEqual(
            image.environment,
            {
                "HF_HUB_CACHE": f"{cache_mount}/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        )
        mounted_volumes = built_app.options["volumes"]
        self.assertEqual(
            mounted_volumes,
            {
                private_sft._PRIOR_VOLUME_MOUNT: volumes[private_sft._PRIOR_VOLUME_NAME],
                private_sft._OUTPUT_VOLUME_MOUNT: volumes[private_sft._OUTPUT_VOLUME_NAME],
                cache_mount: volumes[cache_name],
            },
        )
        self.assertTrue(volumes[private_sft._PRIOR_VOLUME_NAME].read_only)
        self.assertFalse(volumes[private_sft._OUTPUT_VOLUME_NAME].read_only)

    def test_remote_rejects_changed_payload_before_model_imports(self) -> None:
        payload = {"private_marker": "FICTIONAL_PRIVATE_SENTINEL"}
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

        with patch.object(
            modal_private_sft,
            "_REMOTE_OUTPUT_VOLUME",
            SimpleNamespace(commit=lambda: None),
        ):
            remote = modal_private_sft._make_remote_function(payload, "0" * 64)
            with patch("builtins.__import__", side_effect=block_model_import):
                result = remote(request)

        self.assertEqual(result["status"], "remote_failed")
        self.assertEqual(result["error"]["exception_chain"], ["ValueError"])
        self.assertNotIn("CHANGED_SENTINEL", str(result))
        self.assertNotIn("FICTIONAL_PRIVATE_SENTINEL", str(result))

    def test_remote_error_diagnostics_keep_only_bounded_safe_metadata(self) -> None:
        cause = PermissionError(13, "PRIVATE_SENTINEL https://private.invalid/path")
        error = RuntimeError("PRIVATE_PROMPT_SENTINEL")
        error.__cause__ = cause
        error.response = SimpleNamespace(status_code=503)

        diagnostics = private_sft_remote._safe_error_diagnostics(error)

        self.assertEqual(diagnostics["exception_chain"], ["RuntimeError", "PermissionError"])
        self.assertEqual(diagnostics["errno"], 13)
        self.assertEqual(diagnostics["http_status"], 503)
        serialized = json.dumps(diagnostics)
        for forbidden in ("PRIVATE_SENTINEL", "PRIVATE_PROMPT_SENTINEL", "https://"):
            self.assertNotIn(forbidden, serialized)

        deepest: BaseException = ValueError("PRIVATE_CHAIN_SENTINEL")
        for _ in range(8):
            wrapped = RuntimeError("PRIVATE_CHAIN_SENTINEL")
            wrapped.__cause__ = deepest
            deepest = wrapped
        self.assertEqual(
            len(private_sft_remote._safe_error_diagnostics(deepest)["exception_chain"]), 4
        )

        strict_types = RuntimeError("PRIVATE_TYPE_SENTINEL")
        strict_types.errno = True
        strict_types.status_code = True
        strict_types.response = SimpleNamespace(status_code="503")
        strict_diagnostics = private_sft_remote._safe_error_diagnostics(strict_types)
        self.assertNotIn("errno", strict_diagnostics)
        self.assertNotIn("http_status", strict_diagnostics)

    def test_remote_model_loading_uses_precise_failure_stages(self) -> None:
        source = Path(private_sft_remote.__file__).read_text(encoding="utf-8")
        remote_function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_remote_compute"
        )
        stage_lines = {
            node.value.value: node.lineno
            for node in ast.walk(remote_function)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value in {"tokenizer_load", "base_model_load"}
            and any(
                isinstance(target, ast.Name) and target.id == "stage" for target in node.targets
            )
        }
        loading_lines = {
            "tokenizer_load": min(
                node.lineno
                for node in ast.walk(remote_function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "AutoTokenizer"
            ),
            "base_model_load": min(
                node.lineno
                for node in ast.walk(remote_function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "AutoModelForCausalLM"
            ),
        }
        self.assertLess(stage_lines["tokenizer_load"], loading_lines["tokenizer_load"])
        self.assertLess(stage_lines["base_model_load"], loading_lines["base_model_load"])

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

    def test_remote_payload_validator_accepts_exact_balanced_plan(self) -> None:
        payload = self._balanced_payload()
        entry = modal_private_sft._make_remote_function(payload, "0" * 64)
        compute = entry.__globals__["_REMOTE_COMPUTE"]
        validator = compute.__globals__["_VALIDATE_PAYLOAD_SHAPE"]

        self.assertTrue(callable(validator.__globals__["_BUILD_SAMPLING_PLAN"]))
        validated = self._validate_source_free_payload(payload)

        self.assertEqual(validated, payload)

    def test_source_free_validator_accepts_compact_explicit_exposure_schedule(self) -> None:
        payload = self._explicit_exposure_payload()

        validated = self._validate_source_free_payload(payload)

        self.assertEqual(
            validated["sampling_plan"]["row_indices"],
            payload["sampling_plan"]["row_indices"],
        )
        self.assertEqual(len(validated["sampling_plan"]["row_indices"]), 256)

    def test_source_free_validator_rejects_malformed_explicit_exposure_schedules(self) -> None:
        mutations = (
            ("short_indices", lambda plan: plan["row_indices"].pop()),
            ("long_indices", lambda plan: plan["row_indices"].append(0)),
            ("negative_index", lambda plan: plan["row_indices"].__setitem__(0, -1)),
            ("out_of_range_index", lambda plan: plan["row_indices"].__setitem__(0, 3)),
            ("boolean_index", lambda plan: plan["row_indices"].__setitem__(0, True)),
            ("string_index", lambda plan: plan["row_indices"].__setitem__(0, "0")),
            ("wrong_seed", lambda plan: plan.__setitem__("seed", 18)),
            ("float_seed", lambda plan: plan.__setitem__("seed", 17.0)),
            ("unknown_strategy", lambda plan: plan.__setitem__("strategy", "explicit-v2")),
            ("extra_plan_key", lambda plan: plan.__setitem__("note", "unexpected")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                payload = self._explicit_exposure_payload()
                mutate(payload["sampling_plan"])
                with self.assertRaisesRegex(ValueError, "payload is invalid"):
                    self._validate_source_free_payload(payload)

    def test_source_free_validator_keeps_explicit_train_prompts_unique(self) -> None:
        payload = self._explicit_exposure_payload()
        payload["train_rows"][1]["prompt"] = payload["train_rows"][0]["prompt"]

        with self.assertRaisesRegex(ValueError, "payload is invalid"):
            self._validate_source_free_payload(payload)

    def test_source_free_validator_keeps_explicit_train_and_dev_prompts_disjoint(self) -> None:
        payload = self._explicit_exposure_payload()
        payload["dev_prompts"][0] = payload["train_rows"][0]["prompt"]

        with self.assertRaisesRegex(ValueError, "payload is invalid"):
            self._validate_source_free_payload(payload)

    def test_source_free_validator_accepts_and_rejects_execution_ids(self) -> None:
        payload = self._minimal_payload()
        payload["execution_id"] = "retry-2"

        validated = self._validate_source_free_payload(payload)

        self.assertEqual(validated["execution_id"], "retry-2")
        balanced_payload = self._balanced_payload()
        balanced_payload["execution_id"] = "retry-2"
        self.assertEqual(self._validate_source_free_payload(balanced_payload), balanced_payload)
        for invalid_id in (True, "", "Upper", "-retry", "retry_id", "a" * 65):
            with self.subTest(execution_id=invalid_id):
                invalid_payload = self._minimal_payload()
                invalid_payload["execution_id"] = invalid_id
                with self.assertRaisesRegex(ValueError, "payload is invalid"):
                    self._validate_source_free_payload(invalid_payload)

    def test_source_free_validator_rejects_tampered_balanced_plans(self) -> None:
        mutations = (
            ("boolean_index", lambda plan: plan["row_indices"].__setitem__(0, True)),
            ("out_of_range", lambda plan: plan["row_indices"].__setitem__(0, 256)),
            ("swapped_schedule", lambda plan: plan["row_indices"].reverse()),
            ("wrong_seed", lambda plan: plan.__setitem__("seed", 18)),
            ("wrong_strategy", lambda plan: plan.__setitem__("strategy", "support-balanced-v2")),
            ("null_plan", lambda payload: payload.__setitem__("sampling_plan", None)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                payload = self._balanced_payload()
                mutate(payload if name == "null_plan" else payload["sampling_plan"])
                with self.assertRaisesRegex(ValueError, "payload is invalid"):
                    self._validate_source_free_payload(payload)

    def test_balanced_training_metrics_count_schedule_exposures(self) -> None:
        payload = self._balanced_payload()
        rows = payload["train_rows"]
        plan = payload["sampling_plan"]
        indices = plan["row_indices"]
        examples = [
            (
                list(range(1 + index % 3)),
                [-100] * (1 + index % 3) + [101, 2],
            )
            for index in range(len(rows))
        ]

        metrics = private_sft_remote._training_metrics(rows, examples, indices)

        self.assertEqual(metrics["input_tokens"], sum(len(examples[index][0]) for index in indices))
        self.assertEqual(
            metrics["supervised_tokens"],
            sum(sum(label != -100 for label in examples[index][1]) for index in indices),
        )
        self.assertEqual(
            metrics["sampling"]["exposure_counts"],
            {
                "supported": 85,
                "unsupported": 85,
                "uncertain": 86,
            },
        )
        self.assertEqual(
            metrics["sampling"]["unique_source_counts"],
            {"supported": 85, "unsupported": 85, "uncertain": 23},
        )
        self.assertEqual(
            metrics["sampling"]["row_index_sha256"],
            hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("ascii")).hexdigest(),
        )
        self.assertEqual(
            [
                private_sft_remote._accumulation_window_size(position, len(indices), 4)
                for position in range(len(indices))
            ],
            [4] * 256,
        )

    def test_training_metrics_count_compact_explicit_exposures(self) -> None:
        payload = self._explicit_exposure_payload()
        rows = payload["train_rows"]
        indices = payload["sampling_plan"]["row_indices"]
        examples = [
            (list(range(index + 2)), [-100] * index + [101, 2]) for index in range(len(rows))
        ]

        metrics = private_sft_remote._training_metrics(rows, examples, indices)

        self.assertEqual(metrics["input_tokens"], 767)
        self.assertEqual(metrics["supervised_tokens"], 512)
        self.assertEqual(
            metrics["sampling"]["exposure_counts"],
            {"supported": 86, "unsupported": 85, "uncertain": 85},
        )
        self.assertEqual(
            metrics["sampling"]["unique_source_counts"],
            {"supported": 1, "unsupported": 1, "uncertain": 1},
        )
        self.assertEqual(
            metrics["sampling"]["row_index_sha256"],
            hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("ascii")).hexdigest(),
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
        self.assertTrue(callable(entry.__globals__["_SAFE_ERROR_DIAGNOSTICS"]))
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
