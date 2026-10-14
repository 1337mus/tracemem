"""Tests for the offline synthetic QA Modal runner."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.metadata
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch


def _runner_file_hash(runner: Any) -> str:
    runner_path = runner.__file__
    if not isinstance(runner_path, str):
        raise AssertionError("runner module has no source file")
    return hashlib.sha256(Path(runner_path).read_bytes()).hexdigest()


class RemoteTestError(ValueError):
    events: list[str]


def _run_remote_with_fakes(
    *, prompt_lengths: list[int] | None = None, generated_ids: list[int] | None = None
) -> tuple[dict[str, Any], list[str]]:
    runner = importlib.import_module("experiments.modal_synthetic_qa")
    events: list[str] = []
    token_lengths = prompt_lengths or []
    output_ids = generated_ids if generated_ids is not None else [99]

    class Tensor:
        def __init__(self, length: int) -> None:
            self.shape = (1, length)
            self.ndim = 2

        def to(self, device: str) -> Tensor:
            return self

    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class TokenSpan:
        def __init__(self, values: list[int]) -> None:
            self.values = values
            self.shape = (1, len(values))

        def __getitem__(self, index: int) -> Scalar:
            return Scalar(self.values[index])

    class Generated:
        def __getitem__(self, _: object) -> TokenSpan:
            return TokenSpan(output_ids)

    class Event:
        def __init__(self, **_: object) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, _: Event) -> float:
            return 1.0

    class Tokenizer:
        chat_template = "fake pinned template"

        def apply_chat_template(self, *_: object, **__: object) -> Tensor:
            index = sum(event == "tokenized" for event in events)
            events.append("tokenized")
            return Tensor(token_lengths[index] if index < len(token_lengths) else 10)

        def decode(self, *_: object, **__: object) -> str:
            return ""

    class Model:
        device = "cuda"
        generation_config = SimpleNamespace(eos_token_id=99, pad_token_id=None)

        def eval(self) -> None:
            pass

        def generate(self, *_: object, **__: object) -> Generated:
            events.append("generated")
            return Generated()

    model = Model()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> Tokenizer:
            events.append("tokenizer_loaded")
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> Model:
            events.append("model_loaded")
            return model

    cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda _: None,
        current_device=lambda: 0,
        get_device_name=lambda _: "T4 test double",
        get_device_properties=lambda _: SimpleNamespace(major=7, minor=5),
        reset_peak_memory_stats=lambda _: None,
        Event=Event,
        synchronize=lambda: None,
        max_memory_allocated=lambda _: 1234,
    )
    torch = SimpleNamespace(
        Tensor=Tensor,
        cuda=cuda,
        manual_seed=lambda _: None,
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        float16="float16",
        version=SimpleNamespace(cuda="fake-cuda"),
    )
    modal = SimpleNamespace(current_function_call_id=lambda: "fc-test")
    transformers = SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel)
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "modal":
            return modal
        if name == "torch":
            return torch
        if name == "transformers":
            return transformers
        return original_import(name, *args, **kwargs)

    payload = runner.build_payload()
    request = {
        "payload": payload,
        "payload_sha256": runner.payload_sha256(payload),
        "runner_sha256": "a" * 64,
    }
    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch.object(importlib.metadata, "version", return_value="test-version"),
    ):
        try:
            result = runner._remote_compute(request)
        except ValueError as error:
            observed_error = RemoteTestError(str(error))
            observed_error.events = events
            raise observed_error from error
    return result, events


class SyntheticQAPayloadTests(unittest.TestCase):
    def test_builder_reproduces_root_approved_payload(self) -> None:
        try:
            runner = importlib.import_module("experiments.modal_synthetic_qa")
        except ModuleNotFoundError:
            runner = None

        self.assertIsNotNone(runner, "the synthetic QA runner has no payload builder yet")
        if runner is None:
            return
        payload = runner.build_payload()
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac",
        )
        self.assertEqual(len(payload["items"]), 72)

    def test_builder_refuses_a_changed_preparer_source_pin(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        request_bundle, references = runner.prepare_synthetic_qa()
        request_bundle["experiment_source_sha256"] = "0" * 64
        with (
            patch.object(
                runner,
                "prepare_synthetic_qa",
                return_value=(request_bundle, references),
            ),
            self.assertRaisesRegex(ValueError, "preparer source hash"),
        ):
            runner.build_payload()


class SyntheticQACliTests(unittest.TestCase):
    def test_default_invocation_is_offline_dry_run(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        output = StringIO()
        with (
            patch.object(runner, "verify_personal_workspace", create=True) as workspace_check,
            patch.object(runner, "_build_modal_function", create=True) as build_function,
            redirect_stdout(output),
        ):
            exit_code = runner.main([])

        self.assertEqual(exit_code, 0)
        self.assertFalse(json.loads(output.getvalue())["remote_started"])
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_launch_without_approvals_and_show_payload_launch_are_refused_offline(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        stderr = StringIO()
        with (
            patch.object(runner, "verify_personal_workspace") as workspace_check,
            patch.object(runner, "_build_modal_function") as build_function,
            redirect_stderr(stderr),
        ):
            self.assertEqual(runner.main(["--launch"]), 2)
            self.assertEqual(runner.main(["--launch", "--show-payload"]), 2)
        self.assertIn("exact approvals", stderr.getvalue())
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_show_payload_prints_only_the_fixed_payload(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner.main(["--show-payload"]), 0)
        self.assertEqual(json.loads(output.getvalue()), runner.build_payload())


class SyntheticQAOptionsTests(unittest.TestCase):
    def test_modal_options_pin_one_bounded_source_free_call(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        self.assertEqual(
            runner.modal_function_options(),
            {
                "gpu": "T4",
                "cpu": (2, 2),
                "memory": (16_384, 16_384),
                "min_containers": 0,
                "max_containers": 1,
                "buffer_containers": 0,
                "retries": 0,
                "startup_timeout": 300,
                "timeout": 600,
                "scaledown_window": 2,
                "single_use_containers": True,
                "serialized": True,
                "include_source": False,
            },
        )


class SyntheticQARemoteValidationTests(unittest.TestCase):
    def test_malformed_payload_is_rejected_before_heavy_imports(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"heavy inference import happened too early: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaises(ValueError):
                runner._remote_compute({"payload": {}, "payload_sha256": "0" * 64})

    def test_remote_image_matches_local_python_minor(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        image = MagicMock()
        fake_modal = SimpleNamespace(
            App=MagicMock(),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=image)),
        )
        with patch.dict("sys.modules", {"modal": fake_modal}):
            runner._build_modal_function()

        fake_modal.Image.debian_slim.assert_called_once_with(python_version="3.12")

    def test_sdk_serializes_the_registered_remote_body_by_value(self) -> None:
        try:
            from modal._utils.function_utils import FunctionSourceInfo
            from modal._vendor import cloudpickle
        except ModuleNotFoundError as error:
            self.skipTest(f"optional Modal SDK serialization check unavailable: {error.name}")

        runner = importlib.import_module("experiments.modal_synthetic_qa")
        registered: list[Any] = []
        app = MagicMock()

        def register_remote(function: Any) -> Any:
            registered.append(function)
            return function

        app.function.side_effect = lambda **_: register_remote
        image = MagicMock()
        fake_modal = SimpleNamespace(
            App=MagicMock(return_value=app),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=image)),
        )
        with patch.dict("sys.modules", {"modal": fake_modal}):
            _, function = runner._build_modal_function()

        self.assertEqual(len(registered), 1)
        self.assertIs(registered[0], function)
        source = FunctionSourceInfo(function, serialized=True)
        serialized = source.serialized_function()
        restored = cloudpickle.loads(serialized)
        self.assertGreater(len(serialized), 1_000)
        self.assertIsNot(restored, function)
        self.assertNotIn("experiments", restored.__globals__)
        with self.assertRaises(ValueError):
            restored({"payload": {}, "payload_sha256": "0" * 64})

    def test_all_prompts_are_tokenized_before_model_load_and_empty_text_is_valid(self) -> None:
        result, events = _run_remote_with_fakes()
        self.assertEqual(events.count("tokenized"), 72)
        final_tokenization = max(i for i, event in enumerate(events) if event == "tokenized")
        self.assertLess(final_tokenization, events.index("model_loaded"))
        self.assertLess(events.index("model_loaded"), events.index("generated"))
        self.assertEqual(result["request_count"], 72)
        self.assertEqual(result["modal_call_id"], "fc-test")
        self.assertTrue(result["config_sha256"])
        self.assertTrue(result["template_sha256"])
        self.assertEqual(result["outputs"][0]["input_tokens"], 10)
        self.assertEqual(result["outputs"][0]["generated_tokens"], 1)
        self.assertEqual(result["outputs"][0]["raw_output"], "")

    def test_overlimit_prompt_refuses_before_model_load_or_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "2048") as raised:
            _run_remote_with_fakes(prompt_lengths=[2049])
        self.assertIsInstance(raised.exception, RemoteTestError)
        events = cast(RemoteTestError, raised.exception).events
        self.assertNotIn("model_loaded", events)
        self.assertNotIn("generated", events)

    def test_eos_at_token_limit_differs_from_length_truncation(self) -> None:
        eos_result, _ = _run_remote_with_fakes(generated_ids=[7] * 127 + [99])
        length_result, _ = _run_remote_with_fakes(generated_ids=[7] * 128)
        eos_row = eos_result["outputs"][0]
        length_row = length_result["outputs"][0]
        self.assertEqual((eos_row["stop_reason"], eos_row["generated_tokens"]), ("eos", 128))
        self.assertTrue(eos_row["hit_max_new_tokens"])
        self.assertFalse(eos_row["truncated"])
        self.assertEqual(
            (length_row["stop_reason"], length_row["generated_tokens"]), ("length", 128)
        )
        self.assertTrue(length_row["truncated"])

    def test_early_non_eos_termination_is_not_reported_as_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "without EOS or the token limit"):
            _run_remote_with_fakes(generated_ids=[7])


class SyntheticQALaunchTests(unittest.TestCase):
    def test_unsupported_local_python_refuses_before_launch_side_effects(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        payload_hash = runner.payload_sha256(payload)
        runner_hash = _runner_file_hash(runner)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with (
                patch.object(runner.sys, "version_info", SimpleNamespace(major=3, minor=11)),
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function") as build_function,
                self.assertRaisesRegex(RuntimeError, "local Python 3.12"),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=payload_hash,
                    approved_runner_sha256=runner_hash,
                    output_path=output,
                )

            self.assertFalse(output.exists())
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_mutated_payload_is_rejected_even_with_its_matching_hash(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        payload["mode"] = "caller-modified"
        caller_hash = runner.payload_sha256(payload)
        runner_hash = _runner_file_hash(runner)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with (
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function") as build_function,
                self.assertRaises(ValueError),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=caller_hash,
                    approved_runner_sha256=runner_hash,
                    output_path=output,
                )
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_symlink_receipt_path_is_refused_before_workspace_or_app_check(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        runner_hash = _runner_file_hash(runner)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text("do not replace", encoding="utf-8")
            link = Path(directory) / "receipt.json"
            link.symlink_to(target)
            with (
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function") as build_function,
                self.assertRaises(FileExistsError),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=runner.payload_sha256(payload),
                    approved_runner_sha256=runner_hash,
                    output_path=link,
                )
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_existing_receipt_path_is_rejected_before_workspace_or_app_check(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        runner_hash = _runner_file_hash(runner)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_text("existing receipt", encoding="utf-8")
            with (
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function") as build_function,
                self.assertRaises(FileExistsError),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=runner.payload_sha256(payload),
                    approved_runner_sha256=runner_hash,
                    output_path=output,
                )
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_wrong_runner_approval_is_refused_before_workspace_or_app_check(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with (
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function") as build_function,
                redirect_stderr(stderr),
            ):
                exit_code = runner.main(
                    [
                        "--launch",
                        "--approve-payload-sha256",
                        runner.payload_sha256(payload),
                        "--approve-runner-sha256",
                        "0" * 64,
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(exit_code, 2)
        self.assertIn("runner", stderr.getvalue())
        workspace_check.assert_not_called()
        build_function.assert_not_called()
        self.assertFalse(output.exists())

    def test_approved_launch_writes_fresh_private_receipt_after_one_call(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        payload_hash = runner.payload_sha256(payload)
        runner_hash = _runner_file_hash(runner)
        app = MagicMock()
        app.app_id = "ap-synthetic"
        function = MagicMock()
        result = {"payload_sha256": payload_hash, "modal_call_id": "fc-synthetic", "outputs": []}
        function.remote.return_value = result
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with (
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function", return_value=(app, function)),
            ):
                receipt = runner.run_once(
                    payload,
                    approved_payload_sha256=payload_hash,
                    approved_runner_sha256=runner_hash,
                    output_path=output,
                )

            self.assertEqual(
                set(receipt),
                {
                    "schema_version",
                    "payload_sha256",
                    "runner_sha256",
                    "client_elapsed_seconds",
                    "result",
                },
            )
            self.assertEqual(receipt["result"], result)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        workspace_check.assert_called_once_with()
        function.remote.assert_called_once_with(
            {
                "payload": payload,
                "payload_sha256": payload_hash,
                "runner_sha256": runner_hash,
            }
        )

    def test_remote_failure_writes_sanitized_receipt_without_retry(self) -> None:
        runner = importlib.import_module("experiments.modal_synthetic_qa")
        payload = runner.build_payload()
        payload_hash = runner.payload_sha256(payload)
        runner_hash = _runner_file_hash(runner)
        app = MagicMock()
        function = MagicMock()
        function.remote.side_effect = RuntimeError("Bearer synthetic-secret-token")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            with (
                patch.object(runner, "verify_personal_workspace"),
                patch.object(runner, "_build_modal_function", return_value=(app, function)),
                self.assertRaises(RuntimeError),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=payload_hash,
                    approved_runner_sha256=runner_hash,
                    output_path=output,
                )
            failure = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        function.remote.assert_called_once()
        self.assertEqual(failure["payload_sha256"], payload_hash)
        self.assertNotIn("synthetic-secret-token", failure["error"]["message"])
        self.assertIn("[REDACTED]", failure["error"]["message"])


if __name__ == "__main__":
    unittest.main()
