from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import FunctionType, SimpleNamespace


class QAInterpretationAnswerRunnerTests(unittest.TestCase):
    def test_module_exposes_the_pinned_request_contract(self) -> None:
        module_name = "experiments.modal_qa_interpretation_answer"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name),
            "the offline-first interpretation answer producer must exist",
        )
        producer = importlib.import_module(module_name)
        self.assertTrue(callable(getattr(producer, "build_request", None)))
        self.assertTrue(callable(getattr(producer, "request_sha256", None)))
        self.assertTrue(callable(getattr(producer, "validate_request", None)))
        self.assertTrue(callable(getattr(producer, "validate_result", None)))

    def test_request_is_the_exact_36_prompt_source_only_envelope(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        self.assertEqual(producer.request_sha256(request), producer._EXPECTED_REQUEST_SHA256)
        self.assertEqual(
            set(request),
            {"schema_version", "mode", "prepared_payload_sha256", "model", "tokenizer", "items"},
        )
        self.assertEqual(len(request["items"]), 36)
        self.assertTrue(
            all(
                set(item) == {"prompt_id", "messages"}
                and len(item["messages"]) == 2
                and [message["role"] for message in item["messages"]] == ["system", "user"]
                for item in request["items"]
            )
        )
        serialized = producer._canonical_bytes(request).decode("utf-8").lower()
        for excluded in ("reference", "annotation", "ground_truth", "answer_record"):
            self.assertNotIn(excluded, serialized)

    def test_modal_builder_is_source_free_and_uses_exact_fixed_resources(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        captured: dict[str, object] = {}
        events: list[str] = []

        class Image:
            @staticmethod
            def debian_slim(*, python_version: str) -> Image:
                captured["python_version"] = python_version
                return Image()

            def pip_install(self, *packages: str) -> Image:
                captured["packages"] = packages
                return self

        class App:
            def __init__(self, name: str) -> None:
                captured["app_name"] = name

            def function(self, *, image: object, **options: object):
                captured["options"] = options
                return lambda function: function

        fake_modal = SimpleNamespace(Image=Image, App=App)
        original_import = builtins.__import__

        def tracked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "modal":
                events.append("modal-import")
            return original_import(name, *args, **kwargs)

        runner_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        with (
            unittest.mock.patch.dict(sys.modules, {"modal": fake_modal}),
            unittest.mock.patch.object(
                producer,
                "_require_supported_launch_python",
                side_effect=lambda: events.append("python"),
            ),
            unittest.mock.patch.object(
                producer,
                "verify_personal_workspace",
                side_effect=lambda: events.append("workspace"),
            ),
            unittest.mock.patch("builtins.__import__", side_effect=tracked_import),
        ):
            app, function = producer._build_modal_function(
                approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                approved_runner_sha256=runner_hash,
            )
        self.assertIsInstance(app, App)
        self.assertEqual(function.__globals__, {"__name__": "__main__"})
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(events, ["python", "workspace", "modal-import"])
        self.assertEqual(captured["python_version"], "3.12")
        self.assertEqual(
            captured["packages"],
            (
                "torch==2.7.1",
                "transformers==4.51.3",
                "accelerate==1.6.0",
                "tokenizers==0.21.1",
                "jinja2==3.1.6",
            ),
        )
        self.assertEqual(
            captured["options"],
            {
                "gpu": "T4",
                "cpu": (2, 2),
                "memory": (16384, 16384),
                "min_containers": 0,
                "max_containers": 1,
                "buffer_containers": 0,
                "single_use_containers": True,
                "retries": 0,
                "startup_timeout": 300,
                "timeout": 600,
                "scaledown_window": 2,
                "serialized": True,
                "include_source": False,
            },
        )

    def test_modal_import_is_gated_by_both_approvals_and_local_runtime(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        imported: list[str] = []
        original_import = builtins.__import__

        def tracked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "modal":
                imported.append(name)
            return original_import(name, *args, **kwargs)

        runner_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        with unittest.mock.patch("builtins.__import__", side_effect=tracked_import):
            with self.assertRaisesRegex(ValueError, "request approval"):
                producer._build_modal_function(
                    approved_request_sha256="0" * 64,
                    approved_runner_sha256=runner_hash,
                )
            with (
                unittest.mock.patch.object(
                    producer,
                    "_require_supported_launch_python",
                    side_effect=RuntimeError("unsupported Python"),
                ),
                unittest.mock.patch.object(producer, "verify_personal_workspace") as workspace,
            ):
                with self.assertRaisesRegex(RuntimeError, "unsupported Python"):
                    producer._build_modal_function(
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                    )
                workspace.assert_not_called()
        self.assertEqual(imported, [])

    def test_fresh_process_import_does_not_load_model_or_modal_packages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        code = (
            "import sys; import experiments.modal_qa_interpretation_answer; "
            "print(sorted({'modal','torch','transformers','tokenizers'} & sys.modules.keys()))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_remote_tokenizes_all_prompts_before_model_and_records_exact_stops(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        invoke, state = _fake_remote_harness(request)
        result = invoke()
        self.assertEqual(state["tokenized_count"], 36)
        self.assertEqual(state["model_load_count"], 1)
        self.assertEqual(state["generate_count"], 36)
        self.assertEqual(state["tokenized_at_model_load"], [36])
        self.assertTrue(
            all(
                options
                == {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "truncation": False,
                }
                for options in state["template_options"]
            )
        )
        self.assertEqual(result["outputs"][0]["stop_reason"], "eos_token")
        self.assertEqual(result["outputs"][0]["input_token_ids"], [100, 200])
        self.assertEqual(result["outputs"][1]["stop_reason"], "max_new_tokens")
        self.assertTrue(result["outputs"][1]["truncated"])
        self.assertEqual(result["outputs"][1]["generated_tokens"], 128)
        eos_at_cap = result["outputs"][2]
        self.assertEqual(eos_at_cap["stop_reason"], "eos_token")
        self.assertEqual(eos_at_cap["generated_tokens"], 128)
        self.assertTrue(eos_at_cap["hit_max_new_tokens"])
        self.assertFalse(eos_at_cap["truncated"])
        self.assertIs(producer.validate_result(result, request, "a" * 64), result)

    def test_invalid_token_vectors_and_oversize_abort_before_model_loading(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        for invalid_ids in ([], [True], [-1], [1] * 1025):
            with self.subTest(token_count=len(invalid_ids)):
                invoke, state = _fake_remote_harness(request, invalid_ids=invalid_ids)
                with self.assertRaises(ValueError):
                    invoke()
                self.assertEqual(state["model_load_count"], 0)
                self.assertEqual(state["generate_count"], 0)

    def test_tokenizer_pin_drift_aborts_before_model_loading(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        bad_tokenizer_versions = {
            "transformers": "4.51.2",
            "tokenizers": "0.21.1",
            "jinja2": "3.1.6",
            "torch": "2.7.1",
            "accelerate": "1.6.0",
        }
        bad_model_versions = {
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "jinja2": "3.1.6",
            "torch": "2.7.0",
            "accelerate": "1.6.0",
        }
        for versions, expected_message in (
            (bad_tokenizer_versions, "package versions"),
            (bad_model_versions, "model runtime package versions"),
        ):
            with self.subTest(versions=versions):
                invoke, state = _fake_remote_harness(request, package_versions=versions)
                with self.assertRaisesRegex(ValueError, expected_message):
                    invoke()
                self.assertEqual(state["model_load_count"], 0)

    def test_wrong_remote_python_version_fails_before_model_import(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        invoke, state = _fake_remote_harness(producer.build_request())
        with unittest.mock.patch.object(platform, "python_version", return_value="3.11.8"):
            with self.assertRaisesRegex(RuntimeError, "pinned Python 3.12"):
                invoke()
        self.assertEqual(state["model_load_count"], 0)
        self.assertEqual(state["tokenized_count"], 0)

    def test_run_once_persists_raw_result_before_teardown_and_validation(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        runner_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        events: list[str] = []
        original_fsync = os.fsync

        class Function:
            calls = 0
            response: dict[str, object] | None = None
            fail_remote = False
            wrapper: dict[str, object] | None = None

            def remote(self, wrapper: dict[str, object]) -> dict[str, object]:
                self.calls += 1
                events.append("remote")
                self.wrapper = wrapper
                self.assert_wrapper_shape()
                if self.fail_remote:
                    raise RuntimeError("Bearer secret-do-not-log")
                self.response = _fake_valid_receipt(request, runner_hash)
                return self.response

            def assert_wrapper_shape(self) -> None:
                if set(self.wrapper or {}) != {"request", "request_sha256", "runner_sha256"}:
                    raise AssertionError("remote wrapper contains unexpected fields")

        function = Function()

        class RunContext:
            def __init__(self, raw_path: Path, failure: str | None) -> None:
                self.raw_path = raw_path
                self.failure = failure

            def __enter__(self) -> None:
                return None

            def __exit__(self, *_args: object) -> bool:
                events.append("teardown")
                self.assert_raw()
                if self.failure == "teardown":
                    raise RuntimeError("teardown failed")
                return False

            def assert_raw(self) -> None:
                self_outer.assertTrue(self.raw_path.exists())
                if function.fail_remote:
                    self_outer.assertEqual(self.raw_path.read_bytes(), b"")
                    return
                raw = json.loads(self.raw_path.read_text())
                self_outer.assertEqual(
                    set(raw),
                    {"schema_version", "request_sha256", "runner_sha256", "result"},
                )
                self_outer.assertIn("raw-fsync", events)
                self_outer.assertEqual(raw["result"], function.response)

        class App:
            def __init__(self, raw_path: Path, failure: str | None) -> None:
                self.context = RunContext(raw_path, failure)

            def run(self, *, detach: bool) -> RunContext:
                self_outer.assertFalse(detach)
                return self.context

        self_outer = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, failure in (
                ("success", None),
                ("teardown", "teardown"),
                ("validation", "validation"),
                ("remote", "remote"),
            ):
                with self.subTest(failure=failure):
                    events.clear()
                    function.calls = 0
                    function.response = None
                    function.fail_remote = failure == "remote"
                    output_path = root / f"{label}.json"
                    raw_path = Path(f"{output_path}.remote.json")
                    app = App(raw_path, failure)

                    def fake_builder(_app: App = app, **_approvals: object) -> tuple[App, Function]:
                        return _app, function

                    def validate(
                        result: object,
                        fixed_request: object,
                        fixed_hash: str,
                        _raw_path: Path = raw_path,
                        _failure: str | None = failure,
                    ) -> object:
                        events.append("validate")
                        self.assertTrue(_raw_path.exists())
                        self.assertIn("teardown", events)
                        if _failure == "validation":
                            raise ValueError("invalid fake result")
                        return producer._receipt.validate_result(result, fixed_request, fixed_hash)

                    def record_fsync(descriptor: int, _raw_path: Path = raw_path) -> None:
                        raw_text = _raw_path.read_text()
                        events.append("raw-fsync" if raw_text else "output-fsync")
                        original_fsync(descriptor)

                    with (
                        unittest.mock.patch.object(producer, "_require_supported_launch_python"),
                        unittest.mock.patch.object(
                            producer, "_build_modal_function", side_effect=fake_builder
                        ),
                        unittest.mock.patch.object(
                            producer, "validate_result", side_effect=validate
                        ),
                        unittest.mock.patch("os.fsync", side_effect=record_fsync),
                    ):
                        if failure:
                            with self.assertRaises((RuntimeError, ValueError)):
                                producer.run_once(
                                    request,
                                    approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                                    approved_runner_sha256=runner_hash,
                                    output_path=output_path,
                                )
                        else:
                            producer.run_once(
                                request,
                                approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                                approved_runner_sha256=runner_hash,
                                output_path=output_path,
                            )
                    self.assertEqual(function.calls, 1)
                    self.assertEqual(function.wrapper["request"], request)
                    self.assertEqual(
                        function.wrapper["request_sha256"], producer._EXPECTED_REQUEST_SHA256
                    )
                    self.assertEqual(function.wrapper["runner_sha256"], runner_hash)
                    self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(raw_path.stat().st_mode), 0o600)
                    if failure == "remote":
                        self.assertEqual(raw_path.read_bytes(), b"")
                        receipt = json.loads(output_path.read_text())
                        self.assertNotIn("secret-do-not-log", receipt["error"]["message"])
                        self.assertIn("[REDACTED]", receipt["error"]["message"])
                    else:
                        raw = json.loads(raw_path.read_text())
                        self.assertEqual(raw["result"], function.response)
                    if failure is None:
                        self.assertIn("result", json.loads(output_path.read_text()))
                    else:
                        self.assertIn("error", json.loads(output_path.read_text()))

    def test_cli_is_offline_by_default_and_requires_explicit_launch_approvals(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        with (
            unittest.mock.patch.dict(sys.modules, {"modal": None}),
            unittest.mock.patch.object(
                producer,
                "_build_modal_function",
                side_effect=AssertionError("offline CLI must not construct Modal"),
            ),
        ):
            summary_text = io.StringIO()
            with redirect_stdout(summary_text):
                self.assertEqual(producer.main([]), 0)
            summary = json.loads(summary_text.getvalue())
            self.assertEqual(summary["prompt_count"], 36)
            self.assertEqual(summary["request_sha256"], producer._EXPECTED_REQUEST_SHA256)
            self.assertEqual(summary["limits"], {"max_prompt_tokens": 1024, "max_new_tokens": 128})
            self.assertIs(summary["input_tokens_measured"], False)
            self.assertIs(summary["remote_started"], False)

            shown = io.StringIO()
            with redirect_stdout(shown):
                self.assertEqual(producer.main(["--show-request"]), 0)
            self.assertEqual(json.loads(shown.getvalue()), producer.build_request())

            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "approved.json"
                runner_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
                with unittest.mock.patch.object(producer, "run_once") as launch:
                    self.assertEqual(
                        producer.main(
                            [
                                "--launch",
                                "--approve-request-sha256",
                                producer._EXPECTED_REQUEST_SHA256,
                                "--approve-runner-sha256",
                                runner_hash,
                                "--output",
                                str(output),
                            ]
                        ),
                        0,
                    )
                launch.assert_called_once()
                self.assertEqual(
                    launch.call_args.kwargs["approved_request_sha256"],
                    producer._EXPECTED_REQUEST_SHA256,
                )
                self.assertEqual(launch.call_args.kwargs["approved_runner_sha256"], runner_hash)
                self.assertEqual(launch.call_args.kwargs["output_path"], output)

                cases = (
                    ["--approve-request-sha256", "a" * 64],
                    ["--output", str(Path(directory) / "receipt.json")],
                    ["--launch"],
                    [
                        "--launch",
                        "--show-request",
                        "--approve-request-sha256",
                        producer._EXPECTED_REQUEST_SHA256,
                        "--approve-runner-sha256",
                        "b" * 64,
                        "--output",
                        str(Path(directory) / "receipt.json"),
                    ],
                )
                for argv in cases:
                    with self.subTest(argv=argv):
                        errors = io.StringIO()
                        with redirect_stderr(errors):
                            self.assertEqual(producer.main(argv), 2)
                        self.assertTrue(errors.getvalue())

    def test_run_once_rejects_bad_approvals_and_preserves_existing_paths(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        request = producer.build_request()
        runner_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_cases = (
                ("request", "0" * 64, runner_hash, request),
                ("runner", producer._EXPECTED_REQUEST_SHA256, "0" * 64, request),
            )
            with (
                unittest.mock.patch.object(producer, "_require_supported_launch_python"),
                unittest.mock.patch.object(
                    producer,
                    "_build_modal_function",
                    side_effect=AssertionError("bad approval must not construct Modal"),
                ) as build_modal,
                unittest.mock.patch.object(producer, "verify_personal_workspace") as workspace,
            ):
                for label, approved_request, approved_runner, candidate in invalid_cases:
                    output = root / f"{label}.json"
                    with self.subTest(label=label):
                        with self.assertRaises(ValueError):
                            producer.run_once(
                                candidate,
                                approved_request_sha256=approved_request,
                                approved_runner_sha256=approved_runner,
                                output_path=output,
                            )
                        self.assertFalse(output.exists())
                        self.assertFalse(Path(f"{output}.remote.json").exists())

                changed = json.loads(json.dumps(request))
                changed["items"][0]["messages"][1]["content"] = "changed source text"
                output = root / "changed.json"
                with self.assertRaisesRegex(ValueError, "request"):
                    producer.run_once(
                        changed,
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                        output_path=output,
                    )
                self.assertFalse(output.exists())

                with unittest.mock.patch.object(
                    producer, "_EXPECTED_CONTRACT_SOURCE_SHA256", "0" * 64
                ):
                    with self.assertRaisesRegex(ValueError, "contract source differs"):
                        producer.run_once(
                            request,
                            approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                            approved_runner_sha256=runner_hash,
                            output_path=root / "source-drift.json",
                        )
                self.assertFalse((root / "source-drift.json").exists())
                build_modal.assert_not_called()
                workspace.assert_not_called()

                existing = root / "existing.json"
                existing.write_bytes(b"keep this receipt")
                with self.assertRaisesRegex(FileExistsError, "receipt output path must be new"):
                    producer.run_once(
                        request,
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                        output_path=existing,
                    )
                self.assertEqual(existing.read_bytes(), b"keep this receipt")

                target = root / "symlink-target.json"
                target.write_bytes(b"keep this target")
                linked = root / "symlink.json"
                linked.symlink_to(target)
                with self.assertRaisesRegex(FileExistsError, "receipt output path must be new"):
                    producer.run_once(
                        request,
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                        output_path=linked,
                    )
                self.assertTrue(linked.is_symlink())
                self.assertEqual(target.read_bytes(), b"keep this target")

                raw_path = root / "raw-conflict.json.remote.json"
                raw_path.write_bytes(b"keep this raw data")
                receipt_path = root / "raw-conflict.json"
                with self.assertRaisesRegex(FileExistsError, "raw remote output path must be new"):
                    producer.run_once(
                        request,
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                        output_path=receipt_path,
                    )
                self.assertEqual(raw_path.read_bytes(), b"keep this raw data")
                raw_target = root / "raw-target.json"
                raw_target.write_bytes(b"preserve raw symlink target")
                raw_link = root / "raw-link.json.remote.json"
                raw_link.symlink_to(raw_target)
                with self.assertRaisesRegex(FileExistsError, "raw remote output path must be new"):
                    producer.run_once(
                        request,
                        approved_request_sha256=producer._EXPECTED_REQUEST_SHA256,
                        approved_runner_sha256=runner_hash,
                        output_path=root / "raw-link.json",
                    )
                self.assertTrue(raw_link.is_symlink())
                self.assertEqual(raw_target.read_bytes(), b"preserve raw symlink target")
                build_modal.assert_not_called()
                workspace.assert_not_called()

    def test_remote_body_preflights_hashes_before_model_imports(self) -> None:
        producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
        remote_body = getattr(producer, "_remote_compute", None)
        self.assertTrue(callable(remote_body), "source-free remote body is missing")
        remote = FunctionType(remote_body.__code__, {"__name__": "__main__"})
        self.assertEqual(remote.__globals__, {"__name__": "__main__"})
        self.assertEqual(remote.__code__.co_freevars, ())
        request = producer.build_request()
        wrappers = (
            {
                "request": request,
                "request_sha256": "0" * 64,
                "runner_sha256": "a" * 64,
            },
            {
                "request": request,
                "request_sha256": producer._EXPECTED_REQUEST_SHA256,
                "runner_sha256": "bad-runner-hash",
            },
            {
                "request": {**request, "references": {"must_not_leave_local": True}},
                "request_sha256": producer._EXPECTED_REQUEST_SHA256,
                "runner_sha256": "a" * 64,
            },
        )
        with unittest.mock.patch.dict(
            sys.modules, {"transformers": None, "torch": None, "modal": None}
        ):
            for wrapper in wrappers:
                with self.subTest(wrapper=wrapper):
                    with self.assertRaises(ValueError):
                        remote(wrapper)


def _fake_remote_harness(
    request: dict[str, object],
    *,
    invalid_ids: list[int] | None = None,
    package_versions: dict[str, str] | None = None,
) -> tuple[object, dict[str, object]]:
    producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
    state: dict[str, object] = {
        "tokenized_count": 0,
        "tokenized_at_model_load": [],
        "template_options": [],
        "model_load_count": 0,
        "generate_count": 0,
    }
    remote = FunctionType(producer._remote_compute.__code__, {"__name__": "__main__"})
    wrapper = {
        "request": request,
        "request_sha256": producer.request_sha256(request),
        "runner_sha256": "a" * 64,
    }
    expected_template = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    expected_backend = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
    original_sha256 = hashlib.sha256

    class StubHash:
        def __init__(self, digest: str) -> None:
            self.digest = digest

        def hexdigest(self) -> str:
            return self.digest

    def fake_sha256(value: bytes = b"", *args: object, **kwargs: object) -> object:
        if value == b"approved-template":
            return StubHash(expected_template)
        if value == b"approved-backend":
            return StubHash(expected_backend)
        return original_sha256(value, *args, **kwargs)

    class Tokenizer:
        is_fast = True
        chat_template = "approved-template"
        eos_token_id = 151645
        pad_token_id = 0
        backend_tokenizer = SimpleNamespace(to_str=lambda: "approved-backend")

        def apply_chat_template(self, messages: object, **options: object) -> list[int]:
            index = state["tokenized_count"]
            state["tokenized_count"] = index + 1
            state["template_options"].append(options)
            if index == 0 and invalid_ids is not None:
                return invalid_ids
            return [100 + index, 200]

        @staticmethod
        def decode(_token_ids: list[int], *, skip_special_tokens: bool) -> str:
            return "FAKE TEST OUTPUT; no model was run"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
            return Tokenizer()

    class Tensor:
        def __init__(self, data: list[object]) -> None:
            self.data = data

        def tolist(self) -> list[object]:
            return self.data

    class Batch:
        def __init__(self, rows: list[list[int]]) -> None:
            self.rows = rows

        def __getitem__(self, index: int) -> Tensor:
            return Tensor(self.rows[index])

    class Event:
        def record(self) -> None:
            pass

        @staticmethod
        def elapsed_time(_end: object) -> float:
            return 1.0

    class Cuda:
        Event = staticmethod(lambda **_kwargs: Event())

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            pass

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_properties(_device: int) -> object:
            return SimpleNamespace(major=7, minor=5)

        @staticmethod
        def get_device_name(_device: int) -> str:
            return "Tesla T4"

        @staticmethod
        def reset_peak_memory_stats(_device: int) -> None:
            pass

        @staticmethod
        def max_memory_allocated(_device: int) -> int:
            return 1024

        @staticmethod
        def synchronize() -> None:
            pass

    class Model:
        @staticmethod
        def eval() -> None:
            pass

        @staticmethod
        def generate(**kwargs: object) -> Batch:
            index = state["generate_count"]
            state["generate_count"] = index + 1
            if index == 0:
                generated = [300, 151645]
            elif index == 1:
                generated = [301] * 128
            elif index == 2:
                generated = [302] * 127 + [151645]
            else:
                generated = [303, 151645]
            input_ids = kwargs["input_ids"].data[0]
            return Batch([input_ids + generated])

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> Model:
            state["model_load_count"] = state["model_load_count"] + 1
            state["tokenized_at_model_load"].append(state["tokenized_count"])
            return Model()

    versions = package_versions or {
        "transformers": "4.51.3",
        "tokenizers": "0.21.1",
        "jinja2": "3.1.6",
        "torch": "2.7.1",
        "accelerate": "1.6.0",
    }
    fake_transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )
    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        version=SimpleNamespace(cuda="12.6"),
        float16="float16",
        long="long",
        manual_seed=lambda _seed: None,
        tensor=lambda data, **_kwargs: Tensor(data),
        ones_like=lambda value: value,
        inference_mode=nullcontext,
    )
    fake_modal = SimpleNamespace(current_function_call_id=lambda: "fake-modal-call")

    def invoke() -> dict[str, object]:
        with (
            unittest.mock.patch.dict(
                sys.modules,
                {"transformers": fake_transformers, "torch": fake_torch, "modal": fake_modal},
            ),
            unittest.mock.patch.object(
                importlib.metadata, "version", side_effect=lambda name: versions[name]
            ),
            unittest.mock.patch("hashlib.sha256", side_effect=fake_sha256),
        ):
            return remote(wrapper)

    return invoke, state


def _fake_valid_receipt(request: dict[str, object], runner_hash: str) -> dict[str, object]:
    producer = importlib.import_module("experiments.modal_qa_interpretation_answer")
    model = request["model"]
    tokenizer = request["tokenizer"]
    return {
        "schema_version": 1,
        "request_sha256": producer.request_sha256(request),
        "runner_sha256": runner_hash,
        "model": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "generation_protocol_sha256": producer.request_sha256(model),
        },
        "tokenizer": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "template_sha256": tokenizer["template_sha256"],
            "backend_tokenizer_sha256": tokenizer["backend_tokenizer_sha256"],
            "use_fast": True,
            "trust_remote_code": False,
            "package_versions": tokenizer["package_versions"],
            "python_version": platform.python_version(),
            "template_options": tokenizer["template_options"],
            "eos_token_id": tokenizer["eos_token_id"],
        },
        "runtime": {
            "python_version": platform.python_version(),
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
                "generated_tokens": 1,
                "generated_token_ids": [tokenizer["eos_token_id"]],
                "stop_reason": "eos_token",
                "hit_max_new_tokens": False,
                "truncated": False,
            }
            for item in request["items"]
        ],
    }


if __name__ == "__main__":
    unittest.main()
