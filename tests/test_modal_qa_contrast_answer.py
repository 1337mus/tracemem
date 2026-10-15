from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import (
    modal_qa_contrast_answer,
    prepare_qa_contrast_requests,
    qa_contrast_answer_receipt,
)


class QAContrastAnswerRunnerTests(unittest.TestCase):
    def test_build_payload_is_callable(self) -> None:
        self.assertTrue(
            callable(getattr(modal_qa_contrast_answer, "build_payload", None)),
            "the source-only QA answer payload API is missing",
        )

    def test_payload_is_exact_prepared_source_only_input(self) -> None:
        expected, _references, _audit = prepare_qa_contrast_requests.prepare_requests()
        payload = modal_qa_contrast_answer.build_payload()
        self.assertEqual(payload, expected)
        self.assertEqual(len(payload["items"]), 46)
        self.assertEqual(
            modal_qa_contrast_answer.payload_sha256(payload),
            "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e",
        )
        self.assertEqual(
            hashlib.sha256(Path(prepare_qa_contrast_requests.__file__).read_bytes()).hexdigest(),
            payload["provenance"]["preparer_source_sha256"],
        )
        serialized = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "answer_record_ids",
            "authored_answer_record_ids",
            "cell_mapping",
            "references_sha256",
            "scoring",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_validate_payload_checks_the_complete_fixed_prompt_contract(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        self.assertIs(modal_qa_contrast_answer.validate_payload(payload), payload)
        changed = json.loads(json.dumps(payload))
        changed["items"][0]["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "reviewed payload hash"):
            modal_qa_contrast_answer.validate_payload(changed)

    def test_validate_result_binds_all_46_outputs_and_runtime_provenance(self) -> None:
        payload = qa_contrast_answer_receipt.build_payload()
        runner_sha256 = "a" * 64
        result = _valid_result(payload, runner_sha256)
        self.assertIs(
            qa_contrast_answer_receipt.validate_result(result, payload, runner_sha256), result
        )
        self.assertEqual(
            result["model"]["generation_protocol_sha256"],
            "3ed43f94abba3419cd62814aa4c32b5943acfa73684ac1197fabc84b7a58d9f2",
        )
        mutations = (
            ("missing output", "exactly 46 outputs", lambda value: value["outputs"].pop()),
            (
                "input count",
                "remote output identity",
                lambda value: value["outputs"][0].__setitem__("input_tokens", 0),
            ),
            (
                "boolean token count",
                "remote output identity",
                lambda value: value["outputs"][0].__setitem__("input_tokens", True),
            ),
            (
                "runner hash",
                "approval provenance",
                lambda value: value.__setitem__("runner_sha256", "b" * 64),
            ),
            (
                "negative timing",
                "finite and nonnegative",
                lambda value: value["runtime"].__setitem__("elapsed_seconds", -1),
            ),
            (
                "wrong stop flag",
                "stop flags are inconsistent",
                lambda value: value["outputs"][0].__setitem__("truncated", True),
            ),
            (
                "unsorted outputs",
                "remote output identity",
                lambda value: value["outputs"].reverse(),
            ),
            (
                "generation protocol hash",
                "generation protocol hash",
                lambda value: value["model"].__setitem__("generation_protocol_sha256", "d" * 64),
            ),
            (
                "unexpected GPU",
                "runtime provenance",
                lambda value: value["runtime"].__setitem__("gpu_name", "CPU"),
            ),
            (
                "unexpected capability",
                "runtime provenance",
                lambda value: value["runtime"].__setitem__("gpu_capability", "0.0"),
            ),
            (
                "negative GPU time",
                "finite and nonnegative",
                lambda value: value["runtime"].__setitem__("gpu_compute_seconds", -1),
            ),
        )
        for label, message, mutate in mutations:
            with self.subTest(label=label):
                changed = json.loads(json.dumps(result))
                mutate(changed)
                with self.assertRaisesRegex(ValueError, message):
                    qa_contrast_answer_receipt.validate_result(changed, payload, runner_sha256)

        capped = json.loads(json.dumps(result))
        capped_output = capped["outputs"][0]
        capped_output.update(
            generated_tokens=128,
            generated_token_ids=[1] * 128,
            stop_reason="max_new_tokens",
            hit_max_new_tokens=True,
            truncated=True,
        )
        self.assertIs(
            qa_contrast_answer_receipt.validate_result(capped, payload, runner_sha256), capped
        )
        eos_boundary = json.loads(json.dumps(capped))
        eos_boundary["outputs"][0]["generated_token_ids"][-1] = 2
        eos_boundary["outputs"][0]["stop_reason"] = "eos_token"
        eos_boundary["outputs"][0]["truncated"] = False
        self.assertIs(
            qa_contrast_answer_receipt.validate_result(eos_boundary, payload, runner_sha256),
            eos_boundary,
        )
        eos_at_cap = json.loads(json.dumps(capped))
        eos_at_cap["outputs"][0]["generated_token_ids"][-1] = 2
        with self.assertRaisesRegex(ValueError, "token-cap stop flags"):
            qa_contrast_answer_receipt.validate_result(eos_at_cap, payload, runner_sha256)

    def test_remote_payload_preflight_rejects_mutation_before_model_imports(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        payload["items"][0]["prompt_tokens"] += 1
        request = {
            "payload": payload,
            "payload_sha256": "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e",
            "runner_sha256": "a" * 64,
        }
        with patch.dict("sys.modules", {"torch": None, "transformers": None, "modal": None}):
            with self.assertRaisesRegex(ValueError, "payload bytes"):
                modal_qa_contrast_answer._remote_compute(request)

    def test_modal_builder_reconstructs_a_source_free_function_with_fixed_resources(self) -> None:
        captured: dict[str, object] = {}

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
        with patch.dict("sys.modules", {"modal": fake_modal}):
            _app, function = modal_qa_contrast_answer._build_modal_function()
        self.assertEqual(captured["python_version"], "3.12")
        self.assertEqual(function.__globals__, {"__name__": "__main__"})
        self.assertEqual(function.__code__.co_freevars, ())
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
        self.assertEqual(captured["options"]["gpu"], "T4")
        self.assertEqual(captured["options"]["include_source"], False)
        self.assertEqual(captured["options"]["serialized"], True)

    def test_run_once_persists_raw_receipt_before_teardown_and_validation(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        runner_sha256 = hashlib.sha256(
            Path(modal_qa_contrast_answer.__file__).read_bytes()
        ).hexdigest()
        events: list[str] = []
        original_fsync = os.fsync

        class Function:
            calls = 0
            fail_remote = False
            response: dict[str, object] | None = None

            def remote(self, request: dict[str, object]) -> dict[str, object]:
                self.calls += 1
                events.append("remote")
                self.assert_wrapper = request
                self.assert_payload = request["payload"]
                if self.fail_remote:
                    raise RuntimeError("Bearer secret-do-not-log")
                self.response = _valid_result(payload, runner_sha256)
                return self.response

        function = Function()

        class RunContext:
            def __init__(self, raw_path: Path, fail_exit: bool) -> None:
                self.raw_path, self.fail_exit = raw_path, fail_exit

            def __enter__(self) -> None:
                return None

            def __exit__(self, *_args: object) -> bool:
                self.assert_raw()
                events.append("exit")
                if self.fail_exit:
                    raise RuntimeError("teardown failed")
                return False

            def assert_raw(self) -> None:
                self_assert(self.raw_path.exists(), "raw receipt must exist before app teardown")
                if function.fail_remote:
                    self_assert(
                        self.raw_path.read_text() == "",
                        "no raw return is available before a failed remote call",
                    )
                    return
                raw = json.loads(self.raw_path.read_text())
                self_assert(
                    set(raw) == {"schema_version", "payload_sha256", "runner_sha256", "result"},
                    "raw wrapper shape",
                )
                self_assert("fsync" in events, "raw receipt must be fsynced before app teardown")
                self_assert(
                    raw["result"] == function.response, "raw result preserves the return value"
                )

        class App:
            def __init__(self, raw_path: Path, fail_exit: bool) -> None:
                self.context = RunContext(raw_path, fail_exit)

            def run(self, *, detach: bool) -> RunContext:
                self_assert(detach is False, "run must be attached")
                return self.context

        def self_assert(condition: object, message: str) -> None:
            if not condition:
                raise AssertionError(message)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix, failure in (
                ("ok", None),
                ("teardown", "teardown"),
                ("invalid", "validation"),
                ("remote-failure", "remote"),
            ):
                with self.subTest(failure=failure):
                    events.clear()
                    function.calls = 0
                    function.fail_remote = failure == "remote"
                    output_path = root / f"{suffix}.json"
                    raw_path = Path(f"{output_path}.remote.json")
                    app = App(raw_path, failure == "teardown")

                    def validate(
                        result: object,
                        fixed_payload: object,
                        fixed_hash: str,
                        raw_path: Path = raw_path,
                        failure: str | None = failure,
                    ) -> object:
                        events.append("validate")
                        self.assertTrue(raw_path.exists())
                        self.assertIn("exit", events)
                        if failure == "validation":
                            raise ValueError("invalid fixture result")
                        return result

                    with (
                        patch.object(modal_qa_contrast_answer, "verify_personal_workspace"),
                        patch.object(
                            modal_qa_contrast_answer,
                            "_build_modal_function",
                            return_value=(app, function),
                        ),
                        patch.object(
                            modal_qa_contrast_answer, "validate_result", side_effect=validate
                        ),
                        patch(
                            "os.fsync",
                            side_effect=lambda fd: (events.append("fsync"), original_fsync(fd))[1],
                        ),
                    ):
                        if failure:
                            with self.assertRaises((RuntimeError, ValueError)):
                                modal_qa_contrast_answer.run_once(
                                    payload,
                                    approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                                    approved_runner_sha256=runner_sha256,
                                    output_path=output_path,
                                )
                        else:
                            modal_qa_contrast_answer.run_once(
                                payload,
                                approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                                approved_runner_sha256=runner_sha256,
                                output_path=output_path,
                            )
                    self.assertEqual(function.calls, 1)
                    self.assertTrue(raw_path.exists())
                    self.assertEqual(
                        set(function.assert_wrapper),
                        {"payload", "payload_sha256", "runner_sha256"},
                    )
                    self.assertEqual(function.assert_payload, payload)
                    if failure == "remote":
                        self.assertEqual(raw_path.read_text(), "")
                        failure_receipt = json.loads(output_path.read_text())
                        self.assertNotIn("secret-do-not-log", failure_receipt["error"]["message"])
                        self.assertIn("[REDACTED]", failure_receipt["error"]["message"])
                        self.assertLess(events.index("exit"), events.index("fsync"))
                    else:
                        self.assertEqual(
                            json.loads(raw_path.read_text())["runner_sha256"], runner_sha256
                        )
                        self.assertIn("fsync", events)
                    if failure != "teardown":
                        if failure != "remote":
                            self.assertLess(events.index("fsync"), events.index("exit"))
                            self.assertLess(events.index("exit"), events.index("validate"))

    def test_cli_is_offline_by_default_and_show_payload_never_launches(self) -> None:
        with (
            patch.dict("sys.modules", {"modal": None}),
            patch.object(
                modal_qa_contrast_answer,
                "_build_modal_function",
                side_effect=AssertionError("must remain offline"),
            ),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                status = modal_qa_contrast_answer.main([])
            summary = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(summary["item_count"], 46)
            self.assertEqual(
                summary["payload_sha256"],
                "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e",
            )
            self.assertIs(summary["remote_started"], False)

            shown = io.StringIO()
            with redirect_stdout(shown):
                status = modal_qa_contrast_answer.main(["--show-payload"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(shown.getvalue()), modal_qa_contrast_answer.build_payload())

            errors = io.StringIO()
            with redirect_stderr(errors):
                status = modal_qa_contrast_answer.main(["--launch"])
            self.assertEqual(status, 2)
            self.assertIn("requires both exact approvals", errors.getvalue())

    def test_launch_preflight_rejects_bad_approval_and_preserves_existing_targets(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        runner_sha256 = hashlib.sha256(
            Path(modal_qa_contrast_answer.__file__).read_bytes()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(modal_qa_contrast_answer, "verify_personal_workspace") as verify,
                patch.object(
                    modal_qa_contrast_answer,
                    "_build_modal_function",
                    side_effect=AssertionError("must not build Modal"),
                ),
            ):
                fresh = root / "fresh.json"
                with self.assertRaisesRegex(ValueError, "exact reviewed payload approval"):
                    modal_qa_contrast_answer.run_once(
                        payload,
                        approved_payload_sha256="0" * 64,
                        approved_runner_sha256=runner_sha256,
                        output_path=fresh,
                    )
                self.assertFalse(fresh.exists())
                verify.assert_not_called()

                with self.assertRaisesRegex(ValueError, "exact current runner approval"):
                    modal_qa_contrast_answer.run_once(
                        payload,
                        approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                        approved_runner_sha256="0" * 64,
                        output_path=fresh,
                    )
                self.assertFalse(fresh.exists())
                verify.assert_not_called()

                with patch.object(
                    modal_qa_contrast_answer,
                    "_EXPECTED_RECEIPT_SOURCE_SHA256",
                    "0" * 64,
                ):
                    with self.assertRaisesRegex(ValueError, "receipt source differs"):
                        modal_qa_contrast_answer.run_once(
                            payload,
                            approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                            approved_runner_sha256=runner_sha256,
                            output_path=fresh,
                        )
                self.assertFalse(fresh.exists())
                verify.assert_not_called()

                changed = json.loads(json.dumps(payload))
                changed["items"][0]["prompt_tokens"] += 1
                with self.assertRaisesRegex(ValueError, "fixed reviewed payload"):
                    modal_qa_contrast_answer.run_once(
                        changed,
                        approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                        approved_runner_sha256=runner_sha256,
                        output_path=fresh,
                    )
                self.assertFalse(fresh.exists())
                verify.assert_not_called()

                existing = root / "existing.json"
                existing.write_bytes(b"keep me")
                with self.assertRaisesRegex(FileExistsError, "output path must be new"):
                    modal_qa_contrast_answer.run_once(
                        payload,
                        approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                        approved_runner_sha256=runner_sha256,
                        output_path=existing,
                    )
                self.assertEqual(existing.read_bytes(), b"keep me")

                dangling = root / "dangling.json"
                dangling.symlink_to(root / "missing-target")
                with self.assertRaisesRegex(FileExistsError, "output path must be new"):
                    modal_qa_contrast_answer.run_once(
                        payload,
                        approved_payload_sha256=modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
                        approved_runner_sha256=runner_sha256,
                        output_path=dangling,
                    )
                self.assertTrue(dangling.is_symlink())
                verify.assert_not_called()

    def test_remote_tokenizer_identity_failure_precedes_model_loading(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        calls = {"model_load": 0}

        class Tokenizer:
            is_fast = True
            chat_template = "unreviewed template"

            class Backend:
                @staticmethod
                def to_str() -> str:
                    return "unreviewed backend"

            backend_tokenizer = Backend()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
                return Tokenizer()

        class AutoModel:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> None:
                calls["model_load"] += 1

        fake_transformers = SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel
        )
        request = {
            "payload": payload,
            "payload_sha256": modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
            "runner_sha256": "d" * 64,
        }
        with patch.dict(
            "sys.modules", {"transformers": fake_transformers, "torch": None, "modal": None}
        ):
            with self.assertRaisesRegex(ValueError, "template or backend hash"):
                modal_qa_contrast_answer._remote_compute(request)
        self.assertEqual(calls["model_load"], 0)

    def test_reconstructed_remote_body_rejects_count_mismatch_before_model_loading(self) -> None:
        payload = modal_qa_contrast_answer.build_payload()
        calls = {"model_load": 0, "generate": 0}
        expected_template = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        expected_backend = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
        original_sha256 = hashlib.sha256

        class StubHash:
            def __init__(self, digest: str) -> None:
                self.digest = digest

            def hexdigest(self) -> str:
                return self.digest

        class Tokenizer:
            is_fast = True
            chat_template = "approved-template"
            eos_token_id = 2
            pad_token_id = 0

            class Backend:
                @staticmethod
                def to_str() -> str:
                    return "approved-backend"

            backend_tokenizer = Backend()

            @staticmethod
            def apply_chat_template(*_args: object, **_kwargs: object) -> list[int]:
                return [1]

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
                return Tokenizer()

        class AutoModel:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> object:
                calls["model_load"] += 1
                return SimpleNamespace(generate=lambda **_kwargs: calls.__setitem__("generate", 1))

        class Image:
            @staticmethod
            def debian_slim(*_args: object, **_kwargs: object) -> Image:
                return Image()

            def pip_install(self, *_args: object) -> Image:
                return self

        class App:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def function(self, *_args: object, **_kwargs: object):
                return lambda function: function

        fake_transformers = SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel
        )
        fake_modal = SimpleNamespace(Image=Image, App=App)

        def fake_sha256(value: bytes = b"", *args: object, **kwargs: object) -> object:
            if value == b"approved-template":
                return StubHash(expected_template)
            if value == b"approved-backend":
                return StubHash(expected_backend)
            return original_sha256(value, *args, **kwargs)

        request = {
            "payload": payload,
            "payload_sha256": modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
            "runner_sha256": "d" * 64,
        }
        with (
            patch.dict(
                "sys.modules",
                {"transformers": fake_transformers, "modal": fake_modal, "torch": None},
            ),
            patch("hashlib.sha256", side_effect=fake_sha256),
        ):
            _app, reconstructed = modal_qa_contrast_answer._build_modal_function()
            with self.assertRaisesRegex(ValueError, "input count differs"):
                reconstructed(request)
        self.assertEqual(calls, {"model_load": 0, "generate": 0})

    def test_reconstructed_remote_body_emits_protocol_hash_and_gpu_timing(self) -> None:
        import importlib.metadata

        payload = modal_qa_contrast_answer.build_payload()
        prompt_counts = [item["prompt_tokens"] for item in payload["items"]]
        tokenized = {"count": 0}
        calls = {"model_load": 0, "generate": 0}
        expected_template = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        expected_backend = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
        original_sha256 = hashlib.sha256

        class StubHash:
            def __init__(self, digest: str) -> None:
                self.digest = digest

            def hexdigest(self) -> str:
                return self.digest

        class Tensor:
            def __init__(self, data: list[object]) -> None:
                self.data = data

            def tolist(self) -> list[object]:
                return self.data

        class Batch(Tensor):
            def __getitem__(self, index: int) -> Tensor:
                return Tensor(self.data[index])

        class Event:
            def record(self) -> None:
                pass

            @staticmethod
            def elapsed_time(_end: object) -> float:
                return 10.0

        class Cuda:
            version = "12.6"
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

        class Tokenizer:
            is_fast = True
            chat_template = "approved-template"
            eos_token_id = 2
            pad_token_id = 0

            class Backend:
                @staticmethod
                def to_str() -> str:
                    return "approved-backend"

            backend_tokenizer = Backend()

            @staticmethod
            def apply_chat_template(*_args: object, **_kwargs: object) -> list[int]:
                count = prompt_counts[tokenized["count"]]
                tokenized["count"] += 1
                return list(range(count))

            @staticmethod
            def decode(_tokens: list[int], **_kwargs: object) -> str:
                return "Answer"

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
                return Tokenizer()

        class Model:
            config = SimpleNamespace(to_dict=lambda: {})

            @staticmethod
            def eval() -> None:
                pass

            @staticmethod
            def generate(**kwargs: object) -> Batch:
                calls["generate"] += 1
                input_ids = kwargs["input_ids"].data[0]
                return Batch([input_ids + [2]])

        class AutoModel:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> Model:
                calls["model_load"] += 1
                return Model()

        class Image:
            @staticmethod
            def debian_slim(*_args: object, **_kwargs: object) -> Image:
                return Image()

            def pip_install(self, *_args: object) -> Image:
                return self

        class App:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def function(self, *_args: object, **_kwargs: object):
                return lambda function: function

        fake_transformers = SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel
        )
        fake_modal = SimpleNamespace(
            Image=Image, App=App, current_function_call_id=lambda: "fc-fake"
        )

        def fake_sha256(value: bytes = b"", *args: object, **kwargs: object) -> object:
            if value == b"approved-template":
                return StubHash(expected_template)
            if value == b"approved-backend":
                return StubHash(expected_backend)
            return original_sha256(value, *args, **kwargs)

        versions = {
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "jinja2": "3.1.6",
            "torch": "2.7.1",
            "accelerate": "1.6.0",
        }
        request = {
            "payload": payload,
            "payload_sha256": modal_qa_contrast_answer._EXPECTED_PAYLOAD_SHA256,
            "runner_sha256": "e" * 64,
        }
        with (
            patch.dict(
                "sys.modules",
                {"transformers": fake_transformers, "modal": fake_modal, "torch": fake_torch},
            ),
            patch("hashlib.sha256", side_effect=fake_sha256),
            patch.object(importlib.metadata, "version", side_effect=lambda name: versions[name]),
        ):
            _app, reconstructed = modal_qa_contrast_answer._build_modal_function()
            result = reconstructed(request)
        self.assertEqual(calls, {"model_load": 1, "generate": 46})
        self.assertEqual(
            result["model"]["generation_protocol_sha256"],
            "3ed43f94abba3419cd62814aa4c32b5943acfa73684ac1197fabc84b7a58d9f2",
        )
        self.assertEqual(result["runtime"]["gpu_compute_seconds"], 0.46)


def _valid_result(payload: dict[str, object], runner_sha256: str) -> dict[str, object]:
    items = payload["items"]
    generation_protocol_sha256 = hashlib.sha256(
        json.dumps(
            payload["model"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert generation_protocol_sha256 == (
        "3ed43f94abba3419cd62814aa4c32b5943acfa73684ac1197fabc84b7a58d9f2"
    )
    return {
        "schema_version": 1,
        "payload_sha256": qa_contrast_answer_receipt.payload_sha256(payload),
        "runner_sha256": runner_sha256,
        "model": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "generation_protocol_sha256": generation_protocol_sha256,
        },
        "tokenizer": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            "backend_tokenizer_sha256": (
                "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
            ),
            "use_fast": True,
            "trust_remote_code": False,
            "package_versions": {
                "transformers": "4.51.3",
                "tokenizers": "0.21.1",
                "jinja2": "3.1.6",
            },
            "python_version": "3.12.10",
            "template_options": {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "truncation": False,
            },
            "eos_token_id": 2,
        },
        "runtime": {
            "python_version": "3.12.10",
            "torch_version": "2.7.1",
            "accelerate_version": "1.6.0",
            "cuda_version": "12.6",
            "gpu_name": "Tesla T4",
            "gpu_capability": "7.5",
            "model_load_seconds": 2.0,
            "gpu_compute_seconds": 1.0,
            "elapsed_seconds": 4.0,
            "gpu_peak_memory_bytes": 1024,
            "request_count": 46,
            "modal_call_id": "fc-123",
        },
        "outputs": [
            {
                "prompt_id": item["prompt_id"],
                "raw_output": "Answer",
                "input_tokens": item["prompt_tokens"],
                "generated_tokens": 1,
                "generated_token_ids": [2],
                "stop_reason": "eos_token",
                "hit_max_new_tokens": False,
                "truncated": False,
            }
            for item in items
        ],
    }
