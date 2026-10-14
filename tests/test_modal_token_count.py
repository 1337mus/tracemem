"""Focused offline tests for the reviewed Modal tokenizer counter."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import stat
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _runner_file_hash(runner: Any) -> str:
    source_path = runner.__file__
    if not isinstance(source_path, str):
        raise AssertionError("runner module has no source path")
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def _fake_remote_result(payload: dict[str, Any], runner_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "count_receipt": {
            "schema_version": 1,
            "catalog_sha256": payload["catalog_sha256"],
            "counter_id": (
                "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
                "+transformers-4.51.3+template-a55ee1b1"
            ),
            # These fixture integers exercise receipt handling, not real token counts.
            "counts": [
                {"prompt_id": item["prompt_id"], "prompt_tokens": 3} for item in payload["items"]
            ],
        },
        "tokenizer_provenance": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
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
            },
            "backend_tokenizer_sha256": None,
            "elapsed_seconds": 0.25,
            "payload_sha256": "125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88",
            "runner_sha256": runner_sha256,
        },
    }


class TokenCountPayloadTests(unittest.TestCase):
    def test_default_cli_prints_a_dry_summary_without_modal_side_effects(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        main = cast(Callable[[list[str] | None], int], getattr(runner, "main", None))
        self.assertTrue(callable(main), "the offline-default CLI is missing")

        output = StringIO()
        with (
            patch.object(runner, "verify_personal_workspace") as workspace_check,
            patch.object(runner, "_build_modal_function") as build_function,
            redirect_stdout(output),
        ):
            exit_code = main([])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["item_count"], 28)
        self.assertEqual(
            summary["payload_sha256"],
            "125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88",
        )
        self.assertFalse(summary["remote_started"])
        workspace_check.assert_not_called()
        build_function.assert_not_called()

    def test_result_guard_rejects_nonfinite_elapsed_and_malformed_backend_hash(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        payload = runner.build_payload()
        runner_sha256 = _runner_file_hash(runner)
        cases = (
            ("elapsed_nan", "elapsed_seconds", float("nan"), "finite"),
            ("backend_hash", "backend_tokenizer_sha256", "x" * 64, "backend tokenizer hash"),
        )
        for name, key, value, expected_reason in cases:
            with self.subTest(name=name):
                result = _fake_remote_result(payload, runner_sha256)
                result["tokenizer_provenance"][key] = value
                with self.assertRaisesRegex(ValueError, expected_reason):
                    runner._validate_remote_result(result, payload, runner_sha256)

    def test_builder_reproduces_the_reviewed_payload_and_catalog_hashes(self) -> None:
        spec = importlib.util.find_spec("experiments.modal_token_count")
        self.assertIsNotNone(spec, "the reviewed Modal token-count runner is missing")
        runner = importlib.import_module("experiments.modal_token_count")

        payload = runner.build_payload()
        self.assertEqual(
            hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            "125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88",
        )
        self.assertEqual(
            set(payload), {"schema_version", "mode", "catalog_sha256", "tokenizer", "items"}
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["mode"], "synthetic-token-count-v1")
        self.assertEqual(
            payload["catalog_sha256"],
            "85682edbcdd728103e6540c86106de953f400e217a9341ffbd82420d0429e478",
        )
        self.assertEqual(len(payload["items"]), 28)

    def test_launch_guards_reject_before_workspace_or_modal_setup(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        run_once = cast(Callable[..., dict[str, Any]], getattr(runner, "run_once", None))
        self.assertTrue(callable(run_once), "the approved one-shot launch path is missing")

        payload = runner.build_payload()
        payload_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        runner_sha256 = _runner_file_hash(runner)
        changed_payload = dict(payload)
        changed_payload["mode"] = "caller-modified"
        cases = (
            (
                "unsupported_python",
                payload,
                payload_sha256,
                runner_sha256,
                (3, 11),
                False,
                RuntimeError,
                "local Python 3.12",
            ),
            (
                "changed_payload",
                changed_payload,
                hashlib.sha256(_canonical_bytes(changed_payload)).hexdigest(),
                runner_sha256,
                (3, 12),
                False,
                ValueError,
                "caller payload",
            ),
            (
                "wrong_payload_approval",
                payload,
                "0" * 64,
                runner_sha256,
                (3, 12),
                False,
                ValueError,
                "exact reviewed payload approval",
            ),
            (
                "wrong_runner_approval",
                payload,
                payload_sha256,
                "0" * 64,
                (3, 12),
                False,
                ValueError,
                "exact current runner approval",
            ),
            (
                "existing_output",
                payload,
                payload_sha256,
                runner_sha256,
                (3, 12),
                True,
                FileExistsError,
                "receipt output path must be new",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="tracemem-token-counter-guards-") as directory:
            for (
                name,
                value,
                approved_payload,
                approved_runner,
                version,
                existing,
                error_type,
                error_message,
            ) in cases:
                with self.subTest(name=name):
                    output = Path(directory) / f"{name}.json"
                    if existing:
                        output.write_text("preserve", encoding="utf-8")
                    with (
                        patch.object(
                            runner.sys,
                            "version_info",
                            type("Version", (), {"major": version[0], "minor": version[1]})(),
                        ),
                        patch.object(runner, "verify_personal_workspace") as workspace_check,
                        patch.object(runner, "_build_modal_function") as build_function,
                        self.assertRaisesRegex(error_type, error_message),
                    ):
                        run_once(
                            value,
                            approved_payload_sha256=approved_payload,
                            approved_runner_sha256=approved_runner,
                            output_path=output,
                        )
                    workspace_check.assert_not_called()
                    build_function.assert_not_called()
                    self.assertEqual(
                        output.read_text(encoding="utf-8") if existing else output.exists(),
                        "preserve" if existing else False,
                    )

    def test_approved_fake_transport_writes_one_private_success_receipt(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        payload = runner.build_payload()
        runner_sha256 = _runner_file_hash(runner)
        payload_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        result = _fake_remote_result(payload, runner_sha256)
        app = unittest.mock.MagicMock()
        function = unittest.mock.MagicMock()
        function.remote.return_value = result

        with tempfile.TemporaryDirectory(prefix="tracemem-token-counter-success-") as directory:
            output = Path(directory) / "receipt.json"
            with (
                patch.object(
                    runner.sys,
                    "version_info",
                    type("Version", (), {"major": 3, "minor": 12})(),
                ),
                patch.object(runner, "verify_personal_workspace") as workspace_check,
                patch.object(runner, "_build_modal_function", return_value=(app, function)),
            ):
                receipt = runner.run_once(
                    payload,
                    approved_payload_sha256=payload_sha256,
                    approved_runner_sha256=runner_sha256,
                    output_path=output,
                )

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(receipt["result"], result)
        workspace_check.assert_called_once_with()
        app.run.assert_called_once_with(detach=False)
        function.remote.assert_called_once_with(
            {
                "payload": payload,
                "payload_sha256": payload_sha256,
                "runner_sha256": runner_sha256,
            }
        )

    def test_remote_failure_writes_one_sanitized_receipt_without_retry(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        payload = runner.build_payload()
        runner_sha256 = _runner_file_hash(runner)
        payload_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        app = unittest.mock.MagicMock()
        function = unittest.mock.MagicMock()
        function.remote.side_effect = RuntimeError("Bearer synthetic-secret-token")

        with tempfile.TemporaryDirectory(prefix="tracemem-token-counter-failure-") as directory:
            output = Path(directory) / "failure.json"
            with (
                patch.object(
                    runner.sys,
                    "version_info",
                    type("Version", (), {"major": 3, "minor": 12})(),
                ),
                patch.object(runner, "verify_personal_workspace"),
                patch.object(runner, "_build_modal_function", return_value=(app, function)),
                self.assertRaisesRegex(RuntimeError, "Bearer synthetic-secret-token"),
            ):
                runner.run_once(
                    payload,
                    approved_payload_sha256=payload_sha256,
                    approved_runner_sha256=runner_sha256,
                    output_path=output,
                )

            failure = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(failure["payload_sha256"], payload_sha256)
        self.assertEqual(failure["runner_sha256"], runner_sha256)
        self.assertNotIn("synthetic-secret-token", failure["error"]["message"])
        self.assertIn("[REDACTED]", failure["error"]["message"])
        function.remote.assert_called_once()


class RemoteTokenCountTests(unittest.TestCase):
    def test_remote_counter_pins_template_and_rejects_invalid_token_id_lists(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        remote_count = cast(
            Callable[[dict[str, Any]], dict[str, Any]],
            getattr(runner, "_remote_count", None),
        )
        self.assertTrue(callable(remote_count), "the source-free remote counter is missing")

        events: list[tuple[str, Any]] = []
        template = "fake pinned template"
        pin_template_hash = True
        original_sha256 = hashlib.sha256

        class Tokenizer:
            chat_template = template
            is_fast = True
            token_ids: Any = [7, 11, 13]

            def apply_chat_template(self, messages: object, **options: object) -> list[int]:
                events.append(("template", (messages, options)))
                return cast(list[int], self.token_ids)

        tokenizer = Tokenizer()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str, **options: object) -> Tokenizer:
                events.append(("load", (model_id, options)))
                return tokenizer

        transformers = type("Transformers", (), {"AutoTokenizer": AutoTokenizer})
        original_import = __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "transformers":
                return transformers
            return original_import(name, *args, **kwargs)

        def fake_sha256(value: bytes = b"") -> Any:
            if pin_template_hash and value == template.encode("utf-8"):
                return type(
                    "PinnedDigest",
                    (),
                    {
                        "hexdigest": lambda self: (
                            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
                        )
                    },
                )()
            return original_sha256(value)

        payload = runner.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            "runner_sha256": "a" * 64,
        }
        versions = {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}
        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
            patch("hashlib.sha256", side_effect=fake_sha256),
        ):
            result = remote_count(request)

        self.assertEqual(result["count_receipt"]["catalog_sha256"], payload["catalog_sha256"])
        self.assertEqual(
            result["count_receipt"]["counts"],
            [{"prompt_id": item["prompt_id"], "prompt_tokens": 3} for item in payload["items"]],
        )
        self.assertIs(result["tokenizer_provenance"]["use_fast"], True)
        load_events = [event for event in events if event[0] == "load"]
        template_events = [event for event in events if event[0] == "template"]
        self.assertEqual(
            load_events,
            [
                (
                    "load",
                    (
                        "Qwen/Qwen3-4B",
                        {
                            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                            "trust_remote_code": False,
                            "use_fast": True,
                        },
                    ),
                )
            ],
        )
        self.assertEqual(len(template_events), 28)
        self.assertEqual(
            template_events[0][1][1],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
        tokenizer.is_fast = False
        events.clear()
        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
            patch("hashlib.sha256", side_effect=fake_sha256),
            self.assertRaisesRegex(ValueError, "not a fast tokenizer"),
        ):
            remote_count(request)
        self.assertEqual([event[0] for event in events], ["load"])
        tokenizer.is_fast = True
        for invalid_token_ids in ([True], [-1], (7, 11, 13)):
            with self.subTest(token_ids=invalid_token_ids):
                tokenizer.token_ids = invalid_token_ids
                with (
                    patch("builtins.__import__", side_effect=fake_import),
                    patch("importlib.metadata.version", side_effect=versions.__getitem__),
                    patch("hashlib.sha256", side_effect=fake_sha256),
                    self.assertRaisesRegex(ValueError, "invalid token IDs"),
                ):
                    remote_count(request)
        pin_template_hash = False
        events.clear()
        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
            patch("hashlib.sha256", side_effect=fake_sha256),
            self.assertRaisesRegex(ValueError, "template hash differs"),
        ):
            remote_count(request)
        self.assertEqual([event[0] for event in events], ["load"])
        bad_requests = (
            ({**request, "payload_sha256": "0" * 64}, "not the reviewed payload hash"),
            ({**request, "runner_sha256": "not-a-hash"}, "remote runner hash is invalid"),
        )
        for bad_request, expected_reason in bad_requests:
            with self.subTest(remote_request=expected_reason):
                events.clear()
                with (
                    patch("builtins.__import__", side_effect=fake_import),
                    patch("importlib.metadata.version", side_effect=versions.__getitem__),
                    patch("hashlib.sha256", side_effect=fake_sha256),
                    self.assertRaisesRegex(ValueError, expected_reason),
                ):
                    remote_count(bad_request)
                self.assertEqual(events, [])

    def test_modal_function_uses_isolated_cpu_only_single_use_configuration(self) -> None:
        runner = importlib.import_module("experiments.modal_token_count")
        build_function = cast(
            Callable[[], tuple[Any, Any]], getattr(runner, "_build_modal_function", None)
        )
        self.assertTrue(
            callable(build_function), "the source-free Modal function builder is missing"
        )

        registered: list[tuple[object, dict[str, object]]] = []

        class App:
            def function(self, **options: object) -> Any:
                def register(function: object) -> object:
                    registered.append((function, options))
                    return function

                return register

        class Image:
            def pip_install(self, *packages: str) -> str:
                self.packages = packages
                return "fake-image"

        image_builder = Image()
        app = App()
        fake_modal = type(
            "Modal",
            (),
            {
                "App": staticmethod(lambda name: app if name == "tracemem-token-count" else None),
                "Image": type(
                    "Images",
                    (),
                    {
                        "debian_slim": staticmethod(
                            lambda **options: (
                                image_builder if options == {"python_version": "3.12"} else None
                            )
                        )
                    },
                ),
            },
        )
        with patch.dict("sys.modules", {"modal": fake_modal}):
            returned_app, function = build_function()

        self.assertIs(returned_app, app)
        self.assertEqual(len(registered), 1)
        self.assertIs(registered[0][0], function)
        self.assertEqual(function.__globals__, {"__name__": "__main__"})
        self.assertIsNone(function.__closure__)
        options = registered[0][1]
        self.assertEqual(options["cpu"], (1, 2))
        self.assertNotIn("gpu", options)
        self.assertNotIn("mounts", options)
        self.assertNotIn("secrets", options)
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertTrue(options["single_use_containers"])
        self.assertFalse(options["include_source"])
        self.assertEqual(image_builder.packages, runner._IMAGE_PACKAGES)
