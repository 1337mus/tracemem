from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _runner_hash(runner: Any) -> str:
    return hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()


def _fake_result(
    payload: dict[str, Any], runner_sha: str, count_ids: list[str] | None = None
) -> dict[str, Any]:
    ids = count_ids or [item["prompt_id"] for item in payload["items"]]
    return {
        "schema_version": 1,
        "count_receipt": {
            "schema_version": 1,
            "catalog_sha256": payload["catalog_sha256"],
            "counter_id": (
                "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
                "+transformers-4.51.3+template-a55ee1b1"
            ),
            # Fake integers exercise receipt validation, not real token counts.
            "counts": [{"prompt_id": prompt_id, "prompt_tokens": 3} for prompt_id in ids],
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
            "payload_sha256": "7f3055ef98fec05f77676b3b995b27c6c6ebcfc2566fdd668177f065a3d9fcc2",
            "runner_sha256": runner_sha,
        },
    }


class ModalQaContrastTokenCountTests(unittest.TestCase):
    def test_build_payload_matches_reviewed_source_only_catalog(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        payload = runner.build_payload()
        self.assertEqual(payload["mode"], "qa-contrast-dev-token-count-v1")
        self.assertEqual(
            payload["catalog_sha256"],
            "2c18978bea31ba5c4e77d20fa6103a2a7a182bb93c7d94dd5c257b029e9662cf",
        )
        self.assertEqual(
            _payload_hash(payload),
            "7f3055ef98fec05f77676b3b995b27c6c6ebcfc2566fdd668177f065a3d9fcc2",
        )
        self.assertEqual(
            set(payload), {"schema_version", "mode", "catalog_sha256", "tokenizer", "items"}
        )
        self.assertEqual(len(payload["items"]), 399)
        prompt_ids = []
        for item in payload["items"]:
            self.assertEqual(set(item), {"prompt_id", "messages"})
            self.assertEqual([message["role"] for message in item["messages"]], ["system", "user"])
            message_bytes = json.dumps(
                item["messages"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            prompt_id = hashlib.sha256(message_bytes).hexdigest()
            self.assertEqual(item["prompt_id"], prompt_id)
            prompt_ids.append(prompt_id)
        self.assertEqual(prompt_ids, sorted(set(prompt_ids)))

    def test_fixture_digest_mismatch_stops_before_catalog_parsing(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner
        from experiments import prepare_qa_contrast

        class ChangedFixture:
            @staticmethod
            def read_bytes() -> bytes:
                return b"changed fixture bytes"

        with (
            patch.object(runner, "_FIXTURE_PATH", ChangedFixture()),
            patch.object(prepare_qa_contrast, "prepare_contrast") as prepare,
            self.assertRaisesRegex(ValueError, "fixture differs from the reviewed fixture"),
        ):
            runner.build_payload()
        prepare.assert_not_called()

    def test_default_cli_is_offline_and_reports_the_fixed_payload(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        output = StringIO()
        original_import = builtins.__import__

        def reject_runtime_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "transformers", "tokenizers"}:
                raise AssertionError(f"offline CLI imported {name}")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=reject_runtime_import),
            redirect_stdout(output),
        ):
            exit_code = runner.main([])
        self.assertEqual(exit_code, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["item_count"], 399)
        self.assertEqual(
            summary["payload_sha256"],
            "7f3055ef98fec05f77676b3b995b27c6c6ebcfc2566fdd668177f065a3d9fcc2",
        )
        self.assertFalse(summary["remote_started"])
        self.assertRegex(summary["runner_sha256"], r"^[0-9a-f]{64}$")

    def test_launch_preflight_rejects_bad_approvals_python_and_reserved_paths(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        run_once = cast(Any, getattr(runner, "run_once", None))
        self.assertTrue(callable(run_once), "the approved one-shot launch path is missing")
        payload = runner.build_payload()
        payload_sha = _payload_hash(payload)
        runner_sha = _runner_hash(runner)
        changed = dict(payload)
        changed["mode"] = "caller-modified"

        with tempfile.TemporaryDirectory(prefix="qa-contrast-counter-preflight-") as directory:
            root = Path(directory)
            sentinel = root / "sentinel.json"
            sentinel.write_text("preserve target", encoding="utf-8")
            existing = root / "existing.json"
            existing.write_text("preserve output", encoding="utf-8")
            symlink = root / "symlink.json"
            symlink.symlink_to(sentinel)
            raw_existing_output = root / "raw-existing.json"
            raw_existing = Path(f"{raw_existing_output}.remote.json")
            raw_existing.write_text("preserve raw", encoding="utf-8")
            raw_symlink_output = root / "raw-symlink.json"
            raw_symlink = Path(f"{raw_symlink_output}.remote.json")
            raw_symlink.symlink_to(sentinel)

            def rejects(
                name: str,
                value: dict[str, Any],
                output: Path,
                reason: str,
                *,
                version: tuple[int, int] = (3, 12),
                error_type: type[Exception] = ValueError,
                payload_approval: str = payload_sha,
                runner_approval: str = runner_sha,
            ) -> None:
                with self.subTest(name=name):
                    with (
                        patch.object(
                            runner.sys,
                            "version_info",
                            type("Version", (), {"major": version[0], "minor": version[1]})(),
                        ),
                        patch.object(runner, "verify_personal_workspace") as workspace_check,
                        patch.object(runner, "_build_modal_function") as function_build,
                        self.assertRaisesRegex(error_type, reason),
                    ):
                        run_once(
                            value,
                            approved_payload_sha256=payload_approval,
                            approved_runner_sha256=runner_approval,
                            output_path=output,
                        )
                    workspace_check.assert_not_called()
                    function_build.assert_not_called()

            rejects(
                "python",
                payload,
                root / "python.json",
                "local Python 3.12",
                version=(3, 11),
                error_type=RuntimeError,
            )
            rejects("changed", changed, root / "changed.json", "caller payload")
            rejects(
                "payload approval",
                payload,
                root / "payload.json",
                "exact reviewed payload approval",
                payload_approval="0" * 64,
            )
            rejects(
                "runner approval",
                payload,
                root / "runner.json",
                "exact current runner approval",
                runner_approval="0" * 64,
            )
            rejects(
                "existing",
                payload,
                existing,
                "receipt output path must be new",
                error_type=FileExistsError,
            )
            rejects(
                "symlink",
                payload,
                symlink,
                "receipt output path must be new",
                error_type=FileExistsError,
            )
            rejects(
                "existing raw",
                payload,
                raw_existing_output,
                "raw remote output path must be new",
                error_type=FileExistsError,
            )
            rejects(
                "symlink raw",
                payload,
                raw_symlink_output,
                "raw remote output path must be new",
                error_type=FileExistsError,
            )
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve output")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve target")
            self.assertTrue(raw_existing_output.exists())
            self.assertEqual(raw_existing_output.stat().st_size, 0)
            self.assertEqual(raw_existing.read_text(encoding="utf-8"), "preserve raw")
            self.assertTrue(raw_symlink_output.exists())
            self.assertEqual(raw_symlink_output.stat().st_size, 0)
            self.assertTrue(raw_symlink.is_symlink())

    def test_source_free_remote_body_counts_all_prompts_with_pinned_runtime_options(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        remote_body = cast(Any, getattr(runner, "_remote_count", None))
        self.assertTrue(callable(remote_body), "the source-free remote tokenizer body is missing")
        function = types.FunctionType(remote_body.__code__, {"__name__": "__main__"})
        payload = runner.build_payload()
        request = {
            "payload": payload,
            "payload_sha256": _payload_hash(payload),
            "runner_sha256": "a" * 64,
        }
        events: list[tuple[str, Any]] = []
        template = "fake tokenizer chat template"
        fake_tokenizer = types.SimpleNamespace(
            is_fast=True, chat_template=template, backend_tokenizer=None, token_ids=[11, 22, 33]
        )

        def apply_chat_template(messages: object, **options: object) -> list[int]:
            events.append(("template", options))
            # These fixture integers are fake token IDs for the isolated test double.
            return fake_tokenizer.token_ids

        fake_tokenizer.apply_chat_template = apply_chat_template

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str, **options: object) -> object:
                events.append(("load", (model_id, options)))
                return fake_tokenizer

        transformer_stub = types.ModuleType("transformers")
        transformer_stub.AutoTokenizer = AutoTokenizer
        versions = {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}
        real_sha256 = hashlib.sha256

        class TemplateDigest:
            @staticmethod
            def hexdigest() -> str:
                return "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"

        def fake_sha256(raw: bytes = b"", *args: object, **kwargs: object) -> Any:
            if raw == template.encode("utf-8"):
                return TemplateDigest()
            return real_sha256(raw, *args, **kwargs)

        with (
            patch.dict(sys.modules, {"transformers": transformer_stub}),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
            patch("hashlib.sha256", side_effect=fake_sha256),
            patch("platform.python_version", return_value="3.12.10"),
        ):
            result = function(request)
            self.assertEqual(result["tokenizer_provenance"]["package_versions"], versions)
            bad_request = {**request, "payload_sha256": "0" * 64}
            events.clear()
            with self.assertRaisesRegex(ValueError, "not the reviewed payload hash"):
                function(bad_request)
            self.assertEqual(events, [])

            guards = (
                ("slow", "not a fast tokenizer", lambda: setattr(fake_tokenizer, "is_fast", False)),
                (
                    "template",
                    "template hash differs",
                    lambda: setattr(fake_tokenizer, "chat_template", "changed template"),
                ),
                (
                    "bool IDs",
                    "invalid token IDs",
                    lambda: setattr(fake_tokenizer, "token_ids", [True]),
                ),
                (
                    "negative IDs",
                    "invalid token IDs",
                    lambda: setattr(fake_tokenizer, "token_ids", [-1]),
                ),
                (
                    "tuple IDs",
                    "invalid token IDs",
                    lambda: setattr(fake_tokenizer, "token_ids", (1, 2)),
                ),
            )
            for name, reason, mutate in guards:
                with self.subTest(name=name):
                    fake_tokenizer.is_fast = True
                    fake_tokenizer.chat_template = template
                    fake_tokenizer.token_ids = [11, 22, 33]
                    mutate()
                    events.clear()
                    with self.assertRaisesRegex(ValueError, reason):
                        function(request)
                    self.assertEqual([event[0] for event in events].count("load"), 1)
            fake_tokenizer.is_fast = True
            fake_tokenizer.chat_template = template
            fake_tokenizer.token_ids = [11, 22, 33]
            versions["transformers"] = "4.52.0"
            events.clear()
            with self.assertRaisesRegex(ValueError, "package versions differ"):
                function(request)
            self.assertEqual([event[0] for event in events].count("load"), 1)

        self.assertEqual(function.__globals__, {"__name__": "__main__"})
        self.assertIsNone(function.__closure__)
        self.assertEqual([event[0] for event in events].count("load"), 1)
        self.assertEqual([event[0] for event in events].count("template"), 399)
        self.assertEqual(
            events[0],
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
            ),
        )
        expected_options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.assertEqual(events[1], ("template", expected_options))
        self.assertEqual(len(result["count_receipt"]["counts"]), 399)
        self.assertEqual(
            result["count_receipt"]["counts"],
            [{"prompt_id": item["prompt_id"], "prompt_tokens": 3} for item in payload["items"]],
        )
        provenance = result["tokenizer_provenance"]
        self.assertEqual(provenance["template_sha256"], runner._TOKENIZER["template_sha256"])
        self.assertTrue(provenance["python_version"].startswith("3.12."))
        self.assertEqual(
            provenance["package_versions"],
            {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"},
        )

    def test_result_validator_requires_exact_coverage_and_reviewed_provenance(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        validate = cast(Any, getattr(runner, "_validate_remote_result", None))
        self.assertTrue(callable(validate), "the strict remote result validator is missing")
        payload = runner.build_payload()
        runner_sha = "a" * 64
        result = _fake_result(payload, runner_sha)
        self.assertIs(validate(result, payload, runner_sha), result)
        expected_ids = {item["prompt_id"] for item in payload["items"]}
        self.assertEqual(
            {row["prompt_id"] for row in result["count_receipt"]["counts"]}, expected_ids
        )

        def duplicate(row: dict[str, Any]) -> None:
            row["count_receipt"]["counts"][1]["prompt_id"] = row["count_receipt"]["counts"][0][
                "prompt_id"
            ]

        def missing(row: dict[str, Any]) -> None:
            row["count_receipt"]["counts"].pop()

        def unexpected(row: dict[str, Any]) -> None:
            row["count_receipt"]["counts"][0]["prompt_id"] = "f" * 64

        def bool_count(row: dict[str, Any]) -> None:
            row["count_receipt"]["counts"][0]["prompt_tokens"] = True

        def negative_count(row: dict[str, Any]) -> None:
            row["count_receipt"]["counts"][0]["prompt_tokens"] = -1

        cases = (
            ("duplicate", duplicate, "unknown or invalid prompt count"),
            ("missing", missing, "does not cover every reviewed prompt"),
            ("unexpected", unexpected, "unknown or invalid prompt count"),
            ("bool_count", bool_count, "unknown or invalid prompt count"),
            ("negative_count", negative_count, "unknown or invalid prompt count"),
            (
                "catalog",
                lambda row: row["count_receipt"].__setitem__("catalog_sha256", "0" * 64),
                "receipt provenance is invalid",
            ),
            (
                "runner",
                lambda row: row["tokenizer_provenance"].__setitem__("runner_sha256", "b" * 64),
                "provenance differs from the approved configuration",
            ),
            (
                "payload",
                lambda row: row["tokenizer_provenance"].__setitem__("payload_sha256", "b" * 64),
                "provenance differs from the approved configuration",
            ),
            (
                "python",
                lambda row: row["tokenizer_provenance"].__setitem__("python_version", "3.13.0"),
                "provenance differs from the approved configuration",
            ),
            (
                "template",
                lambda row: row["tokenizer_provenance"].__setitem__("template_sha256", "b" * 64),
                "provenance differs from the approved configuration",
            ),
            (
                "settings",
                lambda row: row["tokenizer_provenance"]["template_options"].__setitem__(
                    "enable_thinking", True
                ),
                "provenance differs from the approved configuration",
            ),
            (
                "nonfinite",
                lambda row: row["tokenizer_provenance"].__setitem__(
                    "elapsed_seconds", float("nan")
                ),
                "finite and nonnegative",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                altered = copy.deepcopy(result)
                mutate(altered)
                with self.assertRaisesRegex(ValueError, reason):
                    validate(altered, payload, runner_sha)

    def test_one_shot_launch_preserves_private_success_and_rejected_raw_results(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        payload = runner.build_payload()
        payload_sha = runner._EXPECTED_PAYLOAD_SHA256
        runner_sha = _runner_hash(runner)
        expected_ids = [item["prompt_id"] for item in payload["items"]]
        events: list[str] = []

        class Session:
            def __enter__(self) -> None:
                events.append("app-enter")

            def __exit__(self, *args: object) -> bool:
                events.append("app-exit")
                return False

        class App:
            def run(self, *, detach: bool) -> Session:
                events.append(f"app-run-detach-{detach}")
                return Session()

        class Function:
            response: Any = None
            calls = 0

            def remote(self, request: dict[str, Any]) -> Any:
                self.calls += 1
                events.append("remote")
                if isinstance(self.response, Exception):
                    raise self.response
                return self.response

        app_to_use, function = App(), Function()
        real_fsync = os.fsync

        def track_fsync(descriptor: int) -> None:
            events.append("fsync")
            real_fsync(descriptor)

        with tempfile.TemporaryDirectory(prefix="qa-contrast-counter-one-shot-") as directory:
            root = Path(directory)

            def launch(output: Path) -> dict[str, Any]:
                def check_workspace() -> None:
                    self.assertTrue(output.is_file())
                    self.assertEqual(output.stat().st_size, 0)
                    self.assertTrue(Path(f"{output}.remote.json").is_file())
                    events.append("workspace")

                def build() -> tuple[App, Function]:
                    self.assertTrue(output.is_file())
                    self.assertTrue(Path(f"{output}.remote.json").is_file())
                    events.append("build")
                    return app_to_use, function

                with (
                    patch.object(
                        runner.sys, "version_info", type("Version", (), {"major": 3, "minor": 12})()
                    ),
                    patch.object(runner, "build_payload", return_value=payload),
                    patch.object(runner, "verify_personal_workspace", side_effect=check_workspace),
                    patch.object(runner, "_build_modal_function", side_effect=build),
                    patch.object(runner.os, "fsync", side_effect=track_fsync),
                ):
                    return runner.run_once(
                        payload,
                        approved_payload_sha256=payload_sha,
                        approved_runner_sha256=runner_sha,
                        output_path=output,
                    )

            success_path = root / "success.json"
            good = _fake_result(payload, runner_sha)
            function.response = good
            real_validate = runner._validate_remote_result

            def inspect_raw_before_validation(
                result: dict[str, Any], approved_payload: dict[str, Any], approved_runner: str
            ) -> dict[str, Any]:
                raw_path = Path(f"{success_path}.remote.json")
                self.assertEqual(json.loads(raw_path.read_text(encoding="utf-8")), result)
                self.assertEqual(stat.S_IMODE(raw_path.stat().st_mode), 0o600)
                events.append("validate")
                return real_validate(result, approved_payload, approved_runner)

            with patch.object(
                runner, "_validate_remote_result", side_effect=inspect_raw_before_validation
            ):
                returned = launch(success_path)
            success_receipt = json.loads(success_path.read_text(encoding="utf-8"))
            success_raw = Path(f"{success_path}.remote.json")
            self.assertEqual(returned["result"], good)
            self.assertEqual(success_receipt["result"], good)
            self.assertEqual(json.loads(success_raw.read_text(encoding="utf-8")), good)
            self.assertEqual(stat.S_IMODE(success_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(success_raw.stat().st_mode), 0o600)
            self.assertEqual(
                events,
                [
                    "workspace",
                    "build",
                    "app-run-detach-False",
                    "app-enter",
                    "remote",
                    "fsync",
                    "app-exit",
                    "validate",
                    "fsync",
                ],
            )
            self.assertEqual(function.calls, 1)

            events.clear()
            invalid_path = root / "invalid.json"
            invalid = _fake_result(payload, runner_sha, expected_ids[:-1])
            function.response = invalid
            with self.assertRaisesRegex(ValueError, "does not cover every reviewed prompt"):
                launch(invalid_path)
            invalid_raw = Path(f"{invalid_path}.remote.json")
            invalid_receipt = json.loads(invalid_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(invalid_raw.read_text(encoding="utf-8")), invalid)
            self.assertEqual(invalid_receipt["error"]["type"], "ValueError")
            self.assertIn(
                "does not cover every reviewed prompt", invalid_receipt["error"]["message"]
            )
            self.assertEqual(
                events[:5], ["workspace", "build", "app-run-detach-False", "app-enter", "remote"]
            )
            self.assertEqual(function.calls, 2)

            events.clear()
            failed_path = root / "failure.json"
            function.response = RuntimeError("Authorization: Bearer qa-counter-test-secret")
            with self.assertRaisesRegex(RuntimeError, "qa-counter-test-secret"):
                launch(failed_path)
            failed_receipt = json.loads(failed_path.read_text(encoding="utf-8"))
            failed_raw = Path(f"{failed_path}.remote.json")
            self.assertIn("[REDACTED]", failed_receipt["error"]["message"])
            self.assertNotIn("qa-counter-test-secret", failed_path.read_text(encoding="utf-8"))
            self.assertEqual(failed_raw.read_text(encoding="utf-8"), "")
            self.assertEqual(function.calls, 3)

            class ExitFailureSession(Session):
                def __exit__(self, *args: object) -> bool:
                    events.append("app-exit")
                    raise RuntimeError("Authorization: Bearer teardown-test-secret")

            class ExitFailureApp(App):
                def run(self, *, detach: bool) -> ExitFailureSession:
                    events.append(f"app-run-detach-{detach}")
                    return ExitFailureSession()

            events.clear()
            teardown_path = root / "teardown.json"
            function.response = good
            app_to_use = ExitFailureApp()
            with self.assertRaisesRegex(RuntimeError, "teardown-test-secret"):
                launch(teardown_path)
            teardown_raw = Path(f"{teardown_path}.remote.json")
            teardown_receipt = json.loads(teardown_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(teardown_raw.read_text(encoding="utf-8")), good)
            self.assertEqual(teardown_receipt["error"]["type"], "RuntimeError")
            self.assertIn("[REDACTED]", teardown_receipt["error"]["message"])
            self.assertNotIn("teardown-test-secret", teardown_path.read_text(encoding="utf-8"))
            self.assertEqual(
                events,
                [
                    "workspace",
                    "build",
                    "app-run-detach-False",
                    "app-enter",
                    "remote",
                    "fsync",
                    "app-exit",
                    "fsync",
                ],
            )
            self.assertEqual(function.calls, 4)

    def test_modal_builder_registers_only_isolated_cpu_function_without_uploads(self) -> None:
        from experiments import modal_qa_contrast_token_count as runner

        builder = cast(Any, getattr(runner, "_build_modal_function", None))
        options_builder = cast(Any, getattr(runner, "modal_function_options", None))
        self.assertTrue(callable(builder), "the isolated Modal function builder is missing")
        self.assertTrue(callable(options_builder), "the frozen Modal options are missing")
        registrations: list[tuple[Any, dict[str, Any]]] = []
        events: list[tuple[str, Any]] = []

        class App:
            def __init__(self, name: str) -> None:
                events.append(("app", name))

            def function(self, **options: Any) -> Any:
                def register(function: Any) -> Any:
                    registrations.append((function, options))
                    return function

                return register

        class Image:
            def pip_install(self, *packages: str) -> str:
                events.append(("packages", packages))
                return "fake-image"

        class ImageFactory:
            @staticmethod
            def debian_slim(**options: Any) -> Image:
                events.append(("image", options))
                return Image()

        modal_stub = types.ModuleType("modal")
        modal_stub.App = App
        modal_stub.Image = ImageFactory
        with patch.dict(sys.modules, {"modal": modal_stub}):
            app, function = builder()

        self.assertIsNotNone(app)
        self.assertEqual(events[0], ("image", {"python_version": "3.12"}))
        self.assertEqual(
            events[1],
            (
                "packages",
                (
                    "transformers==4.51.3",
                    "tokenizers==0.21.1",
                    "jinja2==3.1.6",
                    "huggingface-hub==0.30.2",
                    "safetensors==0.5.3",
                ),
            ),
        )
        self.assertEqual(events[2], ("app", "tracemem-qa-contrast-token-count"))
        self.assertEqual(len(registrations), 1)
        self.assertIs(registrations[0][0], function)
        self.assertEqual(function.__globals__, {"__name__": "__main__"})
        self.assertIsNone(function.__closure__)
        options = registrations[0][1]
        self.assertEqual(options.pop("image"), "fake-image")
        self.assertEqual(
            options,
            {
                "cpu": (1, 2),
                "memory": (2048, 4096),
                "max_containers": 1,
                "min_containers": 0,
                "retries": 0,
                "startup_timeout": 300,
                "timeout": 600,
                "scaledown_window": 2,
                "single_use_containers": True,
                "serialized": True,
                "include_source": False,
            },
        )
        self.assertEqual(options, options_builder())
        self.assertNotIn("gpu", options)
        self.assertNotIn("mounts", options)
        self.assertNotIn("secrets", options)
        self.assertNotIn("env", options)
