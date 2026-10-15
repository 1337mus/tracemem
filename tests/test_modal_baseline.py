"""Local payload and launch guards for the synthetic-only Modal runner."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from experiments import modal_baseline
from tracemem.benchmark import load_synthetic_data
from tracemem.remote_vectors import text_sha256


class ModalPayloadTests(unittest.TestCase):
    def test_cross_session_embedding_payload_is_allowlisted_and_default_unchanged(self) -> None:
        original = modal_baseline.build_payload("embeddings")
        self.assertEqual(
            modal_baseline.payload_sha256(original),
            "0f402a3474d7c94e979553a2623f58ef6fd72bc072218feebe872ebe10b7ae53",
        )

        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        raw = Path("examples/synthetic/cross-session-test-v1.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf",
        )
        questions = parse_longmemeval_bytes(raw, namespace="cross-session-test-v1", variant="s")
        expected_texts = {record.text for question in questions for record in question.records}
        expected_texts.update(question.question for question in questions)
        expected_hashes = {text_sha256(text) for text in expected_texts}

        payload = modal_baseline.build_payload("embeddings", fixture="cross-session-test-v1")

        self.assertEqual(payload["input_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            modal_baseline.payload_sha256(payload),
            "95158af854e806eb14c19ee4f3635446c912d5a1efddd7933f7e639f5f43aada",
        )
        self.assertEqual(len(expected_hashes), 78)
        self.assertEqual(len(payload["items"]), 78)
        self.assertEqual({item["text_sha256"] for item in payload["items"]}, expected_hashes)
        self.assertEqual(
            [item["text_sha256"] for item in payload["items"]],
            sorted(expected_hashes),
        )
        self.assertTrue(all(set(item) == {"text_sha256", "text"} for item in payload["items"]))
        self.assertNotIn('"answer"', json.dumps(payload))

    def test_cross_session_fixture_rejects_unknown_fixture_and_extraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixture must be original"):
            modal_baseline.build_payload("embeddings", fixture="unreviewed-fixture")
        with self.assertRaisesRegex(ValueError, "available only for embeddings"):
            modal_baseline.build_payload("extraction", fixture="cross-session-test-v1")

    def test_cross_session_fixture_hash_is_checked_before_parsing(self) -> None:
        from tracemem.adapters import longmemeval

        fixture = Path("examples/synthetic/cross-session-test-v1.json").resolve()
        original_read_bytes = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            if path.resolve() == fixture:
                return original_read_bytes(path) + b"\n"
            return original_read_bytes(path)

        with (
            patch.object(Path, "read_bytes", read_bytes),
            patch.object(longmemeval, "parse_longmemeval_bytes") as parse,
        ):
            with self.assertRaisesRegex(ValueError, "pinned input"):
                modal_baseline.build_payload("embeddings", fixture="cross-session-test-v1")
        parse.assert_not_called()

    def test_cross_session_fixture_cli_prepares_without_launch(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = modal_baseline.main(
                ["--mode", "embeddings", "--fixture", "cross-session-test-v1"]
            )

        prepared = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(prepared["item_count"], 78)
        self.assertEqual(
            prepared["input_sha256"],
            "dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf",
        )
        self.assertFalse(prepared["remote_started"])

    def test_embedding_payload_contains_only_packaged_corpus_and_query_text(self) -> None:
        payload = modal_baseline.build_payload("embeddings")
        records, queries, _ = load_synthetic_data()
        expected_text_hashes = {
            text_sha256(text)
            for text in [*(record.text for record in records), *(q["query"] for q in queries)]
        }

        self.assertEqual(set(payload), {"schema_version", "mode", "input_sha256", "model", "items"})
        self.assertEqual(payload["mode"], "embeddings")
        self.assertEqual({item["text_sha256"] for item in payload["items"]}, expected_text_hashes)
        for item in payload["items"]:
            self.assertEqual(set(item), {"text_sha256", "text"})
            self.assertEqual(item["text_sha256"], text_sha256(item["text"]))

    def test_extraction_payload_uses_rendered_prompt_without_expected_labels(self) -> None:
        case = {
            "case_id": "fictional-case-1",
            "scenario_id": "fictional-scenario-1",
            "split": "diagnostic",
            "tags": ["direct"],
            "messages": [
                {
                    "session_id": "synthetic-session",
                    "message_id": "synthetic-message",
                    "role": "user",
                    "text": "I prefer fictional tea.",
                    "recorded_at": "2026-10-01T12:00:00Z",
                    "content_kind": "mixed_untrusted",
                }
            ],
            "expected": [{"predicate": "PREFERS", "object": "SECRET_EXPECTED_LABEL"}],
        }
        challenge = ModuleType("tracemem.challenge")
        challenge.load_challenge = lambda: ([case], "a" * 64)  # type: ignore[attr-defined]
        seen: list[dict[str, object]] = []

        def render_prompt(prompt_case: dict[str, object]) -> str:
            seen.append(prompt_case)
            return "formatted synthetic prompt"

        challenge.render_prompt = render_prompt  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"tracemem.challenge": challenge}):
            payload = modal_baseline.build_payload("extraction", max_cases=1)

        self.assertEqual(len(seen), 1)
        self.assertNotIn("expected", seen[0])
        self.assertNotIn("tags", seen[0])
        self.assertEqual(
            payload["items"],
            [{"case_id": "fictional-case-1", "prompt": "formatted synthetic prompt"}],
        )
        self.assertNotIn("SECRET_EXPECTED_LABEL", json.dumps(payload))

    def test_explicit_prompt_version_changes_only_rendered_prompts(self) -> None:
        default_payload = modal_baseline.build_payload("extraction")
        v1_payload = modal_baseline.build_payload("extraction", prompt_version="v1")
        v2_payload = modal_baseline.build_payload("extraction", prompt_version="explicit-v2")

        self.assertEqual(default_payload, v1_payload)
        self.assertEqual(
            modal_baseline.payload_sha256(v1_payload),
            "70ec4bb633161983506cfb08e87d552e355e9c26c698a9b7528a94c628b26f0b",
        )
        self.assertEqual(
            modal_baseline.payload_sha256(v2_payload),
            "ee0d99b5c14e98279a7c598c8d25b5f72baf4a8b137dbb03fb1b03bebd4d47e1",
        )
        self.assertEqual(set(v1_payload), set(v2_payload))
        self.assertEqual(v1_payload["input_sha256"], v2_payload["input_sha256"])
        self.assertEqual(v1_payload["model"], v2_payload["model"])
        self.assertEqual(
            [item["case_id"] for item in v1_payload["items"]],
            [item["case_id"] for item in v2_payload["items"]],
        )
        self.assertTrue(
            all(
                first["prompt"] != second["prompt"]
                for first, second in zip(v1_payload["items"], v2_payload["items"], strict=True)
            )
        )
        self.assertIn("speaker_role is the outer role", v2_payload["items"][0]["prompt"])
        with self.assertRaisesRegex(ValueError, "only for extraction"):
            modal_baseline.build_payload("embeddings", prompt_version="explicit-v2")

    def test_runner_cli_prepares_version_and_source_hash_without_launching(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = modal_baseline.main(
                ["--mode", "extraction", "--prompt-version", "explicit-v2"]
            )

        prepared = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(prepared["prompt_version"], "explicit-v2")
        self.assertEqual(
            prepared["runner_sha256"],
            hashlib.sha256(Path(modal_baseline.__file__).read_bytes()).hexdigest(),
        )
        self.assertFalse(prepared["remote_started"])

    def test_extraction_case_limit_cannot_exceed_twenty_four(self) -> None:
        with self.assertRaisesRegex(ValueError, "24"):
            modal_baseline.build_payload("extraction", max_cases=25)

    def test_payload_digest_is_stable_and_binds_payload_contents(self) -> None:
        payload = {"mode": "embeddings", "items": [{"text": "fictional"}]}
        original = modal_baseline.payload_sha256(payload)
        same_content = modal_baseline.payload_sha256(
            {"items": [{"text": "fictional"}], "mode": "embeddings"}
        )
        changed_content = modal_baseline.payload_sha256(
            {"mode": "embeddings", "items": [{"text": "different"}]}
        )

        self.assertEqual(original, same_content)
        self.assertEqual(len(original), 64)
        self.assertNotEqual(original, changed_content)
        self.assertEqual(
            original,
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )


class ModalLaunchGuardTests(unittest.TestCase):
    def test_runtime_error_message_keeps_public_context_and_redacts_credentials(self) -> None:
        error = RuntimeError(
            "Modal request failed\nAuthorization: Bearer modal-secret-value "
            "api_key=private-key-value token=private-token-value"
        )

        message = modal_baseline._safe_exception_message(error)

        self.assertIn("Modal request failed", message)
        self.assertNotIn("modal-secret-value", message)
        self.assertNotIn("private-key-value", message)
        self.assertNotIn("private-token-value", message)
        self.assertNotIn("\n", message)

    def test_workspace_check_finds_modal_cli_beside_venv_interpreter_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            venv_bin = root / ".venv" / "bin"
            bundled_bin = root / "bundled" / "bin"
            venv_bin.mkdir(parents=True)
            bundled_bin.mkdir(parents=True)
            interpreter_target = bundled_bin / "python"
            interpreter_target.write_text("synthetic interpreter", encoding="utf-8")
            interpreter_link = venv_bin / "python"
            interpreter_link.symlink_to(interpreter_target)
            modal_cli = venv_bin / "modal"
            modal_cli.write_text("synthetic modal cli", encoding="utf-8")

            with (
                patch.object(modal_baseline.sys, "executable", str(interpreter_link)),
                patch.object(modal_baseline.sys, "prefix", str(root / ".venv")),
                patch.object(
                    modal_baseline.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="Workspace: synthetic-workspace\n",
                    ),
                ) as run,
            ):
                workspace = modal_baseline._active_workspace()

        self.assertEqual(workspace, "synthetic-workspace")
        self.assertEqual(run.call_args.args[0][0], str(modal_cli))

    def test_prepare_is_default_and_cli_has_no_arbitrary_input_path(self) -> None:
        parser = modal_baseline.build_parser()
        args = parser.parse_args(["--mode", "embeddings"])
        self.assertFalse(args.launch)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--mode", "embeddings", "--input", "/private/transcript.jsonl"])

    def test_modal_function_options_pin_source_free_single_container_execution(self) -> None:
        options = modal_baseline.modal_function_options()

        self.assertEqual(options["gpu"], "T4")
        self.assertEqual(options["cpu"], (2, 2))
        self.assertEqual(options["memory"], (16384, 16384))
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["buffer_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertLessEqual(options["startup_timeout"], 300)
        self.assertLessEqual(options["timeout"], 900)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_workspace_mismatch_blocks_launch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "workspace"):
            modal_baseline.verify_workspace("rajath-61258", "another-workspace")


if __name__ == "__main__":
    unittest.main()
