import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import tracemem.adapters as adapters


class TranscriptAdapterTests(unittest.TestCase):
    def test_claude_multipart_text_and_tool_output_remain_exact(self) -> None:
        read_transcript = getattr(adapters, "read_transcript", None)
        self.assertIsNotNone(read_transcript, "read_transcript is part of the adapter API")
        stats_type = getattr(adapters, "AdapterStats", None)
        self.assertIsNotNone(stats_type, "AdapterStats is part of the adapter API")
        user_text = "A Unicode block: café 🧪"
        assistant_text = "Exact spacing stays.\nSecond line."
        events = [
            {
                "type": "user",
                "sessionId": "session-claude-1",
                "timestamp": "2026-10-13T10:00:00Z",
                "message": {
                    "id": "user-message-1",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "stdout\n✓"}],
                        },
                    ],
                },
            },
            {
                "type": "assistant",
                "sessionId": "session-claude-1",
                "message": {
                    "id": "assistant-message-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "claude.jsonl"
            source_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            stats = stats_type()
            records = list(read_transcript(source_path, "claude", stats=stats))

        self.assertEqual(
            [record.text for record in records], [user_text, "stdout\n✓", assistant_text]
        )
        self.assertEqual([record.role for record in records], ["user", "tool", "assistant"])
        self.assertEqual(
            [record.content_kind for record in records],
            ["mixed_untrusted", "tool_output", "conversation"],
        )
        self.assertEqual(
            [record.message_id for record in records],
            [
                "user-message-1",
                "user-message-1",
                "assistant-message-1",
            ],
        )
        self.assertEqual(stats.role_counts, {"user": 1, "tool": 1, "assistant": 1})
        self.assertEqual(records[0].line_number, 1)
        self.assertNotIn(str(source_path), records[0].source_locator or "")

    def test_codex_messages_and_function_outputs_skip_events_and_partial_tail(self) -> None:
        read_transcript = getattr(adapters, "read_transcript", None)
        self.assertIsNotNone(read_transcript, "read_transcript is part of the adapter API")
        stats_type = getattr(adapters, "AdapterStats", None)
        self.assertIsNotNone(stats_type, "AdapterStats is part of the adapter API")
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "codex.jsonl"
            events = [
                {"type": "session_meta", "payload": {"id": "codex-session-1"}},
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "ignore duplicate event text"},
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-10-13T10:01:00Z",
                    "payload": {
                        "type": "message",
                        "id": "codex-user-1",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Question 🪐"}],
                    },
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-10-13T10:01:01Z",
                    "payload": {
                        "type": "message",
                        "id": "codex-assistant-1",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Answer"}],
                    },
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-10-13T10:01:02Z",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "tool result",
                    },
                },
            ]
            source_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
                + '{"type":"response_item"',
                encoding="utf-8",
            )
            stats = stats_type()

            records = list(read_transcript(source_path, "codex", stats=stats))

        self.assertEqual(
            [record.text for record in records], ["Question 🪐", "Answer", "tool result"]
        )
        self.assertEqual([record.role for record in records], ["user", "assistant", "tool"])
        self.assertEqual(
            [record.session_id for record in records],
            ["codex-session-1", "codex-session-1", "codex-session-1"],
        )
        self.assertEqual(
            [record.message_id for record in records],
            ["codex-user-1", "codex-assistant-1", "call-1"],
        )
        self.assertEqual(stats.malformed_records, 1)
        self.assertEqual(stats.unsupported_records, 1)
        self.assertEqual(stats.role_counts, {"user": 1, "assistant": 1, "tool": 1})

    def test_codex_tool_outputs_keep_only_verbatim_strings(self) -> None:
        events = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "string-call",
                    "output": "exact tool output\n✓",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "custom-call",
                    "output": "custom tool output",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "structured-call",
                    "output": {"text": "derived JSON must not become evidence"},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "codex.jsonl"
            source_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            stats = adapters.AdapterStats()
            records = list(adapters.read_transcript(source_path, "codex", stats=stats))

        self.assertEqual(
            [record.text for record in records],
            ["exact tool output\n✓", "custom tool output"],
        )
        self.assertEqual([record.message_id for record in records], ["string-call", "custom-call"])
        self.assertEqual(stats.unsupported_records, 1)

    def test_record_ids_keep_original_positions_when_prior_blocks_become_supported(self) -> None:
        cases = [
            (
                "claude",
                {
                    "type": "assistant",
                    "message": {
                        "id": "stable-message",
                        "role": "assistant",
                        "content": [
                            {"type": "image", "source": {"type": "base64"}},
                            {"type": "text", "text": "known text"},
                        ],
                    },
                },
                lambda event: event["message"]["content"].__setitem__(
                    0, {"type": "text", "text": "newly supported text"}
                ),
            ),
            (
                "codex",
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "stable-message",
                        "role": "assistant",
                        "content": [
                            {"type": "image", "data": "synthetic"},
                            {"type": "output_text", "text": "known text"},
                        ],
                    },
                },
                lambda event: event["payload"]["content"].__setitem__(
                    0, {"type": "output_text", "text": "newly supported text"}
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "transcript.jsonl"
            for provider, event, support_first_block in cases:
                source_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
                before = next(
                    record
                    for record in adapters.read_transcript(source_path, provider)
                    if record.text == "known text"
                )

                support_first_block(event)
                source_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
                after = next(
                    record
                    for record in adapters.read_transcript(source_path, provider)
                    if record.text == "known text"
                )

                self.assertEqual(after.part_index, before.part_index, provider)
                self.assertEqual(after.record_id, before.record_id, provider)

    def test_claude_tool_result_ids_keep_original_nested_text_positions(self) -> None:
        event = {
            "type": "user",
            "message": {
                "id": "stable-tool-message",
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "image", "source": {"type": "base64"}},
                            {"type": "text", "text": "known tool text"},
                        ],
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "claude.jsonl"
            source_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            before = next(
                record
                for record in adapters.read_transcript(source_path, "claude")
                if record.text == "known tool text"
            )

            event["message"]["content"][0]["content"][0] = {
                "type": "text",
                "text": "newly supported tool text",
            }
            source_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            after = next(
                record
                for record in adapters.read_transcript(source_path, "claude")
                if record.text == "known tool text"
            )

        self.assertEqual(after.part_index, before.part_index)
        self.assertEqual(after.record_id, before.record_id)

    def test_claude_conflicting_role_is_unknown_and_untrusted(self) -> None:
        events = [
            {
                "type": "assistant",
                "message": {
                    "id": "conflicting-role",
                    "role": "user",
                    "content": [{"type": "text", "text": "conflicted attribution"}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "id": "fallback-role",
                    "content": [{"type": "text", "text": "outer role fallback"}],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "claude.jsonl"
            source_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            records = list(adapters.read_transcript(source_path, "claude"))

        self.assertEqual([record.role for record in records], ["unknown", "assistant"])
        self.assertEqual(
            [record.content_kind for record in records], ["mixed_untrusted", "conversation"]
        )

    def test_context_kit_discovery_uses_config_loader_and_transcript_files(self) -> None:
        config_module = ModuleType("ctxkit.config")
        transcripts_module = ModuleType("ctxkit.transcripts")
        config_object = object()
        configured_sources = [
            ("claude", Path("/synthetic/a.jsonl")),
            ("codex", Path("/synthetic/b.jsonl")),
        ]
        config_module.load_config = Mock(return_value=config_object)
        transcripts_module.transcript_files = Mock(return_value=configured_sources)
        package = ModuleType("ctxkit")
        package.__path__ = []
        package.config = config_module
        package.transcripts = transcripts_module
        config_path = Path("/synthetic/context-kit.toml")

        with patch.dict(
            sys.modules,
            {
                "ctxkit": package,
                "ctxkit.config": config_module,
                "ctxkit.transcripts": transcripts_module,
            },
        ):
            result = adapters.discover_context_kit(config_path)

        config_module.load_config.assert_called_once_with(config_path)
        transcripts_module.transcript_files.assert_called_once_with(config_object)
        self.assertEqual(
            result,
            (("claude", "/synthetic/a.jsonl"), ("codex", "/synthetic/b.jsonl")),
        )

    def test_context_kit_missing_dependency_fails_without_echoing_config_path(self) -> None:
        with patch.dict(sys.modules, {"ctxkit": None, "ctxkit.config": None}):
            with self.assertRaises(adapters.ContextKitConfigError) as raised:
                adapters.discover_context_kit("/private/synthetic/config.toml")

        self.assertNotIn("/private/synthetic/config.toml", str(raised.exception))

    @unittest.skipUnless(importlib.util.find_spec("ctxkit"), "optional Context Kit is unavailable")
    def test_context_kit_integration_returns_only_synthetic_source_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            claude_path = directory / "claude-a.jsonl"
            codex_path = directory / "codex-a.jsonl"
            claude_path.write_text("{}\n", encoding="utf-8")
            codex_path.write_text("{}\n", encoding="utf-8")
            config_path = directory / "config.toml"
            config_path.write_text(
                "[sources.claude]\n"
                f"enabled = true\nglob = {json.dumps(str(directory / 'claude-*.jsonl'))}\n"
                "[sources.codex]\n"
                f"enabled = true\nglob = {json.dumps(str(directory / 'codex-*.jsonl'))}\n",
                encoding="utf-8",
            )

            sources = adapters.discover_context_kit(config_path)

        self.assertEqual(
            sources,
            (("claude", str(claude_path)), ("codex", str(codex_path))),
        )


if __name__ == "__main__":
    unittest.main()
