"""Tests for the explicit, read-only ChatGPT export adapter."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tracemem.adapters as adapters
from tracemem import citations
from tracemem.storage import EvidenceStore


class ChatGPTAdapterTests(unittest.TestCase):
    def _reader(self):
        module_spec = importlib.util.find_spec("tracemem.adapters.chatgpt")
        module = importlib.import_module("tracemem.adapters.chatgpt") if module_spec else None
        reader = getattr(module, "read_chatgpt_export", None)
        self.assertIsNotNone(
            reader,
            "read_chatgpt_export must be available from tracemem.adapters.chatgpt",
        )
        return reader

    def _write_export(self, directory: str, export: object) -> Path:
        source_path = Path(directory) / "export.json"
        source_path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
        return source_path

    def test_reads_root_to_current_node_branch(self) -> None:
        export = [
            {
                "id": "conversation-1",
                "current_node": "answer-node",
                "mapping": {
                    "question-node": {
                        "parent": None,
                        "message": {
                            "id": "question-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Question"]},
                        },
                    },
                    "answer-node": {
                        "parent": "question-node",
                        "message": {
                            "id": "answer-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Answer"]},
                        },
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "export.json"
            source_path.write_text(json.dumps(export), encoding="utf-8")
            module_spec = importlib.util.find_spec("tracemem.adapters.chatgpt")
            module = importlib.import_module("tracemem.adapters.chatgpt") if module_spec else None
            reader = getattr(module, "read_chatgpt_export", None)
            self.assertIsNotNone(
                reader,
                "read_chatgpt_export must be available from tracemem.adapters.chatgpt",
            )

            records = list(reader(source_path))

        self.assertEqual([record.text for record in records], ["Question", "Answer"])
        self.assertEqual(
            [record.message_id for record in records], ["question-message", "answer-message"]
        )
        self.assertEqual([record.role for record in records], ["user", "assistant"])

    def test_ignores_sibling_and_uses_parent_links_in_reverse_mapping_order(self) -> None:
        def message(message_id: str, role: str, text: str) -> dict[str, object]:
            return {
                "id": message_id,
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text]},
            }

        export = [
            {
                "id": "branched-conversation",
                "current_node": "leaf",
                "mapping": {
                    "leaf": {
                        "parent": "chosen",
                        "message": message("leaf-id", "assistant", "Leaf"),
                    },
                    "sibling": {
                        "parent": "root",
                        "message": message("sibling-id", "assistant", "Sibling"),
                    },
                    "chosen": {"parent": "root", "message": message("chosen-id", "user", "Chosen")},
                    "root": {"parent": None, "message": message("root-id", "user", "Root")},
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            records = list(self._reader()(source_path))

        self.assertEqual([record.text for record in records], ["Root", "Chosen", "Leaf"])

    def test_conversations_keep_their_own_session_ids(self) -> None:
        export = [
            {
                "id": conversation_id,
                "current_node": "root",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": f"message-{conversation_id}",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": [conversation_id]},
                        },
                    }
                },
            }
            for conversation_id in ("conversation-a", "conversation-b")
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            records = list(self._reader()(source_path))

        self.assertEqual(
            [record.session_id for record in records], ["conversation-a", "conversation-b"]
        )

    def test_null_parent_marks_a_valid_root_node(self) -> None:
        export = [
            {
                "id": "root-conversation",
                "current_node": "only-node",
                "mapping": {
                    "only-node": {
                        "parent": None,
                        "message": {
                            "id": "root-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Root text"]},
                        },
                    }
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            records = list(self._reader()(source_path))

        self.assertEqual([record.text for record in records], ["Root text"])

    def test_malformed_parent_chain_is_skipped_as_a_whole_conversation(self) -> None:
        export = [
            {
                "id": "broken-chain",
                "current_node": "leaf",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "root-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["must not emit"]},
                        },
                    },
                    "leaf": {
                        "parent": "missing-parent",
                        "message": {
                            "id": "leaf-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["also skipped"]},
                        },
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            stats = adapters.AdapterStats()
            records = list(self._reader()(source_path, stats=stats))

        self.assertEqual(records, [])
        self.assertEqual(stats.malformed_records, 1)

    def test_repeated_message_ids_reject_the_selected_conversation(self) -> None:
        export = [
            {
                "id": "duplicate-messages",
                "current_node": "leaf",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "same-id",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Root"]},
                        },
                    },
                    "leaf": {
                        "parent": "root",
                        "message": {
                            "id": "same-id",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Leaf"]},
                        },
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            stats = adapters.AdapterStats()
            records = list(self._reader()(source_path, stats=stats))

        self.assertEqual(records, [])
        self.assertEqual(stats.malformed_records, 1)

    def test_duplicate_conversation_ids_fail_before_any_record_is_yielded(self) -> None:
        conversation = {
            "id": "private-conversation-id",
            "current_node": "root",
            "mapping": {
                "root": {
                    "parent": None,
                    "message": {
                        "id": "message",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Text"]},
                    },
                }
            },
        }
        export = [conversation, conversation]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            reader = self._reader()
            with self.assertRaises(ValueError) as raised:
                next(reader(source_path))

        self.assertIn("duplicate", str(raised.exception).lower())
        self.assertNotIn("private-conversation-id", str(raised.exception))
        self.assertNotIn(str(source_path), str(raised.exception))

    def test_create_time_converts_finite_seconds_and_omits_invalid_values(self) -> None:
        def message(
            message_id: str, role: str, text: str, create_time: object
        ) -> dict[str, object]:
            return {
                "id": message_id,
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": create_time,
            }

        export = [
            {
                "id": "timestamp-conversation",
                "current_node": "out-of-range",
                "mapping": {
                    "fractional": {
                        "parent": None,
                        "message": message("fractional", "user", "Fractional", 1.25),
                    },
                    "boolean": {
                        "parent": "fractional",
                        "message": message("boolean", "assistant", "Boolean", True),
                    },
                    "out-of-range": {
                        "parent": "boolean",
                        "message": message("out-of-range", "assistant", "Large", 1e100),
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            records = list(self._reader()(source_path))

        self.assertEqual(
            [record.recorded_at for record in records],
            ["1970-01-01T00:00:01.250000Z", None, None],
        )

    def test_unsupported_parts_and_roles_keep_text_unserialized_and_untrusted(self) -> None:
        def message(message_id: str, role: str, content: dict[str, object]) -> dict[str, object]:
            return {"id": message_id, "author": {"role": role}, "content": content}

        entries = [
            (
                "user",
                None,
                message(
                    "user-id",
                    "user",
                    {"content_type": "text", "parts": [" café\n🧪 ", 7, {"image": "secret"}, ""]},
                ),
            ),
            (
                "unknown",
                "user",
                message(
                    "unknown-id", "future-role", {"content_type": "text", "parts": ["Unknown"]}
                ),
            ),
            (
                "tool",
                "unknown",
                message("tool-id", "tool", {"content_type": "text", "parts": ["Tool"]}),
            ),
            (
                "system",
                "tool",
                message("system-id", "system", {"content_type": "text", "parts": ["System"]}),
            ),
            (
                "developer",
                "system",
                message(
                    "developer-id", "developer", {"content_type": "text", "parts": ["Developer"]}
                ),
            ),
            (
                "image",
                "developer",
                message(
                    "image-id",
                    "assistant",
                    {"content_type": "image", "parts": [{"data": "secret"}]},
                ),
            ),
            ("null-message", "image", None),
        ]
        mapping: dict[str, object] = {}
        for node_id, parent, current_message in entries:
            mapping[node_id] = {"parent": parent, "message": current_message}
        export = [
            {"id": "content-conversation", "current_node": "null-message", "mapping": mapping}
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            stats = adapters.AdapterStats()
            records = list(self._reader()(source_path, stats=stats))

        self.assertEqual(
            [(record.text, record.part_index) for record in records],
            [
                (" café\n🧪 ", 0),
                ("", 3),
                ("Unknown", 0),
                ("Tool", 0),
                ("System", 0),
                ("Developer", 0),
            ],
        )
        self.assertEqual(
            [(record.role, record.content_kind) for record in records],
            [
                ("user", "mixed_untrusted"),
                ("user", "mixed_untrusted"),
                ("unknown", "mixed_untrusted"),
                ("tool", "tool_output"),
                ("system", "instruction"),
                ("developer", "instruction"),
            ],
        )
        self.assertEqual(stats.unsupported_records, 4)
        self.assertEqual(stats.malformed_records, 0)
        self.assertEqual(
            stats.role_counts,
            {"user": 2, "unknown": 1, "tool": 1, "system": 1, "developer": 1},
        )

    def test_malformed_message_shapes_reject_the_whole_conversation(self) -> None:
        invalid_messages = [
            "not-a-message-object",
            {
                "id": "bad-author",
                "author": "user",
                "content": {"content_type": "text", "parts": []},
            },
            {
                "id": "bad-role",
                "author": {"role": 5},
                "content": {"content_type": "text", "parts": []},
            },
            {"id": "missing-role", "author": {}, "content": {"content_type": "text", "parts": []}},
            {"id": "bad-content", "author": {"role": "user"}, "content": "not-an-object"},
            {"id": "bad-content-type", "author": {"role": "user"}, "content": {"content_type": 4}},
            {
                "id": "bad-parts",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": "not-a-list"},
            },
        ]

        for invalid_message in invalid_messages:
            with (
                self.subTest(invalid_message=invalid_message),
                tempfile.TemporaryDirectory() as directory,
            ):
                export = [
                    {
                        "id": "malformed-message-conversation",
                        "current_node": "invalid",
                        "mapping": {
                            "root": {
                                "parent": None,
                                "message": {
                                    "id": "valid-root",
                                    "author": {"role": "user"},
                                    "content": {"content_type": "text", "parts": ["must not emit"]},
                                },
                            },
                            "invalid": {"parent": "root", "message": invalid_message},
                        },
                    }
                ]
                source_path = self._write_export(directory, export)
                stats = adapters.AdapterStats()
                records = list(self._reader()(source_path, stats=stats))

                self.assertEqual(records, [])
                self.assertEqual(stats.malformed_records, 1)

    def test_oversize_input_is_rejected_before_json_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "oversize.json"
            with source_path.open("wb") as source_file:
                source_file.truncate(64 * 1024 * 1024 + 1)

            with patch("tracemem.adapters.chatgpt.json.loads") as decode:
                with self.assertRaisesRegex(ValueError, "64 MiB") as raised:
                    list(self._reader()(source_path))

        decode.assert_not_called()
        self.assertNotIn(str(source_path), str(raised.exception))

    def test_invalid_json_utf8_duplicate_keys_and_non_list_roots_are_sanitized(self) -> None:
        cases = [
            (b'[ {"id":"private-one", "id":"private-two"} ]', "valid JSON"),
            (b'[ {"id":"private-nan", "value": NaN} ]', "valid JSON"),
            (b'[ {"id":"private-infinity", "value": Infinity} ]', "valid JSON"),
            (b'[ {"id":"private-negative-infinity", "value": -Infinity} ]', "valid JSON"),
            (b"[ private-json-payload", "valid JSON"),
            (b"\xffprivate-utf8-payload", "valid UTF-8"),
            (b'{"id":"private-root-id"}', "root must be a list"),
        ]

        for source_bytes, expected_error in cases:
            with (
                self.subTest(expected_error=expected_error, source=source_bytes),
                tempfile.TemporaryDirectory() as directory,
            ):
                source_path = Path(directory) / "private-source-id.json"
                source_path.write_bytes(source_bytes)

                with self.assertRaisesRegex(ValueError, expected_error) as raised:
                    list(self._reader()(source_path))

                self.assertNotIn(str(source_path), str(raised.exception))
                self.assertNotIn("private-", str(raised.exception))

    def test_cycle_invalid_pointer_and_missing_chain_node_are_malformed(self) -> None:
        mappings = [
            {
                "tip": {"parent": "other", "message": None},
                "other": {"parent": "tip", "message": None},
            },
            {"tip": {"parent": 7, "message": None}},
            {"tip": {"parent": "missing", "message": None}},
            {"tip": {"message": None}},
            {"tip": {"parent": None}},
        ]

        for mapping in mappings:
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as directory:
                source_path = self._write_export(
                    directory,
                    [{"id": "bad-chain", "current_node": "tip", "mapping": mapping}],
                )
                stats = adapters.AdapterStats()
                records = list(self._reader()(source_path, stats=stats))

                self.assertEqual(records, [])
                self.assertEqual(stats.malformed_records, 1)

    def test_malformed_conversation_objects_count_once_and_emit_nothing(self) -> None:
        export = [
            "not-an-object",
            {"id": "", "current_node": "root", "mapping": {}},
            {"id": "missing-current", "mapping": {}},
            {"id": "bad-mapping", "current_node": "root", "mapping": []},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            stats = adapters.AdapterStats()
            records = list(self._reader()(source_path, stats=stats))

        self.assertEqual(records, [])
        self.assertEqual(stats.malformed_records, 4)

    def test_stats_argument_must_use_the_existing_adapter_stats_type(self) -> None:
        export = [
            {
                "id": "stats-conversation",
                "current_node": "root",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "stats-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Text"]},
                        },
                    }
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            with self.assertRaisesRegex(TypeError, "AdapterStats"):
                list(self._reader()(source_path, stats=object()))

    def test_evidence_source_is_read_only_and_locator_hides_its_path(self) -> None:
        export = [
            {
                "id": "read-only-conversation",
                "current_node": "root",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "read-only-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Exact text"]},
                        },
                    }
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export)
            original_bytes = source_path.read_bytes()
            original_hash = hashlib.sha256(original_bytes).hexdigest()
            original_mtime = source_path.stat().st_mtime_ns
            original_entries = set(Path(temporary_directory).iterdir())

            records = list(self._reader()(source_path))

            self.assertEqual(source_path.read_bytes(), original_bytes)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(source_path.stat().st_mtime_ns, original_mtime)
            self.assertEqual(set(Path(temporary_directory).iterdir()), original_entries)
            expected_locator = hashlib.sha256(
                str(source_path.resolve(strict=False)).encode("utf-8")
            ).hexdigest()
            self.assertEqual(records[0].source_locator, expected_locator)
            self.assertNotIn(str(source_path), records[0].source_locator or "")
            self.assertRegex(records[0].source_locator or "", r"^[0-9a-f]{64}$")
            self.assertIsNone(records[0].line_number)

    def test_edited_export_keeps_record_id_and_old_citation_resolvable(self) -> None:
        def export_with_text(text: str) -> list[dict[str, object]]:
            return [
                {
                    "id": "revision-conversation",
                    "current_node": "root",
                    "mapping": {
                        "root": {
                            "parent": None,
                            "message": {
                                "id": "revision-message",
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": [text]},
                            },
                        }
                    },
                }
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(temporary_directory, export_with_text("café 🧪"))
            database_path = Path(temporary_directory) / "evidence.sqlite3"
            original = list(self._reader()(source_path))[0]
            old_citation = f"{original.record_id}@{original.revision}:5:6"

            with EvidenceStore(database_path) as store:
                store.ingest([original])
            with EvidenceStore(database_path) as reopened_store:
                old_quote = citations.fetch_citation(reopened_store, old_citation)
            self.assertEqual(old_quote.quote, "🧪")

            self._write_export(temporary_directory, export_with_text("Edited 🪐"))
            edited = list(self._reader()(source_path))[0]
            self.assertEqual(edited.record_id, original.record_id)
            self.assertNotEqual(edited.revision, original.revision)
            new_citation = f"{edited.record_id}@{edited.revision}:7:8"

            with EvidenceStore(database_path) as store:
                store.ingest([edited])
            with EvidenceStore(database_path) as reopened_store:
                self.assertEqual(
                    citations.fetch_citation(reopened_store, old_citation).quote,
                    "🧪",
                )
                self.assertEqual(
                    citations.fetch_citation(reopened_store, new_citation).quote,
                    "🪐",
                )

    def test_original_part_index_and_node_key_fallback_stay_stable(self) -> None:
        def export_with_first_part(first_part: object) -> list[dict[str, object]]:
            return [
                {
                    "id": "part-index-conversation",
                    "current_node": "stable-node-key",
                    "mapping": {
                        "stable-node-key": {
                            "parent": None,
                            "message": {
                                "id": " ",
                                "author": {"role": "assistant"},
                                "content": {
                                    "content_type": "text",
                                    "parts": [first_part, "Stable quote"],
                                },
                            },
                        }
                    },
                }
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._write_export(
                temporary_directory, export_with_first_part({"image": True})
            )
            before = next(
                record for record in self._reader()(source_path) if record.text == "Stable quote"
            )
            self._write_export(temporary_directory, export_with_first_part("Newly supported"))
            after = next(
                record for record in self._reader()(source_path) if record.text == "Stable quote"
            )

        self.assertEqual(before.message_id, "stable-node-key")
        self.assertEqual(after.message_id, before.message_id)
        self.assertEqual(before.part_index, 1)
        self.assertEqual(after.part_index, before.part_index)
        self.assertEqual(after.record_id, before.record_id)

    def test_lone_surrogate_text_rejects_the_whole_conversation_before_yield(self) -> None:
        export = [
            {
                "id": "surrogate-text-conversation",
                "current_node": "later",
                "mapping": {
                    "earlier": {
                        "parent": None,
                        "message": {
                            "id": "earlier-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Valid earlier text"]},
                        },
                    },
                    "later": {
                        "parent": "earlier",
                        "message": {
                            "id": "later-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["\ud800"]},
                        },
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "surrogate.json"
            source_path.write_text(json.dumps(export, ensure_ascii=True), encoding="utf-8")
            stats = adapters.AdapterStats()
            records = list(self._reader()(source_path, stats=stats))

        self.assertEqual(records, [])
        self.assertEqual(stats.malformed_records, 1)

    def test_lone_surrogate_identity_values_reject_the_whole_conversation(self) -> None:
        cases = [
            ("\ud800", "root", "valid-message-id"),
            ("valid-conversation-id", "root", "\ud800"),
            ("valid-conversation-id", "\ud800", " "),
        ]

        for conversation_id, node_id, message_id in cases:
            with (
                self.subTest(
                    conversation_id=conversation_id,
                    node_id=node_id,
                    message_id=message_id,
                ),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                export = [
                    {
                        "id": conversation_id,
                        "current_node": node_id,
                        "mapping": {
                            node_id: {
                                "parent": None,
                                "message": {
                                    "id": message_id,
                                    "author": {"role": "assistant"},
                                    "content": {
                                        "content_type": "text",
                                        "parts": ["Valid text"],
                                    },
                                },
                            }
                        },
                    }
                ]
                source_path = Path(temporary_directory) / "surrogate-id.json"
                source_path.write_text(json.dumps(export, ensure_ascii=True), encoding="utf-8")
                stats = adapters.AdapterStats()
                records = list(self._reader()(source_path, stats=stats))

                self.assertEqual(records, [])
                self.assertEqual(stats.malformed_records, 1)

    def test_valid_escaped_emoji_text_and_ids_remain_storeable(self) -> None:
        export = [
            {
                "id": "conversation 🧪",
                "current_node": "root",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "message 🧪",
                            "author": {"role": "assistant"},
                            "content": {
                                "content_type": "text",
                                "parts": ["Left 🧪 right"],
                            },
                        },
                    }
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "escaped-emoji.json"
            source_path.write_text(json.dumps(export, ensure_ascii=True), encoding="utf-8")
            records = list(self._reader()(source_path))
            record = records[0]
            record_id = record.record_id
            revision = record.revision
            database_path = Path(temporary_directory) / "evidence.sqlite3"

            with EvidenceStore(database_path) as store:
                self.assertEqual(store.ingest(records), 1)
                self.assertEqual(store.get(record_id, revision), record)

        self.assertEqual(record.text, "Left 🧪 right")
        self.assertEqual(record.session_id, "conversation 🧪")
        self.assertEqual(record.message_id, "message 🧪")

    def test_large_integer_parser_error_uses_generic_json_message(self) -> None:
        large_integer = "9" * 5000
        source_text = (
            '[{"id":"conversation","current_node":"root","mapping":{},"value":'
            + large_integer
            + "}]"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "large-integer.json"
            source_path.write_text(source_text, encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                list(self._reader()(source_path))

        self.assertEqual(str(raised.exception), "ChatGPT export is not valid JSON")

    def test_symlink_loop_is_sanitized_without_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "loop.json"
            source_path.symlink_to(source_path.name)

            with self.assertRaises(ValueError) as raised:
                list(self._reader()(source_path))

        self.assertEqual(str(raised.exception), "cannot read ChatGPT export")
        self.assertNotIn(str(source_path), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
