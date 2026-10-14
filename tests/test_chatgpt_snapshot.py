"""Integration tests for fresh-store ChatGPT export snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tracemem.citations import fetch_citation
from tracemem.cli import main
from tracemem.retrieval import Retriever
from tracemem.snapshots import import_chatgpt_snapshot
from tracemem.storage import EvidenceStore


def _valid_export() -> list[dict[str, object]]:
    return [
        {
            "id": "fictional-conversation",
            "current_node": "leaf",
            "mapping": {
                "leaf": {
                    "parent": None,
                    "message": {
                        "id": "fictional-message",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Fictional marker 東京"],
                        },
                    },
                }
            },
        }
    ]


class ChatGPTSnapshotTests(unittest.TestCase):
    def test_imports_selected_branch_into_searchable_fresh_store(self) -> None:
        selected_text = "Launch date 東京 is 2026-10-14."
        export = [
            {
                "id": "fictional-conversation",
                "current_node": "selected-leaf",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "id": "root-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["When is launch?"]},
                        },
                    },
                    "selected-leaf": {
                        "parent": "root",
                        "message": {
                            "id": "selected-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": [selected_text]},
                        },
                    },
                    "sibling": {
                        "parent": "root",
                        "message": {
                            "id": "sibling-message",
                            "author": {"role": "assistant"},
                            "content": {
                                "content_type": "text",
                                "parts": ["An unselected branch says launch is later."],
                            },
                        },
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            database_path = output_directory / "snapshot.sqlite3"
            source_path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")

            result = import_chatgpt_snapshot(source_path, database_path)
            database_mode = stat.S_IMODE(database_path.stat().st_mode)

            with EvidenceStore(database_path) as store:
                hits = Retriever(store).search("東京", mode="bm25", k=5)
                records = store.records()
                resolved = fetch_citation(store, hits[0].citation)

        self.assertEqual(result["inserted_revisions"], 2)
        self.assertEqual(result["malformed_records"], 0)
        self.assertEqual(result["unsupported_records"], 0)
        self.assertEqual(result["role_counts"], {"assistant": 1, "user": 1})
        self.assertEqual(result["snapshot_policy"], "fresh_store_selected_branch")
        self.assertEqual(database_mode, 0o600)
        self.assertEqual({record.text for record in records}, {"When is launch?", selected_text})
        self.assertEqual([hit.record.text for hit in hits], [selected_text])
        self.assertEqual(resolved.quote, selected_text)
        self.assertEqual(
            resolved.record.text[resolved.char_start : resolved.char_end], selected_text
        )

    def test_changed_current_node_creates_independent_snapshot(self) -> None:
        def conversation(current_node: str) -> list[dict[str, object]]:
            def message(message_id: str, role: str, text: str) -> dict[str, object]:
                return {
                    "id": message_id,
                    "author": {"role": role},
                    "content": {"content_type": "text", "parts": [text]},
                }

            return [
                {
                    "id": "branch-conversation",
                    "current_node": current_node,
                    "mapping": {
                        "root": {"parent": None, "message": message("root", "user", "Prompt")},
                        "chosen": {
                            "parent": "root",
                            "message": message("chosen", "assistant", "Selected branch marker."),
                        },
                        "sibling": {
                            "parent": "root",
                            "message": message("sibling", "assistant", "Sibling branch marker."),
                        },
                    },
                }
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            first_path = output_directory / "first.sqlite3"
            second_path = output_directory / "second.sqlite3"
            source_path.write_text(json.dumps(conversation("chosen")), encoding="utf-8")
            import_chatgpt_snapshot(source_path, first_path)
            first_hash = hashlib.sha256(first_path.read_bytes()).digest()
            first_mtime = first_path.stat().st_mtime_ns

            source_path.write_text(json.dumps(conversation("sibling")), encoding="utf-8")
            import_chatgpt_snapshot(source_path, second_path)

            with (
                EvidenceStore(first_path) as first_store,
                EvidenceStore(second_path) as second_store,
            ):
                first_texts = {record.text for record in first_store.records()}
                second_texts = {record.text for record in second_store.records()}
            first_hash_after = hashlib.sha256(first_path.read_bytes()).digest()
            first_mtime_after = first_path.stat().st_mtime_ns

        self.assertEqual(first_texts, {"Prompt", "Selected branch marker."})
        self.assertEqual(second_texts, {"Prompt", "Sibling branch marker."})
        self.assertEqual(first_hash_after, first_hash)
        self.assertEqual(first_mtime_after, first_mtime)

    def test_existing_outputs_of_every_kind_are_rejected_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            source_path.write_text(json.dumps(_valid_export()), encoding="utf-8")

            database_path = output_directory / "existing.sqlite3"
            with EvidenceStore(database_path):
                pass
            database_before = hashlib.sha256(database_path.read_bytes()).digest()

            file_path = output_directory / "arbitrary.bin"
            file_path.write_bytes(b"preserve arbitrary output")
            file_before = hashlib.sha256(file_path.read_bytes()).digest()

            symlink_target = output_directory / "symlink-target.bin"
            symlink_target.write_bytes(b"preserve symlink target")
            symlink_path = output_directory / "dangling-or-live-link"
            symlink_path.symlink_to(symlink_target)
            symlink_before = os.readlink(symlink_path)
            target_before = hashlib.sha256(symlink_target.read_bytes()).digest()
            dangling_path = output_directory / "dangling-link"
            dangling_path.symlink_to(output_directory / "missing-target")
            dangling_before = os.readlink(dangling_path)

            directory_path = output_directory / "existing-directory"
            directory_path.mkdir()
            directory_child = directory_path / "keep.txt"
            directory_child.write_bytes(b"preserve directory content")
            directory_before = hashlib.sha256(directory_child.read_bytes()).digest()

            for destination in (
                database_path,
                file_path,
                symlink_path,
                dangling_path,
                directory_path,
            ):
                with self.subTest(destination=destination.name):
                    with self.assertRaisesRegex(ValueError, "ChatGPT snapshot import failed"):
                        import_chatgpt_snapshot(source_path, destination)

            self.assertEqual(hashlib.sha256(database_path.read_bytes()).digest(), database_before)
            self.assertEqual(hashlib.sha256(file_path.read_bytes()).digest(), file_before)
            self.assertEqual(os.readlink(symlink_path), symlink_before)
            self.assertEqual(os.readlink(dangling_path), dangling_before)
            self.assertEqual(hashlib.sha256(symlink_target.read_bytes()).digest(), target_before)
            self.assertEqual(
                hashlib.sha256(directory_child.read_bytes()).digest(), directory_before
            )

    def test_source_must_be_a_file_and_destination_must_be_outside_its_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            nested_output_directory = source_directory / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            nested_output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            source_bytes = json.dumps(_valid_export()).encode("utf-8")
            source_path.write_bytes(source_bytes)
            source_mtime = source_path.stat().st_mtime_ns
            nested_destination = nested_output_directory / "snapshot.sqlite3"
            external_destination = output_directory / "snapshot.sqlite3"
            missing_parent_destination = root / "missing" / "snapshot.sqlite3"

            for source, destination in (
                (source_path, source_path),
                (source_path, nested_destination),
                (source_directory, external_destination),
                (source_path, missing_parent_destination),
            ):
                with self.subTest(source=source.name, destination=destination.name):
                    with self.assertRaisesRegex(ValueError, "ChatGPT snapshot import failed"):
                        import_chatgpt_snapshot(source, destination)

            self.assertEqual(source_path.read_bytes(), source_bytes)
            self.assertEqual(source_path.stat().st_mtime_ns, source_mtime)
            self.assertFalse(nested_destination.exists())
            self.assertFalse(external_destination.exists())
            self.assertEqual(list(nested_output_directory.iterdir()), [])
            self.assertEqual(list(output_directory.iterdir()), [])
            self.assertFalse(missing_parent_destination.parent.exists())

    def test_existing_sqlite_sidecars_are_rejected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            source_path.write_text(json.dumps(_valid_export()), encoding="utf-8")

            for suffix in ("-journal", "-wal", "-shm"):
                with self.subTest(sidecar=suffix):
                    destination = output_directory / f"snapshot-{suffix[1:]}.sqlite3"
                    sidecar = destination.with_name(destination.name + suffix)
                    contents = f"preserve {suffix} bytes".encode()
                    sidecar.write_bytes(contents)

                    with self.assertRaisesRegex(ValueError, "ChatGPT snapshot import failed"):
                        import_chatgpt_snapshot(source_path, destination)

                    self.assertFalse(destination.exists())
                    self.assertEqual(sidecar.read_bytes(), contents)
                    self.assertEqual(list(output_directory.glob(".tracemem-chatgpt-*")), [])

    def test_invalid_exports_leave_no_output_or_temporary_debris(self) -> None:
        malformed_conversation = _valid_export()[0]
        malformed_conversation["id"] = "malformed-conversation"
        malformed_conversation["current_node"] = "missing-selected-node"
        one_conversation = _valid_export()[0]
        cases = (
            (
                "malformed-chain",
                json.dumps([*_valid_export(), malformed_conversation]).encode("utf-8"),
            ),
            ("duplicate-ids", json.dumps([one_conversation, one_conversation]).encode("utf-8")),
            ("invalid-json", b"{ private-export-marker: invalid JSON"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"

            for name, contents in cases:
                with self.subTest(case=name):
                    source_path.write_bytes(contents)
                    source_before = source_path.read_bytes()
                    source_mtime = source_path.stat().st_mtime_ns
                    destination = output_directory / f"{name}.sqlite3"

                    with self.assertRaises(ValueError) as raised:
                        import_chatgpt_snapshot(source_path, destination)

                    message = str(raised.exception)
                    self.assertNotIn(str(source_path), message)
                    self.assertNotIn("private-export-marker", message)
                    self.assertFalse(destination.exists())
                    self.assertEqual(source_path.read_bytes(), source_before)
                    self.assertEqual(source_path.stat().st_mtime_ns, source_mtime)
                    self.assertEqual(list(output_directory.glob(".tracemem-chatgpt-*")), [])

    def test_unsupported_content_is_skipped_and_counted(self) -> None:
        export = _valid_export()
        conversation = export[0]
        mapping = conversation["mapping"]
        node = mapping["leaf"]
        message = node["message"]
        content = message["content"]
        content["parts"].append({"image": "not serialized"})

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            destination = output_directory / "snapshot.sqlite3"
            source_path.write_text(json.dumps(export), encoding="utf-8")

            result = import_chatgpt_snapshot(source_path, destination)

            with EvidenceStore(destination) as store:
                records = store.records()

        self.assertEqual(result["inserted_revisions"], 1)
        self.assertEqual(result["unsupported_records"], 1)
        self.assertEqual(result["malformed_records"], 0)
        self.assertEqual(result["role_counts"], {"assistant": 1})
        self.assertEqual([record.text for record in records], ["Fictional marker 東京"])

    def test_empty_valid_export_publishes_an_empty_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            destination = output_directory / "empty.sqlite3"
            source_path.write_text("[]", encoding="utf-8")

            result = import_chatgpt_snapshot(source_path, destination)

            with EvidenceStore(destination) as store:
                records = store.records()

        self.assertEqual(result["inserted_revisions"], 0)
        self.assertEqual(result["malformed_records"], 0)
        self.assertEqual(result["unsupported_records"], 0)
        self.assertEqual(result["role_counts"], {})
        self.assertEqual(records, [])

    def test_cli_import_emits_only_aggregate_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            destination = output_directory / "snapshot.sqlite3"
            source_path.write_text(json.dumps(_valid_export()), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    status = main(
                        ["import-chatgpt", "--input", str(source_path), "--db", str(destination)]
                    )
                except SystemExit as exit_error:
                    status = int(exit_error.code)

            payload = json.loads(stdout.getvalue()) if status == 0 else {}
            inventory_stdout = io.StringIO()
            with redirect_stdout(inventory_stdout):
                inventory_status = main(["inventory", "--db", str(destination)])
            inventory_payload = json.loads(inventory_stdout.getvalue())
            search_stdout = io.StringIO()
            with redirect_stdout(search_stdout):
                search_status = main(
                    ["search", "--db", str(destination), "--query", "東京", "--mode", "bm25"]
                )
            search_payload = json.loads(search_stdout.getvalue())

        self.assertEqual(status, 0, stderr.getvalue())
        self.assertEqual(
            payload,
            {
                "inserted_revisions": 1,
                "malformed_records": 0,
                "unsupported_records": 0,
                "role_counts": {"assistant": 1},
                "snapshot_policy": "fresh_store_selected_branch",
            },
        )
        self.assertNotIn("Fictional marker 東京", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(inventory_status, 0)
        self.assertEqual(inventory_payload["revisions"], 1)
        self.assertEqual(inventory_payload["current_records"], 1)
        self.assertEqual(search_status, 0)
        self.assertEqual(search_payload["hits"][0]["text"], "Fictional marker 東京")

    def test_cli_import_failure_is_generic_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "private-export.json"
            destination = output_directory / "snapshot.sqlite3"
            source_path.write_text("private-export-marker invalid JSON", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    ["import-chatgpt", "--input", str(source_path), "--db", str(destination)]
                )

            error_output = stderr.getvalue()
            destination_exists = destination.exists()

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn(str(source_path), error_output)
        self.assertNotIn("private-export-marker", error_output)
        self.assertEqual(
            error_output,
            "tracemem: import-chatgpt failed; no transcript contents were printed\n",
        )
        self.assertFalse(destination_exists)

    def test_publication_collision_preserves_the_new_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            output_directory = root / "derived"
            source_directory.mkdir()
            output_directory.mkdir()
            source_path = source_directory / "conversations.json"
            destination = output_directory / "snapshot.sqlite3"
            source_path.write_text(json.dumps(_valid_export()), encoding="utf-8")
            real_link = os.link
            collision_bytes = b"concurrent winner"

            def create_collision(
                source: str | os.PathLike[str], target: str | os.PathLike[str]
            ) -> None:
                Path(target).write_bytes(collision_bytes)
                real_link(source, target)

            with patch("tracemem.snapshots.os.link", side_effect=create_collision):
                with self.assertRaisesRegex(ValueError, "ChatGPT snapshot import failed"):
                    import_chatgpt_snapshot(source_path, destination)

            self.assertEqual(destination.read_bytes(), collision_bytes)
            self.assertEqual(list(output_directory.glob(".tracemem-chatgpt-*")), [])
