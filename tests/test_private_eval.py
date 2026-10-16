"""Fictional-fixture checks for the local private evaluation snapshot tool."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from experiments import private_eval
from tracemem.adapters import AdapterStats, read_transcript

WORKSPACE = Path(__file__).resolve().parents[1]


def _source_path(base: Path, name: str) -> Path:
    directory = base / "sources"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory / name


def _write_transcript(
    path: Path, messages: list[tuple[str, str, str]], session_id: str | None = None
) -> None:
    session_id = session_id or f"session-{path.stem}"
    lines = [json.dumps({"type": "session_meta", "payload": {"id": session_id}})]
    for role, message_id, text in messages:
        block_type = "input_text" if role == "user" else "output_text"
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": role,
                        "id": message_id,
                        "content": [{"type": block_type, "text": text}],
                    },
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_multi_session_transcript(path: Path) -> None:
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "session-one"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "id": "one-user",
                    "content": [{"type": "input_text", "text": "First session question."}],
                },
            }
        ),
        json.dumps({"type": "session_meta", "payload": {"id": "session-two"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "id": "two-answer",
                    "content": [{"type": "output_text", "text": "Second session answer."}],
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(path: Path, source: Path, project_group: str = "project-fixture") -> None:
    _manifest_many(path, [(source, project_group)])


def _manifest_many(path: Path, sources: list[tuple[Path, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "provenance": "private_source",
                "discovery": {"fixture": True},
                "sources": [
                    {
                        "provider": "codex",
                        "path": str(source_path),
                        "project_group": project_group,
                    }
                    for source_path, project_group in sources
                ],
            }
        ),
        encoding="utf-8",
    )


class PrivateEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_temp = tempfile.TemporaryDirectory()
        self.addCleanup(workspace_temp.cleanup)
        workspace = Path(workspace_temp.name).resolve()
        local_root = workspace / ".local"
        local_root.mkdir(mode=0o700)

        for patcher in (
            patch.object(sys.modules[__name__], "WORKSPACE", workspace),
            patch.object(private_eval, "WORKSPACE_ROOT", workspace),
            patch.object(private_eval, "LOCAL_ROOT", local_root),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_snapshot_preserves_complete_evidence_ids_anchors_and_unicode(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "fictional.jsonl")
            manifest = base / "manifest.json"
            output = base / "prepared"
            _write_transcript(
                source,
                [
                    ("user", "question-id", "What does café 🌿 mean?"),
                    ("assistant", "answer-id", "It means coffee ☕."),
                ],
            )
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            original_records = list(
                read_transcript(source, "codex", stats=AdapterStats(), require_quiescent=True)
            )
            _manifest(manifest, source)

            from experiments.private_eval import prepare_private_eval

            aggregate = prepare_private_eval(manifest, output, seed=17)
            snapshot = json.loads((output / "snapshot.json").read_text(encoding="utf-8"))
            windows = snapshot["windows"]
            self.assertEqual(aggregate["annotation_status"], "pending")
            self.assertEqual(len(windows), 1)
            self.assertEqual(
                windows[0]["records"],
                [
                    {
                        **asdict(record),
                        "record_id": record.record_id,
                        "revision": record.revision,
                        "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
                        "privacy_indicators": [],
                    }
                    for record in original_records
                ],
            )
            self.assertEqual(windows[0]["records"][0]["line_number"], 2)
            self.assertEqual(windows[0]["records"][0]["text"], "What does café 🌿 mean?")
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)

    def test_project_groups_are_deterministically_assigned_as_units(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            sources: list[tuple[Path, str]] = []
            for name, group, content in (
                ("a1", "group-a", "Alpha one"),
                ("a2", "group-a", "Alpha two"),
                ("b1", "group-b", "Beta one"),
                ("b2", "group-b", "Beta two"),
            ):
                source = _source_path(base, f"{name}.jsonl")
                _write_transcript(source, [("user", f"{name}-id", content)])
                sources.append((source, group))
            manifest = base / "manifest.json"
            _manifest_many(manifest, sources)

            from experiments.private_eval import prepare_private_eval

            prepare_private_eval(manifest, base / "prepared", seed=23)
            windows = json.loads((base / "prepared" / "snapshot.json").read_text())["windows"]

        assignments: dict[str, set[str]] = {}
        for window in windows:
            assignments.setdefault(window["project_group"], set()).add(window["split"])
        self.assertEqual(set(assignments), {"group-a", "group-b"})
        self.assertEqual(assignments["group-a"].__len__(), 1)
        self.assertEqual(assignments["group-b"].__len__(), 1)
        self.assertNotEqual(next(iter(assignments["group-a"])), next(iter(assignments["group-b"])))

    def test_duplicate_complete_windows_across_groups_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            first = _source_path(base, "first.jsonl")
            second = _source_path(base, "second.jsonl")
            same_text = "A short fictional source phrase."
            _write_transcript(first, [("user", "first-message", same_text)], "session-first")
            _write_transcript(second, [("user", "second-message", same_text)], "session-second")
            manifest = base / "manifest.json"
            _manifest_many(manifest, [(first, "group-first"), (second, "group-second")])
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "duplicate evaluation window"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_shared_session_identity_across_groups_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            first = _source_path(base, "first.jsonl")
            second = _source_path(base, "second.jsonl")
            _write_transcript(first, [("user", "first-message", "First source.")], "same-session")
            _write_transcript(
                second, [("user", "second-message", "Second source.")], "same-session"
            )
            manifest = base / "manifest.json"
            _manifest_many(manifest, [(first, "group-first"), (second, "group-second")])
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "shared session identity"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_source_change_during_read_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "changing.jsonl")
            _write_transcript(source, [("user", "message-original", "Original fixture text.")])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "must-not-exist"
            original_reader = read_transcript

            def changing_reader(*args: object, **kwargs: object):
                yield from original_reader(*args, **kwargs)  # type: ignore[arg-type]
                source.write_text("changed after adapter read\n", encoding="utf-8")

            from experiments.private_eval import prepare_private_eval

            with patch("experiments.private_eval.read_transcript", changing_reader):
                with self.assertRaisesRegex(ValueError, "source changed during preparation"):
                    prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_long_normalized_record_overlap_across_splits_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            repeated = "A fictional paragraph shared across projects. " * 5
            first = _source_path(base, "first.jsonl")
            second = _source_path(base, "second.jsonl")
            _write_transcript(
                first,
                [("user", "first-user", repeated), ("assistant", "first-answer", "Answer one.")],
                "session-first",
            )
            _write_transcript(
                second,
                [("user", "second-user", repeated), ("assistant", "second-answer", "Answer two.")],
                "session-second",
            )
            manifest = base / "manifest.json"
            _manifest_many(manifest, [(first, "group-first"), (second, "group-second")])
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "normalized text overlap across splits"):
                prepare_private_eval(manifest, output, seed=17)

            self.assertFalse(output.exists())

    def test_more_than_24_sources_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            sources: list[tuple[Path, str]] = []
            for index in range(25):
                source = _source_path(base, f"source-{index:02d}.jsonl")
                _write_transcript(source, [("user", f"message-{index}", f"Fiction {index}")])
                sources.append((source, f"group-{index:02d}"))
            manifest = base / "manifest.json"
            _manifest_many(manifest, sources)
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "at most 24 source files"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_source_larger_than_8_mib_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "oversized.jsonl")
            source.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "8 MiB limit"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_two_windows_use_deterministic_interior_user_turn_quantiles(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "many-turns.jsonl")
            messages: list[tuple[str, str, str]] = []
            for index in range(12):
                messages.extend(
                    [
                        ("user", f"user-{index}", f"Question fixture {index}."),
                        ("assistant", f"assistant-{index}", f"Answer fixture {index}."),
                    ]
                )
            _write_transcript(source, messages)
            manifest = base / "manifest.json"
            _manifest(manifest, source)

            from experiments.private_eval import prepare_private_eval

            first_output = base / "first-output"
            second_output = base / "second-output"
            prepare_private_eval(manifest, first_output, seed=41)
            prepare_private_eval(manifest, second_output, seed=41)
            first_snapshot = (first_output / "snapshot.json").read_bytes()
            second_snapshot = (second_output / "snapshot.json").read_bytes()
            windows = json.loads(first_snapshot)["windows"]

        self.assertEqual(first_snapshot, second_snapshot)
        self.assertEqual(len(windows), 2)
        anchors = [window["records"][0]["message_id"] for window in windows]
        self.assertEqual(len(set(anchors)), 2)
        self.assertNotIn("user-0", anchors)
        self.assertNotIn("user-11", anchors)
        self.assertTrue(all(len(window["records"]) <= 6 for window in windows))

    def test_window_character_limit_skips_whole_candidate_and_advances(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "window-limit.jsonl")
            oversized = "x" * 8001
            messages = [
                ("user", "user-0", "small opening"),
                ("assistant", "assistant-0", "small response"),
                ("user", "user-1", oversized),
                ("assistant", "assistant-1", "small response"),
                ("user", "user-2", "complete later turn"),
            ]
            _write_transcript(source, messages)
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "prepared"

            from experiments.private_eval import prepare_private_eval

            aggregate = prepare_private_eval(manifest, output)
            windows = json.loads((output / "snapshot.json").read_text())["windows"]

        self.assertEqual(aggregate["skipped"]["windows"], 1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["records"][0]["message_id"], "user-2")
        self.assertEqual(windows[0]["records"][0]["text"], "complete later turn")

    def test_exactly_8000_characters_are_preserved_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "exact-window-limit.jsonl")
            text = "x" * 8000
            _write_transcript(source, [("user", "exact-limit-user", text)])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "prepared"

            from experiments.private_eval import prepare_private_eval

            prepare_private_eval(manifest, output)
            windows = json.loads((output / "snapshot.json").read_text())["windows"]

        self.assertEqual(windows[0]["records"][0]["text"], text)
        self.assertEqual(len(windows[0]["records"][0]["text"]), 8000)

    def test_private_outputs_escape_source_and_keep_annotations_pending(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "html-fixture.jsonl")
            _write_transcript(
                source,
                [
                    ("user", "html-user", "<script>alert('fixture')</script> & quoted text"),
                    ("assistant", "html-answer", "A fictional reply."),
                ],
            )
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "prepared"

            from experiments.private_eval import prepare_private_eval

            aggregate = prepare_private_eval(manifest, output)
            snapshot = json.loads((output / "snapshot.json").read_text(encoding="utf-8"))
            annotation = json.loads(
                (output / "annotation-template.json").read_text(encoding="utf-8")
            )
            html = (output / "review.html").read_text(encoding="utf-8")
            output_mode = os.stat(output).st_mode & 0o777
            file_modes = {
                name: os.stat(output / name).st_mode & 0o777
                for name in ("snapshot.json", "annotation-template.json", "review.html")
            }

        self.assertEqual(snapshot["annotation_status"], "pending")
        self.assertFalse(aggregate["held_out_available"])
        self.assertEqual(aggregate["test_count"], 0)
        self.assertFalse(snapshot["held_out_available"])
        self.assertEqual(annotation["annotation_status"], "pending")
        self.assertEqual(annotation["windows"][0]["claim_annotations"], [])
        self.assertEqual(annotation["windows"][0]["qa_annotations"], [])
        self.assertEqual(annotation["windows"][0]["authorship_review"]["status"], "pending")
        self.assertEqual(
            annotation["windows"][0]["authorship_review"]["review_options"],
            ["direct_authored", "quoted_or_instructions"],
        )
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("Review locally", html)
        self.assertIn("annotation-template.json", html)
        self.assertIn("edit a copy", html)
        self.assertEqual(output_mode, 0o700)
        self.assertEqual(set(file_modes.values()), {0o600})

    def test_privacy_indicators_mark_exact_evidence_without_exposing_matches(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "privacy-fixture.jsonl")
            private_fixture = "fixture-person@example.invalid"
            text = f"Fictional contact {private_fixture} for scanner coverage."
            _write_transcript(source, [("user", "privacy-user", text)])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "prepared"

            from experiments.private_eval import prepare_private_eval

            aggregate = prepare_private_eval(manifest, output)
            snapshot = json.loads((output / "snapshot.json").read_text(encoding="utf-8"))
            html = (output / "review.html").read_text(encoding="utf-8")
            aggregate_text = json.dumps(aggregate)
            saved_record = snapshot["windows"][0]["records"][0]

        self.assertEqual(saved_record["text"], text)
        self.assertEqual(
            saved_record["privacy_indicators"][0]["category"],
            "email",
        )
        finding = saved_record["privacy_indicators"][0]
        self.assertEqual(text[finding["char_start"] : finding["char_end"]], private_fixture)
        self.assertIn("Privacy indicators", html)
        self.assertIn("email", html)
        self.assertNotIn(private_fixture, aggregate_text)
        self.assertEqual(aggregate["privacy_indicators"]["findings"], 1)

    def test_output_outside_workspace_local_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as input_temp:
            base = Path(input_temp)
            source = _source_path(base, "containment.jsonl")
            _write_transcript(source, [("user", "containment-user", "Fictional text.")])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            with tempfile.TemporaryDirectory() as outside_temp:
                outside_base = Path(outside_temp).resolve()
                output = outside_base / "prepared"

                from experiments.private_eval import prepare_private_eval

                with self.assertRaisesRegex(ValueError, "new directory under workspace .local"):
                    prepare_private_eval(manifest, output)

                self.assertFalse(output.exists())

    def test_output_inside_a_source_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "output-separation.jsonl")
            _write_transcript(source, [("user", "separation-user", "Fictional text.")])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = source.parent / "prepared"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "inside a source directory"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_output_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as input_temp:
            base = Path(input_temp)
            source = _source_path(base, "output-symlink.jsonl")
            _write_transcript(source, [("user", "symlink-user", "Fictional text.")])
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            with tempfile.TemporaryDirectory() as outside_temp:
                outside_base = Path(outside_temp).resolve()
                link = base / "output-link"
                link.symlink_to(outside_base, target_is_directory=True)
                output = link / "prepared"

                from experiments.private_eval import prepare_private_eval

                with self.assertRaisesRegex(ValueError, "output path must not contain symlinks"):
                    prepare_private_eval(manifest, output)

                self.assertFalse((outside_base / "prepared").exists())

    def test_source_symlink_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            actual = _source_path(base, "actual.jsonl")
            _write_transcript(actual, [("user", "source-link-user", "Fictional text.")])
            source_link = actual.parent / "link.jsonl"
            source_link.symlink_to(actual)
            manifest = base / "manifest.json"
            _manifest(manifest, source_link)
            output = base / "must-not-exist"

            from experiments.private_eval import prepare_private_eval

            with self.assertRaisesRegex(ValueError, "source path must not contain symlinks"):
                prepare_private_eval(manifest, output)

            self.assertFalse(output.exists())

    def test_windows_never_mix_native_session_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / ".local") as temporary:
            base = Path(temporary)
            source = _source_path(base, "multi-session.jsonl")
            _write_multi_session_transcript(source)
            manifest = base / "manifest.json"
            _manifest(manifest, source)
            output = base / "prepared"

            from experiments.private_eval import prepare_private_eval

            prepare_private_eval(manifest, output)
            windows = json.loads((output / "snapshot.json").read_text())["windows"]

        self.assertTrue(windows)
        for window in windows:
            session_ids = {record["session_id"] for record in window["records"]}
            self.assertEqual(session_ids, {window["session_id"]})


if __name__ == "__main__":
    unittest.main()
