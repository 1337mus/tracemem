"""Focused tests for the local-only SFT preparation boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tracemem import prompts
from training.prepare_sft import main, prepare_dataset


def _source(split: str) -> dict[str, object]:
    case_id = f"case-{split}"
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": [
            {
                "case_id": case_id,
                "scenario_id": f"scenario-{split}",
                "split": split,
                "tags": ["fixture"],
                "messages": [
                    {
                        "session_id": f"session-{split}",
                        "message_id": f"message-{split}",
                        "role": "user",
                        "text": f"I prefer tea in {split}.",
                        "recorded_at": None,
                        "content_kind": "conversation",
                    }
                ],
                "expected": [],
            }
        ],
        "case_metadata": {
            case_id: {
                "scenario_group": f"scenario-group-{split}",
                "template_group": f"template-group-{split}",
                "entity_ids": [f"entity-{split}"],
            }
        },
    }


def _three_splits() -> list[dict[str, object]]:
    return [_source(split) for split in ("train", "dev", "test")]


def _add_gold_claim(source: dict[str, object], sentinel: str) -> None:
    case = source["cases"][0]
    message = case["messages"][0]
    case["expected"] = [
        {
            "subject": "user",
            "predicate": "PREFERS",
            "object": sentinel,
            "speaker_role": "user",
            "attribution": "direct_user_statement",
            "modality": "preference",
            "polarity": "positive",
            "valid_from": None,
            "valid_to": None,
            "evidence": [
                {
                    "session_id": message["session_id"],
                    "message_id": message["message_id"],
                    "char_start": 0,
                    "char_end": len(message["text"]),
                    "quote": message["text"],
                }
            ],
        }
    ]


class PrepareSftTests(unittest.TestCase):
    def test_prepare_dataset_renders_three_explicit_partitions(self) -> None:
        sources = _three_splits()
        bundle = prepare_dataset(sources)

        self.assertEqual(set(bundle["files"]), {"train.jsonl", "dev.jsonl", "test.jsonl"})
        row = json.loads(bundle["files"]["train.jsonl"])
        self.assertEqual(
            set(row),
            {"case_id", "split", "scenario_group", "template_group", "messages"},
        )
        self.assertEqual(row["messages"][0]["role"], "user")
        self.assertIn("I prefer tea in train.", row["messages"][0]["content"])
        self.assertNotIn("case-train", row["messages"][0]["content"])
        self.assertNotIn("fixture", row["messages"][0]["content"])
        self.assertNotIn("scenario-group-train", row["messages"][0]["content"])
        self.assertEqual(row["messages"][1], {"role": "assistant", "content": '{"claims":[]}'})
        self.assertEqual(
            bundle["manifest"]["partitions"]["train"],
            {
                "rows": 1,
                "claims": 0,
                "empty_targets": 1,
                "family_counts": {"template-group-train": 1},
            },
        )
        canonical = json.dumps(
            sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            bundle["manifest"]["source"]["sha256"], hashlib.sha256(canonical).hexdigest()
        )
        prompt_source = Path(prompts.__file__).read_bytes()
        self.assertEqual(
            bundle["manifest"]["prompt_source_sha256"], hashlib.sha256(prompt_source).hexdigest()
        )
        policy = bundle["manifest"]["annotation_policy"]
        self.assertEqual(policy["support_status"], "provisional_agent_authored")
        self.assertEqual(policy["gate_a"], "not_assessed")
        self.assertEqual(policy["human_validity"], "not_assessed")
        self.assertIn("does not establish truth", policy["empty_target"])

    def test_rejects_duplicate_case_ids_across_payloads(self) -> None:
        sources = _three_splits()
        sources[1]["cases"][0]["case_id"] = "case-train"
        metadata = sources[1]["case_metadata"]
        metadata["case-train"] = metadata.pop("case-dev")

        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            prepare_dataset(sources)

    def test_rejects_duplicate_global_message_identity(self) -> None:
        sources = _three_splits()
        sources[2]["cases"][0]["messages"][0]["session_id"] = "session-train"
        sources[2]["cases"][0]["messages"][0]["message_id"] = "message-train"

        with self.assertRaisesRegex(ValueError, "duplicate global message identity"):
            prepare_dataset(sources)

    def test_rejects_blank_or_duplicate_entity_metadata(self) -> None:
        for entity_ids in ([], ["entity", "entity"]):
            with self.subTest(entity_ids=entity_ids):
                sources = _three_splits()
                sources[0]["case_metadata"]["case-train"]["entity_ids"] = entity_ids

                with self.assertRaisesRegex(ValueError, "entity_ids"):
                    prepare_dataset(sources)

    def test_requires_nonempty_train_dev_and_test_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete bundle"):
            prepare_dataset(_three_splits()[:2])

    def test_rejects_whitespace_normalized_duplicate_conversations(self) -> None:
        sources = _three_splits()
        sources[1]["cases"][0]["messages"][0]["text"] = "I   prefer tea in train."

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            prepare_dataset(sources)

    def test_rejects_groups_shared_across_splits(self) -> None:
        for field in ("scenario_id", "scenario_group", "template_group", "entity_ids"):
            with self.subTest(field=field):
                sources = _three_splits()
                train_metadata = sources[0]["case_metadata"]["case-train"]
                dev_metadata = sources[1]["case_metadata"]["case-dev"]
                if field == "scenario_id":
                    sources[1]["cases"][0]["scenario_id"] = sources[0]["cases"][0]["scenario_id"]
                else:
                    dev_metadata[field] = (
                        list(train_metadata[field])
                        if field == "entity_ids"
                        else train_metadata[field]
                    )

                with self.assertRaisesRegex(ValueError, "multiple splits"):
                    prepare_dataset(sources)

    def test_rejects_unknown_root_field(self) -> None:
        sources = _three_splits()
        sources[0]["source_path"] = "/private/corpus.json"

        with self.assertRaisesRegex(ValueError, "source root"):
            prepare_dataset(sources)

    def test_rejects_case_with_unknown_schema_field(self) -> None:
        sources = _three_splits()
        sources[0]["cases"][0]["private_note"] = "must not be exported"

        with self.assertRaisesRegex(ValueError, "case_metadata|challenge case"):
            prepare_dataset(sources)

    def test_rejects_expected_claim_with_corrupt_evidence_quote(self) -> None:
        sources = _three_splits()
        _add_gold_claim(sources[0], "gold")
        sources[0]["cases"][0]["expected"][0]["evidence"][0]["quote"] = "not the source quote"

        with self.assertRaisesRegex(ValueError, "evidence"):
            prepare_dataset(sources)

    def test_rejects_non_fiction_provenance(self) -> None:
        sources = _three_splits()
        sources[1]["provenance"] = "existing_corpus"

        with self.assertRaisesRegex(ValueError, "provenance"):
            prepare_dataset(sources)

    def test_cli_rejects_duplicate_json_keys_and_nonfinite_constants(self) -> None:
        for name, malformed in (
            ("duplicate", b'{"version":1,"version":1}'),
            ("nan", b'{"version":NaN}'),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths: dict[str, Path] = {}
                for split in ("train", "dev", "test"):
                    path = root / f"{split}.json"
                    if split == "train":
                        path.write_bytes(malformed)
                    else:
                        path.write_text(json.dumps(_source(split)), encoding="utf-8")
                    paths[split] = path
                output_dir = root / "bundle"

                result = main(
                    [
                        "--train",
                        str(paths["train"]),
                        "--dev",
                        str(paths["dev"]),
                        "--test",
                        str(paths["test"]),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

                self.assertEqual(result, 2)
                self.assertFalse(output_dir.exists())

    def test_cli_rejects_case_relabeling_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [_source("dev"), _source("train"), _source("test")]
            paths: list[Path] = []
            for index, payload in enumerate(sources):
                path = root / f"input-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            output_dir = root / "bundle"

            result = main(
                [
                    "--train",
                    str(paths[0]),
                    "--dev",
                    str(paths[1]),
                    "--test",
                    str(paths[2]),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 2)
            self.assertFalse(output_dir.exists())

    def test_cli_refuses_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_paths: dict[str, Path] = {}
            for payload in _three_splits():
                split = payload["cases"][0]["split"]
                path = root / f"{split}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                input_paths[split] = path
            output_dir = root / "already-there"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("existing", encoding="utf-8")

            result = main(
                [
                    "--train",
                    str(input_paths["train"]),
                    "--dev",
                    str(input_paths["dev"]),
                    "--test",
                    str(input_paths["test"]),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 2)
            self.assertEqual(list(output_dir.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")

    def test_cli_writes_explicit_partitions_and_matching_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = _three_splits()
            input_paths: dict[str, Path] = {}
            input_hashes: dict[str, str] = {}
            for payload in sources:
                split = payload["cases"][0]["split"]
                path = root / f"{split}.json"
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                path.write_bytes(raw)
                input_paths[split] = path
                input_hashes[split] = hashlib.sha256(raw).hexdigest()
            expected_files = prepare_dataset(sources)["files"]
            output_dir = root / "bundle"

            result = main(
                [
                    "--train",
                    str(input_paths["train"]),
                    "--dev",
                    str(input_paths["dev"]),
                    "--test",
                    str(input_paths["test"]),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 0)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["sha256_by_partition"], input_hashes)
            for split in ("train", "dev", "test"):
                output = (output_dir / f"{split}.jsonl").read_bytes()
                self.assertEqual(
                    hashlib.sha256(output).hexdigest(), manifest["output_sha256"][f"{split}.jsonl"]
                )
                self.assertEqual(json.loads(output), json.loads(expected_files[f"{split}.jsonl"]))
                self.assertNotIn(temporary.encode(), output)
            self.assertEqual(manifest["token_count"], None)
            self.assertEqual(manifest["token_count_status"], "unavailable_locally")
            self.assertIn("not token counts", manifest["character_counts"]["note"])
            self.assertNotIn(temporary, (output_dir / "manifest.json").read_text(encoding="utf-8"))

    def test_gold_claim_is_completion_only(self) -> None:
        sources = _three_splits()
        sentinel = "GOLD_ONLY_SENTINEL_8274"
        _add_gold_claim(sources[0], sentinel)

        bundle = prepare_dataset(sources)
        row = json.loads(bundle["files"]["train.jsonl"])

        self.assertNotIn(sentinel, row["messages"][0]["content"])
        self.assertEqual(json.loads(row["messages"][1]["content"])["claims"][0]["object"], sentinel)


if __name__ == "__main__":
    unittest.main()
