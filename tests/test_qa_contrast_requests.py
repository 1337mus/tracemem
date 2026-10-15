from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import (
    prepare_qa_contrast,
    prepare_qa_contrast_requests,
    replay_qa_contrast_tokens,
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class QAContrastRequestPreparationTests(unittest.TestCase):
    def test_prepare_requests_api_is_callable(self) -> None:
        self.assertTrue(
            callable(getattr(prepare_qa_contrast_requests, "prepare_requests", None)),
            "the source-only QA request preparation API is missing",
        )

    def test_provenance_distinguishes_this_preparer_from_historical_preparation(self) -> None:
        payload, _, _ = prepare_qa_contrast_requests.prepare_requests()
        request_source_sha256 = hashlib.sha256(
            Path(prepare_qa_contrast_requests.__file__).read_bytes()
        ).hexdigest()
        provenance = payload["provenance"]
        self.assertEqual(provenance["preparer_source_sha256"], request_source_sha256)
        self.assertEqual(
            provenance["preparation_source_sha256"],
            replay_qa_contrast_tokens.replay_from_counts(
                prepare_qa_contrast_requests._RECEIPT_PATH.read_bytes()
            )["provenance"]["preparation_source_sha256"],
        )
        self.assertNotEqual(
            provenance["preparer_source_sha256"], provenance["preparation_source_sha256"]
        )

    def test_payload_has_46_exact_source_only_prompts_and_frozen_settings(self) -> None:
        payload, _, _ = prepare_qa_contrast_requests.prepare_requests()
        self.assertEqual(
            set(payload), {"schema_version", "mode", "model", "tokenizer", "provenance", "items"}
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["mode"], "qa-contrast-dev-source-only-requests-v1")
        self.assertEqual(len(payload["items"]), 46)
        prompt_ids = [item["prompt_id"] for item in payload["items"]]
        self.assertEqual(prompt_ids, sorted(set(prompt_ids)))
        for item in payload["items"]:
            self.assertEqual(set(item), {"prompt_id", "messages", "prompt_tokens"})
            self.assertEqual(len(item["messages"]), 2)
            self.assertEqual([message["role"] for message in item["messages"]], ["system", "user"])
            self.assertTrue(
                all(set(message) == {"role", "content"} for message in item["messages"])
            )
            self.assertTrue(all(type(message["content"]) is str for message in item["messages"]))
            self.assertEqual(
                hashlib.sha256(canonical_bytes(item["messages"])).hexdigest(), item["prompt_id"]
            )
            self.assertGreaterEqual(item["prompt_tokens"], 0)
            self.assertLessEqual(item["prompt_tokens"], 1024)
        self.assertEqual(
            payload["model"],
            {
                "model_id": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "precision": "float16",
                "attn_implementation": "eager",
                "trust_remote_code": False,
                "use_safetensors": True,
                "use_fast": True,
                "enable_thinking": False,
                "do_sample": False,
                "num_beams": 1,
                "seed": 0,
                "max_new_tokens": 128,
                "max_prompt_tokens": 1024,
            },
        )
        self.assertEqual(payload["tokenizer"]["settings"]["revision"], payload["model"]["revision"])
        self.assertEqual(
            payload["tokenizer"]["measurement"]["template_options"]["enable_thinking"], False
        )
        self.assertNotIn("elapsed_seconds", canonical_bytes(payload).decode())
        self.assertEqual(
            set(payload["provenance"]),
            {
                "fixture_sha256",
                "receipt_sha256",
                "catalog_sha256",
                "replay_sha256",
                "replay_source_sha256",
                "prompt_template_sha256",
                "preparer_source_sha256",
                "preparation_source_sha256",
            },
        )
        serialized = canonical_bytes(payload).decode("utf-8")
        for private_field in (
            "answer_record_ids",
            "authored_answer_record_ids",
            "question_type",
            "cell_mapping",
            "scoring",
        ):
            self.assertNotIn(private_field, serialized)

    def test_full_cell_mapping_and_authored_references_stay_local(self) -> None:
        payload, references, audit = prepare_qa_contrast_requests.prepare_requests()
        replay = replay_qa_contrast_tokens.replay_from_counts(
            prepare_qa_contrast_requests._RECEIPT_PATH.read_bytes()
        )
        expected_items: dict[str, dict[str, object]] = {}
        expected_mapping = []
        for cell in replay["cells"]:
            prompt = cell["final_prompt"]
            expected_items[prompt["prompt_id"]] = {
                "prompt_id": prompt["prompt_id"],
                "messages": prompt["messages"],
                "prompt_tokens": cell["used_tokens"],
            }
            expected_mapping.append(
                {
                    "cell_id": cell["cell_id"],
                    "question_id": cell["question_id"],
                    "split": cell["split"],
                    "mode": cell["mode"],
                    "ceiling": cell["ceiling"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_tokens": cell["used_tokens"],
                    "selected_citation_ids": cell["selected_citation_ids"],
                    "selected_record_ids": cell["selected_record_ids"],
                    "scoring": cell["scoring"],
                }
            )
        self.assertEqual(
            payload["items"],
            [expected_items[prompt_id] for prompt_id in sorted(expected_items)],
        )
        mapping = audit["cell_mapping"]
        self.assertEqual(len(mapping), 72)
        self.assertEqual(mapping, expected_mapping)
        self.assertEqual({row["mode"] for row in mapping}, {"bm25", "vector", "hybrid"})
        self.assertEqual({row["ceiling"] for row in mapping}, {128, 256, 512, 1024})
        self.assertEqual(
            {row["question_id"] for row in mapping},
            {q["question_id"] for q in references["questions"]},
        )
        payload_items = {row["prompt_id"]: row for row in payload["items"]}
        self.assertTrue(all(row["prompt_id"] in payload_items for row in mapping))
        self.assertTrue(all("scoring" in row and "selected_record_ids" in row for row in mapping))
        self.assertTrue(all("scoring" not in row for row in payload["items"]))
        self.assertEqual(
            references["payload_sha256"], hashlib.sha256(canonical_bytes(payload)).hexdigest()
        )
        _, expected_reference_bundle, _ = prepare_qa_contrast.prepare_contrast(
            prepare_qa_contrast_requests._FIXTURE_PATH.read_bytes()
        )
        self.assertEqual(references["questions"], expected_reference_bundle["questions"])
        self.assertEqual(references["test"], [])
        self.assertTrue(references["dev_only"] and references["provisional"])
        self.assertEqual(references["human_label_count"], 0)
        self.assertTrue(audit["qa_unmeasured"] and not audit["qa_measured"])
        self.assertEqual(
            audit["totals"],
            {
                "cells": 72,
                "unique_prompts": 46,
                "input_tokens_all_cells": 28944,
                "input_tokens_unique_prompts": 23533,
                "generation_cap_tokens": 5888,
                "sum_unique_input_plus_generation_cap": 29421,
                "bounds_only_not_billing_or_runtime": True,
            },
        )
        self.assertEqual(audit["deduplication"]["duplicate_cells"], 26)
        self.assertFalse(audit["deduplication"]["cells_are_independent_qa_trials"])
        self.assertEqual(audit["provenance"]["payload_sha256"], references["payload_sha256"])

    def test_fixture_receipt_and_full_replay_pins_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.json"
            fixture.write_bytes(prepare_qa_contrast_requests._FIXTURE_PATH.read_bytes() + b" ")
            receipt = Path(directory) / "receipt.json"
            receipt.write_bytes(prepare_qa_contrast_requests._RECEIPT_PATH.read_bytes() + b" ")
            guards = (
                (
                    "fixture",
                    patch.object(prepare_qa_contrast_requests, "_FIXTURE_PATH", fixture),
                    "pinned fixture",
                ),
                (
                    "receipt",
                    patch.object(prepare_qa_contrast_requests, "_RECEIPT_PATH", receipt),
                    "pinned measured receipt",
                ),
                (
                    "replay",
                    patch.object(replay_qa_contrast_tokens, "replay_from_counts", return_value={}),
                    "pinned canonical replay",
                ),
            )
            for name, guard_patch, rejection in guards:
                with self.subTest(guard=name), guard_patch:
                    with self.assertRaisesRegex(ValueError, rejection):
                        prepare_qa_contrast_requests.prepare_requests()

    def test_cli_summary_is_deterministic_and_new_outputs_are_exclusive(self) -> None:
        payload, references, audit = prepare_qa_contrast_requests.prepare_requests()
        with patch.object(
            prepare_qa_contrast_requests,
            "prepare_requests",
            return_value=(payload, references, audit),
        ):
            first, second = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(first):
                self.assertEqual(prepare_qa_contrast_requests.main([]), 0)
            with contextlib.redirect_stdout(second):
                self.assertEqual(prepare_qa_contrast_requests.main([]), 0)
            self.assertEqual(first.getvalue(), second.getvalue())
            summary = json.loads(first.getvalue())
            self.assertEqual(
                set(summary),
                {"payload_sha256", "references_sha256", "audit_sha256", "cells", "unique_prompts"},
            )

            with tempfile.TemporaryDirectory() as parent:
                output_dir = Path(parent) / "new"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        prepare_qa_contrast_requests.main(["--output-dir", str(output_dir)]), 0
                    )
                self.assertEqual(os.stat(output_dir).st_mode & 0o777, 0o700)
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()},
                    {"payload.json", "references.json", "audit.json"},
                )
                self.assertEqual(
                    {path.name: os.stat(path).st_mode & 0o777 for path in output_dir.iterdir()},
                    {"payload.json": 0o600, "references.json": 0o600, "audit.json": 0o600},
                )
                self.assertEqual(json.loads((output_dir / "payload.json").read_bytes()), payload)
                self.assertEqual(
                    json.loads((output_dir / "references.json").read_bytes()), references
                )
                self.assertEqual(json.loads((output_dir / "audit.json").read_bytes()), audit)
                existing = Path(parent) / "existing"
                existing.mkdir()
                sentinel = existing / "keep.txt"
                sentinel.write_bytes(b"preserve")
                file_target = Path(parent) / "existing-file"
                file_target.write_bytes(b"preserve-file")
                link_target = Path(parent) / "dangling-link"
                link_target.symlink_to(Path(parent) / "missing")
                for target, expected in (
                    (existing, b"preserve"),
                    (file_target, b"preserve-file"),
                    (link_target, None),
                ):
                    with self.subTest(target=target), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            prepare_qa_contrast_requests.main(["--output-dir", str(target)])
                    if expected is not None:
                        self.assertEqual(
                            target.read_bytes() if target.is_file() else sentinel.read_bytes(),
                            expected,
                        )
                    else:
                        self.assertTrue(target.is_symlink())
