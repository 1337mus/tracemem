"""Tests for preparing blinded reviews of generated scenario candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

from experiments import scenario_payload
from experiments.prepare_scenario_reviews import main, prepare_reviews
from tracemem.annotations import _canonical_json
from tracemem.scenarios import load_scenarios


def _fixture() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    return load_scenarios()


def _receipt(
    model_key: str,
    overrides: dict[str, str] | None = None,
) -> bytes:
    cases, _, _ = _fixture()
    payload = scenario_payload.build_payload(model_key)
    eos_ids = [32007, 32001, 32000] if model_key == "phi-3.5-mini" else [42, 43]
    overrides = {} if overrides is None else overrides
    outputs = []
    for item in payload["items"]:
        raw_output = overrides.get(item["case_id"], '{"claims":[]}')
        outputs.append(
            {
                "case_id": item["case_id"],
                "raw_output": raw_output,
                "input_tokens": 200,
                "generated_tokens": 20,
                "hit_max_new_tokens": False,
                "output_characters": len(raw_output),
                "finish_reason": "eos",
                "generated_last_token_id": eos_ids[-1],
            }
        )
    receipt = {
        "schema_version": 1,
        "mode": "scenarios",
        "input_sha256": payload["input_sha256"],
        "prompt_source_sha256": payload["prompt_source_sha256"],
        "payload_sha256": scenario_payload.payload_sha256(payload),
        "prompt_sha256": payload["prompt_sha256"],
        "prompt_version": payload["prompt_version"],
        "model": payload["model"],
        "package_versions": {"torch": "2.7.1", "transformers": "4.51.3"},
        "gpu_compute_seconds": 12.5,
        "model_load_seconds": 4.0,
        "remote_function_elapsed_seconds": 20.0,
        "peak_gpu_memory_bytes": 3_000_000,
        "python_version": "3.12.0",
        "torch_cuda_version": "12.6",
        "gpu_name": "Tesla T4",
        "modal_function_call_id": "fc-test",
        "modal_app_id": "ap-test",
        "client_call_wall_seconds_including_image_and_startup": 30.0,
        "runner_sha256": "b" * 64,
        "generation_eos_token_ids": eos_ids,
        "generation_pad_token_id": eos_ids[-1],
        "case_count": len(outputs),
        "generated_outputs": outputs,
    }
    return json.dumps(receipt, ensure_ascii=False).encode("utf-8")


def _candidate(case_id: str, object_text: str) -> dict[str, Any]:
    cases, _, _ = _fixture()
    case = next(case for case in cases if case["case_id"] == case_id)
    claim = copy.deepcopy(case["expected"][0])
    claim["object"] = object_text
    return claim


def _json_claims(*claims: dict[str, Any]) -> str:
    return json.dumps({"claims": list(claims)}, ensure_ascii=False, separators=(",", ":"))


class ScenarioReviewPreparationTests(unittest.TestCase):
    def test_deduplicates_after_anchoring_and_keeps_each_receipt_origin_private(self) -> None:
        cases, _, _ = _fixture()
        source_case = next(case for case in cases if case["expected"])
        case_id = source_case["case_id"]
        exact = _candidate(case_id, "generated review candidate")
        shifted = copy.deepcopy(exact)
        shifted["evidence"][0]["char_start"] += 1
        shifted["evidence"][0]["char_end"] += 1
        qwen_receipt = _receipt("qwen3-4b", {case_id: _json_claims(shifted, shifted)})
        phi_receipt = _receipt("phi-3.5-mini", {case_id: _json_claims(exact)})

        bundle, key, provenance = prepare_reviews([qwen_receipt, phi_receipt], seed=11)

        self.assertEqual(len(bundle["tasks"]), 1)
        task = bundle["tasks"][0]
        self.assertEqual(task["candidate"]["object"], "generated review candidate")
        self.assertEqual(len(key["task_keys"]), 1)
        self.assertEqual(
            key["task_keys"][0],
            {
                "task_id": task["task_id"],
                "case_id": case_id,
                "split": source_case["split"],
                "membership": "unlabeled",
                "reference_label": None,
            },
        )

        blind_json = json.dumps(bundle, ensure_ascii=False)
        for private_value in (
            case_id,
            source_case["messages"][0]["session_id"],
            source_case["messages"][0]["message_id"],
            "qwen3-4b",
            "phi-3.5-mini",
            '"split"',
            '"membership"',
            '"reference_label"',
        ):
            self.assertNotIn(private_value, blind_json)

        task_rows = provenance["task_provenance"]
        self.assertEqual({row["task_id"] for row in task_rows}, {task["task_id"]})
        self.assertEqual(
            {row["task_id"] for row in key["task_keys"]},
            {row["task_id"] for row in task_rows},
        )
        origins = task_rows[0]["origins"]
        self.assertEqual(
            {(origin["model_key"], origin["receipt_sha256"]) for origin in origins},
            {
                ("qwen3-4b", hashlib.sha256(qwen_receipt).hexdigest()),
                ("phi-3.5-mini", hashlib.sha256(phi_receipt).hexdigest()),
            },
        )
        self.assertEqual(len(origins), 3)
        self.assertEqual(
            {origin["claim_index"] for origin in origins if origin["model_key"] == "qwen3-4b"},
            {0, 1},
        )
        output_statuses = {
            row["model_key"]: (row["raw_status"], row["anchored_status"])
            for row in provenance["outputs"]
            if row["case_id"] == case_id
        }
        self.assertEqual(output_statuses["qwen3-4b"], ("valid_schema", "repaired"))
        self.assertEqual(output_statuses["phi-3.5-mini"], ("valid_schema", "unchanged"))
        self.assertEqual(provenance["exclusions"], [])
        mapping_body = {
            "bundle_sha256": bundle["bundle_sha256"],
            "task_provenance": task_rows,
        }
        self.assertEqual(
            provenance["task_mapping_sha256"],
            hashlib.sha256(_canonical_json(mapping_body).encode("utf-8")).hexdigest(),
        )
        unsigned_provenance = {
            key: value for key, value in provenance.items() if key != "provenance_sha256"
        }
        self.assertEqual(
            provenance["provenance_sha256"],
            hashlib.sha256(_canonical_json(unsigned_provenance).encode("utf-8")).hexdigest(),
        )

    def test_parse_and_schema_failures_are_separate_exclusions_not_empty_predictions(self) -> None:
        cases, _, _ = _fixture()
        selected = [case for case in cases if case["expected"]][:3]
        parse_case, schema_case, valid_case = selected
        candidate = _candidate(valid_case["case_id"], "valid candidate")
        original = _receipt(
            "qwen3-4b",
            {
                parse_case["case_id"]: "not JSON",
                schema_case["case_id"]: '{"claims":[{"unexpected":"shape"}]}',
                valid_case["case_id"]: _json_claims(candidate),
            },
        )

        bundle, key, provenance = prepare_reviews([original], seed=3)

        self.assertEqual(len(bundle["tasks"]), 1)
        self.assertEqual(len(key["task_keys"]), 1)
        self.assertEqual(key["task_keys"][0]["case_id"], valid_case["case_id"])
        self.assertEqual(
            {(row["case_id"], row["reason"]) for row in provenance["exclusions"]},
            {
                (parse_case["case_id"], "invalid_json"),
                (schema_case["case_id"], "invalid_candidate_schema"),
            },
        )
        self.assertNotIn(
            parse_case["case_id"], {row["case_id"] for row in provenance["task_provenance"]}
        )
        self.assertNotIn(
            schema_case["case_id"], {row["case_id"] for row in provenance["task_provenance"]}
        )

    def test_authored_expected_claims_are_not_inserted_as_review_candidates(self) -> None:
        cases, _, _ = _fixture()
        source_case = next(case for case in cases if case["expected"])
        authored_claim = copy.deepcopy(source_case["expected"][0])
        receipt = _receipt("qwen3-4b", {source_case["case_id"]: _json_claims(authored_claim)})

        bundle, key, _ = prepare_reviews([receipt], seed=7)

        self.assertEqual(len(bundle["tasks"]), 1)
        self.assertEqual(key["task_keys"][0]["membership"], "unlabeled")
        self.assertIsNone(key["task_keys"][0]["reference_label"])
        self.assertNotIn("gold", json.dumps(bundle))

    def test_fixture_hash_or_receipt_coverage_mismatch_is_rejected(self) -> None:
        parsed = json.loads(_receipt("qwen3-4b"))
        parsed["input_sha256"] = "0" * 64
        bad_receipt = json.dumps(parsed).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "input hash"):
            prepare_reviews([bad_receipt], seed=0)

        parsed = json.loads(_receipt("qwen3-4b"))
        parsed["generated_outputs"].pop()
        with self.assertRaisesRegex(ValueError, "generated output count"):
            prepare_reviews([json.dumps(parsed).encode("utf-8")], seed=0)

    def test_repeated_receipts_and_multiple_receipts_for_one_model_are_rejected(self) -> None:
        receipt = _receipt("qwen3-4b")
        with self.assertRaisesRegex(ValueError, "duplicate receipt SHA-256"):
            prepare_reviews([receipt, receipt], seed=0)

        changed = json.loads(receipt)
        changed["gpu_compute_seconds"] += 1
        second_run = json.dumps(changed).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "more than one receipt for model"):
            prepare_reviews([receipt, second_run], seed=0)

    def test_cli_writes_exclusively_and_rejects_aliases_without_mutating_receipts(self) -> None:
        cases, _, _ = _fixture()
        source_case = next(case for case in cases if case["expected"])
        claim = _candidate(source_case["case_id"], "cli candidate")
        raw_receipt = _receipt("qwen3-4b", {source_case["case_id"]: _json_claims(claim)})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(raw_receipt)
            outputs = [root / "bundle.json", root / "key.json", root / "provenance.json"]
            args = [
                "--receipts",
                str(receipt_path),
                "--bundle",
                str(outputs[0]),
                "--answer-key",
                str(outputs[1]),
                "--provenance",
                str(outputs[2]),
                "--seed",
                "5",
            ]

            self.assertEqual(main(args), 0)
            self.assertEqual(receipt_path.read_bytes(), raw_receipt)
            before = [path.read_bytes() for path in outputs]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(args), 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual([path.read_bytes() for path in outputs], before)

            alias_args = [
                "--receipts",
                str(receipt_path),
                "--bundle",
                str(receipt_path),
                "--answer-key",
                str(root / "alias-key.json"),
                "--provenance",
                str(root / "alias-provenance.json"),
                "--seed",
                "5",
            ]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(alias_args), 2)
            self.assertIn("aliases an input", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), raw_receipt)


if __name__ == "__main__":
    unittest.main()
