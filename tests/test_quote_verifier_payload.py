from __future__ import annotations

import builtins
import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import experiments.quote_verifier_payload as quote_verifier_payload


class QuoteVerifierPayloadTests(unittest.TestCase):
    def test_preparation_selects_the_pinned_cited_quote_v2_candidates(self) -> None:
        prepared = quote_verifier_payload.build_preparation()
        payload = prepared["payload"]
        audit = prepared["audit"]

        self.assertEqual(
            [item["case_id"] for item in payload["items"]],
            [f"V4D-{index:02d}" for index in range(1, 10)] + ["V4D-12"],
        )
        self.assertEqual(len(payload["items"]), 10)
        self.assertEqual(sum(item["candidate_count"] for item in payload["items"]), 39)
        self.assertEqual(len(audit["cases"]), 12)
        self.assertEqual(
            audit["input_sha256"],
            "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0",
        )
        self.assertEqual(
            audit["generator_receipt_sha256"],
            "d47debe18192dae62759ed097017d2a9bcef92c148692b2b9a1e41e3805b2ac3",
        )
        by_case = {case["case_id"]: case for case in audit["cases"]}
        self.assertEqual(by_case["V4D-10"]["candidate_indices"], [])
        self.assertEqual(by_case["V4D-10"]["rejected"][0]["source_index"], 0)
        self.assertEqual(by_case["V4D-11"]["candidate_indices"], [])
        self.assertEqual(by_case["V4D-10"]["status"], "no-valid-candidates")
        self.assertEqual(by_case["V4D-11"]["status"], "invalid-json")
        self.assertEqual(by_case["V4D-10"]["pipeline"], "deterministic-only")
        self.assertEqual(by_case["V4D-01"]["pipeline"], "semantic-pipeline")
        self.assertIn("not valid UTF-8 JSON", by_case["V4D-11"]["generator_error"])

    def test_payload_is_minimal_and_preserves_candidate_order_without_labels(self) -> None:
        prepared = quote_verifier_payload.build_preparation()
        payload = prepared["payload"]
        audit = prepared["audit"]

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "mode",
                "input_sha256",
                "generator_receipt_sha256",
                "candidate_sha256",
                "items_sha256",
                "renderer_source_sha256",
                "payload_source_sha256",
                "runner_source_sha256",
                "model",
                "settings",
                "items",
            },
        )
        self.assertEqual(payload["mode"], "semantic-pipeline")
        self.assertEqual(payload["model"]["revision"], "b968826d9c46dd6066d109eabc6255188de91218")
        self.assertEqual(payload["model"]["license"], "apache-2.0")
        self.assertFalse(payload["model"]["gated"])
        self.assertEqual(payload["settings"]["precision"], "bfloat16")
        self.assertEqual(payload["settings"]["max_prompt_tokens"], 4096)
        self.assertEqual(payload["settings"]["max_new_tokens"], 1024)
        self.assertFalse(payload["settings"]["enable_thinking"])
        self.assertTrue(
            all(
                set(item) == {"case_id", "item_id", "prompt", "candidate_count"}
                for item in payload["items"]
            )
        )
        self.assertNotIn("expected", str(payload))
        self.assertNotIn("candidate_indices", str(payload))
        self.assertNotIn("annotation", str(payload).lower())

        case = next(case for case in audit["cases"] if case["case_id"] == "V4D-05")
        item = next(item for item in payload["items"] if item["case_id"] == "V4D-05")
        self.assertEqual(case["candidate_indices"], list(range(6)))
        serialized = json.dumps(case["candidates"], ensure_ascii=False, separators=(",", ":"))
        self.assertTrue(item["prompt"].endswith(f"Candidates: {serialized}"))

    def test_strict_generation_gate_rejects_fences_schema_errors_and_parseable_truncation(
        self,
    ) -> None:
        cases, receipt, _, _ = quote_verifier_payload._load_inputs()
        case = next(case for case in cases if case["case_id"] == "V4D-01")
        row = next(
            row for row in receipt["generated_outputs"] if row.get("item_id") == "V4D-01::quote-v2"
        )

        fenced = quote_verifier_payload._select_case(
            case, dict(row, raw_output=f"```json\n{row['raw_output']}\n```")
        )
        malformed = json.loads(row["raw_output"])
        malformed["claims"][0]["unexpected"] = "field"
        invalid_schema = quote_verifier_payload._select_case(
            case, dict(row, raw_output=json.dumps(malformed))
        )
        truncated = quote_verifier_payload._select_case(
            case, dict(row, hit_max_new_tokens=True, finish_reason="length")
        )

        self.assertEqual(fenced["status"], "invalid-json")
        self.assertEqual(fenced["candidates"], [])
        self.assertEqual(invalid_schema["status"], "invalid-schema")
        self.assertEqual(invalid_schema["candidates"], [])
        self.assertEqual(truncated["status"], "truncated")
        self.assertEqual(truncated["candidates"], [])

    def test_expected_targets_do_not_change_prepared_prompts_or_payload(self) -> None:
        cases, receipt, dev_sha, generator_sha = quote_verifier_payload._load_inputs()
        original = (cases, receipt, dev_sha, generator_sha)
        changed_cases = copy.deepcopy(cases)
        for case in changed_cases:
            case["expected"] = [{"claim": "TARGET-LEAK-SENTINEL"}]

        with patch.object(quote_verifier_payload, "_load_inputs", return_value=original):
            baseline = quote_verifier_payload.build_preparation()
        with patch.object(
            quote_verifier_payload,
            "_load_inputs",
            return_value=(changed_cases, receipt, dev_sha, generator_sha),
        ):
            changed = quote_verifier_payload.build_preparation()

        self.assertEqual(baseline["payload"], changed["payload"])
        self.assertNotIn("TARGET-LEAK-SENTINEL", json.dumps(changed))

    def test_source_hash_drift_changes_payload_hash_and_stale_payload_is_rejected(self) -> None:
        payload = quote_verifier_payload.build_payload()
        original_digest = quote_verifier_payload.payload_sha256(payload)
        source_hash = quote_verifier_payload._source_sha256

        def changed_source(path: Path) -> str:
            if path == Path(quote_verifier_payload.__file__):
                return "0" * 64
            return source_hash(path)

        with patch.object(quote_verifier_payload, "_source_sha256", side_effect=changed_source):
            changed = quote_verifier_payload.build_payload()
        self.assertNotEqual(original_digest, quote_verifier_payload.payload_sha256(changed))

        stale = dict(payload, renderer_source_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "renderer source hash"):
            quote_verifier_payload.validate_payload(stale)

    def test_preparation_does_not_import_modal_or_model_runtimes(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "transformers"}:
                raise AssertionError(f"unexpected runtime import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            prepared = quote_verifier_payload.build_preparation()
        self.assertEqual(len(prepared["payload"]["items"]), 10)

    def test_generator_truncation_metadata_must_be_boolean(self) -> None:
        cases, receipt, _, _ = quote_verifier_payload._load_inputs()
        case = next(case for case in cases if case["case_id"] == "V4D-01")
        row = next(
            row for row in receipt["generated_outputs"] if row.get("item_id") == "V4D-01::quote-v2"
        )
        malformed = dict(row, hit_max_new_tokens=1)

        selected = quote_verifier_payload._select_case(case, malformed)

        self.assertEqual(selected["status"], "invalid-generator-metadata")
        self.assertEqual(selected["candidates"], [])
        self.assertIn("hit_max_new_tokens", selected["generator_error"])


if __name__ == "__main__":
    unittest.main()
