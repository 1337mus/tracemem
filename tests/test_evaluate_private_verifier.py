"""Offline scoring tests for the private verifier receipt contract."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from typing import Any

from experiments import evaluate_private_verifier


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    text = "Fictional demo: a cobalt cup sits beside a paper star."
    source = {
        "record_id": "record_fiction",
        "revision": "revision_1",
        "session_id": "session_fiction",
        "message_id": "message_fiction",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_locator": "fiction/record.jsonl",
        "line_number": 1,
        "role": "user",
        "recorded_at": "2026-10-01T12:00:00Z",
        "content_kind": "conversation",
    }
    message = {
        key: source[key]
        for key in (
            "message_id",
            "revision",
            "session_id",
            "text",
            "text_sha256",
            "source_locator",
            "line_number",
            "role",
            "recorded_at",
            "content_kind",
        )
    }
    message["source_record_id"] = source["record_id"]
    quote = "Fictional demo"
    labels = (
        ("supported", "eligible"),
        ("unsupported", "eligible"),
        ("supported", "ineligible"),
        ("uncertain", "uncertain"),
    )
    candidates = []
    for index, (support, eligibility) in enumerate(labels, start=1):
        candidates.append(
            {
                "candidate_id": f"candidate_{index}",
                "claim": f"Fictional candidate {index}.",
                "message_id": source["message_id"],
                "source_record_id": source["record_id"],
                "quote": quote,
                "char_start": 0,
                "char_end": len(quote),
                "valid_from": None,
                "valid_to": None,
                "support": support,
                "eligibility": eligibility,
                "support_reason": "Fictional support rationale.",
                "eligibility_reason": "Fictional eligibility rationale.",
                "label_status": "agent_provisional",
                "annotator": {"reviewer_id": "agent_fiction", "reviewer_type": "agent"},
                "reviews": [],
            }
        )
    case = {
        "case_id": "case_fiction",
        "session_id": source["session_id"],
        "split": "dev",
        "window_id": "window_fiction",
        "privacy_clearance": {
            "status": "cleared",
            "reviewer_id": "agent_clearance_fiction",
            "reviewer_type": "agent",
            "reviewed_at": "2026-10-02T12:00:00Z",
        },
        "messages": [message],
        "candidates": candidates,
    }
    withheld = copy.deepcopy(case)
    withheld["case_id"] = "case_withheld"
    withheld["window_id"] = "window_withheld"
    withheld["privacy_clearance"] = {
        "status": "pending",
        "reviewer_id": "agent_clearance_fiction",
        "reviewer_type": "agent",
        "reviewed_at": None,
    }
    withheld["candidates"] = [
        {**candidate, "candidate_id": f"withheld_{index}"}
        for index, candidate in enumerate(candidates, start=1)
    ]
    dataset = {
        "version": 1,
        "license": "private-not-for-redistribution",
        "annotation_status": "agent_provisional",
        "provenance": "private_source",
        "scope_policy": "personal-memory-research-v1",
        "cases": [case, withheld],
    }
    source_records = {source["record_id"]: source}

    from training.private_verifier_data import render_private_eval_prompts

    prompts = render_private_eval_prompts(dataset, source_records, "dev")
    items = [
        {"item_id": f"opaque_{index}", "prompt": prompt} for index, prompt in enumerate(prompts)
    ]
    prior_adapter = {
        "volume_name": "fictional-volume",
        "volume_mount": "/mnt/fictional",
        "relative_path": "prior/adapter",
        "safetensors_sha256": "d" * 64,
        "config_sha256": "e" * 64,
    }
    continued_adapter = {
        "volume_name": "fictional-volume",
        "volume_mount": "/mnt/fictional",
        "relative_path": "continued/adapter",
        "artifact_sha256": {"adapter_config.json": "f" * 64, "adapter_model.safetensors": "0" * 64},
    }
    model = {"model_id": "fictional-model", "revision": "fictional-revision"}
    settings = {"do_sample": False, "max_new_tokens": 128}
    provenance = {"source_content_in_payload": "minimized_prompts_only"}
    candidate_manifest = {
        "training_payload_sha256": "1" * 64,
        "training_receipt_sha256": "2" * 64,
        "adapter_relative_path": "continued/adapter",
        "adapter_artifact_sha256": continued_adapter["artifact_sha256"],
    }
    payload = {
        "schema_version": 1,
        "mode": "private_inference_only",
        "provenance": provenance,
        "runner_sha256": "a" * 64,
        "model": model,
        "settings": settings,
        "candidate_manifest": candidate_manifest,
        "prior_adapter": prior_adapter,
        "continued_adapter": continued_adapter,
        "items": items,
        "training_performed": False,
    }
    truths = {item["item_id"]: pair for item, pair in zip(items, labels, strict=True)}

    def outputs(overrides: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        rows = []
        for item in items:
            support, eligibility = truths[item["item_id"]]
            row = {
                "item_id": item["item_id"],
                "raw_output": json.dumps({"support": support, "eligibility": eligibility}),
                "finish_reason": "eos",
                "output_tokens": 4,
            }
            row.update((overrides or {}).get(item["item_id"], {}))
            rows.append(row)
        return rows

    receipt = {
        "schema_version": 1,
        "status": "success",
        "mode": "private_inference_only",
        "payload_sha256": _canonical_sha(payload),
        "runner_sha256": payload["runner_sha256"],
        "provenance": provenance,
        "candidate_manifest": candidate_manifest,
        "model": model,
        "settings": settings,
        "prior_adapter": prior_adapter,
        "continued_adapter": continued_adapter,
        "adapter_artifact_sha256": {
            "prior": {
                "adapter_config.json": prior_adapter["config_sha256"],
                "adapter_model.safetensors": prior_adapter["safetensors_sha256"],
            },
            "continued": continued_adapter["artifact_sha256"],
        },
        "training_performed": False,
        "base_outputs": outputs(),
        "prior_outputs": outputs(
            {"opaque_1": {"raw_output": "not json"}, "opaque_2": {"finish_reason": "length"}}
        ),
        "continued_outputs": outputs(),
    }
    return dataset, source_records, payload, receipt


class EvaluatePrivateVerifierTests(unittest.TestCase):
    def test_scores_three_arms_and_reports_single_session_agent_only_scope(self) -> None:
        dataset, source_records, payload, receipt = _fixture()
        receipt["base_outputs"].reverse()

        report = evaluate_private_verifier.evaluate_report(
            dataset, source_records, payload, receipt
        )

        self.assertEqual(set(report["arms"]), {"base", "prior", "continued"})
        self.assertEqual(report["scope"]["split"], "dev")
        self.assertEqual(report["scope"]["session_count"], 1)
        self.assertFalse(report["scope"]["held_out"])
        self.assertEqual(report["scope"]["label_basis"], "agent-only provisional annotations")
        self.assertIn("one-session dev challenge", report["scope"]["interpretation"])
        self.assertIn("not held out", report["scope"]["interpretation"])
        base = report["arms"]["base"]
        self.assertEqual(base["summary"]["candidate_count"], 4)
        self.assertEqual(base["summary"]["support"]["confusion"]["supported"]["supported"], 2)
        self.assertEqual(base["summary"]["support"]["confusion"]["unsupported"]["unsupported"], 1)
        self.assertEqual(base["summary"]["support"]["confusion"]["uncertain"]["uncertain"], 1)
        self.assertEqual(base["summary"]["eligibility"]["confusion"]["eligible"]["eligible"], 2)
        self.assertEqual(base["summary"]["eligibility"]["confusion"]["ineligible"]["ineligible"], 1)
        self.assertEqual(base["summary"]["eligibility"]["confusion"]["uncertain"]["uncertain"], 1)
        self.assertEqual(base["joint_acceptance"]["useful_supported_eligible"]["accepted"], 1)
        self.assertEqual(
            base["joint_acceptance"]["unsupported_eligible_false_accepts"]["accepted"], 0
        )
        serialized = json.dumps(report)
        for forbidden in (
            "Fictional demo",
            "Fictional candidate",
            "fiction/record.jsonl",
            "private_source",
            "opaque_1",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_malformed_and_truncated_outputs_are_counted_as_invalid(self) -> None:
        dataset, source_records, payload, receipt = _fixture()

        report = evaluate_private_verifier.evaluate_report(
            dataset, source_records, payload, receipt
        )
        prior = report["arms"]["prior"]["summary"]

        self.assertEqual(prior["validity"]["invalid"], 2)
        self.assertEqual(prior["validity"]["malformed"], 1)
        self.assertEqual(prior["validity"]["truncated"], 1)
        self.assertEqual(prior["support"]["confusion"]["supported"]["invalid"], 1)
        self.assertEqual(prior["support"]["confusion"]["unsupported"]["invalid"], 1)

    def test_rejects_changed_rendered_prompt_even_when_receipt_hash_is_updated(self) -> None:
        dataset, source_records, payload, receipt = _fixture()
        payload["items"][0]["prompt"] += " changed"
        receipt["payload_sha256"] = _canonical_sha(payload)

        with self.assertRaisesRegex(ValueError, "rendered dev prompts"):
            evaluate_private_verifier.evaluate_report(dataset, source_records, payload, receipt)

    def test_rejects_receipt_hash_model_settings_and_adapter_mismatches(self) -> None:
        mutations = (
            ("payload_sha256", lambda receipt: receipt.update(payload_sha256="0" * 64)),
            ("runner_sha256", lambda receipt: receipt.update(runner_sha256="b" * 64)),
            ("model", lambda receipt: receipt.update(model={"model_id": "other"})),
            ("settings", lambda receipt: receipt.update(settings={"max_new_tokens": 64})),
            (
                "prior_adapter",
                lambda receipt: receipt.update(prior_adapter={"relative_path": "other"}),
            ),
            (
                "continued_adapter",
                lambda receipt: receipt.update(continued_adapter={"relative_path": "other"}),
            ),
            (
                "adapter_artifact_sha256",
                lambda receipt: receipt["adapter_artifact_sha256"]["continued"].update(
                    **{"adapter_model.safetensors": "9" * 64}
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                dataset, source_records, payload, receipt = _fixture()
                mutate(receipt)
                with self.assertRaises(ValueError):
                    evaluate_private_verifier.evaluate_report(
                        dataset, source_records, payload, receipt
                    )

    def test_rejects_missing_duplicate_and_extra_output_ids(self) -> None:
        mutations = (
            ("missing", lambda outputs: outputs.pop()),
            ("duplicate", lambda outputs: outputs.append(copy.deepcopy(outputs[0]))),
            (
                "extra",
                lambda outputs: outputs.append(
                    {**copy.deepcopy(outputs[0]), "item_id": "opaque_extra"}
                ),
            ),
        )
        for name, mutate in mutations:
            for arm, output_field in (
                ("base", "base_outputs"),
                ("prior", "prior_outputs"),
                ("continued", "continued_outputs"),
            ):
                with self.subTest(name=name, arm=arm):
                    dataset, source_records, payload, receipt = _fixture()
                    mutate(receipt[output_field])
                    with self.assertRaisesRegex(ValueError, "output IDs"):
                        evaluate_private_verifier.evaluate_report(
                            dataset, source_records, payload, receipt
                        )


if __name__ == "__main__":
    unittest.main()
