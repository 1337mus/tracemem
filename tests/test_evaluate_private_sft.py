"""Offline scoring tests for the private SFT receipt contract."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from typing import Any
from unittest.mock import patch

from experiments import evaluate_private_sft, private_sft
from tests.test_private_sft import _private_fixture, _synthetic_replay_fixture


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _fixture(
    synthetic_replay_dataset: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    dataset, source_records = _private_fixture(include_test=True)
    payload = private_sft.build_payload(dataset, source_records, synthetic_replay_dataset)
    approved_runner_sha256 = "a" * 64
    labels = [("uncertain", "uncertain"), ("supported", "ineligible")]
    row_count = len(payload["train_rows"])
    optimizer_steps = (row_count + 3) // 4

    def outputs() -> list[dict[str, Any]]:
        rows = []
        for index, (support, eligibility) in enumerate(labels):
            input_token_ids = [101, 200 + index, 102]
            output_token_ids = [301 + index, 302]
            rows.append(
                {
                    "item_id": f"dev_{index:03d}",
                    "raw_output": json.dumps(
                        {"support": support, "eligibility": eligibility}, separators=(",", ":")
                    ),
                    "finish_reason": "eos",
                    "input_tokens": len(input_token_ids),
                    "output_tokens": len(output_token_ids),
                    "input_token_ids": input_token_ids,
                    "input_token_ids_sha256": _canonical_sha(input_token_ids),
                    "output_token_ids": output_token_ids,
                    "output_token_ids_sha256": _canonical_sha(output_token_ids),
                }
            )
        return rows

    receipt = {
        "schema_version": 1,
        "status": "success",
        "mode": "private_sft",
        "training_performed": True,
        "payload_sha256": _canonical_sha(payload),
        "runner_sha256": approved_runner_sha256,
        "model": copy.deepcopy(payload["model"]),
        "settings": copy.deepcopy(payload["settings"]),
        "prior_adapter": copy.deepcopy(payload["prior_adapter"]),
        "source_sha256": copy.deepcopy(payload["source_sha256"]),
        "provenance": copy.deepcopy(payload["provenance"]),
        "training": {
            "row_count": row_count,
            "microsteps": row_count,
            "optimizer_steps": optimizer_steps,
            "input_tokens": 10,
            "supervised_tokens": 4,
            "mean_loss": 0.3,
            "optimizer_step_losses": [0.3] * optimizer_steps,
            "trainable_parameters": 12,
            "adapter_hash_before_training": "b" * 64,
            "adapter_hash_after_training": "c" * 64,
            "adapter_changed": True,
            "seconds": 3.5,
        },
        "artifact_manifest": {
            "volume_name": payload["output_volume"]["name"],
            "volume_mount": payload["output_volume"]["mount"],
            "relative_path": f"runs/{_canonical_sha(payload)}/adapter",
            "artifact_sha256": {
                "adapter_config.json": "d" * 64,
                "adapter_model.safetensors": "e" * 64,
            },
        },
        "base_outputs": outputs(),
        "prior_outputs": outputs(),
        "trained_outputs": outputs(),
    }
    return dataset, source_records, payload, receipt, approved_runner_sha256


class EvaluatePrivateSftTests(unittest.TestCase):
    @staticmethod
    def _explicit_schedule_fixture() -> tuple[
        list[dict[str, str]], dict[str, object], dict[str, object]
    ]:
        supports = ("supported", "unsupported", "uncertain")
        rows = [
            {
                "prompt": f"fictional explicit schedule prompt {index}",
                "completion": json.dumps(
                    {"support": support, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, support in enumerate(supports)
        ]
        indices = [0] * 128 + [1] * 64 + [2] * 64
        plan: dict[str, object] = {
            "strategy": "explicit-exposures-v1",
            "seed": 17,
            "row_indices": indices,
        }
        sampling = {
            "row_index_sha256": _canonical_sha(indices),
            "exposure_counts": {"supported": 128, "unsupported": 64, "uncertain": 64},
            "unique_source_counts": {"supported": 1, "unsupported": 1, "uncertain": 1},
        }
        training: dict[str, object] = {
            "row_count": len(rows),
            "microsteps": len(indices),
            "optimizer_steps": 64,
            "input_tokens": 767,
            "supervised_tokens": 512,
            "mean_loss": 0.3,
            "optimizer_step_losses": [0.3] * 64,
            "trainable_parameters": 12,
            "adapter_hash_before_training": "b" * 64,
            "adapter_hash_after_training": "c" * 64,
            "adapter_changed": True,
            "seconds": 3.5,
            "sampling": sampling,
        }
        return rows, plan, training

    @staticmethod
    def _balanced_fixture() -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str
    ]:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        supports = ["supported"] * 117 + ["unsupported"] * 116 + ["uncertain"] * 23
        rows = [
            {
                "prompt": f"fictional balanced prompt {index}",
                "completion": json.dumps(
                    {"support": support, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, support in enumerate(supports)
        ]
        payload["train_rows"] = rows
        payload["sampling_plan"] = private_sft.build_sampling_plan(rows, seed=17)
        payload_sha = _canonical_sha(payload)
        receipt["payload_sha256"] = payload_sha
        receipt["artifact_manifest"]["relative_path"] = f"runs/{payload_sha}/adapter"
        receipt["training"].update(
            row_count=256,
            microsteps=256,
            optimizer_steps=64,
            optimizer_step_losses=[0.3] * 64,
            input_tokens=1200,
            supervised_tokens=600,
            sampling=evaluate_private_sft._expected_sampling_metrics(
                rows, payload["sampling_plan"]
            ),
        )
        return dataset, source_records, payload, receipt, runner_sha

    def test_explicit_schedule_metrics_and_training_use_256_exposures(self) -> None:
        rows, plan, training = self._explicit_schedule_fixture()

        expected = evaluate_private_sft._expected_sampling_metrics(rows, plan)
        validated = evaluate_private_sft._validate_training(
            training, len(rows), 4, sampling_plan=plan, train_rows=rows
        )

        self.assertEqual(expected, training["sampling"])
        self.assertEqual(validated["row_count"], 3)
        self.assertEqual(validated["microsteps"], 256)
        self.assertEqual(validated["optimizer_steps"], 64)

    def test_scheduled_training_row_count_must_match_source_pool(self) -> None:
        rows, plan, training = self._explicit_schedule_fixture()
        training["row_count"] = 10

        with self.assertRaisesRegex(ValueError, "row_count"):
            evaluate_private_sft._validate_training(
                training, 10, 4, sampling_plan=plan, train_rows=rows
            )

    def test_explicit_schedule_accepts_one_to_256_unique_sources(self) -> None:
        for pool_size in (1, 256):
            with self.subTest(pool_size=pool_size):
                supports = ("supported", "unsupported", "uncertain")
                rows = [
                    {
                        "prompt": f"fictional bounded-pool prompt {index}",
                        "completion": json.dumps(
                            {
                                "support": supports[index % len(supports)],
                                "eligibility": "eligible",
                            },
                            separators=(",", ":"),
                        ),
                    }
                    for index in range(pool_size)
                ]
                indices = [index % pool_size for index in range(256)]
                plan = {
                    "strategy": "explicit-exposures-v1",
                    "seed": 17,
                    "row_indices": indices,
                }

                metrics = evaluate_private_sft._expected_sampling_metrics(rows, plan)

                self.assertEqual(sum(metrics["exposure_counts"].values()), 256)
                self.assertEqual(sum(metrics["unique_source_counts"].values()), pool_size)

    def test_explicit_schedule_rejects_malformed_plan_before_training_steps(self) -> None:
        mutations = (
            ("short_indices", lambda plan: plan["row_indices"].pop()),
            ("long_indices", lambda plan: plan["row_indices"].append(0)),
            ("negative_index", lambda plan: plan["row_indices"].__setitem__(0, -1)),
            ("out_of_range_index", lambda plan: plan["row_indices"].__setitem__(0, 3)),
            ("boolean_index", lambda plan: plan["row_indices"].__setitem__(0, True)),
            ("string_index", lambda plan: plan["row_indices"].__setitem__(0, "0")),
            ("wrong_seed", lambda plan: plan.__setitem__("seed", 18)),
            ("float_seed", lambda plan: plan.__setitem__("seed", 17.0)),
            ("unknown_strategy", lambda plan: plan.__setitem__("strategy", "explicit-v2")),
            ("extra_plan_key", lambda plan: plan.__setitem__("unexpected", True)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                rows, plan, training = self._explicit_schedule_fixture()
                training["microsteps"] = 255
                mutate(plan)

                with self.assertRaisesRegex(ValueError, "payload sampling"):
                    evaluate_private_sft._validate_training(
                        training, len(rows), 4, sampling_plan=plan, train_rows=rows
                    )

        rows, plan, training = self._explicit_schedule_fixture()
        rows[1]["prompt"] = rows[0]["prompt"]
        training["microsteps"] = 255
        with self.assertRaisesRegex(ValueError, "payload sampling rows"):
            evaluate_private_sft._validate_training(
                training, len(rows), 4, sampling_plan=plan, train_rows=rows
            )

    def test_explicit_schedule_rejects_tampered_receipt_schedule_metrics_and_steps(self) -> None:
        mutations = (
            (
                "schedule_index",
                lambda plan, training: plan["row_indices"].__setitem__(0, 2),
                "row-index hash",
            ),
            ("microsteps", lambda plan, training: training.__setitem__("microsteps", 255), "steps"),
            (
                "optimizer_steps",
                lambda plan, training: training.__setitem__("optimizer_steps", 63),
                "steps",
            ),
            (
                "hash",
                lambda plan, training: training["sampling"].__setitem__(
                    "row_index_sha256", "0" * 64
                ),
                "row-index hash",
            ),
            (
                "exposure_counts",
                lambda plan, training: training["sampling"]["exposure_counts"].update(
                    supported=127
                ),
                "exposure_counts",
            ),
            (
                "unique_source_counts",
                lambda plan, training: training["sampling"]["unique_source_counts"].update(
                    supported=2
                ),
                "unique_source_counts",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                rows, plan, training = self._explicit_schedule_fixture()
                mutate(plan, training)

                with self.assertRaisesRegex(ValueError, message):
                    evaluate_private_sft._validate_training(
                        training, len(rows), 4, sampling_plan=plan, train_rows=rows
                    )

    def test_scores_three_arms_with_aggregate_only_axis_metrics(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()

        report = evaluate_private_sft.evaluate_report(
            dataset,
            source_records,
            payload,
            receipt,
            approved_runner_sha256=runner_sha,
        )

        self.assertEqual(set(report["arms"]), {"base", "prior", "trained"})
        self.assertEqual(report["scope"]["split"], "dev")
        self.assertEqual(report["scope"]["candidate_count"], 2)
        self.assertEqual(report["scope"]["case_count"], 1)
        self.assertEqual(
            report["arms"]["trained"]["summary"]["support"]["confusion"]["uncertain"]["uncertain"],
            1,
        )
        serialized = json.dumps(report)
        for forbidden in (
            "TEST_TEXT_SENTINEL",
            "TEST_LABEL_SENTINEL",
            "RATIONALE_SENTINEL",
            "REVIEWER_SENTINEL",
            "case-dev-a",
            "dev_000",
            "train-a",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_rebuilds_payload_with_explicit_execution_id(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        payload["execution_id"] = "retry-2"
        payload_sha = _canonical_sha(payload)
        receipt["payload_sha256"] = payload_sha
        receipt["artifact_manifest"]["relative_path"] = f"runs/{payload_sha}/adapter"

        report = evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        self.assertEqual(report["payload_sha256"], payload_sha)

    def test_accepts_balanced_schedule_metrics_and_returns_them_in_report(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = self._balanced_fixture()

        with patch.object(private_sft, "build_payload", return_value=payload) as build_payload:
            report = evaluate_private_sft.evaluate_report(
                dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
            )

        build_payload.assert_called_once_with(
            dataset,
            source_records,
            None,
            sampling_strategy="support-balanced-v1",
        )
        self.assertEqual(report["training"]["sampling"], receipt["training"]["sampling"])

    def test_rejects_tampered_balanced_plan_and_receipt_sampling_metrics(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = self._balanced_fixture()
        valid_payload = copy.deepcopy(payload)
        tampered_payload = copy.deepcopy(payload)
        tampered_payload["sampling_plan"]["row_indices"][0] = (
            tampered_payload["sampling_plan"]["row_indices"][0] + 1
        ) % 256
        with patch.object(private_sft, "build_payload", return_value=valid_payload):
            with self.assertRaisesRegex(ValueError, "exact exporter output"):
                evaluate_private_sft.evaluate_report(
                    dataset,
                    source_records,
                    tampered_payload,
                    receipt,
                    approved_runner_sha256=runner_sha,
                )

        for mutate, message in (
            (lambda sampling: sampling.update(row_index_sha256="0" * 64), "row-index hash"),
            (
                lambda sampling: sampling["exposure_counts"].update(supported=84),
                "exposure_counts",
            ),
            (
                lambda sampling: sampling["unique_source_counts"].update(uncertain=True),
                "unique_source_counts",
            ),
        ):
            with self.subTest(message=message):
                bad_receipt = copy.deepcopy(receipt)
                mutate(bad_receipt["training"]["sampling"])
                with patch.object(private_sft, "build_payload", return_value=payload):
                    with self.assertRaisesRegex(ValueError, message):
                        evaluate_private_sft.evaluate_report(
                            dataset,
                            source_records,
                            payload,
                            bad_receipt,
                            approved_runner_sha256=runner_sha,
                        )

    def test_normalizes_only_verified_terminal_eos_and_preserves_raw_validity(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["tokenizer"] = {"eos_token_id": 151645}
        for arm in ("base_outputs", "prior_outputs", "trained_outputs"):
            for row in receipt[arm]:
                row["raw_output"] += "<|im_end|>"
                row["output_token_ids"].append(151645)
                row["output_tokens"] += 1
                row["output_token_ids_sha256"] = _canonical_sha(row["output_token_ids"])
        original_receipt = copy.deepcopy(receipt)
        original_receipt_sha = _canonical_sha(receipt)

        report = evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        self.assertEqual(
            report["scoring_convention"], "content_after_verified_terminal_eos_removal"
        )
        self.assertEqual(report["normalization"]["terminal_eos_markers_removed"], 6)
        self.assertEqual(report["normalization"]["by_arm"]["trained"], 2)
        self.assertEqual(report["raw_validity"]["trained"]["invalid"], 2)
        self.assertEqual(report["raw_validity"]["trained"]["valid"], 0)
        self.assertEqual(report["arms"]["trained"]["summary"]["validity"]["valid"], 2)
        self.assertEqual(report["receipt_sha256"], original_receipt_sha)
        self.assertEqual(receipt, original_receipt)

    def test_rejects_marker_with_spoofed_token_id_or_tokenizer_metadata(self) -> None:
        for bad_id, bad_metadata in ((151644, 151645), (151645, 151644)):
            with self.subTest(token_id=bad_id, metadata=bad_metadata):
                dataset, source_records, payload, receipt, runner_sha = _fixture()
                receipt["tokenizer"] = {"eos_token_id": bad_metadata}
                row = receipt["base_outputs"][0]
                row["raw_output"] += "<|im_end|>"
                row["output_token_ids"].append(bad_id)
                row["output_tokens"] += 1
                row["output_token_ids_sha256"] = _canonical_sha(row["output_token_ids"])

                with self.assertRaisesRegex(ValueError, "terminal EOS marker"):
                    evaluate_private_sft.evaluate_report(
                        dataset,
                        source_records,
                        payload,
                        receipt,
                        approved_runner_sha256=runner_sha,
                    )

    def test_trailing_prose_after_marker_is_not_normalized(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["tokenizer"] = {"eos_token_id": 151645}
        row = receipt["trained_outputs"][0]
        row["raw_output"] += "<|im_end|> trailing prose"
        row["output_token_ids"].append(151645)
        row["output_tokens"] += 1
        row["output_token_ids_sha256"] = _canonical_sha(row["output_token_ids"])

        report = evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        self.assertEqual(report["normalization"]["terminal_eos_markers_removed"], 0)
        self.assertEqual(report["raw_validity"]["trained"]["invalid"], 1)
        self.assertEqual(report["arms"]["trained"]["summary"]["validity"]["invalid"], 1)

    def test_accepts_runtime_metadata_and_hashes_the_full_receipt(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt.update(
            versions={"python": "3.12", "torch": "fictional"},
            tokenizer={"name": "fictional-tokenizer"},
            timings={"total_seconds": 4.5},
            hardware={"accelerator": "fictional"},
            modal_ids={"app": "fictional-app"},
            client_time="2026-10-16T12:00:00Z",
            launch_checks={"passed": True},
        )

        report = evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        self.assertEqual(report["receipt_sha256"], _canonical_sha(receipt))
        self.assertNotIn("fictional-app", json.dumps(report))

    def test_requires_the_replay_dataset_used_to_build_payload(self) -> None:
        replay = _synthetic_replay_fixture()
        dataset, source_records, payload, receipt, runner_sha = _fixture(replay)
        evaluate_private_sft.evaluate_report(
            dataset,
            source_records,
            payload,
            receipt,
            approved_runner_sha256=runner_sha,
            synthetic_replay_dataset=replay,
        )

        with self.assertRaisesRegex(ValueError, "payload does not match exact exporter output"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

        changed_replay = copy.deepcopy(replay)
        changed_replay["cases"][0]["messages"][0]["text"] += " changed"
        with self.assertRaisesRegex(ValueError, "payload does not match exact exporter output"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
                synthetic_replay_dataset=changed_replay,
            )

    def test_rejects_impossible_negative_mean_loss(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["training"]["mean_loss"] = -0.01

        with self.assertRaisesRegex(ValueError, "mean_loss"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

    def test_requires_one_epoch_step_counts_and_changed_adapter(self) -> None:
        mutations = (
            lambda training: training.update(microsteps=training["microsteps"] - 1),
            lambda training: training.update(
                optimizer_steps=training["optimizer_steps"] + 1,
                optimizer_step_losses=training["optimizer_step_losses"] + [0.1],
            ),
            lambda training: training.update(
                adapter_hash_after_training=training["adapter_hash_before_training"],
                adapter_changed=False,
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                dataset, source_records, payload, receipt, runner_sha = _fixture()
                mutate(receipt["training"])
                with self.assertRaisesRegex(ValueError, "steps|adapter"):
                    evaluate_private_sft.evaluate_report(
                        dataset,
                        source_records,
                        payload,
                        receipt,
                        approved_runner_sha256=runner_sha,
                    )

    def test_requires_input_token_count_on_every_output(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        del receipt["base_outputs"][0]["input_tokens"]

        with self.assertRaisesRegex(ValueError, "output row has an invalid shape"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

    def test_rejects_prompt_and_payload_hash_mismatches(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        payload["dev_prompts"][0] += " changed"
        receipt["payload_sha256"] = _canonical_sha(payload)

        with self.assertRaisesRegex(ValueError, "exact exporter output"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["payload_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "payload SHA-256"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

    def test_exact_payload_comparison_rejects_bool_integer_equivalence(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        payload["schema_version"] = True
        receipt["payload_sha256"] = _canonical_sha(payload)

        with self.assertRaisesRegex(ValueError, "exact exporter output"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

    def test_rejects_unapproved_runner_digest(self) -> None:
        dataset, source_records, payload, receipt, _ = _fixture()

        with self.assertRaisesRegex(ValueError, "approved runner"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256="f" * 64,
            )

    def test_rejects_missing_or_duplicate_arm_outputs(self) -> None:
        for field, mutate in (
            ("base_outputs", lambda rows: rows.pop()),
            ("prior_outputs", lambda rows: rows.append(copy.deepcopy(rows[0]))),
            ("trained_outputs", lambda rows: rows.append({**rows[0], "item_id": "dev_999"})),
        ):
            with self.subTest(field=field):
                dataset, source_records, payload, receipt, runner_sha = _fixture()
                mutate(receipt[field])
                with self.assertRaisesRegex(ValueError, "output IDs must match"):
                    evaluate_private_sft.evaluate_report(
                        dataset,
                        source_records,
                        payload,
                        receipt,
                        approved_runner_sha256=runner_sha,
                    )

    def test_malformed_and_truncated_outputs_count_as_invalid(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["trained_outputs"][0]["raw_output"] = "not JSON"
        receipt["trained_outputs"][1]["finish_reason"] = "length"

        report = evaluate_private_sft.evaluate_report(
            dataset,
            source_records,
            payload,
            receipt,
            approved_runner_sha256=runner_sha,
        )

        validity = report["arms"]["trained"]["summary"]["validity"]
        self.assertEqual(validity["invalid"], 2)
        self.assertEqual(validity["malformed"], 1)
        self.assertEqual(validity["truncated"], 1)

    def test_empty_output_is_scored_invalid(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        empty = receipt["trained_outputs"][0]
        empty["raw_output"] = ""
        empty["finish_reason"] = "empty"
        empty["output_token_ids"] = []
        empty["output_tokens"] = 0
        empty["output_token_ids_sha256"] = _canonical_sha([])

        report = evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        validity = report["arms"]["trained"]["summary"]["validity"]
        self.assertEqual(validity["invalid"], 1)
        self.assertEqual(validity["finish_reasons"]["empty"], 1)

    def test_manifest_allows_readme_and_rejects_path_outside_payload_run(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["artifact_manifest"]["artifact_sha256"]["README.md"] = "f" * 64
        evaluate_private_sft.evaluate_report(
            dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
        )

        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["artifact_manifest"]["relative_path"] = "runs/other/adapter"
        with self.assertRaisesRegex(ValueError, "path does not match payload"):
            evaluate_private_sft.evaluate_report(
                dataset, source_records, payload, receipt, approved_runner_sha256=runner_sha
            )

    def test_rejects_bad_token_hashes_and_cross_arm_input_mismatch(self) -> None:
        dataset, source_records, payload, receipt, runner_sha = _fixture()
        receipt["base_outputs"][0]["output_token_ids_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "output_token_ids_sha256"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )

        dataset, source_records, payload, receipt, runner_sha = _fixture()
        changed = [101, 999, 102]
        receipt["prior_outputs"][0]["input_token_ids"] = changed
        receipt["prior_outputs"][0]["input_tokens"] = len(changed)
        receipt["prior_outputs"][0]["input_token_ids_sha256"] = _canonical_sha(changed)
        with self.assertRaisesRegex(ValueError, "input token IDs differ"):
            evaluate_private_sft.evaluate_report(
                dataset,
                source_records,
                payload,
                receipt,
                approved_runner_sha256=runner_sha,
            )
