from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _payload(split: str, family_count: int, variants_per_family: int) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for family in range(family_count):
        for variant in range(variants_per_family):
            case_id = f"{split}-case-{family:02d}-{variant:02d}"
            message_id = f"{split}-message-{family:02d}-{variant:02d}"
            text = f"Fictional source line for {case_id}."
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "scenario_group": f"{split}-scenario-{family:02d}",
                    "template_group": f"{split}-template-{family:02d}",
                    "entity_ids": [f"{split}-entity-{family:02d}-{variant:02d}"],
                    "tags": ["synthetic"],
                    "messages": [
                        {
                            "session_id": f"{split}-session-{family:02d}-{variant:02d}",
                            "message_id": message_id,
                            "role": "user",
                            "text": text,
                            "recorded_at": "2026-01-02T10:00:00Z",
                            "content_kind": "conversation",
                        }
                    ],
                    "expected": [
                        {
                            "claim": f"The user mentioned {case_id}.",
                            "message_id": message_id,
                            "quote": text,
                            "valid_from": None,
                            "valid_to": None,
                        }
                    ],
                }
            )
    return {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
    }


def _build_from_payloads(
    train_payload: dict[str, object], dev_payload: dict[str, object]
) -> dict[str, object]:
    from experiments import modal_sft

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        train_path = root / "train.json"
        dev_path = root / "dev.json"
        train_path.write_text(json.dumps(train_payload), encoding="utf-8")
        dev_path.write_text(json.dumps(dev_payload), encoding="utf-8")
        with (
            patch.object(modal_sft, "_QUOTE_TRAIN_PATH", train_path, create=True),
            patch.object(modal_sft, "_QUOTE_DEV_PATH", dev_path, create=True),
        ):
            return modal_sft.build_payload(profile="quote-v2")


class QuoteSftPayloadTests(unittest.TestCase):
    def test_quote_v2_payload_is_fixed_profile_and_pins_only_train_dev_sources(self) -> None:
        from experiments import modal_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.json"
            dev_path = root / "dev.json"
            train_bytes = json.dumps(_payload("train", 48, 16)).encode("utf-8")
            dev_bytes = json.dumps(_payload("dev", 24, 1)).encode("utf-8")
            train_path.write_bytes(train_bytes)
            dev_path.write_bytes(dev_bytes)
            with (
                patch.object(modal_sft, "_QUOTE_TRAIN_PATH", train_path, create=True),
                patch.object(modal_sft, "_QUOTE_DEV_PATH", dev_path, create=True),
            ):
                job = modal_sft.build_payload(profile="quote-v2")
                from experiments import sft_launch

                self.assertEqual(sft_launch._validate_approved_job(job, job["job_sha256"]), job)

        request = job["request"]
        self.assertEqual(
            set(request), {"mode", "profile", "train_rows", "dev_prompts", "remote_run_dir"}
        )
        self.assertEqual(request["profile"], "quote-v2")
        self.assertEqual(len(request["train_rows"]), 768)
        self.assertEqual(len(request["dev_prompts"]), 24)
        target = json.loads(request["train_rows"][0]["messages"][1]["content"])
        self.assertEqual(set(target), {"claims"})
        self.assertEqual(
            set(target["claims"][0]), {"claim", "message_id", "quote", "valid_from", "valid_to"}
        )
        self.assertEqual(
            [row["case_id"] for row in request["dev_prompts"]],
            [case["case_id"] for case in _payload("dev", 24, 1)["cases"]],
        )
        from training.train_sft import select_training_rows

        source_rows = [
            {"case_id": f"train-case-{family:02d}-{variant:02d}"}
            for family in range(48)
            for variant in range(16)
        ]
        self.assertEqual(
            [row["case_id"] for row in request["train_rows"]],
            [row["case_id"] for row in select_training_rows(source_rows, max_rows=768, seed=17)],
        )
        self.assertEqual(
            job["source_sha256"],
            {
                "train": hashlib.sha256(train_bytes).hexdigest(),
                "dev": hashlib.sha256(dev_bytes).hexdigest(),
            },
        )
        expected_modules = {
            "training/prepare_sft.py",
            "training/train_sft.py",
            "training/remote_sft.py",
            "experiments/sft_launch.py",
            "experiments/scenario_payload.py",
            "src/tracemem/challenge.py",
            "src/tracemem/prompts.py",
            "training/quote_data.py",
            "training/build_quote_data.py",
        }
        self.assertEqual(set(job["module_sha256"]), expected_modules)
        for relative_path, digest in job["module_sha256"].items():
            self.assertEqual(
                digest,
                hashlib.sha256((modal_sft._ROOT / relative_path).read_bytes()).hexdigest(),
            )

    def test_quote_v2_run_directory_is_bound_to_its_profile_volume(self) -> None:
        from experiments import modal_sft, sft_launch

        job = modal_sft.build_payload("train", run_id="quote-run")
        job["request"]["profile"] = "quote-v2"
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v2/adapters/quote-run"
        job["job_sha256"] = sft_launch.canonical_job_sha256(job)

        self.assertEqual(sft_launch._run_id_from_job(job), "quote-run")

    def test_quote_v2_launcher_uses_its_own_volume_app_and_two_worker_files(self) -> None:
        from experiments import modal_sft, sft_launch

        job = modal_sft.build_payload("train", run_id="quote-run")
        job["request"]["profile"] = "quote-v2"
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v2/adapters/quote-run"

        image = MagicMock()
        image.pip_install.return_value = image
        image.add_local_file.return_value = image
        app = MagicMock()
        app.function.return_value = lambda function: function
        volume = object()
        modal = MagicMock()
        modal.Image.debian_slim.return_value = image
        modal.Volume.from_name.return_value = volume
        modal.App.return_value = app

        with patch.dict("sys.modules", {"modal": modal}):
            built_app, _function = sft_launch._build_modal_function(job)

        self.assertIs(built_app, app)
        modal.Volume.from_name.assert_called_once_with(
            "tracemem-sft-quote-v2", create_if_missing=True
        )
        modal.App.assert_called_once_with("tracemem-sft-quote-v2")
        self.assertEqual(image.add_local_file.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in image.add_local_file.call_args_list},
            {
                modal_sft._ROOT / "training/train_sft.py",
                modal_sft._ROOT / "training/remote_sft.py",
            },
        )
        self.assertEqual(
            {call.kwargs["remote_path"] for call in image.add_local_file.call_args_list},
            {"/root/training/train_sft.py", "/root/training/remote_sft.py"},
        )
        self.assertEqual(
            app.function.call_args.kwargs["volumes"],
            {"/mnt/tracemem-sft-quote-v2": volume},
        )

    def test_quote_v2_accepts_shared_assistant_filler_in_distinct_conversations(self) -> None:
        train_payload = _payload("train", 48, 16)
        for case in train_payload["cases"]:
            message_number = len(case["messages"]) + 1
            case["messages"].append(
                {
                    "session_id": f"{case['case_id']}-session",
                    "message_id": f"{case['case_id']}-assistant-{message_number}",
                    "role": "assistant",
                    "text": "Thanks for sharing.",
                    "recorded_at": "2026-01-02T10:01:00Z",
                    "content_kind": "conversation",
                }
            )

        job = _build_from_payloads(train_payload, _payload("dev", 24, 1))

        self.assertEqual(len(job["request"]["train_rows"]), 768)

    def test_quote_v2_rejects_duplicate_full_train_conversations_with_new_ids(self) -> None:
        train_payload = _payload("train", 48, 16)
        first, duplicate = train_payload["cases"][:2]
        duplicate["messages"][0]["text"] = first["messages"][0]["text"]
        duplicate["expected"][0]["quote"] = first["messages"][0]["text"]

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            _build_from_payloads(train_payload, _payload("dev", 24, 1))

    def test_quote_v2_rejects_duplicate_full_conversation_across_train_and_dev(self) -> None:
        train_payload = _payload("train", 48, 16)
        dev_payload = _payload("dev", 24, 1)
        train_case, dev_case = train_payload["cases"][0], dev_payload["cases"][0]
        dev_case["messages"][0]["text"] = train_case["messages"][0]["text"]
        dev_case["expected"][0]["quote"] = train_case["messages"][0]["text"]

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            _build_from_payloads(train_payload, dev_payload)

    def test_quote_v2_retains_global_message_id_and_group_split_guards(self) -> None:
        for guarded_field in ("scenario_group", "template_group", "entity_ids"):
            with self.subTest(guarded_field=guarded_field):
                train_payload = _payload("train", 48, 16)
                dev_payload = _payload("dev", 24, 1)
                train_case, dev_case = train_payload["cases"][0], dev_payload["cases"][0]
                dev_case[guarded_field] = train_case[guarded_field]
                with self.assertRaisesRegex(ValueError, "appears in both train and dev"):
                    _build_from_payloads(train_payload, dev_payload)

        train_payload = _payload("train", 48, 16)
        dev_payload = _payload("dev", 24, 1)
        dev_payload["cases"][0]["messages"][0]["message_id"] = train_payload["cases"][0][
            "messages"
        ][0]["message_id"]
        dev_payload["cases"][0]["expected"][0]["message_id"] = dev_payload["cases"][0]["messages"][
            0
        ]["message_id"]
        with self.assertRaisesRegex(ValueError, "duplicate global message identity"):
            _build_from_payloads(train_payload, dev_payload)


if __name__ == "__main__":
    unittest.main()
