"""Bounded source-free continuation runner for synthetic verifier v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import FunctionType
from typing import Any

from experiments.verifier_v2_remote import _collect_parity_differences, _remote_compute

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _ROOT / "examples/synthetic/verifier-v2-train-dev.json"
_TRANSFER_DATASET_PATH = _ROOT / "examples/synthetic/verifier-transfer-v1.json"
_PAIRED_DATASET_PATH = _ROOT / "examples/synthetic/verifier-pairs-v1.json"
_PRIOR_REFERENCE_PATH = _ROOT / "examples/synthetic/verifier-transfer-remote-v1.json"
_REMOTE_HELPER_PATH = _ROOT / "experiments/verifier_v2_remote.py"
_RENDERER_PATH = _ROOT / "training/verifier_data.py"
_SOURCE_HASHES = {
    "examples/synthetic/verifier-v2-train-dev.json": (
        "61e28eca9cb516efae7cc7c88db7b12e0b959505b01b7b540db7791e586c3c75"
    ),
    "examples/synthetic/verifier-transfer-v1.json": (
        "6696c78b35499933310eeabdf626d683d5af122524fb5e50640d85043e7d20e9"
    ),
    "examples/synthetic/verifier-pairs-v1.json": (
        "9442f2703ccf98c602579009f1759a41377fd860f5648626c2b26da2f8ea0358"
    ),
    "examples/synthetic/verifier-transfer-remote-v1.json": (
        "070ad8665bfd34fc7192c67943b0e27141c9cf714dfff83abb42a1de7ed4766f"
    ),
    "experiments/verifier_v2_remote.py": (
        "ea904a3b568280fa038eb4b3179e85659eb141416ca7652b40f77d80dae1ee00"
    ),
    "training/verifier_data.py": (
        "ada22823e053042d6d20e533f35410d852855607c8f6d51fd3cab93d8a4234c6"
    ),
}
_PRIOR_ADAPTER: dict[str, Any] = {
    "volume_name": "tracemem-verifier-pairs-v1",
    "volume_mount": "/mnt/tracemem-verifier-pairs-v1",
    "relative_path": (
        "runs/534139fb7e076e7511e85928c51dfe80b2a625138f3bebe50e36be3b77a24106/adapter"
    ),
    "safetensors_sha256": "014a21986d0b2fb6fe884ab8346a1708012558632b5dbbe4325c701394e8f9b2",
    "config_sha256": "9c79cb0e2879e658f96bbad4cc41aa17c603fe03405f70100b3adc86b8692153",
}
_MODEL = {
    "model_id": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "quantization": {
        "load_in_4bit": True,
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    },
}
_SETTINGS = {
    "do_sample": False,
    "enable_thinking": False,
    "max_new_tokens": 128,
    "max_input_tokens": 2048,
    "attn_implementation": "eager",
    "seed": 17,
    "optimizer": "AdamW",
    "learning_rate": 0.00005,
    "gradient_accumulation_steps": 4,
    "micro_batch_size": 1,
    "epochs": 2,
    "microsteps": 384,
    "optimizer_steps": 96,
    "gradient_checkpointing": True,
    "completion_only_loss": True,
    "completion_eos_loss": True,
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    },
}
_OUTPUT_VOLUME_NAME = "tracemem-verifier-v2"
_OUTPUT_VOLUME_MOUNT = "/mnt/tracemem-verifier-v2"
_COHORT_COUNTS = {"new_dev": 48, "transfer_regression": 48, "original_paired_dev": 12}
_PAYLOAD_COUNTS = {
    "train_cases": 96,
    "train_rows": 192,
    "new_dev_cases": 24,
    "new_dev_candidates": 48,
    "transfer_regression_candidates": 48,
    "original_paired_dev_candidates": 12,
    "evaluation_candidates": 108,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_payload() -> dict[str, Any]:
    from training.verifier_data import (
        render_candidate_prompt,
        render_training_rows,
        validate_verifier_dataset,
    )

    paths = {
        "examples/synthetic/verifier-v2-train-dev.json": _DATASET_PATH,
        "examples/synthetic/verifier-transfer-v1.json": _TRANSFER_DATASET_PATH,
        "examples/synthetic/verifier-pairs-v1.json": _PAIRED_DATASET_PATH,
        "examples/synthetic/verifier-transfer-remote-v1.json": _PRIOR_REFERENCE_PATH,
        "experiments/verifier_v2_remote.py": _REMOTE_HELPER_PATH,
        "training/verifier_data.py": _RENDERER_PATH,
    }
    raw_sources: dict[str, bytes] = {}
    for name, path in paths.items():
        expected = _SOURCE_HASHES[name]
        if expected == "UNREVIEWED":
            raise ValueError(f"verifier v2 source pin is UNREVIEWED: {name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"verifier v2 source SHA-256 is not allowlisted: {name}")
        raw_sources[name] = raw

    new_dataset = json.loads(raw_sources["examples/synthetic/verifier-v2-train-dev.json"])
    new_cases = validate_verifier_dataset(new_dataset)
    split_counts = {
        split: sum(case["split"] == split for case in new_cases)
        for split in ("train", "dev", "test")
    }
    if split_counts != {"train": 96, "dev": 24, "test": 0}:
        raise ValueError("verifier v2 corpus must contain exactly 96 train and 24 dev cases")
    train_rows = render_training_rows(new_dataset, "train")
    if len(train_rows) != 192:
        raise ValueError("verifier v2 corpus must render exactly 192 train rows")

    transfer_dataset = json.loads(raw_sources["examples/synthetic/verifier-transfer-v1.json"])
    transfer_cases = validate_verifier_dataset(transfer_dataset)
    if len(transfer_cases) != 24 or any(case["split"] != "dev" for case in transfer_cases):
        raise ValueError("transfer regression corpus must contain exactly 24 dev cases")
    paired_dataset = json.loads(raw_sources["examples/synthetic/verifier-pairs-v1.json"])
    paired_cases = validate_verifier_dataset(paired_dataset)
    paired_dev = [case for case in paired_cases if case["split"] == "dev"]
    if len(paired_dev) != 6 or sum(case["split"] == "test" for case in paired_cases) != 6:
        raise ValueError("original paired corpus dev/test counts changed")

    def make_items(cases: list[dict[str, Any]], cohort: str) -> list[dict[str, str]]:
        return [
            {
                "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
                "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
                "cohort": cohort,
            }
            for case in cases
            for candidate in case["candidates"]
        ]

    items = (
        make_items([case for case in new_cases if case["split"] == "dev"], "new_dev")
        + make_items(transfer_cases, "transfer_regression")
        + make_items(paired_dev, "original_paired_dev")
    )
    if len(items) != _PAYLOAD_COUNTS["evaluation_candidates"]:
        raise ValueError("verifier v2 evaluation must contain exactly 108 candidates")
    cohort_counts = {name: sum(item["cohort"] == name for item in items) for name in _COHORT_COUNTS}
    if cohort_counts != _COHORT_COUNTS or len({item["item_id"] for item in items}) != len(items):
        raise ValueError("verifier v2 evaluation cohort counts or item IDs changed")

    reference = json.loads(raw_sources["examples/synthetic/verifier-transfer-remote-v1.json"])
    if (
        reference.get("status") != "success"
        or reference.get("mode") != "transfer"
        or reference.get("split") != "dev"
    ):
        raise ValueError("pinned transfer reference receipt has an invalid envelope")
    transfer_ids = {item["item_id"] for item in items if item["cohort"] == "transfer_regression"}
    reference_outputs: dict[str, list[dict[str, Any]]] = {}
    for key in ("baseline_outputs", "tuned_outputs"):
        rows = reference.get(key)
        if type(rows) is not list or {row.get("item_id") for row in rows} != transfer_ids:
            raise ValueError("pinned transfer reference output IDs do not match transfer prompts")
        reference_outputs[key] = deepcopy(rows)

    return {
        "schema_version": 1,
        "mode": "continue_train",
        "split": "dev",
        "source_sha256": deepcopy(_SOURCE_HASHES),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": deepcopy(_MODEL),
        "settings": deepcopy(_SETTINGS),
        "prior_adapter": deepcopy(_PRIOR_ADAPTER),
        "counts": deepcopy(_PAYLOAD_COUNTS),
        "items": items,
        "train_rows": train_rows,
        "prior_reference_sha256": _SOURCE_HASHES[
            "examples/synthetic/verifier-transfer-remote-v1.json"
        ],
        "prior_reference_outputs": reference_outputs,
    }


def modal_function_options() -> dict[str, Any]:
    from experiments.modal_verifier_pairs import modal_function_options as paired_options

    options = paired_options()
    options["timeout"] = 1800
    return options


def _make_remote_function(
    output_volume: Any,
    runner_sha256: str | None = None,
    payload: dict[str, Any] | None = None,
) -> FunctionType:
    payload = build_payload() if payload is None else payload
    runner_hash = runner_sha256 or payload["runner_sha256"]
    prior_adapter = payload["prior_adapter"]
    prior_adapter_root = f"{prior_adapter['volume_mount']}/{prior_adapter['relative_path']}"
    function = FunctionType(
        _remote_compute.__code__,
        {
            "__name__": "__main__",
            "_collect_parity_differences": FunctionType(
                _collect_parity_differences.__code__,
                {"__name__": "__main__", "__builtins__": __builtins__},
                name="_collect_parity_differences",
            ),
            "_EXPECTED_COUNTS": deepcopy(payload["counts"]),
            "_EXPECTED_COHORT_COUNTS": deepcopy(_COHORT_COUNTS),
            "_EXPECTED_MODEL": deepcopy(payload["model"]),
            "_EXPECTED_OUTPUT_MOUNT": _OUTPUT_VOLUME_MOUNT,
            "_EXPECTED_PRIOR_ADAPTER": deepcopy(payload["prior_adapter"]),
            "_EXPECTED_PRIOR_ADAPTER_ROOT": prior_adapter_root,
            "_EXPECTED_PAYLOAD_SHA256": payload_sha256(payload),
            "_EXPECTED_PRIOR_REFERENCE_OUTPUTS_SHA256": hashlib.sha256(
                _canonical_bytes(payload["prior_reference_outputs"])
            ).hexdigest(),
            "_EXPECTED_PRIOR_REFERENCE_SHA256": payload["prior_reference_sha256"],
            "_EXPECTED_RUNNER_SHA256": runner_hash,
            "_EXPECTED_SETTINGS": deepcopy(payload["settings"]),
            "_EXPECTED_SOURCE_HASHES": deepcopy(payload["source_sha256"]),
            "_REMOTE_OUTPUT_VOLUME": output_volume,
        },
        name="remote_verifier_v2",
    )
    function.__doc__ = _remote_compute.__doc__
    return function


def _build_modal_function(payload: dict[str, Any]) -> tuple[Any, Any]:
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-verifier-v2")
    prior_volume = modal.Volume.from_name(
        _PRIOR_ADAPTER["volume_name"], create_if_missing=False
    ).with_mount_options(read_only=True)
    output_volume = modal.Volume.from_name(_OUTPUT_VOLUME_NAME, create_if_missing=True)
    options = modal_function_options()
    options["volumes"] = {
        _PRIOR_ADAPTER["volume_mount"]: prior_volume,
        _OUTPUT_VOLUME_MOUNT: output_volume,
    }
    function = app.function(image=image, **options)(
        _make_remote_function(output_volume, payload["runner_sha256"], payload)
    )
    return app, function


def _validate_current_payload(
    payload: dict[str, Any], approved_payload_sha256: str, approved_runner_sha256: str
) -> tuple[dict[str, Any], str, str]:
    if type(payload) is not dict:
        raise ValueError("payload differs from the current fixed verifier v2 payload")
    expected = build_payload()
    if _canonical_bytes(payload) != _canonical_bytes(expected):
        raise ValueError("payload differs from the current fixed verifier v2 payload")
    digest = payload_sha256(payload)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        approved_payload_sha256 != digest
        or approved_runner_sha256 != runner_hash
        or payload.get("runner_sha256") != runner_hash
    ):
        raise ValueError(
            "exact current verifier v2 payload and runner SHA-256 approvals are required"
        )
    return expected, digest, runner_hash


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    expected, digest, runner_hash = _validate_current_payload(
        payload, approved_payload_sha256, approved_runner_sha256
    )
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    from experiments.sft_launch import (
        _acquire_launch_lock,
        _reserve_output,
        _write_receipt,
        verify_modal_inventory,
        verify_personal_workspace,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd = _reserve_output(output_path)
    lock_fd: int | None = None
    started = time.perf_counter()
    remote_started = False
    try:
        lock_fd = _acquire_launch_lock()
        verify_personal_workspace()
        inventory = verify_modal_inventory()
        app, function = _build_modal_function(expected)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(
                {
                    "payload": expected,
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                }
            )
        if type(result) is not dict or result.get("mode") != "continue_train":
            raise TypeError("Modal returned an invalid verifier v2 result")
        if result.get("status") == "protocol_failed":
            training = result.get("training")
            if (
                result.get("split") != "dev"
                or result.get("payload_sha256") != digest
                or result.get("runner_sha256") != runner_hash
                or type(result.get("baseline_outputs")) is not list
                or len(result["baseline_outputs"]) != 108
                or type(result.get("prior_outputs")) is not list
                or len(result["prior_outputs"]) != 108
                or type(result.get("parity_differences")) is not list
                or not result["parity_differences"]
                or result.get("training_performed") is not False
                or type(training) is not dict
                or any(
                    training.get(key) != 0 for key in ("row_count", "microsteps", "optimizer_steps")
                )
            ):
                raise TypeError("Modal returned an invalid verifier v2 protocol failure")
        elif result.get("status") != "success" or any(
            type(result.get(key)) is not list or len(result[key]) != 108
            for key in ("baseline_outputs", "prior_outputs", "tuned_outputs")
        ):
            raise TypeError("Modal returned an invalid verifier v2 result")
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-verifier-v2",
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "personal_workspace_verified": True,
                "modal_inventory_checked": True,
                "modal_app_count": len(inventory),
            },
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        failure = {
            "schema_version": 1,
            "status": "launch_failed",
            "mode": "continue_train",
            "payload_sha256": digest,
            "runner_sha256": runner_hash,
            "remote_started": remote_started,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": {"type": type(error).__name__, "message": safe_exception_message(error)},
        }
        _write_receipt(output_fd, failure)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(output_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-verifier-v2")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build a reviewable offline payload")
    prepare.add_argument("--show-payload", action="store_true")
    launch = commands.add_parser("launch", help="run one exactly approved continuation")
    launch.add_argument("--approve-payload-sha256", required=True)
    launch.add_argument("--approve-runner-sha256", required=True)
    launch.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload()
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if payload["runner_sha256"] != runner_hash:
            raise ValueError("verifier v2 runner SHA-256 changed while preparing the payload")
        if args.command == "prepare":
            result: dict[str, Any] = {
                "mode": payload["mode"],
                "split": payload["split"],
                "item_count": len(payload["items"]),
                "train_row_count": len(payload["train_rows"]),
                "source_sha256": payload["source_sha256"],
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "remote_started": False,
            }
            if args.show_payload:
                result["payload"] = payload
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        if args.approve_payload_sha256 != digest or args.approve_runner_sha256 != runner_hash:
            raise ValueError(
                "exact current verifier v2 payload and runner SHA-256 approvals are required"
            )
        receipt = run_once(
            payload,
            approved_payload_sha256=digest,
            approved_runner_sha256=runner_hash,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "receipt_path": str(args.output),
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                    "remote_started": True,
                    "modal_app_id": receipt.get("modal_app_id"),
                    "modal_function_call_id": receipt.get("modal_function_call_id"),
                    "status": receipt.get("status"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if receipt.get("status") == "success" else 2
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        print(f"{type(error).__name__}: {safe_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
