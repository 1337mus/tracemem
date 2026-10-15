"""Offline preparation and explicitly approved remote SFT pilot launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tracemem.challenge import _validate_case
from tracemem.prompts import render_explicit_prompt
from training.prepare_sft import _read_payload, _validate_metadata, _validate_root
from training.train_sft import select_training_rows

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PATH = _ROOT / "datasets/sft-pilot/train.json"
_DEV_PATH = _ROOT / "datasets/sft-pilot/eval.json"
_QUOTE_TRAIN_PATH = _ROOT / "datasets/sft-quote-v2/train.json"
_QUOTE_DEV_PATH = _ROOT / "datasets/sft-quote-v2/dev.json"
_REMOTE_VOLUME_PATH = "/mnt/tracemem-sft-pilot"
_QUOTE_REMOTE_VOLUME_PATH = "/mnt/tracemem-sft-quote-v2"
_RUN_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_FIRST_DEV_VARIANT = re.compile(r"-V01\Z")
_COMMON_PACKAGES = (
    "transformers==4.51.3",
    "tokenizers==0.21.1",
    "jinja2==3.1.6",
    "huggingface-hub==0.30.2",
    "safetensors==0.5.3",
)
_TRAIN_PACKAGES = (
    "torch==2.7.1",
    *_COMMON_PACKAGES,
    "accelerate==1.6.0",
    "peft==0.15.2",
    "bitsandbytes==0.45.5",
)
_MODULE_PATHS = (
    "training/prepare_sft.py",
    "training/train_sft.py",
    "training/remote_sft.py",
    "experiments/sft_launch.py",
    "experiments/scenario_payload.py",
    "src/tracemem/challenge.py",
    "src/tracemem/prompts.py",
)
_QUOTE_MODULE_PATHS = ("training/quote_data.py", "training/build_quote_data.py")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def modal_function_options(mode: str) -> dict[str, Any]:
    """Return the fixed single-use resource envelope for one SFT stage."""
    if mode == "preflight":
        return {
            "cpu": (2, 2),
            "memory": (4096, 4096),
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "retries": 0,
            "startup_timeout": 300,
            "timeout": 600,
            "scaledown_window": 2,
            "single_use_containers": True,
            "serialized": True,
            "include_source": False,
        }
    if mode == "train":
        return {
            "gpu": "A10G",
            "cpu": (4, 4),
            "memory": (32768, 32768),
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "retries": 0,
            "startup_timeout": 300,
            "timeout": 3600,
            "scaledown_window": 2,
            "single_use_containers": True,
            "serialized": True,
            "include_source": False,
        }
    raise ValueError("mode must be preflight or train")


def dependency_pins(mode: str) -> tuple[str, ...]:
    """Return only pinned remote image dependencies; never install locally."""
    if mode == "preflight":
        return _COMMON_PACKAGES
    if mode == "train":
        return _TRAIN_PACKAGES
    raise ValueError("mode must be preflight or train")


def build_payload(
    mode: str = "preflight", *, run_id: str | None = None, profile: str = "pilot"
) -> dict[str, Any]:
    """Build a fixed, validated request without importing Modal or ML packages."""
    if mode not in {"preflight", "train"}:
        raise ValueError("mode must be preflight or train")
    if type(profile) is not str or profile not in {"pilot", "quote-v2"}:
        raise ValueError("profile must be pilot or quote-v2")
    if profile == "quote-v2":
        if mode == "train":
            if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
                raise ValueError("train mode requires a unique lowercase run-id slug")
            remote_run_dir = f"{_QUOTE_REMOTE_VOLUME_PATH}/adapters/{run_id}"
        else:
            if run_id is not None:
                raise ValueError("run-id is only valid for train mode")
            remote_run_dir = None
        return _build_quote_v2_payload(mode, remote_run_dir)
    if mode == "train":
        if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("train mode requires a unique lowercase run-id slug")
        remote_run_dir = f"{_REMOTE_VOLUME_PATH}/adapters/{run_id}"
    else:
        if run_id is not None:
            raise ValueError("run-id is only valid for train mode")
        remote_run_dir = None

    raw_cases: dict[str, list[dict[str, Any]]] = {}
    source_bytes: dict[str, bytes] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    message_ids: set[tuple[str, str]] = set()
    content_keys: set[tuple[tuple[str, str], ...]] = set()
    group_splits: dict[tuple[str, str], str] = {}

    for source_split, path in (("train", _TRAIN_PATH), ("dev", _DEV_PATH)):
        payload, raw = _read_payload(path)
        source_bytes[source_split] = raw
        _validate_root(payload)
        cases, metadata = payload["cases"], payload["case_metadata"]
        if type(cases) is not list or not cases or type(metadata) is not dict:
            raise ValueError(f"{source_split} cases and metadata must be non-empty objects")
        for case in cases:
            _validate_case(case)
        ids_in_source = {case["case_id"] for case in cases}
        if set(metadata) != ids_in_source:
            raise ValueError(f"{source_split} case_metadata keys must match case IDs")

        for case in cases:
            case_id = case["case_id"]
            if case["split"] != source_split:
                raise ValueError(f"{source_split} source contains a non-{source_split} case")
            if case_id in ids:
                raise ValueError(f"duplicate case_id across train and dev: {case_id}")
            ids.add(case_id)
            case_metadata = _validate_metadata(metadata[case_id], case_id)
            metadata_by_id[case_id] = case_metadata
            for kind, value in (
                ("scenario_id", case["scenario_id"]),
                ("scenario_group", case_metadata["scenario_group"]),
                ("template_group", case_metadata["template_group"]),
                *(("entity_id", entity_id) for entity_id in case_metadata["entity_ids"]),
            ):
                key = (kind, value)
                prior = group_splits.setdefault(key, source_split)
                if prior != source_split:
                    raise ValueError(f"{kind} {value} appears in both train and dev")

            ordered_content = tuple(
                (message["role"], " ".join(message["text"].split())) for message in case["messages"]
            )
            if ordered_content in content_keys:
                raise ValueError(f"duplicate conversation content: {case_id}")
            content_keys.add(ordered_content)
            for message in case["messages"]:
                identity = (message["session_id"], message["message_id"])
                if identity in message_ids:
                    raise ValueError(f"duplicate global message identity: {identity}")
                message_ids.add(identity)
        raw_cases[source_split] = cases

    if len(raw_cases["train"]) != 512 or len(raw_cases["dev"]) != 32:
        raise ValueError("pilot sources must contain 512 train and 32 dev cases")
    train_groups = Counter(
        metadata_by_id[case["case_id"]]["scenario_group"] for case in raw_cases["train"]
    )
    dev_groups = Counter(
        metadata_by_id[case["case_id"]]["scenario_group"] for case in raw_cases["dev"]
    )
    if len(train_groups) != 32 or set(train_groups.values()) != {16}:
        raise ValueError("train source must contain 32 scenario families with 16 variants each")
    if len(dev_groups) != 8 or set(dev_groups.values()) != {4}:
        raise ValueError("dev source must contain eight scenario families with four variants each")

    train_rows: list[dict[str, Any]] = []
    for case in raw_cases["train"]:
        case_metadata = metadata_by_id[case["case_id"]]
        train_rows.append(
            {
                "case_id": case["case_id"],
                "split": "train",
                "scenario_group": case_metadata["scenario_group"],
                "template_group": case_metadata["template_group"],
                "messages": [
                    {"role": "user", "content": render_explicit_prompt(case)},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"claims": case["expected"]},
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    },
                ],
            }
        )
    selected_train_rows = select_training_rows(train_rows, max_rows=128, seed=17)

    first_dev_case: dict[str, dict[str, Any]] = {}
    for case in raw_cases["dev"]:
        group = metadata_by_id[case["case_id"]]["scenario_group"]
        first_dev_case.setdefault(group, case)
    dev_prompts: list[dict[str, str]] = []
    for group, case in first_dev_case.items():
        if _FIRST_DEV_VARIANT.search(case["case_id"]) is None:
            raise ValueError(f"first dev variant for {group} must be V01")
        dev_prompts.append({"case_id": case["case_id"], "prompt": render_explicit_prompt(case)})
    if len(dev_prompts) != 8:
        raise ValueError("dev source must produce exactly eight prompt-only objects")

    request = {
        "mode": mode,
        "train_rows": selected_train_rows,
        "dev_prompts": dev_prompts,
        "remote_run_dir": remote_run_dir,
    }
    job: dict[str, Any] = {
        "schema_version": 1,
        "request": request,
        "source_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()
        },
        "module_sha256": {
            relative: hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
            for relative in _MODULE_PATHS
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_options": modal_function_options(mode),
        "dependency_pins": dependency_pins(mode),
    }
    job["job_sha256"] = hashlib.sha256(_canonical_bytes(job)).hexdigest()
    return job


def _build_quote_v2_payload(mode: str, remote_run_dir: str | None) -> dict[str, Any]:
    from training.quote_data import render_quote_prompt, validate_dataset

    cases_by_split: dict[str, list[dict[str, Any]]] = {}
    source_bytes: dict[str, bytes] = {}
    case_ids: set[str] = set()
    message_ids: set[str] = set()
    conversation_contents: set[tuple[tuple[str, str, str], ...]] = set()
    group_splits: dict[tuple[str, str], str] = {}

    for split, path in (("train", _QUOTE_TRAIN_PATH), ("dev", _QUOTE_DEV_PATH)):
        payload, raw = _read_payload(path)
        source_bytes[split] = raw
        cases = validate_dataset(payload, split)
        for case in cases:
            case_id = case["case_id"]
            if case_id in case_ids:
                raise ValueError(f"duplicate case_id across train and dev: {case_id}")
            case_ids.add(case_id)
            for kind, values in (
                ("scenario_group", [case["scenario_group"]]),
                ("template_group", [case["template_group"]]),
                ("entity_id", case["entity_ids"]),
            ):
                for value in values:
                    key = (kind, value)
                    prior = group_splits.setdefault(key, split)
                    if prior != split:
                        raise ValueError(f"{kind} {value} appears in both train and dev")

            ordered_content: list[tuple[str, str, str]] = []
            for message in case["messages"]:
                message_id = message["message_id"]
                if message_id in message_ids:
                    raise ValueError(f"duplicate global message identity: {message_id}")
                message_ids.add(message_id)
                normalized_text = " ".join(message["text"].split())
                ordered_content.append((message["role"], message["content_kind"], normalized_text))
            content_key = tuple(ordered_content)
            if content_key in conversation_contents:
                raise ValueError(f"duplicate conversation content: {case_id}")
            conversation_contents.add(content_key)
        cases_by_split[split] = cases

    train_cases = cases_by_split["train"]
    dev_cases = cases_by_split["dev"]
    if len(train_cases) != 768 or len(dev_cases) != 24:
        raise ValueError("quote-v2 sources must contain 768 train and 24 dev cases")
    train_groups = Counter(case["scenario_group"] for case in train_cases)
    dev_groups = Counter(case["scenario_group"] for case in dev_cases)
    if len(train_groups) != 48 or set(train_groups.values()) != {16}:
        raise ValueError("quote-v2 train source must contain 48 families with 16 variants each")
    if len(dev_groups) != 24 or set(dev_groups.values()) != {1}:
        raise ValueError("quote-v2 dev source must contain 24 families with one case each")

    train_rows = [
        {
            "case_id": case["case_id"],
            "split": "train",
            "messages": [
                {"role": "user", "content": render_quote_prompt(case)},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"claims": case["expected"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
            ],
        }
        for case in train_cases
    ]
    selected_train_rows = select_training_rows(train_rows, max_rows=768, seed=17)
    dev_prompts = [
        {"case_id": case["case_id"], "prompt": render_quote_prompt(case)} for case in dev_cases
    ]
    request = {
        "mode": mode,
        "profile": "quote-v2",
        "train_rows": selected_train_rows,
        "dev_prompts": dev_prompts,
        "remote_run_dir": remote_run_dir,
    }
    module_paths = (*_MODULE_PATHS, *_QUOTE_MODULE_PATHS)
    job: dict[str, Any] = {
        "schema_version": 1,
        "request": request,
        "source_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()
        },
        "module_sha256": {
            relative: hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
            for relative in module_paths
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_options": modal_function_options(mode),
        "dependency_pins": dependency_pins(mode),
    }
    job["job_sha256"] = hashlib.sha256(_canonical_bytes(job)).hexdigest()
    return job


def build_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Return launch preparation evidence without exposing prompts or labels."""
    request = job["request"]
    summary = {
        "schema_version": job["schema_version"],
        "mode": request["mode"],
        "train_row_count": len(request["train_rows"]),
        "dev_prompt_count": len(request["dev_prompts"]),
        "job_sha256": job["job_sha256"],
        "runner_sha256": job["runner_sha256"],
        "source_sha256": job["source_sha256"],
        "module_sha256": job["module_sha256"],
        "remote_started": False,
    }
    if "profile" in request:
        summary["profile"] = request["profile"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-sft")
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--profile", choices=("pilot", "quote-v2"), default="pilot")
    parser.add_argument("--run-id", help="unique lowercase run slug required for train mode")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument("--approve-job-sha256", help="exact job hash from offline preparation")
    parser.add_argument("--output", type=Path, help="new path for the local raw receipt")
    return parser


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        job = build_payload(args.mode, run_id=args.run_id, profile=args.profile)
        summary = build_summary(job)
        if not args.launch:
            if args.approve_job_sha256 is not None or args.output is not None:
                raise ValueError("approval and output flags require --launch")
            _write_json(summary)
            return 0

        if args.approve_job_sha256 is None or args.output is None:
            raise ValueError("--launch requires an exact job approval and a new --output path")
        if args.approve_job_sha256 != job["job_sha256"]:
            raise ValueError("exact current job approval is required")

        from experiments import sft_launch

        sft_launch.run_once(
            job,
            approved_job_sha256=args.approve_job_sha256,
            output_path=args.output,
        )
        _write_json({**summary, "remote_started": True})
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
