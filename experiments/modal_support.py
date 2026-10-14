"""Prepare, launch, or locally evaluate the synthetic NLI support diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from experiments.modal_baseline import (
    _IMAGE_PACKAGES,
    _ROOT,
    _active_workspace,
    _expected_workspace,
    _safe_exception_message,
    payload_sha256,
    verify_workspace,
)
from tracemem.support import load_support_challenge

_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
_REV = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
_LABEL_MAPPING = {"0": "contradiction", "1": "entailment", "2": "neutral"}
_MAX_PAIRS = 64


def _model() -> dict[str, Any]:
    config = json.loads((_ROOT / "configs" / "support-model.json").read_text(encoding="utf-8"))
    if type(config) is not dict or config.get("model_id") != _MODEL_ID:
        raise ValueError("support model configuration is invalid")
    if config.get("revision") != _REV or config.get("label_mapping") != _LABEL_MAPPING:
        raise ValueError("support model revision or label mapping is not allowlisted")
    if config.get("license_tag") != "apache-2.0" or config.get("selected") is not False:
        raise ValueError("support model provenance is not approved for this diagnostic")
    if config.get("standalone_license_file_found") is not False:
        raise ValueError("support model license verification status changed")
    if config.get("weights_downloaded_locally") is not False:
        raise ValueError("local support model weights are prohibited")
    return {"model_id": _MODEL_ID, "revision": _REV, "label_mapping": _LABEL_MAPPING}


def build_payload(*, max_cases: int = _MAX_PAIRS) -> dict[str, Any]:
    """Build the allowlisted fixture-only payload, excluding all labels and tags."""
    if (
        isinstance(max_cases, bool)
        or not isinstance(max_cases, int)
        or not 2 <= max_cases <= _MAX_PAIRS
        or max_cases % 2
    ):
        raise ValueError("support evaluation accepts an even prefix between 2 and 64 pairs")
    cases, fixture_hash = load_support_challenge()
    selected = cases[:max_cases]
    scenarios = [case["scenario_id"] for case in selected]
    if len(selected) % 2 or any(scenarios.count(scenario) != 2 for scenario in set(scenarios)):
        raise ValueError("support payload must include complete two-pair scenarios")
    items = [{key: case[key] for key in ("case_id", "premise", "hypothesis")} for case in selected]
    if len(items) < 2 or len(items) > _MAX_PAIRS:
        raise ValueError("support fixture must yield between 2 and 64 pairs")
    return {"schema_version": 1, "input_sha256": fixture_hash, "model": _model(), "items": items}


def modal_function_options() -> dict[str, Any]:
    return {
        "gpu": "T4",
        "cpu": (2, 2),
        "memory": (16_384, 16_384),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 300,
        "timeout": 300,
        "scaledown_window": 2,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Self-contained source-free remote function; all runtime imports are local."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import math as remote_math
    import platform
    import re as remote_re
    import time as remote_time

    import modal as modal_sdk
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = remote_time.perf_counter()
    model_id = "cross-encoder/nli-deberta-v3-base"
    revision = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
    labels = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    max_pairs, max_tokens, batch_size = 64, 512, 8
    if type(request) is not dict or set(request) != {"payload", "payload_sha256"}:
        raise ValueError("remote request keys are invalid")
    payload, approved = request["payload"], request["payload_sha256"]
    if not isinstance(approved, str) or remote_re.fullmatch(r"[0-9a-f]{64}", approved) is None:
        raise ValueError("remote payload hash is invalid")
    canonical = remote_json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    if remote_hashlib.sha256(canonical).hexdigest() != approved:
        raise ValueError("remote payload hash does not match approval")
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "input_sha256",
        "model",
        "items",
    }:
        raise ValueError("remote payload keys are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("remote payload version is invalid")
    if (
        not isinstance(payload["input_sha256"], str)
        or remote_re.fullmatch(r"[0-9a-f]{64}", payload["input_sha256"]) is None
    ):
        raise ValueError("remote fixture hash is invalid")
    expected_model = {"model_id": model_id, "revision": revision, "label_mapping": labels}
    if payload["model"] != expected_model:
        raise ValueError("remote model or label mapping is not allowlisted")
    items = payload["items"]
    if not isinstance(items, list) or not 1 <= len(items) <= max_pairs:
        raise ValueError("remote pair count exceeds the limit")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"case_id", "premise", "hypothesis"}:
            raise ValueError("remote pair keys are invalid")
        case_id, premise, hypothesis = (item[key] for key in ("case_id", "premise", "hypothesis"))
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("remote case IDs are invalid")
        if any(
            not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 20_000
            for text in (premise, hypothesis)
        ):
            raise ValueError("remote pair text is invalid or too large")
        seen.add(case_id)

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal T4 GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).to("cuda")
    model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    id2label = getattr(model.config, "id2label", None)
    observed = (
        {str(index): str(label).strip().casefold() for index, label in id2label.items()}
        if isinstance(id2label, dict)
        else {}
    )
    if observed != labels:
        raise ValueError("loaded model label mapping does not match the pinned order")
    label2id = getattr(model.config, "label2id", None)
    if isinstance(label2id, dict) and {
        str(name).strip().casefold(): int(index) for name, index in label2id.items()
    } != {name: int(index) for index, name in labels.items()}:
        raise ValueError("loaded model reverse label mapping is inconsistent")

    outputs: list[dict[str, Any]] = []
    gpu_seconds = 0.0
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        encoded = tokenizer(
            [item["premise"] for item in batch],
            [item["hypothesis"] for item in batch],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
        if (
            any(length > max_tokens for length in lengths)
            or int(encoded["attention_mask"].shape[1]) > max_tokens
        ):
            raise ValueError("support pair exceeds the 512-token limit")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1)
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds += start_event.elapsed_time(end_event) / 1000
        raw_logits, raw_probs = logits.cpu().tolist(), probabilities.cpu().tolist()
        for item, count, row_logits, row_probs in zip(
            batch, lengths, raw_logits, raw_probs, strict=True
        ):
            if (
                len(row_logits) != 3
                or len(row_probs) != 3
                or not all(remote_math.isfinite(float(x)) for x in (*row_logits, *row_probs))
            ):
                raise ValueError("model returned non-finite or malformed class scores")
            outputs.append(
                {
                    "case_id": item["case_id"],
                    "input_tokens": count,
                    "logits": [float(value) for value in row_logits],
                    "probabilities": [float(value) for value in row_probs],
                }
            )
    torch.cuda.synchronize()
    versions = dict(
        sorted(
            (
                (d.metadata.get("Name"), d.version)
                for d in metadata.distributions()
                if d.metadata.get("Name")
            ),
            key=lambda pair: pair[0].casefold(),
        )
    )
    return {
        "schema_version": 1,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": approved,
        "model": expected_model,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_seconds": round(gpu_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "remote_function_elapsed_seconds": round(remote_time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "case_count": len(outputs),
        "outputs": outputs,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
    }


def evaluate_receipt(receipt_bytes: bytes, *, max_cases: int = _MAX_PAIRS) -> dict[str, Any]:
    from experiments.evaluate_support import evaluate_receipt as evaluate

    return evaluate(receipt_bytes, max_cases=max_cases)


def _build_modal_function() -> tuple[Any, Any]:
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-independent-support-diagnostic")
    return app, app.function(image=image, **modal_function_options())(_remote_compute)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-support")
    parser.add_argument("--max-cases", type=int, default=_MAX_PAIRS)
    parser.add_argument("--launch", action="store_true", help="run one reviewed remote call")
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument("--evaluate", type=Path, help="evaluate an existing raw receipt locally")
    parser.add_argument("--output", type=Path, help="write local evaluation to a separate file")
    return parser


def _write_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.evaluate is not None:
            if args.launch or args.approve_payload_sha256 is not None:
                raise ValueError("receipt evaluation cannot be combined with launch approval")
            if args.output is None:
                raise ValueError("--evaluate requires --output PATH")
            from experiments.evaluate_support import output_aliases_input

            if output_aliases_input(args.evaluate, args.output):
                raise ValueError("evaluation output must not overwrite the input receipt")
            report = evaluate_receipt(args.evaluate.read_bytes(), max_cases=args.max_cases)
            args.output.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            return 0
        if args.output is not None:
            raise ValueError("--output is available only together with --evaluate")
        payload = build_payload(max_cases=args.max_cases)
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not args.launch:
            if args.approve_payload_sha256 is not None:
                raise ValueError("payload approval is only accepted together with --launch")
            _write_json(
                {
                    "model": payload["model"],
                    "item_count": len(payload["items"]),
                    "input_sha256": payload["input_sha256"],
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                    "remote_started": False,
                }
            )
            return 0
        if args.approve_payload_sha256 != digest:
            raise ValueError("--launch requires the exact prepared payload SHA-256")
        verify_workspace(_expected_workspace(), _active_workspace())
        app, function = _build_modal_function()
        started = time.perf_counter()
        with app.run(detach=False):
            result = function.remote({"payload": payload, "payload_sha256": digest})
        output = dict(result)
        output["modal_app_id"] = app.app_id
        output["runner_sha256"] = runner_hash
        output["client_call_wall_seconds_including_image_and_startup"] = round(
            time.perf_counter() - started, 6
        )
        _write_json(output)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {_safe_exception_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
