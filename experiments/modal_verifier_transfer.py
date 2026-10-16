"""Matched base-versus-frozen-adapter transfer inference, without training."""

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

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _ROOT / "examples/synthetic/verifier-transfer-v1.json"
_RENDERER_PATH = _ROOT / "training/verifier_data.py"
# Root may update the corpus after its final semantic review; this is the only
# corpus pin to change when that review produces a replacement hash.
_DATASET_SHA256 = "6696c78b35499933310eeabdf626d683d5af122524fb5e50640d85043e7d20e9"
_RENDERER_SHA256 = "ada22823e053042d6d20e533f35410d852855607c8f6d51fd3cab93d8a4234c6"
_ADAPTER = {
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
}
_EXPECTED_ADAPTER: dict[str, Any] = _ADAPTER
_EXPECTED_ADAPTER_ROOT = f"{_ADAPTER['volume_mount']}/{_ADAPTER['relative_path']}"
_EXPECTED_DATASET_SHA256 = _DATASET_SHA256
_EXPECTED_MODEL: dict[str, Any] = _MODEL
_EXPECTED_RENDERER_SHA256 = _RENDERER_SHA256
_EXPECTED_RUNNER_SHA256 = ""
_EXPECTED_SETTINGS: dict[str, Any] = _SETTINGS


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def modal_function_options() -> dict[str, Any]:
    """Reuse the frozen verifier envelope with the transfer timeout cap."""
    from experiments.modal_verifier_pairs import modal_function_options as paired_options

    options = paired_options()
    options["timeout"] = 1200
    return options


def build_payload() -> dict[str, Any]:
    """Build the exact offline payload approved for one transfer comparison."""
    dataset_raw = _DATASET_PATH.read_bytes()
    if hashlib.sha256(dataset_raw).hexdigest() != _DATASET_SHA256:
        raise ValueError("transfer dataset SHA-256 is not allowlisted")
    renderer_hash = hashlib.sha256(_RENDERER_PATH.read_bytes()).hexdigest()
    if renderer_hash != _RENDERER_SHA256:
        raise ValueError("verifier prompt renderer SHA-256 is not allowlisted")

    from training.verifier_data import render_candidate_prompt, validate_verifier_dataset

    dataset = json.loads(dataset_raw)
    cases = validate_verifier_dataset(dataset)
    if len(cases) != 24 or any(case["split"] != "dev" for case in cases):
        raise ValueError("transfer dataset must contain exactly 24 dev cases")
    counts = {
        name: sum(case["phenomenon"] == name for case in cases)
        for name in ("attribution", "condition")
    }
    if counts != {"attribution": 12, "condition": 12}:
        raise ValueError("transfer dataset phenomenon counts changed")
    items = [
        {
            "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
            "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
        }
        for case in cases
        for candidate in case["candidates"]
    ]
    if len(items) != 48:
        raise ValueError("transfer dataset must contain exactly 48 candidate prompts")
    return {
        "schema_version": 1,
        "mode": "transfer",
        "split": "dev",
        "source_sha256": {
            "examples/synthetic/verifier-transfer-v1.json": _DATASET_SHA256,
            "training/verifier_data.py": renderer_hash,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": deepcopy(_MODEL),
        "settings": deepcopy(_SETTINGS),
        "adapter": deepcopy(_ADAPTER),
        "items": items,
        "train_rows": [],
    }


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate an approved transfer payload and compare the frozen model pair."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re as remote_re
    import time as remote_time
    from pathlib import Path as RemotePath
    from stat import S_ISDIR, S_ISREG
    from typing import cast

    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote transfer request keys are invalid")
    payload = request["payload"]
    approved_payload = request["payload_sha256"]
    approved_runner = request["runner_sha256"]
    for digest in (approved_payload, approved_runner):
        if type(digest) is not str or remote_re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("remote transfer approval hashes are invalid")
    try:
        payload_bytes = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote transfer payload is not canonical JSON") from exc
    if len(payload_bytes) > 2_000_000:
        raise ValueError("remote transfer payload exceeds the byte limit")
    if remote_hashlib.sha256(payload_bytes).hexdigest() != approved_payload:
        raise ValueError("remote transfer payload SHA-256 does not match approval")

    expected_keys = {
        "schema_version",
        "mode",
        "split",
        "source_sha256",
        "runner_sha256",
        "model",
        "settings",
        "adapter",
        "items",
        "train_rows",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("remote transfer payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "transfer"
        or payload["split"] != "dev"
    ):
        raise ValueError("remote transfer mode, split, or schema version is invalid")
    expected_sources = {
        "examples/synthetic/verifier-transfer-v1.json": _EXPECTED_DATASET_SHA256,
        "training/verifier_data.py": _EXPECTED_RENDERER_SHA256,
    }
    if payload["source_sha256"] != expected_sources:
        raise ValueError("remote transfer data or prompt renderer hashes are not allowlisted")
    if payload["runner_sha256"] != approved_runner:
        raise ValueError("remote transfer runner SHA-256 does not match approval")
    if _EXPECTED_RUNNER_SHA256 and approved_runner != _EXPECTED_RUNNER_SHA256:
        raise ValueError("remote transfer runner SHA-256 is not allowlisted")
    if payload["model"] != _EXPECTED_MODEL or payload["settings"] != _EXPECTED_SETTINGS:
        raise ValueError("remote transfer model or settings changed")
    if payload["adapter"] != _EXPECTED_ADAPTER:
        raise ValueError("remote transfer adapter metadata changed")
    if payload["train_rows"] != []:
        raise ValueError("remote transfer payload must not contain training rows")

    items = payload["items"]
    if type(items) is not list or len(items) != 48:
        raise ValueError("remote transfer requires exactly 48 dev candidate items")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("remote transfer item shape is invalid")
        item_id, prompt = item["item_id"], item["prompt"]
        if (
            type(item_id) is not str
            or remote_re.fullmatch(r"[A-Za-z0-9_-]+::[A-Za-z0-9_-]+", item_id) is None
            or item_id in seen
        ):
            raise ValueError("remote transfer item IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote transfer prompt is empty or exceeds its byte limit")
        seen.add(item_id)

    volume_mount = RemotePath(_EXPECTED_ADAPTER["volume_mount"])
    adapter_root = RemotePath(_EXPECTED_ADAPTER_ROOT)
    try:
        canonical_volume = volume_mount.resolve(strict=True)
        if (
            not S_ISDIR(canonical_volume.stat().st_mode)
            or not S_ISDIR(adapter_root.lstat().st_mode)
            or adapter_root.resolve(strict=True)
            != canonical_volume / _EXPECTED_ADAPTER["relative_path"]
        ):
            raise ValueError("remote transfer adapter directory is invalid")
        expected_artifacts = {
            "adapter_config.json": _EXPECTED_ADAPTER["config_sha256"],
            "adapter_model.safetensors": _EXPECTED_ADAPTER["safetensors_sha256"],
        }
        actual_artifacts: dict[str, str] = {}
        for filename, expected_hash in expected_artifacts.items():
            artifact_path = adapter_root / filename
            if not S_ISREG(artifact_path.lstat().st_mode):
                raise ValueError("remote transfer adapter artifact is not a regular file")
            artifact_hash = remote_hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if artifact_hash != expected_hash:
                raise ValueError(f"remote transfer adapter {filename} SHA-256 mismatch")
            actual_artifacts[filename] = artifact_hash
    except (OSError, RuntimeError) as exc:
        raise ValueError("remote transfer adapter files are unavailable or unsafe") from exc

    # ML imports happen only after request, payload, and mounted adapter checks pass.
    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from peft import (  # type: ignore[import-not-found]
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    pinned_packages = (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
    )
    versions = {
        "modal": getattr(modal_sdk, "__version__", "injected runtime"),
        **{package: metadata.version(package) for package in pinned_packages},
    }
    if tuple(versions[package] for package in pinned_packages) != (
        "2.7.1",
        "4.51.3",
        "0.21.1",
        "1.6.0",
        "0.15.2",
        "0.45.5",
        "0.5.3",
    ):
        raise RuntimeError("remote transfer dependency versions do not match the pinned image")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("remote transfer requires Python 3.12")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("remote transfer requires an A10G CUDA device with BF16 support")

    started = remote_time.perf_counter()
    torch.manual_seed(_EXPECTED_SETTINGS["seed"])
    torch.cuda.manual_seed_all(_EXPECTED_SETTINGS["seed"])
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        _EXPECTED_MODEL["model_id"],
        revision=_EXPECTED_MODEL["revision"],
        trust_remote_code=False,
        local_files_only=False,
    )
    template = tokenizer.chat_template
    if type(template) is not str or not template:
        raise ValueError("pinned Qwen tokenizer chat template is unavailable")
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("pinned Qwen tokenizer EOS token ID is invalid")
    pad_id = tokenizer.pad_token_id if type(tokenizer.pad_token_id) is int else eos_id
    tokenizer_provenance = {
        "class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "chat_template_sha256": remote_hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
    }
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        _EXPECTED_MODEL["model_id"],
        revision=_EXPECTED_MODEL["revision"],
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=_EXPECTED_SETTINGS["attn_implementation"],
        quantization_config=quantization,
        device_map="cuda",
        local_files_only=False,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = True
    base_model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    input_device = base_model.get_input_embeddings().weight.device

    def encode_prompt(prompt: str) -> list[int]:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("Qwen chat template returned an invalid token tensor")
        token_ids = encoded[0].detach().cpu().tolist()
        if not token_ids or len(token_ids) > _EXPECTED_SETTINGS["max_input_tokens"]:
            raise ValueError("verifier prompt exceeds 2048 tokens; truncation is disabled")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("Qwen chat template returned invalid input token IDs")
        return cast(list[int], token_ids)

    prepared_items = [(item, encode_prompt(item["prompt"])) for item in items]

    def generate_outputs(
        active_model: Any, prepared: list[tuple[dict[str, Any], list[int]]]
    ) -> list[dict[str, Any]]:
        active_model.eval()
        outputs: list[dict[str, Any]] = []
        for item, input_ids in prepared:
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=input_device)
            attention = torch.ones_like(input_tensor)
            with torch.inference_mode():
                generated = active_model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention,
                    do_sample=_EXPECTED_SETTINGS["do_sample"],
                    num_beams=1,
                    max_new_tokens=_EXPECTED_SETTINGS["max_new_tokens"],
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            if (
                not isinstance(generated, torch.Tensor)
                or generated.ndim != 2
                or generated.shape[0] != 1
            ):
                raise ValueError("Qwen generation returned an invalid token tensor")
            output_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
            if any(type(token_id) is not int or token_id < 0 for token_id in output_ids):
                raise ValueError("Qwen generation returned invalid output token IDs")
            ended_with_eos = bool(output_ids and output_ids[-1] == eos_id)
            hit_max = len(output_ids) >= _EXPECTED_SETTINGS["max_new_tokens"]
            outputs.append(
                {
                    "item_id": item["item_id"],
                    "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "finish_reason": (
                        "eos" if ended_with_eos else "length" if hit_max else "unknown"
                    ),
                    "input_tokens": len(input_ids),
                    "output_tokens": len(output_ids),
                    "input_token_ids": input_ids,
                    "output_token_ids": output_ids,
                    "ended_with_eos": ended_with_eos,
                    "hit_max_new_tokens": hit_max,
                }
            )
        torch.cuda.synchronize()
        return outputs

    baseline_started = remote_time.perf_counter()
    baseline_outputs = generate_outputs(base_model, prepared_items)
    baseline_seconds = remote_time.perf_counter() - baseline_started

    tuned_started = remote_time.perf_counter()
    tuned_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_root),
        is_trainable=False,
        local_files_only=True,
    )
    if any(parameter.requires_grad for parameter in tuned_model.parameters()):
        raise RuntimeError("frozen transfer adapter unexpectedly has trainable parameters")
    tuned_model.eval()
    tuned_outputs = generate_outputs(tuned_model, prepared_items)
    tuned_seconds = remote_time.perf_counter() - tuned_started
    torch.cuda.synchronize()

    return {
        "schema_version": 1,
        "status": "success",
        "mode": "transfer",
        "split": "dev",
        "payload_sha256": approved_payload,
        "runner_sha256": approved_runner,
        "model": _EXPECTED_MODEL,
        "settings": _EXPECTED_SETTINGS,
        "adapter": _EXPECTED_ADAPTER,
        "adapter_artifact_sha256": actual_artifacts,
        "versions": {**versions, "python": platform.python_version()},
        "tokenizer": tokenizer_provenance,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
        "baseline_outputs": baseline_outputs,
        "tuned_outputs": tuned_outputs,
        "timings": {
            "model_load_seconds": round(model_load_seconds, 6),
            "baseline_seconds": round(baseline_seconds, 6),
            "tuned_seconds": round(tuned_seconds, 6),
            "total_seconds": round(remote_time.perf_counter() - started, 6),
        },
        "memory": {
            "gpu_name": gpu_name,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }


def _make_remote_function(volume: Any, runner_sha256: str | None = None) -> FunctionType:
    function = FunctionType(
        _remote_compute.__code__,
        {
            "__name__": "__main__",
            "_EXPECTED_ADAPTER": _ADAPTER,
            "_EXPECTED_ADAPTER_ROOT": f"{_ADAPTER['volume_mount']}/{_ADAPTER['relative_path']}",
            "_EXPECTED_DATASET_SHA256": _DATASET_SHA256,
            "_EXPECTED_MODEL": _MODEL,
            "_EXPECTED_RENDERER_SHA256": _RENDERER_SHA256,
            "_EXPECTED_RUNNER_SHA256": runner_sha256 or "",
            "_EXPECTED_SETTINGS": _SETTINGS,
            "_REMOTE_VOLUME": volume,
        },
        name="remote_verifier_transfer",
    )
    function.__doc__ = _remote_compute.__doc__
    return function


def _build_modal_function(payload: dict[str, Any]) -> tuple[Any, Any]:
    """Build a source-free one-call Modal app with the frozen adapter mounted read-only."""
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-verifier-transfer-v1")
    volume = modal.Volume.from_name(
        _ADAPTER["volume_name"], create_if_missing=False
    ).with_mount_options(read_only=True)
    options = modal_function_options()
    options["volumes"] = {_ADAPTER["volume_mount"]: volume}
    function = app.function(image=image, **options)(
        _make_remote_function(volume, payload["runner_sha256"])
    )
    return app, function


def _validate_current_payload(
    payload: dict[str, Any], approved_payload_sha256: str, approved_runner_sha256: str
) -> tuple[dict[str, Any], str, str]:
    if type(payload) is not dict:
        raise ValueError("payload differs from the current fixed transfer payload")
    expected_payload = build_payload()
    if _canonical_bytes(payload) != _canonical_bytes(expected_payload):
        raise ValueError("payload differs from the current fixed transfer payload")
    digest = payload_sha256(payload)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        approved_payload_sha256 != digest
        or approved_runner_sha256 != runner_hash
        or payload.get("runner_sha256") != runner_hash
    ):
        raise ValueError("exact current payload and runner SHA-256 approvals are required")
    return expected_payload, digest, runner_hash


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one approved transfer payload with workspace, inventory, lock, and receipt guards."""
    expected_payload, digest, runner_hash = _validate_current_payload(
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
        app, function = _build_modal_function(expected_payload)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(
                {
                    "payload": expected_payload,
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                }
            )
        if type(result) is not dict or result.get("status") != "success":
            raise TypeError("Modal returned an invalid transfer result")
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-verifier-transfer-v1",
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
            "mode": "transfer",
            "split": "dev",
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
    parser = argparse.ArgumentParser(prog="modal-verifier-transfer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build an offline, reviewable payload")
    prepare.add_argument("--show-payload", action="store_true")
    launch = commands.add_parser("launch", help="run one exactly approved transfer payload")
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
            raise ValueError("current transfer runner SHA-256 changed while preparing the payload")
        if args.command == "prepare":
            result: dict[str, Any] = {
                "mode": "transfer",
                "split": "dev",
                "item_count": len(payload["items"]),
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
            raise ValueError("exact current payload and runner SHA-256 approvals are required")
        receipt = run_once(
            payload,
            approved_payload_sha256=args.approve_payload_sha256,
            approved_runner_sha256=args.approve_runner_sha256,
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
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        print(f"{type(error).__name__}: {safe_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
