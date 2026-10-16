"""Prepare or run one explicitly approved, source-free quote-dev inference call."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from experiments.quote_baseline_payload import build_payload, payload_sha256
from experiments.scenario_payload import safe_exception_message, verify_personal_workspace

_IMAGE_PACKAGES = ("torch==2.7.1", "transformers==4.51.3", "accelerate==1.6.0")


def modal_function_options() -> dict[str, Any]:
    """Return the bounded A10G, source-free, single-use Modal envelope."""
    return {
        "gpu": "A10G",
        "cpu": 2,
        "memory": 32_768,
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 600,
        "timeout": 2400,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete source-free request before importing model runtimes."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re
    import time as remote_time

    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote quote baseline request keys are invalid")
    payload = request["payload"]
    approved_hash = request["payload_sha256"]
    approved_runner_hash = request["runner_sha256"]
    if (
        type(approved_hash) is not str
        or re.fullmatch(r"[0-9a-f]{64}", approved_hash) is None
        or type(approved_runner_hash) is not str
        or re.fullmatch(r"[0-9a-f]{64}", approved_runner_hash) is None
    ):
        raise ValueError("remote quote baseline approval hashes are invalid")
    try:
        canonical_payload = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("remote quote baseline payload is not canonical JSON") from exc
    if remote_hashlib.sha256(canonical_payload).hexdigest() != approved_hash:
        raise ValueError("remote quote baseline payload hash does not match approval")

    payload_keys = {
        "schema_version",
        "dataset",
        "input_sha256",
        "prompt_source_sha256",
        "items_sha256",
        "runner_source_sha256",
        "model",
        "settings",
        "items",
    }
    if type(payload) is not dict or set(payload) != payload_keys:
        raise ValueError("remote quote baseline payload keys are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("remote quote baseline schema version is invalid")
    if (
        payload["dataset"] != "datasets/sft-quote-v4/dev.json"
        or payload["input_sha256"]
        != "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0"
    ):
        raise ValueError("remote quote-dev fixture path or SHA-256 is not allowlisted")
    if (
        payload["prompt_source_sha256"]
        != "c00cdb2a56c2a1132bade2ad6dda4ad573ac5e49502fe537553195e50e0eb5ce"
    ):
        raise ValueError("remote quote prompt source hash is not allowlisted")
    if (
        payload["runner_source_sha256"] != approved_runner_hash
        or re.fullmatch(r"[0-9a-f]{64}", payload["runner_source_sha256"]) is None
    ):
        raise ValueError("remote quote runner hash does not match the approved runner")
    prompt_versions = {"quote-v2", "quote-coverage-v1"}
    model_pins = {
        "qwen3-4b": {
            "key": "qwen3-4b",
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "license": "apache-2.0",
            "gated": False,
        },
        "qwen3-8b": {
            "key": "qwen3-8b",
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "license": "apache-2.0",
            "gated": False,
        },
    }
    model = payload["model"]
    if type(model) is not dict or model not in model_pins.values():
        raise ValueError("remote quote baseline model is not allowlisted")
    max_new_tokens = 1024
    max_prompt_tokens = 4096
    settings = {
        "precision": "bfloat16",
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "do_sample": False,
        "num_beams": 1,
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "use_safetensors": True,
        "enable_thinking": False,
    }
    if payload["settings"] != settings:
        raise ValueError("remote quote generation settings changed")
    items = payload["items"]
    if type(items) is not list or len(items) != 24:
        raise ValueError("remote quote baseline must contain exactly 24 prompt items")
    try:
        canonical_items = remote_json.dumps(
            items, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("remote quote prompt items are not canonical JSON") from exc
    if remote_hashlib.sha256(canonical_items).hexdigest() != payload["items_sha256"]:
        raise ValueError("remote quote item hash does not match its prompts")

    case_variants: dict[str, set[str]] = {}
    item_ids: set[str] = set()
    max_prompt_bytes = 100_000
    for item in items:
        if type(item) is not dict or set(item) != {
            "item_id",
            "case_id",
            "prompt_version",
            "prompt",
        }:
            raise ValueError("remote quote items may contain only IDs, version, and prompt")
        item_id = item["item_id"]
        case_id = item["case_id"]
        version = item["prompt_version"]
        prompt = item["prompt"]
        if type(case_id) is not str or not case_id.strip():
            raise ValueError("remote quote case ID is invalid")
        if type(version) is not str or version not in prompt_versions:
            raise ValueError("remote quote item prompt version is not allowlisted")
        variants = case_variants.setdefault(case_id, set())
        if version in variants:
            raise ValueError("remote quote case has a duplicate prompt variant")
        if item_id != f"{case_id}::{version}" or item_id in item_ids:
            raise ValueError("remote quote item IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("remote quote prompt must be non-empty text")
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            raise ValueError("remote quote prompt exceeds the payload byte limit")
        variants.add(version)
        item_ids.add(item_id)
    if len(case_variants) != 12 or any(
        found != prompt_versions for found in case_variants.values()
    ):
        raise ValueError("each remote quote case must have both prompt variants exactly once")

    model_id = model["model_id"]
    revision = model["revision"]
    started = remote_time.perf_counter()
    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal A10G GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_fast=True,
    )
    chat_template = tokenizer.chat_template
    tokenizer_is_fast = tokenizer.is_fast
    if type(chat_template) is not str or not chat_template or type(tokenizer_is_fast) is not bool:
        raise ValueError("pinned Qwen tokenizer has invalid backend or chat template metadata")
    tokenizer_metadata = {
        "class": type(tokenizer).__name__,
        "backend": "fast" if tokenizer_is_fast else "slow",
        "chat_template_sha256": remote_hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda",
    )
    loaded_model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    generation = loaded_model.generation_config
    raw_eos = generation.eos_token_id
    eos_ids = [raw_eos] if type(raw_eos) is int else raw_eos
    if (
        type(eos_ids) is not list
        or not eos_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("pinned Qwen generation config has invalid eos_token_id")
    raw_pad = generation.pad_token_id
    pad_token_id = raw_pad if type(raw_pad) is int and raw_pad >= 0 else eos_ids[0]

    prepared: list[tuple[dict[str, str], Any, int, list[int]]] = []
    for item in items:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError("Qwen tokenizer returned an invalid prompt tensor")
        input_tokens = int(input_ids.shape[-1])
        if input_tokens <= 0 or input_tokens > max_prompt_tokens:
            raise ValueError("quote prompt exceeds the 4096-token limit; truncation is disabled")
        input_token_rows = input_ids.detach().cpu().tolist()
        if (
            type(input_token_rows) is not list
            or len(input_token_rows) != 1
            or type(input_token_rows[0]) is not list
            or len(input_token_rows[0]) != input_tokens
            or any(type(token_id) is not int for token_id in input_token_rows[0])
        ):
            raise ValueError("Qwen tokenizer returned invalid prompt token IDs")
        prepared.append((item, input_ids, input_tokens, input_token_rows[0]))

    outputs: list[dict[str, Any]] = []
    gpu_seconds = 0.0
    for item, input_ids, input_tokens, input_token_ids in prepared:
        input_ids = input_ids.to(loaded_model.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            generated = loaded_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_ids,
            )
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds += start_event.elapsed_time(end_event) / 1000
        generated_ids = generated[0, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        output_token_ids = generated_ids.detach().cpu().tolist()
        if (
            type(output_token_ids) is not list
            or len(output_token_ids) != output_tokens
            or any(type(token_id) is not int for token_id in output_token_ids)
        ):
            raise ValueError("Qwen model returned invalid completion token IDs")
        last_token_id = int(generated_ids[-1].item()) if output_tokens else None
        ended_with_eos = last_token_id in eos_ids if last_token_id is not None else False
        hit_max_new_tokens = output_tokens >= max_new_tokens
        finish_reason = (
            "empty"
            if output_tokens == 0
            else ("eos" if ended_with_eos else "length" if hit_max_new_tokens else "unknown")
        )
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if type(raw_output) is not str:
            raise ValueError("Qwen tokenizer returned a non-text output")
        outputs.append(
            {
                "item_id": item["item_id"],
                "case_id": item["case_id"],
                "prompt_version": item["prompt_version"],
                "raw_output": raw_output,
                "input_token_ids": input_token_ids,
                "output_token_ids": output_token_ids,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "last_token_id": last_token_id,
                "ended_with_eos": ended_with_eos,
                "hit_max_new_tokens": hit_max_new_tokens,
                "finish_reason": finish_reason,
            }
        )

    package_versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        package_name = distribution.metadata["Name"] if "Name" in distribution.metadata else None
        if package_name:
            package_versions[package_name] = distribution.version
    versions = dict(sorted(package_versions.items(), key=lambda row: row[0].casefold()))
    torch.cuda.synchronize()
    return {
        "schema_version": 1,
        "mode": "quote-baseline-prompt-only",
        "input_sha256": payload["input_sha256"],
        "payload_sha256": approved_hash,
        "items_sha256": payload["items_sha256"],
        "prompt_source_sha256": payload["prompt_source_sha256"],
        "runner_sha256": approved_runner_hash,
        "model": model,
        "settings": settings,
        "tokenizer": tokenizer_metadata,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_seconds": round(gpu_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "remote_function_elapsed_seconds": round(remote_time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "generation_eos_token_ids": eos_ids,
        "generation_pad_token_id": pad_token_id,
        "case_count": len(case_variants),
        "item_count": len(outputs),
        "generated_outputs": outputs,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
    }


def _build_modal_function() -> tuple[Any, Any]:
    """Build the only remote function; source uploads remain disabled."""
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-quote-baseline")
    function = app.function(image=image, **modal_function_options())(_remote_compute)
    return app, function


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-quote-baseline")
    parser.add_argument("--model", choices=("qwen3-4b", "qwen3-8b"), default="qwen3-4b")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument("--approve-runner-sha256")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--output", type=Path, help="exclusive path for the raw inference receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(args.model)
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if runner_hash != payload["runner_source_sha256"]:
            raise ValueError("current runner SHA-256 changed while preparing the payload")
        if not args.launch:
            if args.approve_payload_sha256 is not None or args.approve_runner_sha256 is not None:
                raise ValueError("hash approvals are accepted only together with --launch")
            if args.output is not None:
                raise ValueError("--output is available only together with --launch")
            prepared: dict[str, Any] = {
                "model": payload["model"],
                "item_count": len(payload["items"]),
                "input_sha256": payload["input_sha256"],
                "items_sha256": payload["items_sha256"],
                "prompt_source_sha256": payload["prompt_source_sha256"],
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "remote_started": False,
            }
            if args.show_payload:
                prepared["payload"] = payload
            _write_json(prepared)
            return 0
        if args.show_payload:
            raise ValueError("--show-payload is available only during preparation")
        if args.output is None:
            raise ValueError("--launch requires --output PATH for the exclusive receipt")
        if args.approve_payload_sha256 != digest:
            raise ValueError("--launch requires the exact prepared payload SHA-256")
        if args.approve_runner_sha256 != runner_hash:
            raise ValueError("--launch requires the exact prepared runner SHA-256")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as receipt_file:
            remote_started = False
            try:
                verify_personal_workspace()
                app, function = _build_modal_function()
                client_started = time.perf_counter()
                with app.run(detach=False):
                    remote_started = True
                    result = function.remote(
                        {
                            "payload": payload,
                            "payload_sha256": digest,
                            "runner_sha256": runner_hash,
                        }
                    )
                if type(result) is not dict:
                    raise ValueError("Modal returned an invalid quote baseline receipt")
                receipt = dict(result)
                receipt.update(
                    {
                        "modal_app_id": app.app_id,
                        "runner_sha256": runner_hash,
                        "client_call_wall_seconds_including_image_and_startup": round(
                            time.perf_counter() - client_started, 6
                        ),
                    }
                )
                receipt_file.write(
                    json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                )
                receipt_file.flush()
            except Exception as exc:
                failure = {
                    "status": "launch_failed",
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                    "remote_started": remote_started,
                    "error": safe_exception_message(exc),
                }
                receipt_file.write(
                    json.dumps(failure, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
                receipt_file.flush()
                raise
        _write_json(
            {
                "receipt_path": str(args.output),
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "remote_started": True,
            }
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {safe_exception_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
