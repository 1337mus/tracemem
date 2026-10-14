"""Prepare or run one bounded, source-free scenario inference call on Modal."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from experiments.scenario_payload import (
    build_payload,
    payload_sha256,
    safe_exception_message,
    verify_launch_preconditions,
    verify_personal_workspace,
)

_IMAGE_PACKAGES = ("torch==2.7.1", "transformers==4.51.3", "accelerate==1.6.0")


def modal_function_options() -> dict[str, Any]:
    """Return the reviewed single-use, source-free Modal resource envelope."""
    return {
        "gpu": "T4",
        "cpu": (2, 2),
        "memory": (16_384, 16_384),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 300,
        "timeout": 1200,
        "scaledown_window": 2,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Source-free remote body; imports and all runtime state are local to this function."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re as remote_re
    import time as remote_time

    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    started = remote_time.perf_counter()
    if type(request) is not dict or set(request) != {"payload", "payload_sha256"}:
        raise ValueError("remote request keys are invalid")
    payload, approved_hash = request["payload"], request["payload_sha256"]
    if (
        type(approved_hash) is not str
        or remote_re.fullmatch(r"[0-9a-f]{64}", approved_hash) is None
    ):
        raise ValueError("remote payload hash is invalid")
    canonical = remote_json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if remote_hashlib.sha256(canonical).hexdigest() != approved_hash:
        raise ValueError("remote payload hash does not match approval")
    payload_keys = {
        "schema_version",
        "input_sha256",
        "prompt_source_sha256",
        "prompt_sha256",
        "prompt_version",
        "model",
        "items",
    }
    if type(payload) is not dict or set(payload) != payload_keys:
        raise ValueError("remote payload keys are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("remote payload schema version is invalid")
    if payload["prompt_version"] != "explicit-v2":
        raise ValueError("remote prompt version is not allowlisted")
    for name in ("input_sha256", "prompt_sha256"):
        value = payload[name]
        if type(value) is not str or remote_re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"remote {name} is invalid")
    if (
        payload["prompt_source_sha256"]
        != "2d866dd3203b5352f0d80ab2227a5ab8cf32efc2d759da0bcaf5e4af54525c57"
    ):
        raise ValueError("remote explicit-v2 prompt source hash is not allowlisted")

    settings = {
        "precision": "float16",
        "max_new_tokens": 1024,
        "max_prompt_tokens": 4096,
        "do_sample": False,
        "num_beams": 1,
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "use_safetensors": True,
        "enable_thinking": False,
    }
    supported = {
        "qwen3-4b": (
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
        ),
        "phi-3.5-mini": (
            "microsoft/Phi-3.5-mini-instruct",
            "2fe192450127e6a83f7441aef6e3ca586c338b77",
        ),
    }
    model_info = payload["model"]
    if type(model_info) is not dict:
        raise ValueError("remote model settings are invalid")
    model_key = model_info.get("key")
    if model_key not in supported:
        raise ValueError("remote model is not allowlisted")
    model_id, revision = supported[model_key]
    expected_model = {
        "key": model_key,
        "model_id": model_id,
        "revision": revision,
        **settings,
    }
    if model_info != expected_model:
        raise ValueError("remote model revision or generation settings are invalid")

    items = payload["items"]
    if type(items) is not list or len(items) != 48:
        raise ValueError("remote scenario payload must contain exactly 48 cases")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"case_id", "prompt"}:
            raise ValueError("remote payload items may contain only case_id and prompt")
        case_id, prompt = item["case_id"], item["prompt"]
        if type(case_id) is not str or not case_id.strip() or case_id in seen:
            raise ValueError("remote case IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote prompt is invalid or too large")
        seen.add(case_id)
    prompt_bytes = remote_json.dumps(
        items, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if remote_hashlib.sha256(prompt_bytes).hexdigest() != payload["prompt_sha256"]:
        raise ValueError("remote prompt hash does not match its items")

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal T4 GPU is unavailable")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda",
    )
    model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started

    generation = model.generation_config
    raw_eos = generation.eos_token_id
    eos_ids = [raw_eos] if type(raw_eos) is int else raw_eos
    if (
        type(eos_ids) is not list
        or not eos_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("pinned model generation config has invalid eos_token_id")
    raw_pad = generation.pad_token_id
    pad_token_id = raw_pad if type(raw_pad) is int and raw_pad >= 0 else eos_ids[0]

    prepared: list[tuple[dict[str, str], Any, int]] = []
    for item in items:
        template_kwargs = {"enable_thinking": False} if model_key == "qwen3-4b" else {}
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        )
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError("tokenizer returned an invalid prompt tensor")
        input_tokens = int(input_ids.shape[-1])
        if input_tokens <= 0 or input_tokens > model_info["max_prompt_tokens"]:
            raise ValueError("scenario prompt exceeds the 4096-token limit; truncation is disabled")
        prepared.append((item, input_ids, input_tokens))

    outputs: list[dict[str, Any]] = []
    gpu_seconds = 0.0
    for item, input_ids, input_tokens in prepared:
        input_ids = input_ids.to(model.device)
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=model_info["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_ids,
            )
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds += start_event.elapsed_time(end_event) / 1000
        generated_ids = generated[0, input_tokens:]
        generated_tokens = int(generated_ids.shape[-1])
        last_token_id = int(generated_ids[-1].item()) if generated_tokens else None
        finish_reason = "eos" if last_token_id in eos_ids else "length"
        hit_max_new_tokens = generated_tokens >= model_info["max_new_tokens"]
        if finish_reason == "length" and not hit_max_new_tokens:
            raise ValueError("generation stopped before eos or the configured token limit")
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not isinstance(raw_output, str) or len(raw_output) > 100_000:
            raise ValueError("model output is invalid or too large")
        outputs.append(
            {
                "case_id": item["case_id"],
                "raw_output": raw_output,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "hit_max_new_tokens": hit_max_new_tokens,
                "output_characters": len(raw_output),
                "finish_reason": finish_reason,
                "generated_last_token_id": last_token_id,
            }
        )

    package_versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        package_name = distribution.metadata["Name"]
        if package_name:
            package_versions[package_name] = distribution.version
    versions = dict(sorted(package_versions.items(), key=lambda row: row[0].casefold()))
    torch.cuda.synchronize()
    return {
        "schema_version": 1,
        "mode": "scenarios",
        "input_sha256": payload["input_sha256"],
        "payload_sha256": approved_hash,
        "prompt_source_sha256": payload["prompt_source_sha256"],
        "prompt_sha256": payload["prompt_sha256"],
        "prompt_version": payload["prompt_version"],
        "model": expected_model,
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
        "case_count": len(outputs),
        "generated_outputs": outputs,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
    }


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _build_modal_function() -> tuple[Any, Any]:
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-scenario-inference")
    function = app.function(image=image, **modal_function_options())(_remote_compute)
    return app, function


def _write_evaluation(input_path: Path, output_path: Path) -> None:
    from experiments.evaluate_scenarios import evaluate_receipt

    if input_path.resolve() == output_path.resolve():
        raise ValueError("evaluation output must not point to the input receipt")
    if output_path.exists() and input_path.exists() and output_path.samefile(input_path):
        raise ValueError("evaluation output must not alias the input receipt")
    report = evaluate_receipt(input_path.read_bytes())
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-scenarios")
    parser.add_argument("--model", choices=("qwen3-4b", "phi-3.5-mini"), default="qwen3-4b")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument(
        "--show-payload", action="store_true", help="print the complete review payload"
    )
    parser.add_argument("--evaluate", type=Path, help="evaluate one completed raw receipt locally")
    parser.add_argument(
        "--output", type=Path, help="exclusive path for the separate evaluation report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.evaluate is not None:
            if args.launch or args.approve_payload_sha256 is not None or args.show_payload:
                raise ValueError("receipt evaluation cannot be combined with launch options")
            if args.output is None:
                raise ValueError("--evaluate requires --output PATH")
            _write_evaluation(args.evaluate, args.output)
            return 0
        if args.output is not None:
            raise ValueError("--output is available only together with --evaluate")
        payload = build_payload(args.model)
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not args.launch:
            if args.approve_payload_sha256 is not None:
                raise ValueError("payload approval is only accepted together with --launch")
            prepared = {
                "model": payload["model"],
                "item_count": len(payload["items"]),
                "input_sha256": payload["input_sha256"],
                "prompt_source_sha256": payload["prompt_source_sha256"],
                "prompt_sha256": payload["prompt_sha256"],
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "remote_started": False,
            }
            if args.show_payload:
                prepared["payload"] = payload
            _write_json(prepared)
            return 0
        if args.approve_payload_sha256 != digest:
            raise ValueError("--launch requires the exact prepared payload SHA-256")
        if args.show_payload:
            raise ValueError("--show-payload is available only during preparation")
        verify_launch_preconditions(payload)
        verify_personal_workspace()
        app, function = _build_modal_function()
        started = time.perf_counter()
        with app.run(detach=False):
            result = function.remote({"payload": payload, "payload_sha256": digest})
        receipt = dict(result)
        receipt["modal_app_id"] = app.app_id
        receipt["runner_sha256"] = runner_hash
        receipt["client_call_wall_seconds_including_image_and_startup"] = round(
            time.perf_counter() - started, 6
        )
        _write_json(receipt)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {safe_exception_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
