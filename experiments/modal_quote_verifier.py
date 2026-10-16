"""Prepare or run one explicitly approved, source-free quote verifier inference call."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from experiments.quote_verifier_payload import build_preparation, payload_sha256
from experiments.scenario_payload import safe_exception_message, verify_personal_workspace

_IMAGE_PACKAGES = ("torch==2.7.1", "transformers==4.51.3", "accelerate==1.6.0")
_DEFAULT_AUDIT = (
    Path(__file__).resolve().parents[1]
    / ".local"
    / "quote-verifier-dev-20261016"
    / "selection-audit.json"
)


def modal_function_options() -> dict[str, Any]:
    """Return the bounded, single-use source-free inference envelope."""
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
    """Validate the exact approved payload before importing model runtimes."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re
    import time as remote_time

    if type(request) is not dict or set(request) != {"payload", "payload_sha256", "runner_sha256"}:
        raise ValueError("remote quote verifier request keys are invalid")
    payload = request["payload"]
    approved_hash = request["payload_sha256"]
    approved_runner = request["runner_sha256"]
    if any(
        type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (approved_hash, approved_runner)
    ):
        raise ValueError("remote quote verifier approval hashes are invalid")
    try:
        canonical = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote quote verifier payload is not canonical JSON") from exc
    if remote_hashlib.sha256(canonical).hexdigest() != approved_hash:
        raise ValueError("remote quote verifier payload hash does not match approval")

    keys = {
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
    }
    if type(payload) is not dict or set(payload) != keys:
        raise ValueError("remote quote verifier payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "semantic-pipeline"
    ):
        raise ValueError("remote quote verifier version or mode is invalid")
    if (
        payload["input_sha256"]
        != "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0"
        or payload["generator_receipt_sha256"]
        != "d47debe18192dae62759ed097017d2a9bcef92c148692b2b9a1e41e3805b2ac3"
    ):
        raise ValueError("remote quote verifier data hashes are not allowlisted")
    if (
        payload["candidate_sha256"]
        != "2e60e4c0b5eadff29b517814f6b1040d3e10e6b453ed796d6015fce5df8f3041"
    ):
        raise ValueError("remote quote verifier candidate hash is not allowlisted")
    if (
        payload["renderer_source_sha256"]
        != "69c754fea1ba097de98a36dd918c2598c49f5df617f9097ce735d9653743d82f"
        or payload["payload_source_sha256"]
        != "5dcf30b9aa5948c11942db383512008076abb3bb201ec7a1fa63ee105558cef2"
    ):
        raise ValueError("remote quote verifier renderer or builder hash is not allowlisted")
    if payload["runner_source_sha256"] != approved_runner:
        raise ValueError("remote quote verifier runner hash does not match approval")

    model = {
        "key": "qwen3-8b",
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "license": "apache-2.0",
        "gated": False,
    }
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
    if payload["model"] != model or payload["settings"] != settings:
        raise ValueError("remote quote verifier model or generation settings changed")
    counts = {
        "V4D-01": 3,
        "V4D-02": 4,
        "V4D-03": 4,
        "V4D-04": 3,
        "V4D-05": 6,
        "V4D-06": 4,
        "V4D-07": 7,
        "V4D-08": 2,
        "V4D-09": 4,
        "V4D-12": 2,
    }
    items = payload["items"]
    try:
        item_bytes = remote_json.dumps(
            items, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote quote verifier items are not canonical JSON") from exc
    item_hash = remote_hashlib.sha256(item_bytes).hexdigest()
    if (
        item_hash != payload["items_sha256"]
        or item_hash != "232465b09dde770e17d3811b4ac2384a961ebf09ad5b9adbbdf69d082cca88b6"
    ):
        raise ValueError("remote quote verifier item hash is not allowlisted")
    if type(items) is not list or len(items) != len(counts):
        raise ValueError("remote quote verifier must contain exactly 10 items")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {
            "case_id",
            "item_id",
            "prompt",
            "candidate_count",
        }:
            raise ValueError("remote quote verifier item keys are invalid")
        case_id, item_id, prompt, count = (
            item["case_id"],
            item["item_id"],
            item["prompt"],
            item["candidate_count"],
        )
        if (
            type(case_id) is not str
            or case_id not in counts
            or case_id in seen
            or item_id != f"{case_id}::quote-verifier-v1"
        ):
            raise ValueError("remote quote verifier item IDs are invalid or duplicated")
        if type(count) is not int or count != counts[case_id]:
            raise ValueError("remote quote verifier candidate count is invalid")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote quote verifier prompt is empty or exceeds the byte limit")
        seen.add(case_id)
    if seen != set(counts) or sum(item["candidate_count"] for item in items) != 39:
        raise ValueError("remote quote verifier IDs or candidate total are invalid")

    started = remote_time.perf_counter()
    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("Modal A10G GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model["model_id"], revision=model["revision"], trust_remote_code=False, use_fast=True
    )
    template = tokenizer.chat_template
    if type(template) is not str or not template or type(tokenizer.is_fast) is not bool:
        raise ValueError("pinned Qwen tokenizer metadata is invalid")
    tokenizer_info = {
        "class": type(tokenizer).__name__,
        "backend": "fast" if tokenizer.is_fast else "slow",
        "chat_template_sha256": remote_hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model["model_id"],
        revision=model["revision"],
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda",
    )
    loaded_model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    raw_eos = loaded_model.generation_config.eos_token_id
    eos_ids = [raw_eos] if type(raw_eos) is int else raw_eos
    if (
        type(eos_ids) is not list
        or not eos_ids
        or any(type(token) is not int or token < 0 for token in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("pinned Qwen EOS token IDs are invalid")
    raw_pad = loaded_model.generation_config.pad_token_id
    pad_id = raw_pad if type(raw_pad) is int and raw_pad >= 0 else eos_ids[0]
    prepared: list[tuple[dict[str, Any], Any, int, list[int]]] = []
    for item in items:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Qwen tokenizer returned an invalid prompt tensor")
        count = int(ids.shape[-1])
        if count <= 0 or count > max_prompt_tokens:
            raise ValueError("quote prompt exceeds 4096 tokens; truncation is disabled")
        token_rows = ids.detach().cpu().tolist()
        if (
            type(token_rows) is not list
            or len(token_rows) != 1
            or type(token_rows[0]) is not list
            or len(token_rows[0]) != count
            or any(type(token) is not int for token in token_rows[0])
        ):
            raise ValueError("Qwen tokenizer returned invalid prompt token IDs")
        prepared.append((item, ids, count, token_rows[0]))

    outputs: list[dict[str, Any]] = []
    gpu_seconds = 0.0
    for item, ids, input_count, input_tokens in prepared:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            generated = loaded_model.generate(
                ids.to(loaded_model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
            )
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds += start_event.elapsed_time(end_event) / 1000
        completion = generated[0, input_count:]
        output_count = int(completion.shape[-1])
        output_tokens = completion.detach().cpu().tolist()
        if (
            type(output_tokens) is not list
            or len(output_tokens) != output_count
            or any(type(token) is not int for token in output_tokens)
        ):
            raise ValueError("Qwen returned invalid completion token IDs")
        last = int(completion[-1].item()) if output_count else None
        eos = last in eos_ids if last is not None else False
        hit_limit = output_count >= max_new_tokens
        finish = (
            "empty" if not output_count else "eos" if eos else "length" if hit_limit else "unknown"
        )
        raw_output = tokenizer.decode(completion, skip_special_tokens=True)
        if type(raw_output) is not str:
            raise ValueError("Qwen tokenizer returned non-text output")
        outputs.append(
            {
                "case_id": item["case_id"],
                "item_id": item["item_id"],
                "raw_output": raw_output,
                "input_token_ids": input_tokens,
                "output_token_ids": output_tokens,
                "input_tokens": input_count,
                "output_tokens": output_count,
                "last_token_id": last,
                "ended_with_eos": eos,
                "hit_max_new_tokens": hit_limit,
                "finish_reason": finish,
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
        "mode": "quote-verifier-semantic",
        "input_sha256": payload["input_sha256"],
        "generator_receipt_sha256": payload["generator_receipt_sha256"],
        "candidate_sha256": payload["candidate_sha256"],
        "payload_sha256": approved_hash,
        "items_sha256": item_hash,
        "renderer_source_sha256": payload["renderer_source_sha256"],
        "payload_source_sha256": payload["payload_source_sha256"],
        "runner_sha256": approved_runner,
        "model": model,
        "settings": settings,
        "tokenizer": tokenizer_info,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "device": str(loaded_model.device),
        "gpu_compute_seconds": round(gpu_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "remote_function_elapsed_seconds": round(remote_time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "generation_eos_token_ids": eos_ids,
        "generation_pad_token_id": pad_id,
        "case_count": len(seen),
        "item_count": len(outputs),
        "generated_outputs": outputs,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
    }


def _build_modal_function() -> tuple[Any, Any]:
    """Create the only remote function with source uploads disabled."""
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-quote-verifier")
    function = app.function(image=image, **modal_function_options())(_remote_compute)
    return app, function


def _write_audit(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    try:
        with path.open("x", encoding="utf-8") as audit_file:
            audit_file.write(content)
    except FileExistsError as exc:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("selection audit exists with different contents") from exc


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-quote-verifier")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument("--approve-runner-sha256")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--output", type=Path, help="exclusive path for the raw inference receipt")
    parser.add_argument("--audit-output", type=Path, default=_DEFAULT_AUDIT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = build_preparation()
        payload = prepared["payload"]
        audit = prepared["audit"]
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if runner_hash != payload["runner_source_sha256"]:
            raise ValueError("current runner SHA-256 changed while preparing the payload")
        if not args.launch:
            if args.approve_payload_sha256 is not None or args.approve_runner_sha256 is not None:
                raise ValueError("hash approvals are accepted only together with --launch")
            if args.output is not None:
                raise ValueError("--output is available only together with --launch")
            _write_audit(args.audit_output, audit)
            result: dict[str, Any] = {
                "model": payload["model"],
                "item_count": len(payload["items"]),
                "input_sha256": payload["input_sha256"],
                "items_sha256": payload["items_sha256"],
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "audit_path": str(args.audit_output),
                "remote_started": False,
            }
            if args.show_payload:
                result["payload"] = payload
            _write_json(result)
            return 0
        if args.show_payload:
            raise ValueError("--show-payload is available only during preparation")
        if args.output is None:
            raise ValueError("--launch requires --output PATH for the exclusive receipt")
        if args.approve_payload_sha256 != digest:
            raise ValueError("--launch requires the exact prepared payload SHA-256")
        if args.approve_runner_sha256 != runner_hash:
            raise ValueError("--launch requires the exact prepared runner SHA-256")

        _write_audit(args.audit_output, audit)
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
                        {"payload": payload, "payload_sha256": digest, "runner_sha256": runner_hash}
                    )
                if type(result) is not dict:
                    raise ValueError("Modal returned an invalid quote verifier receipt")
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
