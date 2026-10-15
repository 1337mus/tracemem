"""Run the approved source-free QA contrast answering diagnostic."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from experiments import qa_contrast_answer_receipt as _receipt

_EXPECTED_RECEIPT_SOURCE_SHA256 = "63e9bcfa37447f9b6e81b091d4f928bb1ba36668426260c5d99239156a7b0461"
_EXPECTED_PAYLOAD_SHA256 = _receipt._EXPECTED_PAYLOAD_SHA256


def _verify_receipt_source() -> None:
    digest = hashlib.sha256(Path(_receipt.__file__).read_bytes()).hexdigest()
    if digest != _EXPECTED_RECEIPT_SOURCE_SHA256:
        raise ValueError("QA contrast answer receipt source differs from the reviewed source")


def _canonical_bytes(value: Any) -> bytes:
    _verify_receipt_source()
    return _receipt._canonical_bytes(value)


def payload_sha256(payload: dict[str, Any]) -> str:
    _verify_receipt_source()
    return _receipt.payload_sha256(payload)


def build_payload() -> dict[str, Any]:
    _verify_receipt_source()
    return _receipt.build_payload()


def validate_payload(payload: Any) -> dict[str, Any]:
    _verify_receipt_source()
    return _receipt.validate_payload(payload)


def validate_result(result: Any, payload: dict[str, Any], runner_sha256: str) -> dict[str, Any]:
    _verify_receipt_source()
    return _receipt.validate_result(result, payload, runner_sha256)


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Source-free inference body; all imports and dependencies are local to this function."""
    import hashlib
    import importlib.metadata
    import json
    import platform
    import re
    import time

    def canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")

    expected_hash = "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e"
    expected_model = {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "precision": "float16",
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "use_safetensors": True,
        "use_fast": True,
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "seed": 0,
        "max_new_tokens": 128,
        "max_prompt_tokens": 1024,
    }
    if type(request) is not dict or set(request) != {"payload", "payload_sha256", "runner_sha256"}:
        raise ValueError("remote wrapper keys are invalid")
    payload, payload_hash, runner_hash = (
        request["payload"],
        request["payload_sha256"],
        request["runner_sha256"],
    )
    if type(payload_hash) is not str or payload_hash != expected_hash:
        raise ValueError("remote payload hash is not the reviewed hash")
    if type(runner_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", runner_hash) is None:
        raise ValueError("remote runner hash is invalid")
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != expected_hash:
        raise ValueError("remote payload bytes do not match the reviewed payload hash")
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "mode",
        "model",
        "tokenizer",
        "provenance",
        "items",
    }:
        raise ValueError("remote payload shape is invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "qa-contrast-dev-source-only-requests-v1"
        or payload["model"] != expected_model
    ):
        raise ValueError("remote payload settings differ from the reviewed configuration")
    items = payload["items"]
    if type(items) is not list or len(items) != 46:
        raise ValueError("remote payload must contain exactly 46 prompts")
    prompt_ids, total_tokens = [], 0
    for item in items:
        if type(item) is not dict or set(item) != {"prompt_id", "messages", "prompt_tokens"}:
            raise ValueError("remote prompt fields are invalid")
        prompt_id, messages, count = item["prompt_id"], item["messages"], item["prompt_tokens"]
        if (
            type(prompt_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", prompt_id) is None
            or type(messages) is not list
            or len(messages) != 2
            or any(
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["role"]) is not str
                or type(message["content"]) is not str
                for message in messages
            )
            or [message["role"] for message in messages] != ["system", "user"]
            or hashlib.sha256(canonical_bytes(messages)).hexdigest() != prompt_id
            or type(count) is not int
            or not 1 <= count <= 1024
        ):
            raise ValueError("remote prompt identity, messages, or token count are invalid")
        prompt_ids.append(prompt_id)
        total_tokens += count
    if prompt_ids != sorted(set(prompt_ids)) or total_tokens != 23533:
        raise ValueError("remote prompt IDs or measured input token total are invalid")

    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        expected_model["model_id"],
        revision=expected_model["revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.is_fast is not True or tokenizer.chat_template is None:
        raise ValueError("pinned fast tokenizer or chat template is unavailable")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    backend = tokenizer.backend_tokenizer.to_str()
    backend_hash = hashlib.sha256(backend.encode("utf-8")).hexdigest()
    expected_template = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    expected_backend = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"
    if template_hash != expected_template or backend_hash != expected_backend:
        raise ValueError(
            "pinned tokenizer template or backend hash differs from the reviewed tokenizer"
        )
    tokenized = []
    for item in items:
        ids = tokenizer.apply_chat_template(
            item["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if type(ids) is not list or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("tokenizer returned invalid token IDs")
        if len(ids) != item["prompt_tokens"]:
            raise ValueError(f"tokenizer input count differs for prompt {item['prompt_id']}")
        tokenized.append(ids)
    package_versions = {
        name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")
    }
    if package_versions != {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}:
        raise ValueError("tokenizer package versions differ from the reviewed pins")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if type(eos_id) is not int or type(pad_id) is not int:
        raise ValueError("tokenizer EOS and pad token IDs must be explicit integers")

    import modal
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal T4 GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        expected_model["model_id"],
        revision=expected_model["revision"],
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    gpu_compute_seconds = 0.0
    outputs = []
    with torch.inference_mode():
        for item, ids in zip(items, tokenized, strict=True):
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            sequence = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )[0].tolist()
            end_event.record()
            torch.cuda.synchronize()
            gpu_compute_seconds += start_event.elapsed_time(end_event) / 1000
            generated_ids = sequence[len(ids) :]
            if len(generated_ids) > 128:
                raise ValueError("model generated more than the fixed token cap")
            if generated_ids and generated_ids[-1] == eos_id:
                stop_reason = "eos_token"
            elif len(generated_ids) == 128:
                stop_reason = "max_new_tokens"
            else:
                raise ValueError("generation stopped without EOS or the token ceiling")
            hit_cap = len(generated_ids) == 128
            outputs.append(
                {
                    "prompt_id": item["prompt_id"],
                    "raw_output": tokenizer.decode(generated_ids, skip_special_tokens=True),
                    "input_tokens": len(ids),
                    "generated_tokens": len(generated_ids),
                    "generated_token_ids": generated_ids,
                    "stop_reason": stop_reason,
                    "hit_max_new_tokens": hit_cap,
                    "truncated": stop_reason == "max_new_tokens",
                }
            )
    runtime = {
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "accelerate_version": importlib.metadata.version("accelerate"),
        "cuda_version": torch.version.cuda or "",
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": f"{properties.major}.{properties.minor}",
        "model_load_seconds": round(load_seconds, 6),
        "gpu_compute_seconds": round(gpu_compute_seconds, 6),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "request_count": len(outputs),
        "modal_call_id": modal.current_function_call_id(),
    }
    return {
        "schema_version": 1,
        "payload_sha256": payload_hash,
        "runner_sha256": runner_hash,
        "model": {
            "model_id": expected_model["model_id"],
            "revision": expected_model["revision"],
            "generation_protocol_sha256": hashlib.sha256(
                canonical_bytes(payload["model"])
            ).hexdigest(),
        },
        "tokenizer": {
            "model_id": expected_model["model_id"],
            "revision": expected_model["revision"],
            "template_sha256": template_hash,
            "backend_tokenizer_sha256": backend_hash,
            "use_fast": tokenizer.is_fast,
            "trust_remote_code": False,
            "package_versions": package_versions,
            "python_version": platform.python_version(),
            "template_options": {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "truncation": False,
            },
            "eos_token_id": eos_id,
        },
        "runtime": runtime,
        "outputs": outputs,
    }


def modal_function_options() -> dict[str, Any]:
    return {
        "gpu": "T4",
        "cpu": (2, 2),
        "memory": (16384, 16384),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "single_use_containers": True,
        "retries": 0,
        "startup_timeout": 300,
        "timeout": 600,
        "scaledown_window": 2,
        "serialized": True,
        "include_source": False,
    }


def _build_modal_function() -> tuple[Any, Any]:
    """Construct the source-free, single-container Modal function after local approval."""
    import types

    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.7.1",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "tokenizers==0.21.1",
        "jinja2==3.1.6",
    )
    app = modal.App("tracemem-qa-contrast-answer")
    remote_function = types.FunctionType(_remote_compute.__code__, {"__name__": "__main__"})
    remote_function.__doc__ = _remote_compute.__doc__
    function = app.function(image=image, **modal_function_options())(remote_function)
    return app, function


def verify_personal_workspace() -> None:
    """Apply the repository's existing personal-workspace gate only at launch time."""
    from experiments.scenario_payload import verify_personal_workspace as verify

    verify()


def _require_supported_launch_python() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "approved Modal launches require local Python 3.12 "
            f"(found {sys.version_info.major}.{sys.version_info.minor})"
        )


def _reserve_output(path: Path, message: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        if isinstance(error, FileExistsError) or error.errno == errno.ELOOP:
            raise FileExistsError(message) from None
        raise
    os.fchmod(descriptor, 0o600)
    return descriptor


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Perform one explicitly approved remote call and preserve its raw result immediately."""
    _require_supported_launch_python()
    expected_payload = build_payload()
    if type(payload) is not dict or _canonical_bytes(payload) != _canonical_bytes(expected_payload):
        raise ValueError("caller payload differs from the fixed reviewed payload")
    if approved_payload_sha256 != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("exact reviewed payload approval is required")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if approved_runner_sha256 != runner_hash:
        raise ValueError("exact current runner approval is required")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    raw_path = Path(f"{output_path}.remote.json")
    output_fd = _reserve_output(output_path, "receipt output path must be new")
    try:
        raw_fd = _reserve_output(raw_path, "raw remote output path must be new")
    except Exception:
        os.close(output_fd)
        raise
    started = time.perf_counter()
    with (
        os.fdopen(output_fd, "w", encoding="utf-8") as output,
        os.fdopen(raw_fd, "w", encoding="utf-8") as raw,
    ):
        try:
            verify_personal_workspace()
            app, function = _build_modal_function()
            with app.run(detach=False):
                remote_result = function.remote(
                    {
                        "payload": expected_payload,
                        "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                        "runner_sha256": runner_hash,
                    }
                )
                raw_wrapper = {
                    "schema_version": 1,
                    "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                    "runner_sha256": runner_hash,
                    "result": remote_result,
                }
                raw.write(
                    json.dumps(raw_wrapper, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
                )
                raw.flush()
                os.fsync(raw.fileno())
            result = validate_result(remote_result, expected_payload, runner_hash)
            receipt = {
                "schema_version": 1,
                "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                "runner_sha256": runner_hash,
                "client_elapsed_seconds": round(time.perf_counter() - started, 6),
                "result": result,
            }
            output.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            return receipt
        except Exception as error:
            # A returned raw result is durable and must survive teardown or validation errors.
            from experiments.scenario_payload import safe_exception_message

            failure = {
                "schema_version": 1,
                "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                "runner_sha256": runner_hash,
                "client_elapsed_seconds": round(time.perf_counter() - started, 6),
                "error": {"type": type(error).__name__, "message": safe_exception_message(error)},
            }
            output.write(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-qa-contrast-answer")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument(
        "--show-payload", action="store_true", help="print the fixed offline payload"
    )
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument("--approve-runner-sha256")
    parser.add_argument("--output", type=Path, help="new path for the local receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload()
        payload_hash = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not args.launch:
            if args.approve_payload_sha256 or args.approve_runner_sha256 or args.output:
                raise ValueError("approval flags and --output require --launch")
            value = (
                payload
                if args.show_payload
                else {
                    "item_count": len(payload["items"]),
                    "payload_sha256": payload_hash,
                    "runner_sha256": runner_hash,
                    "remote_started": False,
                }
            )
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        if args.show_payload:
            raise ValueError("--show-payload is only available during offline preparation")
        if (
            args.approve_payload_sha256 is None
            or args.approve_runner_sha256 is None
            or args.output is None
        ):
            raise ValueError("--launch requires both exact approvals and a new --output path")
        run_once(
            payload,
            approved_payload_sha256=args.approve_payload_sha256,
            approved_runner_sha256=args.approve_runner_sha256,
            output_path=args.output,
        )
        return 0
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        print(f"{type(error).__name__}: {safe_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
