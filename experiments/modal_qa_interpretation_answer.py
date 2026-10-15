"""Offline-first, source-free Modal producer for fictional QA controls answers."""

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

from experiments import qa_interpretation_answer_receipt as _receipt

_EXPECTED_CONTRACT_SOURCE_SHA256 = (
    "8f03a1c2b7767cdc5095fc51d28e066fa32750c127954ec0f6946862c8ee7278"
)
_EXPECTED_REQUEST_SHA256 = "df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871"
_EXPECTED_PAYLOAD_SHA256 = "36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1"


def _verify_contract_source() -> None:
    source = Path(_receipt.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != _EXPECTED_CONTRACT_SOURCE_SHA256:
        raise ValueError(
            "QA interpretation answer contract source differs from the reviewed source"
        )


def _canonical_bytes(value: Any) -> bytes:
    _verify_contract_source()
    return _receipt._canonical_bytes(value)


def request_sha256(request: Any) -> str:
    _verify_contract_source()
    return _receipt.request_sha256(request)


def build_request() -> dict[str, Any]:
    _verify_contract_source()
    request = _receipt.build_request()
    if _receipt.request_sha256(request) != _EXPECTED_REQUEST_SHA256:
        raise ValueError("QA interpretation answer request differs from the reviewed request")
    return request


def validate_request(request: Any) -> dict[str, Any]:
    _verify_contract_source()
    return _receipt.validate_request(request)


def validate_result(result: Any, request: Any, runner_sha256: str) -> dict[str, Any]:
    _verify_contract_source()
    return _receipt.validate_result(result, request, runner_sha256)


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


def verify_personal_workspace() -> None:
    from experiments.scenario_payload import verify_personal_workspace as verify

    verify()


def _require_supported_launch_python() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "approved Modal launches require local Python 3.12 "
            f"(found {sys.version_info.major}.{sys.version_info.minor})"
        )


def _build_modal_function(
    *, approved_request_sha256: str, approved_runner_sha256: str
) -> tuple[Any, Any]:
    """Build one source-free Modal function only after local launch approvals."""
    build_request()
    if approved_request_sha256 != _EXPECTED_REQUEST_SHA256:
        raise ValueError("exact reviewed request approval is required")
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if approved_runner_sha256 != runner_sha256:
        raise ValueError("exact current runner approval is required")
    _require_supported_launch_python()
    verify_personal_workspace()

    import types

    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.7.1",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "tokenizers==0.21.1",
        "jinja2==3.1.6",
    )
    app = modal.App("tracemem-qa-interpretation-answer")
    function_body = types.FunctionType(_remote_compute.__code__, {"__name__": "__main__"})
    function_body.__doc__ = _remote_compute.__doc__
    function = app.function(image=image, **modal_function_options())(function_body)
    return app, function


def _reserve_output(path: Path, message: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        if isinstance(error, FileExistsError) or error.errno == errno.ELOOP:
            raise FileExistsError(message) from None
        raise
    try:
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def run_once(
    request: dict[str, Any],
    *,
    approved_request_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Make one approved remote call and persist its raw return before teardown."""
    _require_supported_launch_python()
    expected_request = validate_request(request)
    if approved_request_sha256 != _EXPECTED_REQUEST_SHA256:
        raise ValueError("exact reviewed request approval is required")
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if approved_runner_sha256 != runner_sha256:
        raise ValueError("exact current runner approval is required")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    raw_path = Path(f"{output_path}.remote.json")
    output_fd = _reserve_output(output_path, "receipt output path must be new")
    try:
        raw_fd = _reserve_output(raw_path, "raw remote output path must be new")
    except BaseException:
        os.close(output_fd)
        raise
    started = time.perf_counter()
    with (
        os.fdopen(output_fd, "w", encoding="utf-8") as output,
        os.fdopen(raw_fd, "w", encoding="utf-8") as raw,
    ):
        try:
            app, function = _build_modal_function(
                approved_request_sha256=approved_request_sha256,
                approved_runner_sha256=approved_runner_sha256,
            )
            with app.run(detach=False):
                remote_result = function.remote(
                    {
                        "request": expected_request,
                        "request_sha256": _EXPECTED_REQUEST_SHA256,
                        "runner_sha256": runner_sha256,
                    }
                )
                raw_wrapper = {
                    "schema_version": 1,
                    "request_sha256": _EXPECTED_REQUEST_SHA256,
                    "runner_sha256": runner_sha256,
                    "result": remote_result,
                }
                raw.write(
                    json.dumps(raw_wrapper, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
                )
                raw.flush()
                os.fsync(raw.fileno())
            result = validate_result(remote_result, expected_request, runner_sha256)
            receipt = {
                "schema_version": 1,
                "request_sha256": _EXPECTED_REQUEST_SHA256,
                "runner_sha256": runner_sha256,
                "client_elapsed_seconds": round(time.perf_counter() - started, 6),
                "result": result,
            }
            output.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            return receipt
        except Exception as error:
            from experiments.scenario_payload import safe_exception_message

            failure = {
                "schema_version": 1,
                "request_sha256": _EXPECTED_REQUEST_SHA256,
                "runner_sha256": runner_sha256,
                "client_elapsed_seconds": round(time.perf_counter() - started, 6),
                "error": {
                    "type": type(error).__name__,
                    "message": safe_exception_message(error),
                },
            }
            output.write(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            raise


def _remote_compute(wrapper: dict[str, Any]) -> dict[str, Any]:
    """Source-free inference body; imports and dependencies stay local to this function."""
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

    expected_request_sha256 = "df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871"
    if type(wrapper) is not dict or set(wrapper) != {
        "request",
        "request_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote wrapper fields are invalid")
    request = wrapper["request"]
    request_hash = wrapper["request_sha256"]
    runner_hash = wrapper["runner_sha256"]
    if type(request_hash) is not str or request_hash != expected_request_sha256:
        raise ValueError("remote request hash is not the reviewed hash")
    if type(runner_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", runner_hash) is None:
        raise ValueError("remote runner hash is invalid")
    try:
        computed_request_hash = hashlib.sha256(canonical_bytes(request)).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("remote request is not canonical JSON") from None
    if computed_request_hash != expected_request_sha256:
        raise ValueError("remote request bytes differ from the reviewed hash")
    if type(request) is not dict or set(request) != {
        "schema_version",
        "mode",
        "prepared_payload_sha256",
        "model",
        "tokenizer",
        "items",
    }:
        raise ValueError("remote request envelope is invalid")
    if (
        type(request["schema_version"]) is not int
        or request["schema_version"] != 1
        or request["mode"] != "qa-interpretation-controls-answer-request-v1"
        or type(request["model"]) is not dict
        or type(request["tokenizer"]) is not dict
    ):
        raise ValueError("remote request settings differ from the reviewed configuration")
    # This full-request digest binds nested settings, prompt bytes, and item order.
    model_settings = request["model"]
    tokenizer_settings = request["tokenizer"]
    items = request["items"]
    if type(items) is not list or len(items) != 36:
        raise ValueError("remote request must contain exactly 36 prompts")
    for item in items:
        if (
            type(item) is not dict
            or set(item) != {"prompt_id", "messages"}
            or type(item["prompt_id"]) is not str
            or type(item["messages"]) is not list
            or len(item["messages"]) != 2
            or any(
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["role"]) is not str
                or type(message["content"]) is not str
                for message in item["messages"]
            )
            or [message["role"] for message in item["messages"]] != ["system", "user"]
        ):
            raise ValueError("remote source-only prompt shape is invalid")
    python_version = platform.python_version()
    if re.fullmatch(r"3\.12\.\d+", python_version) is None:
        raise RuntimeError("the Modal runtime must use the pinned Python 3.12 image")

    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_settings["model_id"],
        revision=model_settings["revision"],
        trust_remote_code=model_settings["trust_remote_code"],
        use_fast=model_settings["use_fast"],
    )
    if tokenizer.is_fast is not True or type(tokenizer.chat_template) is not str:
        raise ValueError("pinned fast tokenizer or chat template is unavailable")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    backend_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    if (
        template_hash != tokenizer_settings["template_sha256"]
        or backend_hash != tokenizer_settings["backend_tokenizer_sha256"]
    ):
        raise ValueError("pinned tokenizer template or backend hash differs from the request")
    package_versions = {
        name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")
    }
    if package_versions != tokenizer_settings["package_versions"]:
        raise ValueError("tokenizer package versions differ from the request")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if type(eos_id) is not int or eos_id != tokenizer_settings["eos_token_id"]:
        raise ValueError("tokenizer EOS ID differs from the request")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("tokenizer EOS and pad IDs must be nonnegative integers")
    template_options = tokenizer_settings["template_options"]
    tokenized = []
    for item in items:
        ids = tokenizer.apply_chat_template(item["messages"], **template_options)
        if type(ids) is not list or any(
            type(token_id) is not int or token_id < 0 for token_id in ids
        ):
            raise ValueError("tokenizer returned invalid prompt token IDs")
        if not 1 <= len(ids) <= model_settings["max_prompt_tokens"]:
            raise ValueError("tokenized prompt is empty or exceeds the fixed 1024-token limit")
        tokenized.append(ids)

    import torch

    torch_version = importlib.metadata.version("torch")
    accelerate_version = importlib.metadata.version("accelerate")
    if torch_version != "2.7.1" or accelerate_version != "1.6.0":
        raise ValueError("model runtime package versions differ from the pinned image")
    if not torch.cuda.is_available():
        raise RuntimeError("the Modal T4 GPU is unavailable")
    torch.manual_seed(model_settings["seed"])
    torch.cuda.manual_seed_all(model_settings["seed"])
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_capability = f"{properties.major}.{properties.minor}"
    if gpu_name != "Tesla T4" or gpu_capability != "7.5":
        raise RuntimeError("the Modal function did not receive the pinned Tesla T4 GPU")
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_settings["model_id"],
        revision=model_settings["revision"],
        trust_remote_code=model_settings["trust_remote_code"],
        use_safetensors=model_settings["use_safetensors"],
        torch_dtype=torch.float16,
        attn_implementation=model_settings["attn_implementation"],
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
                max_new_tokens=model_settings["max_new_tokens"],
                do_sample=model_settings["do_sample"],
                num_beams=model_settings["num_beams"],
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )[0].tolist()
            end_event.record()
            torch.cuda.synchronize()
            gpu_compute_seconds += start_event.elapsed_time(end_event) / 1000
            if type(sequence) is not list or sequence[: len(ids)] != ids:
                raise ValueError("model output does not retain the complete prompt token prefix")
            generated_ids = sequence[len(ids) :]
            if not 1 <= len(generated_ids) <= model_settings["max_new_tokens"] or any(
                type(token_id) is not int or token_id < 0 for token_id in generated_ids
            ):
                raise ValueError("model output token count is outside the fixed 128-token limit")
            eos_count = generated_ids.count(eos_id)
            if eos_count:
                if generated_ids[-1] != eos_id or eos_count != 1:
                    raise ValueError("model output has an invalid EOS stop sequence")
                stop_reason = "eos_token"
            elif len(generated_ids) == model_settings["max_new_tokens"]:
                stop_reason = "max_new_tokens"
            else:
                raise ValueError("generation stopped without EOS or the fixed token ceiling")
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if type(raw_output) is not str or len(raw_output) > 65_536:
                raise ValueError("decoded output is not a bounded string")
            outputs.append(
                {
                    "prompt_id": item["prompt_id"],
                    "raw_output": raw_output,
                    "input_tokens": len(ids),
                    "input_token_ids": ids,
                    "generated_tokens": len(generated_ids),
                    "generated_token_ids": generated_ids,
                    "stop_reason": stop_reason,
                    "hit_max_new_tokens": len(generated_ids) == model_settings["max_new_tokens"],
                    "truncated": stop_reason == "max_new_tokens",
                }
            )

    import modal

    return {
        "schema_version": 1,
        "request_sha256": request_hash,
        "runner_sha256": runner_hash,
        "model": {
            "model_id": model_settings["model_id"],
            "revision": model_settings["revision"],
            "generation_protocol_sha256": hashlib.sha256(
                canonical_bytes(request["model"])
            ).hexdigest(),
        },
        "tokenizer": {
            "model_id": model_settings["model_id"],
            "revision": model_settings["revision"],
            "template_sha256": template_hash,
            "backend_tokenizer_sha256": backend_hash,
            "use_fast": tokenizer.is_fast,
            "trust_remote_code": model_settings["trust_remote_code"],
            "package_versions": package_versions,
            "python_version": python_version,
            "template_options": template_options,
            "eos_token_id": eos_id,
        },
        "runtime": {
            "python_version": python_version,
            "torch_version": torch_version,
            "accelerate_version": accelerate_version,
            "cuda_version": torch.version.cuda or "",
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
            "model_load_seconds": round(load_seconds, 6),
            "gpu_compute_seconds": round(gpu_compute_seconds, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            "request_count": len(outputs),
            "modal_call_id": modal.current_function_call_id(),
        },
        "outputs": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-qa-interpretation-answer")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument(
        "--show-request", action="store_true", help="print the fixed offline request"
    )
    parser.add_argument("--approve-request-sha256")
    parser.add_argument("--approve-runner-sha256")
    parser.add_argument("--output", type=Path, help="new local receipt path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = build_request()
        request_hash = _EXPECTED_REQUEST_SHA256
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not args.launch:
            if (
                args.approve_request_sha256 is not None
                or args.approve_runner_sha256 is not None
                or args.output is not None
            ):
                raise ValueError("launch approval flags and --output require --launch")
            value = (
                request
                if args.show_request
                else {
                    "prompt_count": len(request["items"]),
                    "request_sha256": request_hash,
                    "runner_sha256": runner_hash,
                    "limits": {"max_prompt_tokens": 1024, "max_new_tokens": 128},
                    "input_tokens_measured": False,
                    "outputs_measured": False,
                    "remote_started": False,
                }
            )
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        if args.show_request:
            raise ValueError("--show-request is available only in offline mode")
        if (
            args.approve_request_sha256 is None
            or args.approve_runner_sha256 is None
            or args.output is None
        ):
            raise ValueError("--launch requires both exact approvals and a new --output path")
        run_once(
            request,
            approved_request_sha256=args.approve_request_sha256,
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
