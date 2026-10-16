"""Pure SFT preprocessing plus an explicitly remote-only training entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
SEED = 17
MAX_TRAIN_ROWS = 128
MAX_TRAIN_STEPS = 32
GRADIENT_ACCUMULATION_STEPS = 4
MICROBATCH_SIZE = 1
LEARNING_RATE = 2e-4
MAX_TRAIN_CONTEXT_TOKENS = 2048
_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_PREFLIGHT_DEPENDENCIES = {
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "jinja2": "3.1.6",
    "huggingface-hub": "0.30.2",
    "safetensors": "0.5.3",
}
_TRAIN_DEPENDENCIES = {
    **_PREFLIGHT_DEPENDENCIES,
    "torch": "2.7.1",
    "accelerate": "1.6.0",
    "peft": "0.15.2",
    "bitsandbytes": "0.45.5",
}
_DEV_MODEL_CONTEXT = 4096
_DEV_MAX_NEW_TOKENS = 1024
_QUOTE_V3_COHORT_SIZES = {"regression": 24, "dev": 24, "test": 48}
_QUOTE_V3_PRIOR_ADAPTER = {
    "path": "/mnt/tracemem-sft-quote-v2/adapters/quote-v2-20261015-01/adapter",
    "sha256": {
        "adapter_model.safetensors": (
            "3b30cbf98a685792734d8c16ffd7f31e000707f8f2b72a74e5ca9bc364214093"
        ),
        "adapter_config.json": "d6cb445350cae1da7b9508d1cf4cc9d20f9b638983e7c27a886127db0b28125a",
    },
}


@dataclass(frozen=True)
class SFTProfile:
    name: str
    train_rows: int
    train_steps: int
    dev_prompts: int
    dev_max_new_tokens: int
    train_context_tokens: int
    dev_context_tokens: int


_SFT_PROFILES: Mapping[str, SFTProfile] = MappingProxyType(
    {
        "pilot": SFTProfile(
            "pilot",
            MAX_TRAIN_ROWS,
            MAX_TRAIN_STEPS,
            8,
            _DEV_MAX_NEW_TOKENS,
            MAX_TRAIN_CONTEXT_TOKENS,
            _DEV_MODEL_CONTEXT,
        ),
        "quote-v2": SFTProfile("quote-v2", 768, 192, 24, 512, 2048, 4096),
        "quote-v3": SFTProfile("quote-v3", 768, 192, 96, 512, 2048, 4096),
    }
)


def resolve_sft_profile(name: str = "pilot") -> SFTProfile:
    if type(name) is not str or name not in _SFT_PROFILES:
        raise ValueError("profile must be pilot, quote-v2, or quote-v3")
    return _SFT_PROFILES[name]


def encode_sft_example(
    row: Mapping[str, Any], tokenizer: Any, *, max_context_tokens: int = 2048
) -> dict[str, Any]:
    if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int):
        raise TypeError("max_context_tokens must be a positive integer")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be a positive integer")
    encoded = _encode_unbounded_example(row, tokenizer)
    if len(encoded["input_ids"]) > max_context_tokens:
        case_id = row.get("case_id", "<unknown>")
        raise ValueError(
            f"case {case_id} has {len(encoded['input_ids'])} tokens, "
            f"over the {max_context_tokens}-token limit"
        )
    return encoded


def _encode_unbounded_example(row: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = row.get("messages")
    if type(messages) is not list or len(messages) != 2:
        raise ValueError("an SFT row must contain exactly one user prompt and one assistant target")
    user_message, assistant_message = messages
    if (
        not isinstance(user_message, Mapping)
        or user_message.get("role") != "user"
        or type(user_message.get("content")) is not str
    ):
        raise ValueError("the first SFT message must be a user prompt")
    if (
        not isinstance(assistant_message, Mapping)
        or assistant_message.get("role") != "assistant"
        or type(assistant_message.get("content")) is not str
    ):
        raise ValueError("the second SFT message must be an assistant target")

    prefix_ids = _encode_user_prompt(user_message["content"], tokenizer)
    completion_ids = _as_token_ids(
        tokenizer.encode(assistant_message["content"], add_special_tokens=False),
        "assistant completion",
    )
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    input_ids = prefix_ids + completion_ids + [eos_token_id]
    return {
        "case_id": row.get("case_id"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + completion_ids + [eos_token_id],
        "prefix_tokens": len(prefix_ids),
        "completion_tokens": len(completion_ids),
        "supervised_tokens": len(completion_ids) + 1,
    }


def _encode_user_prompt(prompt: str, tokenizer: Any) -> list[int]:
    if type(prompt) is not str:
        raise ValueError("user prompt must be a string")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_token_ids(rendered, "chat template")


def _as_token_ids(value: Any, name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if type(value) is tuple:
        value = list(value)
    if type(value) is not list:
        raise ValueError(f"{name} tokenization must return a one-dimensional ID list")
    if value and type(value[0]) is list:
        if len(value) != 1:
            raise ValueError(f"{name} tokenization returned more than one sequence")
        value = value[0]
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in value):
        raise ValueError(f"{name} tokenization returned a non-integer token ID")
    return cast(list[int], value)


def collate_examples(
    examples: Sequence[Mapping[str, Any]], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)) or not examples:
        raise ValueError("examples must be a non-empty sequence")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("pad_token_id must be an integer")
    lengths: list[int] = []
    for example in examples:
        input_ids = example.get("input_ids")
        attention_mask = example.get("attention_mask")
        labels = example.get("labels")
        if (
            type(input_ids) is not list
            or type(attention_mask) is not list
            or type(labels) is not list
            or not input_ids
            or len(input_ids) != len(attention_mask)
            or len(input_ids) != len(labels)
        ):
            raise ValueError("example IDs, masks, and labels must be non-empty equal-length lists")
        lengths.append(len(input_ids))
    max_length = max(lengths)
    batch_ids: list[list[int]] = []
    batch_masks: list[list[int]] = []
    batch_labels: list[list[int]] = []
    for example, length in zip(examples, lengths, strict=True):
        pad_length = max_length - length
        batch_ids.append([*example["input_ids"], *([pad_token_id] * pad_length)])
        batch_masks.append([*example["attention_mask"], *([0] * pad_length)])
        batch_labels.append([*example["labels"], *([-100] * pad_length)])
    return {"input_ids": batch_ids, "attention_mask": batch_masks, "labels": batch_labels}


def select_training_rows(
    rows: Sequence[Mapping[str, Any]], *, max_rows: int = 128, seed: int = 17
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mappings")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:max_rows]


def preflight_train_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    dev_prompts: Sequence[Mapping[str, Any]],
    max_context_tokens: int = 2048,
    max_rows: int | None = None,
    seed: int = 17,
    selected_in_order: bool = False,
    profile: str = "pilot",
) -> dict[str, Any]:
    config = resolve_sft_profile(profile)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("training rows must be a non-empty sequence")
    if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int):
        raise TypeError("max_context_tokens must be a positive integer")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be a positive integer")
    if config.name != "pilot" and max_context_tokens != config.train_context_tokens:
        raise ValueError(f"{profile} training context limit is fixed")
    if config.name != "pilot" and seed != SEED:
        raise ValueError(f"{profile} seed is fixed")
    effective_max_rows = config.train_rows if max_rows is None else max_rows
    if config.name != "pilot" and effective_max_rows != config.train_rows:
        raise ValueError(f"{profile} training row limit is fixed")
    clean_dev_prompts = _validated_dev_prompts(dev_prompts, config)

    train_report: list[dict[str, Any]] = []
    oversized_train: list[str] = []
    selected = (
        list(rows)
        if selected_in_order
        else select_training_rows(rows, max_rows=effective_max_rows, seed=seed)
    )
    if selected_in_order and len(selected) != effective_max_rows:
        raise ValueError(
            f"preselected training rows must contain exactly {effective_max_rows} rows"
        )
    selected_ids: set[str] = set()
    for row in selected:
        if not isinstance(row, Mapping) or row.get("split") != "train":
            raise ValueError("preflight training rows must have split='train'")
        case_id = row.get("case_id")
        if type(case_id) is not str or not case_id.strip():
            raise ValueError("preflight training rows must have a non-blank case_id")
        if case_id in selected_ids:
            raise ValueError("preflight training row case IDs must be unique")
        selected_ids.add(case_id)
        encoded = _encode_unbounded_example(row, tokenizer)
        total_tokens = len(encoded["input_ids"])
        within_limit = total_tokens <= max_context_tokens
        if not within_limit:
            oversized_train.append(case_id)
        train_report.append(
            {
                "case_id": case_id,
                "prefix_tokens": encoded["prefix_tokens"],
                "completion_tokens": encoded["completion_tokens"],
                "eos_tokens": 1,
                "total_tokens": total_tokens,
                "supervised_tokens": encoded["supervised_tokens"],
                "prefix_masked": all(
                    label == -100 for label in encoded["labels"][: encoded["prefix_tokens"]]
                ),
                "eos_supervised": encoded["labels"][-1] == tokenizer.eos_token_id
                and encoded["labels"][-1] != -100,
                "within_context_limit": within_limit,
            }
        )

    dev_report: list[dict[str, Any]] = []
    oversized_dev: list[str] = []
    for case_id, prompt_text in clean_dev_prompts:
        prefix_tokens = len(_encode_user_prompt(prompt_text, tokenizer))
        required_context = prefix_tokens + config.dev_max_new_tokens
        within_limit = required_context <= config.dev_context_tokens
        if not within_limit:
            oversized_dev.append(case_id)
        dev_report.append(
            {
                "case_id": case_id,
                "prefix_tokens": prefix_tokens,
                "max_new_tokens": config.dev_max_new_tokens,
                "required_context_tokens": required_context,
                "within_context_limit": within_limit,
            }
        )

    rejected = bool(oversized_train or oversized_dev)
    limits: dict[str, Any] = {
        "train_context_tokens": max_context_tokens,
        "dev_context_tokens": config.dev_context_tokens,
        "dev_max_new_tokens": config.dev_max_new_tokens,
        "max_train_rows": effective_max_rows,
        "seed": seed,
        "selection_order": "input" if selected_in_order else "seeded_shuffle",
    }
    result: dict[str, Any] = {
        "status": "rejected_oversized" if rejected else "ready",
        "limits": limits,
        "train": {
            "selected_rows": len(train_report),
            "selected_case_ids": [item["case_id"] for item in train_report],
            "input_tokens": sum(item["total_tokens"] for item in train_report),
            "supervised_tokens": sum(item["supervised_tokens"] for item in train_report),
            "oversized_case_ids": oversized_train,
            "rows": train_report,
        },
        "dev": {
            "prompt_count": len(dev_report),
            "oversized_case_ids": oversized_dev,
            "rows": dev_report,
        },
    }
    if config.name != "pilot":
        limits["max_train_steps"] = config.train_steps
        limits["profile"] = config.name
    return result


def run_remote_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if os.environ.get("TRACEMEM_SFT_REMOTE") != "1":
        raise RuntimeError("SFT request requires the configured remote runtime")
    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    mode = request.get("mode")
    if type(mode) is not str or mode not in {"preflight", "train"}:
        raise ValueError("mode must be preflight or train")
    request_keys = {"mode", "train_rows", "dev_prompts", "remote_run_dir"}
    profile_name = request.get("profile", "pilot")
    if "profile" in request:
        if profile_name not in {"quote-v2", "quote-v3"}:
            raise ValueError("profile field is only valid for quote-v2 or quote-v3")
        expected_keys = request_keys | {"profile"}
        if profile_name == "quote-v3":
            expected_keys |= {"eval_cohorts", "prior_adapter"}
    else:
        expected_keys = request_keys
    config = resolve_sft_profile(profile_name)
    if set(request) != expected_keys:
        raise ValueError("request must contain exactly the fixed fields for its SFT profile")
    train_rows = request["train_rows"]
    dev_prompts = request["dev_prompts"]
    if type(train_rows) is not list or any(not isinstance(row, Mapping) for row in train_rows):
        raise ValueError("train_rows must be a list of row objects")
    if type(dev_prompts) is not list:
        raise ValueError("dev_prompts must be a list of prompt objects")
    if config.name == "quote-v3":
        _validate_quote_v3_metadata(request, train_rows, dev_prompts)
    remote_run_dir = request["remote_run_dir"]
    if mode == "preflight" and remote_run_dir is not None:
        raise ValueError("remote_run_dir must be null for preflight")
    if mode == "train":
        if type(remote_run_dir) is not str or not remote_run_dir.strip():
            raise ValueError("remote_run_dir must be a non-blank absolute path for training")
        run_path = Path(remote_run_dir)
        if not run_path.is_absolute() or ".." in run_path.parts:
            raise ValueError("remote_run_dir must be a normalized absolute path")
        if run_path.exists() or run_path.is_symlink():
            raise ValueError("remote_run_dir must be fresh and unused")
    if mode == "preflight":
        versions = _check_dependency_versions(_PREFLIGHT_DEPENDENCIES)
        tokenizer = _load_pinned_tokenizer()
        tokenizer_provenance = _tokenizer_provenance(tokenizer, versions)
        report = preflight_train_rows(
            train_rows,
            tokenizer,
            dev_prompts=dev_prompts,
            selected_in_order=True,
            profile=config.name,
        )
        return {
            "mode": "preflight",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dependency_versions": versions,
            "tokenizer_provenance": tokenizer_provenance,
            **report,
        }
    return _run_remote_training(request)


def _validated_dev_prompts(
    prompts: Sequence[Mapping[str, Any]], profile: SFTProfile | None = None
) -> list[tuple[str, str]]:
    config = profile or resolve_sft_profile()
    if type(prompts) is not list or len(prompts) != config.dev_prompts:
        raise ValueError(f"preflight requires exactly {config.dev_prompts} DEV prompt-only objects")
    validated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for prompt in prompts:
        if type(prompt) is not dict or set(prompt) != {"case_id", "prompt"}:
            raise ValueError("each DEV prompt must contain only case_id and prompt")
        case_id, prompt_text = prompt["case_id"], prompt["prompt"]
        if type(case_id) is not str or not case_id.strip() or case_id in seen:
            raise ValueError("DEV prompt case IDs must be non-blank and unique")
        if type(prompt_text) is not str or not prompt_text.strip():
            raise ValueError("DEV prompt text must be non-blank")
        seen.add(case_id)
        validated.append((case_id, prompt_text))
    return validated


def _validate_quote_v3_metadata(
    request: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    dev_prompts: Sequence[Mapping[str, Any]],
) -> None:
    if request["prior_adapter"] != _QUOTE_V3_PRIOR_ADAPTER:
        raise ValueError("quote-v3 prior_adapter must match the fixed quote-v2 checkpoint")
    cohorts = request["eval_cohorts"]
    if type(cohorts) is not dict or set(cohorts) != set(_QUOTE_V3_COHORT_SIZES):
        raise ValueError("quote-v3 eval_cohorts must contain regression, dev, and test")
    cohort_ids: list[str] = []
    seen: set[str] = set()
    for name, expected_count in _QUOTE_V3_COHORT_SIZES.items():
        values = cohorts[name]
        if type(values) is not list or len(values) != expected_count:
            raise ValueError(f"quote-v3 {name} cohort must contain {expected_count} case IDs")
        for case_id in values:
            if type(case_id) is not str or not case_id.strip() or case_id in seen:
                raise ValueError("quote-v3 cohort case IDs must be non-blank and globally unique")
            seen.add(case_id)
            cohort_ids.append(case_id)
    if any(
        type(prompt) is not dict or set(prompt) != {"case_id", "prompt"} for prompt in dev_prompts
    ):
        raise ValueError("each DEV prompt must contain only case_id and prompt")
    prompt_ids = [prompt["case_id"] for prompt in dev_prompts]
    if prompt_ids != cohort_ids:
        raise ValueError("quote-v3 dev_prompts must follow eval_cohorts order exactly")
    train_ids = {row.get("case_id") for row in train_rows if type(row.get("case_id")) is str}
    if seen & train_ids:
        raise ValueError("quote-v3 evaluation case IDs must be disjoint from training rows")


def _check_dependency_versions(required: Mapping[str, str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in required.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required remote package is missing: {package}=={expected}"
            ) from exc
        if actual != expected:
            raise RuntimeError(f"remote package {package} must be {expected}; found {actual}")
        versions[package] = actual
    return versions


def _tokenizer_provenance(tokenizer: Any, versions: Mapping[str, str]) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if type(chat_template) is not str:
        raise RuntimeError("pinned tokenizer must expose its chat template as text")
    chat_template_hash = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if chat_template_hash != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise RuntimeError(
            "pinned tokenizer chat template fingerprint mismatch: "
            f"expected {EXPECTED_CHAT_TEMPLATE_SHA256}, found {chat_template_hash}"
        )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise RuntimeError("pinned tokenizer must expose its backend tokenizer JSON")
    backend_json = to_str()
    if type(backend_json) is not str:
        raise RuntimeError("backend tokenizer JSON must be text")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "chat_template_sha256": chat_template_hash,
        "backend_tokenizer_sha256": hashlib.sha256(backend_json.encode("utf-8")).hexdigest(),
        "dependency_versions": dict(versions),
    }


def _load_pinned_tokenizer() -> Any:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        use_fast=True,
    )


def _load_remote_sft() -> Any:
    remote_path = Path(__file__).with_name("remote_sft.py")
    spec = importlib.util.spec_from_file_location("_tracemem_remote_sft", remote_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load remote SFT trainer from {remote_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_remote_training(request: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _load_remote_sft().run_training(request, sys.modules[__name__]))
