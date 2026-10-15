"""Prepare source-only prompts for the frozen fictional QA contrast exercise."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments import modal_qa_contrast_token_count as token_counter
from experiments import prepare_qa_contrast, replay_qa_contrast_tokens

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "examples/synthetic/qa-contrast-dev-v1.json"
_RECEIPT_PATH = _ROOT / "examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json"
_FIXTURE_SHA256 = "b4fb6ea6242c294574ca9ed15549f5b524a77d3154b4c87b9831d2f357beefed"
_RECEIPT_SHA256 = "a48defa04fad82435a979980a3d327232b83260adb7eafe42aa165bef35400bd"
_REPLAY_SHA256 = "0b7efb6dc7e54d9e099333445ae6d5ae692d0f92e4f37e8c336e1c68246a27f4"
_MODES = ("bm25", "vector", "hybrid")
_CEILINGS = (128, 256, 512, 1024)
_MAX_PROMPT_TOKENS = 1024
_GENERATION_TOKENS = 128
_MODEL = {
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
    "max_new_tokens": _GENERATION_TOKENS,
    "max_prompt_tokens": _MAX_PROMPT_TOKENS,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _checked_replay(receipt_raw: bytes) -> dict[str, Any]:
    replay = replay_qa_contrast_tokens.replay_from_counts(receipt_raw)
    if _sha256(_canonical_bytes(replay)) != _REPLAY_SHA256:
        raise ValueError("complete token replay differs from the pinned canonical replay")
    return replay


def prepare_requests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return remote prompts, separate local references, and a local audit."""
    fixture_raw = _FIXTURE_PATH.read_bytes()
    if _sha256(fixture_raw) != _FIXTURE_SHA256:
        raise ValueError("QA contrast development fixture differs from the pinned fixture")
    receipt_raw = _RECEIPT_PATH.read_bytes()
    if _sha256(receipt_raw) != _RECEIPT_SHA256:
        raise ValueError("token count receipt differs from the pinned measured receipt")

    replay = _checked_replay(receipt_raw)
    catalog, authored_references, _ = prepare_qa_contrast.prepare_contrast(fixture_raw)
    provenance = replay["provenance"]
    catalog_sha256 = _sha256(_canonical_bytes(catalog))
    replay_source_sha256 = _sha256(Path(replay_qa_contrast_tokens.__file__).read_bytes())
    if (
        catalog_sha256 != provenance["catalog_sha256"]
        or catalog["prompt_sha256"] == ""
        or catalog["source_sha256"] != provenance["preparation_source_sha256"]
        or replay_source_sha256 != provenance["replay_source_sha256"]
    ):
        raise ValueError("source, catalog, or prompt-template provenance differs from replay")

    prompt_items: dict[str, dict[str, Any]] = {}
    mapping: list[dict[str, Any]] = []
    cells = replay["cells"]
    if len(cells) != 72:
        raise ValueError("pinned replay must contain exactly 72 default cells")
    seen_cells: set[tuple[str, str, int]] = set()
    for cell in cells:
        if cell.get("status") != "feasible":
            raise ValueError("all default QA contrast cells must be feasible")
        question_id, split = cell["question_id"], cell["split"]
        mode, ceiling = cell["mode"], cell["ceiling"]
        key = (question_id, mode, ceiling)
        if split != "dev" or mode not in _MODES or ceiling not in _CEILINGS or key in seen_cells:
            raise ValueError("replay cell identity is invalid or duplicated")
        seen_cells.add(key)

        prompt = cell["final_prompt"]
        if type(prompt) is not dict or set(prompt) != {"prompt_id", "messages"}:
            raise ValueError("feasible replay prompt fields are invalid")
        messages = prompt["messages"]
        if (
            type(messages) is not list
            or len(messages) != 2
            or any(
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["content"]) is not str
                for message in messages
            )
            or [message["role"] for message in messages] != ["system", "user"]
        ):
            raise ValueError("prompt must contain only system/user role and content strings")
        prompt_id = prompt["prompt_id"]
        if type(prompt_id) is not str or _sha256(_canonical_bytes(messages)) != prompt_id:
            raise ValueError("prompt ID does not match the canonical full messages")
        prompt_tokens = cell["used_tokens"]
        if (
            type(prompt_tokens) is not int
            or prompt_tokens < 0
            or prompt_tokens > ceiling
            or prompt_tokens > _MAX_PROMPT_TOKENS
        ):
            raise ValueError("prompt token count exceeds its fixed ceiling")
        item = {"prompt_id": prompt_id, "messages": messages, "prompt_tokens": prompt_tokens}
        previous = prompt_items.setdefault(prompt_id, item)
        if previous != item:
            raise ValueError("repeated prompt IDs differ in messages or exact token count")

        selected_citations = cell["selected_citation_ids"]
        selected_records = cell["selected_record_ids"]
        scoring = cell["scoring"]
        if type(selected_citations) is not list or type(selected_records) is not list:
            raise ValueError("replay selected citation and record lists are invalid")
        if type(scoring) is not dict:
            raise ValueError("replay scoring record is invalid")
        mapping.append(
            {
                "cell_id": cell["cell_id"],
                "question_id": question_id,
                "split": split,
                "mode": mode,
                "ceiling": ceiling,
                "prompt_id": prompt_id,
                "prompt_tokens": prompt_tokens,
                "selected_citation_ids": selected_citations,
                "selected_record_ids": selected_records,
                "scoring": scoring,
            }
        )

    if len(seen_cells) != 72 or len(prompt_items) != 46:
        raise ValueError("default replay must map 72 cells to exactly 46 unique prompts")
    items = [prompt_items[key] for key in sorted(prompt_items)]
    payload = {
        "schema_version": 1,
        "mode": "qa-contrast-dev-source-only-requests-v1",
        "model": _MODEL.copy(),
        "tokenizer": {
            "settings": token_counter._TOKENIZER.copy(),
            "measurement": provenance["tokenizer_provenance"].copy(),
        },
        "provenance": {
            "fixture_sha256": _FIXTURE_SHA256,
            "receipt_sha256": _RECEIPT_SHA256,
            "catalog_sha256": catalog_sha256,
            "replay_sha256": _REPLAY_SHA256,
            "replay_source_sha256": replay_source_sha256,
            "prompt_template_sha256": catalog["prompt_sha256"],
            "preparer_source_sha256": _sha256(Path(__file__).read_bytes()),
            "preparation_source_sha256": catalog["source_sha256"],
        },
        "items": items,
    }
    payload_sha256 = _sha256(_canonical_bytes(payload))
    references = {
        "schema_version": authored_references["schema_version"],
        "kind": "tracemem-authored-qa-contrast-request-references",
        "provenance": "authored-fictional-agent-only",
        "fixture_sha256": _FIXTURE_SHA256,
        "payload_sha256": payload_sha256,
        "split": "dev",
        "dev_only": True,
        "test": [],
        "held_out_test_cases": False,
        "label_provenance": "authored-agent-only",
        "provisional": True,
        "human_label_count": 0,
        "questions": authored_references["questions"],
    }
    references_sha256 = _sha256(_canonical_bytes(references))
    input_tokens = sum(cell["prompt_tokens"] for cell in mapping)
    unique_input_tokens = sum(item["prompt_tokens"] for item in items)
    if (input_tokens, unique_input_tokens) != (28944, 23533):
        raise ValueError("pinned replay token totals differ from their reviewed values")
    audit = {
        "schema_version": 1,
        "kind": "offline-qa-contrast-request-audit",
        "provenance": {
            **payload["provenance"],
            "payload_sha256": payload_sha256,
            "references_sha256": references_sha256,
        },
        "dev_only": True,
        "provisional": True,
        "human_label_count": 0,
        "qa_measured": False,
        "qa_unmeasured": True,
        "test": [],
        "held_out_test_cases": False,
        "retrieval": provenance["retrieval"],
        "embedder": provenance["embedder"],
        "packing": {
            "algorithm": "pack_context_by_tokens",
            "counter_id": provenance["counter_id"],
            "max_prompt_tokens": _MAX_PROMPT_TOKENS,
            "prompt_truncation": False,
            "ordered_hits": True,
            "skip_over_budget_hits": True,
        },
        "deduplication": {
            "identity": "prompt_id plus exact messages and prompt_tokens",
            "unique_prompts": 46,
            "duplicate_cells": 26,
            "shared_answers_are_common_input_reuse": True,
            "cells_are_independent_qa_trials": False,
        },
        "totals": {
            "cells": 72,
            "unique_prompts": 46,
            "input_tokens_all_cells": input_tokens,
            "input_tokens_unique_prompts": unique_input_tokens,
            "generation_cap_tokens": 5888,
            "sum_unique_input_plus_generation_cap": 29421,
            "bounds_only_not_billing_or_runtime": True,
        },
        "cell_mapping": mapping,
    }
    return payload, references, audit


def _json_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _summary(
    payload: dict[str, Any], references: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "payload_sha256": _sha256(_canonical_bytes(payload)),
        "references_sha256": _sha256(_canonical_bytes(references)),
        "audit_sha256": _sha256(_canonical_bytes(audit)),
        "cells": audit["totals"]["cells"],
        "unique_prompts": audit["totals"]["unique_prompts"],
    }


def _write_new_output(directory: Path, files: dict[str, bytes]) -> None:
    os.mkdir(directory, 0o700)
    os.chmod(directory, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for name, contents in files.items():
        descriptor = os.open(directory / name, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-qa-contrast-requests")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, references, audit = prepare_requests()
        if args.output_dir is not None:
            _write_new_output(
                args.output_dir,
                {
                    "payload.json": _json_bytes(payload),
                    "references.json": _json_bytes(references),
                    "audit.json": _json_bytes(audit),
                },
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(_summary(payload, references, audit), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
