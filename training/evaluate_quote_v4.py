"""Offline evaluator for the provisional quote-v4 SFT comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.evaluate_quote_v3 import _evaluate_report, _EvaluatorContract
from training.train_sft import _QUOTE_V4_PRIOR_ADAPTER, MODEL_ID, MODEL_REVISION

GENERATION = {"max_new_tokens": 1024, "do_sample": False}
COHORT_SIZES = {"dev": 12, "test": 24}
ARMS = ("base", "prior", "tuned")

_V4_CONTRACT = _EvaluatorContract(
    profile="quote-v4",
    prior_adapter=_QUOTE_V4_PRIOR_ADAPTER,
    prior_checkpoint_label="quote-v3",
    cohort_sizes=tuple(COHORT_SIZES.items()),
    generation=GENERATION,
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    report_type="provisional quote-v4 cohort development diagnostic",
    train_row_count=768,
    training_counts=(
        ("max_steps", 192),
        ("optimizer_steps", 192),
        ("examples_consumed", 768),
    ),
    prompt_fixture_description="two evaluation fixtures",
)


def evaluate_report(
    job_path: Path,
    receipt_path: Path,
    *,
    dev_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    """Validate and score the quote-v4 dev and test cohorts."""
    return _evaluate_report(
        job_path,
        receipt_path,
        fixture_paths={"dev": dev_path, "test": test_path},
        contract=_V4_CONTRACT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dev-fixture", type=Path, required=True)
    parser.add_argument("--test-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(
            args.job,
            args.receipt,
            dev_path=args.dev_fixture,
            test_path=args.test_fixture,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        print(f"Quote-v4 evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"output": str(args.output), "cohort_counts": COHORT_SIZES},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
