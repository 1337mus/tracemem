"""Write an offline synthetic context-budget diagnostic to a new JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tracemem.benchmark import load_synthetic_data
from tracemem.budget_benchmark import run_context_budget_benchmark
from tracemem.storage import EvidenceStore


def evaluate_context_budgets() -> dict[str, object]:
    """Run the fixed-budget diagnostic using only the authored synthetic fixture."""
    records, queries, fixture_sha256 = load_synthetic_data()
    with TemporaryDirectory(prefix="tracemem-context-budget-") as directory:
        with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
            store.ingest(records)
            return run_context_budget_benchmark(
                store,
                queries,
                input_sha256=fixture_sha256,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new diagnostic JSON path")
    args = parser.parse_args(argv)
    try:
        report = evaluate_context_budgets()
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
