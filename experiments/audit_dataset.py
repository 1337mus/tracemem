"""Print local readiness diagnostics for the bundled claim challenge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from tracemem.challenge import load_challenge
from tracemem.dataset_audit import audit_cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    cases, fixture_sha256 = load_challenge()
    report = audit_cases(cases)
    report["fixture_sha256"] = fixture_sha256
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
