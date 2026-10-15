"""Print local readiness diagnostics for claim datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from tracemem.challenge import load_challenge
from tracemem.dataset_audit import audit_cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="*", type=Path, metavar="FIXTURE")
    args = parser.parse_args(argv)
    if not args.fixtures:
        cases, fixture_sha256 = load_challenge()
        report = audit_cases(cases)
        report["fixture_sha256"] = fixture_sha256
    else:
        cases = []
        case_metadata: dict[str, dict[str, object]] = {}
        metadata_was_declared = False
        input_sha256 = []
        try:
            for fixture_path in args.fixtures:
                raw = fixture_path.read_bytes()
                payload = json.loads(
                    raw,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_non_finite,
                )
                if type(payload) is not dict or type(payload.get("cases")) is not list:
                    raise ValueError("fixture must be an object containing a cases list")
                cases.extend(payload["cases"])
                if "case_metadata" in payload:
                    metadata_was_declared = True
                    fixture_metadata = payload["case_metadata"]
                    if type(fixture_metadata) is not dict:
                        raise ValueError("case_metadata must be an object")
                    for case_id, row in fixture_metadata.items():
                        if case_id in case_metadata:
                            raise ValueError("duplicate case_metadata case_id")
                        case_metadata[case_id] = row
                input_sha256.append(hashlib.sha256(raw).hexdigest())
            report = audit_cases(
                cases,
                case_metadata=case_metadata if metadata_was_declared else None,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        report["input_sha256"] = input_sha256
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
