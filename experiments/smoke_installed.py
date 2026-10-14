"""Read-only installed-wheel smoke; all SQLite artifacts stay in --work-dir."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

FROZEN = {
    "legacy_claim_fixture": "d1a7e6f91b8ea81c60f318f0472fb85f4e230d3ac5742e1d6d76c0148f6ffe67",
    "scenarios_fixture": "5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214",
    "support_fixture": "582612fc24e60868c8b0b1f37745188dbf143a41330a5777152b5cce9adc90e9",
    "nli_receipt": "16be7f4c542bd174e49f975bfc3d8baa7ef9995fdd5d8cfcf2e3df03893f3871",
    "support_report": "9d2891c622c1644baddd8a7cf87faa410d393f5a7e70e6ded80a32c6daf828e2",
}
CONTEXT_BUDGET_REPORT_SHA256 = "9e14de94e09893f400e17226fa2cebceeec377c1493f4f263fa325d49a055c01"
REVIEW_BUNDLE_SHA256 = "65ed4a2e0bbd227c3a521899676c4db98085f855083ec2d2a16aecbdda55bda5"
REVIEW_HTML_SHA256 = "07521f971f5554e5476d15a1299eb75cf0b5b20fbc6854c79b6053a1aecc474e"
REVIEW_TASK_COUNT = 84


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _first_mismatch_path(actual: Any, expected: Any, path: str = "report") -> str | None:
    if type(actual) is not type(expected):
        return path
    if isinstance(actual, Mapping):
        for key in sorted(set(actual) | set(expected)):
            child_path = f"{path}.{key}"
            if key not in actual or key not in expected:
                return child_path
            mismatch = _first_mismatch_path(actual[key], expected[key], child_path)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}.length"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            mismatch = _first_mismatch_path(actual_item, expected_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if actual == expected else path


def _compare_context_budget_reports(
    actual: Mapping[str, Any], recorded: Mapping[str, Any]
) -> dict[str, Any]:
    actual_environment = actual.get("environment")
    recorded_environment = recorded.get("environment")
    if not isinstance(actual_environment, Mapping) or not isinstance(recorded_environment, Mapping):
        raise SystemExit("context-budget report environment must be an object")
    actual_environment_fields = set(actual_environment)
    recorded_environment_fields = set(recorded_environment)
    if actual_environment_fields != recorded_environment_fields:
        raise SystemExit(
            "context-budget environment schema changed: "
            f"actual={sorted(actual_environment_fields)}, "
            f"recorded={sorted(recorded_environment_fields)}"
        )
    for field in sorted(actual_environment_fields):
        if type(actual_environment[field]) is not type(recorded_environment[field]):
            raise SystemExit(f"context-budget environment field type changed: environment.{field}")

    allowed_environment_values = {"python", "sqlite"}
    stable_actual = dict(actual)
    stable_recorded = dict(recorded)
    stable_actual["environment"] = {
        key: value
        for key, value in actual_environment.items()
        if key not in allowed_environment_values
    }
    stable_recorded["environment"] = {
        key: value
        for key, value in recorded_environment.items()
        if key not in allowed_environment_values
    }
    mismatch = _first_mismatch_path(stable_actual, stable_recorded)
    if mismatch is not None:
        raise SystemExit(f"context-budget report changed at {mismatch}")
    return {
        "stable_fields_match": True,
        "environment": {
            "actual": dict(sorted(actual_environment.items())),
            "recorded": dict(sorted(recorded_environment.items())),
        },
    }


def _require_frozen_sha256(label: str, raw: bytes, expected: str) -> str:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SystemExit(f"{label} SHA-256 changed: expected {expected}, got {actual}")
    return actual


def _exercise_new_apis(source_root: Path, work_dir: Path) -> dict[str, Any]:
    from tracemem.benchmark import load_synthetic_data
    from tracemem.budget_benchmark import run_context_budget_benchmark
    from tracemem.review_html import render_review_html
    from tracemem.storage import EvidenceStore

    report_path = source_root / "docs/baseline-context-budgets.json"
    bundle_path = source_root / "examples/synthetic/reviews/scenario-review-blind.json"
    try:
        recorded_report_bytes = report_path.read_bytes()
        bundle_bytes = bundle_path.read_bytes()
    except OSError as error:
        raise SystemExit(f"missing installed API smoke input: {error.filename}") from error

    report_sha256 = _require_frozen_sha256(
        "context-budget baseline report", recorded_report_bytes, CONTEXT_BUDGET_REPORT_SHA256
    )
    bundle_sha256 = _require_frozen_sha256(
        "blind review bundle", bundle_bytes, REVIEW_BUNDLE_SHA256
    )
    recorded_report = json.loads(recorded_report_bytes)
    review_bundle = json.loads(bundle_bytes)
    if not isinstance(recorded_report, Mapping):
        raise SystemExit("context-budget baseline report must be a JSON object")
    if not isinstance(review_bundle, Mapping):
        raise SystemExit("blind review bundle must be a JSON object")
    tasks = review_bundle.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != REVIEW_TASK_COUNT:
        raise SystemExit(f"blind review bundle must contain {REVIEW_TASK_COUNT} tasks")

    database_path = work_dir / "context-budget-evidence.sqlite3"
    if database_path.exists():
        raise SystemExit(f"context-budget smoke database already exists: {database_path.name}")
    records, queries, fixture_sha256 = load_synthetic_data()
    with EvidenceStore(database_path) as store:
        store.ingest(records)
        actual_report = run_context_budget_benchmark(store, queries, input_sha256=fixture_sha256)
    budget_comparison = _compare_context_budget_reports(actual_report, recorded_report)

    rendered_html = render_review_html(review_bundle)
    html_sha256 = _require_frozen_sha256(
        "rendered blind review HTML", rendered_html.encode("utf-8"), REVIEW_HTML_SHA256
    )
    return {
        "context_budget": {
            "report_sha256": report_sha256,
            "fixture_sha256": actual_report["fixture_sha256"],
            **budget_comparison,
        },
        "review_html": {
            "bundle_sha256": bundle_sha256,
            "html_sha256": html_sha256,
            "task_count": len(tasks),
        },
    }


def preflight(source_root: Path, work_dir: Path) -> tuple[Path, Path]:
    if "PYTHONPATH" in os.environ or "PYTHONHOME" in os.environ:
        raise SystemExit("PYTHONPATH and PYTHONHOME must be unset")
    source = source_root.expanduser().resolve(strict=True)
    work = work_dir.expanduser().resolve(strict=True)
    cwd = Path.cwd().resolve()
    if not source.is_dir() or inside(cwd, source):
        raise SystemExit("run from a working directory outside --source-root")
    temp_root = Path(tempfile.gettempdir()).resolve()
    if (
        not work.is_dir()
        or not inside(work, temp_root)
        or inside(work, source)
        or any(work.iterdir())
    ):
        raise SystemExit("--work-dir must be an empty directory outside --source-root")
    if any(
        not (source / relative).is_file()
        for relative in (
            "examples/synthetic/nli-deberta-remote.json",
            "docs/baseline-nli-support.json",
        )
    ):
        raise SystemExit("missing frozen NLI input under --source-root")
    return source, work


def installed_module_paths(source_root: Path) -> dict[str, str]:
    paths = {
        name: Path(getattr(module, "__file__", "") or "").resolve()
        for name, module in sorted(sys.modules.items())
        if name == "tracemem" or name.startswith("tracemem.")
    }
    if "tracemem" not in paths or any(
        "site-packages" not in path.parts or inside(path, source_root) for path in paths.values()
    ):
        raise SystemExit(
            "tracemem imports must come from installed site-packages outside source-root"
        )
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    source, work = preflight(args.source_root, args.work_dir)

    from tracemem.annotation_report import build_review_report
    from tracemem.annotations import export_review_bundle
    from tracemem.challenge import _case_records, load_challenge
    from tracemem.citations import fetch_entry_evidence
    from tracemem.claims import parse_candidates
    from tracemem.ledger import CandidateLedger
    from tracemem.scenarios import load_scenarios
    from tracemem.storage import EvidenceStore
    from tracemem.support import load_support_challenge, score_support_cases

    cases, legacy_hash = load_challenge()
    if legacy_hash != FROZEN["legacy_claim_fixture"]:
        raise SystemExit("legacy claim fixture hash changed")
    legacy_count = 0
    first = None
    legacy_ledger_path = work / "legacy-ledger.sqlite3"
    with CandidateLedger(legacy_ledger_path) as ledger:
        for case in cases:
            records = _case_records(case)
            for claim in parse_candidates({"claims": case["expected"]}):
                entry = ledger.append(
                    claim, records, "installed-gold-fixture", "2026-10-14T10:00:00Z"
                )
                assert entry.verification.status == "REVIEW"
                if first is None:
                    first = (entry, records)
                legacy_count += 1
        assert legacy_count == 26 and first is not None
        original, records = first
        correction = ledger.append(
            original.claim,
            records,
            "installed-correction-proposal",
            "2026-10-14T10:01:00Z",
            supersedes=(original.entry_id,),
        )
        assert correction.supersedes == (original.entry_id,)
    with CandidateLedger(legacy_ledger_path) as reopened:
        assert len(reopened.entries()) == 27
        assert reopened.get(original.entry_id) == original
        assert reopened.get(correction.entry_id) == correction

    scenarios, manifest, scenario_hash = load_scenarios()
    if scenario_hash != FROZEN["scenarios_fixture"]:
        raise SystemExit("packaged scenario fixture hash changed")
    scenario_db = work / "scenario-evidence.sqlite3"
    scenario_ledger_path = work / "scenario-ledger.sqlite3"
    scenario_claims = 0
    historical = None
    with EvidenceStore(scenario_db) as evidence, CandidateLedger(scenario_ledger_path) as ledger:
        for case in scenarios:
            records = _case_records(case)
            evidence.ingest(records)
            for claim in parse_candidates({"claims": case["expected"]}):
                entry = ledger.append(
                    claim, records, "installed-scenario-fixture", "2026-10-14T10:00:00Z"
                )
                resolved = fetch_entry_evidence(evidence, entry)
                if historical is None:
                    historical = (entry, resolved[0].record)
                scenario_claims += 1
        assert scenario_claims == 43 and historical is not None
        old_entry, old_record = historical
        newer_record = replace(old_record, text=old_record.text + " [later revision]")
        evidence.ingest([newer_record])

    with (
        EvidenceStore(scenario_db) as reopened_evidence,
        CandidateLedger(scenario_ledger_path) as reopened_ledger,
    ):
        assert len(reopened_ledger.entries()) == 43
        reopened_entry = reopened_ledger.get(old_entry.entry_id)
        exact_old = fetch_entry_evidence(reopened_evidence, reopened_entry)
        assert exact_old[0].record == old_record
        assert reopened_evidence.get(old_record.record_id, newer_record.revision) == newer_record

    bundle, answer_key = export_review_bundle(
        scenarios,
        scenario_hash,
        seed=1729,
        case_metadata=manifest["case_metadata"],
    )
    assert len(scenarios) == len(answer_key["case_ids"]) == 48 and len(bundle["tasks"]) == 43
    report = build_review_report(bundle, answer_key, [])
    reviewers = {kind: report["by_reviewer_kind"][kind]["reviewers"] for kind in ("human", "agent")}
    assert not any(reviewers.values())

    support_cases, support_hash = load_support_challenge()
    receipt_path = source / "examples/synthetic/nli-deberta-remote.json"
    report_path = source / "docs/baseline-nli-support.json"
    receipt_hash, report_hash = sha256(receipt_path), sha256(report_path)
    if support_hash != FROZEN["support_fixture"] or receipt_hash != FROZEN["nli_receipt"]:
        raise SystemExit("support fixture or NLI receipt hash changed")
    if report_hash != FROZEN["support_report"]:
        raise SystemExit("frozen support report hash changed")
    receipt = json.loads(receipt_path.read_text())
    frozen_report = json.loads(report_path.read_text())
    assert receipt["input_sha256"] == support_hash == frozen_report["input_sha256"]
    assert frozen_report["raw_receipt_sha256"] == receipt_hash
    outputs = {row["case_id"]: row["probabilities"] for row in receipt["outputs"]}
    replay = score_support_cases(support_cases, outputs)
    assert replay == frozen_report["score"]

    new_api_checks = _exercise_new_apis(source, work)
    modules = installed_module_paths(source)

    receipt = {
        "installed_module_paths": modules,
        "sha256": {**FROZEN},
        "new_api_checks": new_api_checks,
        "counts": {
            "legacy": {"gold_claims": legacy_count, "entries_after_reopen": 27},
            "nli_support_cases": len(support_cases),
            "scenarios": {
                "cases": len(scenarios),
                "claims_verified": scenario_claims,
                "entries_after_reopen": 43,
            },
            "annotations": {
                "tasks": len(bundle["tasks"]),
                "case_keys": len(answer_key["case_ids"]),
                "human_reviewer_documents": reviewers["human"],
                "agent_reviewer_documents": reviewers["agent"],
            },
        },
        "conditions": {
            "citation_revision_survives_newer_source_and_reopen": True,
            "preflight_passed": True,
            "frozen_nli_scores_match": True,
            "model_or_remote_calls": 0,
            "semantic_acceptance_claimed": False,
        },
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
