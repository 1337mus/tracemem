"""Read-only installed-wheel smoke; all SQLite artifacts stay in --work-dir."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

FROZEN = {
    "legacy_claim_fixture": "d1a7e6f91b8ea81c60f318f0472fb85f4e230d3ac5742e1d6d76c0148f6ffe67",
    "scenarios_fixture": "5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214",
    "support_fixture": "582612fc24e60868c8b0b1f37745188dbf143a41330a5777152b5cce9adc90e9",
    "nli_receipt": "16be7f4c542bd174e49f975bfc3d8baa7ef9995fdd5d8cfcf2e3df03893f3871",
    "support_report": "9d2891c622c1644baddd8a7cf87faa410d393f5a7e70e6ded80a32c6daf828e2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


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

    modules = installed_module_paths(source)
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

    receipt = {
        "installed_module_paths": modules,
        "sha256": {**FROZEN},
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
