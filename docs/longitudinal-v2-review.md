# Longitudinal v2 review slice

The usable full packet is at
`.local/longitudinal-v2-review-ready/reviewer/review.html` (26 tasks). Share
only its `reviewer/` directory; keep the answer key in `curator/`. The page is
an interface for annotation, not itself a human annotation. To regenerate the
packet, use the [full-packet export recipe](longitudinal-revision-v2.md#export-separate-reviewer-and-curator-files)
and the [offline HTML instructions](offline-review.md).

## Scope and result

The frozen v2 fixture is `examples/synthetic/longitudinal-dev-v2.json` (SHA-256
`ea6b8c5a43c5c2c022ebf10b76e12581b46b1b92ff515f5e9be72340a5c0016e`). This
review used an explicit, nonrandom subset: `long-dev-case-03`, `04`, `07`, and
`10`, with 10 tasks (6 gold, 3 foil, 1 unlabeled). It is not representative of
the fixture or a benchmark sample.

A fresh fork-none `gpt-6-astra` agent saw only the probe's blind JSON. It
returned 6 supported, 3 unsupported, and 1 uncertain. Agreement was 9/9 on
decisive tasks with authored gold/foil labels; the unlabeled task is excluded
from that denominator. This is descriptive agreement with provisional labels,
not accuracy, calibration, or proof of truth. Earlier labels were
authored/reviewed by Luna; this comparison does not establish model-family or
version independence.

The unblinded root audit retained the original response, agrees on all 10
labels, and checked 13 evidence spans against the fixture. It is not an
independent review or human ground truth, and no reference labels were
promoted. The audit recommends a neutral key-keeper pronoun in future
curation; preserve the frozen v2 fixture bytes.

There are zero human labels and Gate A remains unmet. The targeted agent
annotation was performed, but no new model-serving experiment, training, GPU
work, private-source ingestion, or runtime change was part of this review. See
the [validation record](validation-longitudinal-v2-review.json) and
[copied review artifacts](../examples/synthetic/reviews/longitudinal-v2/manifest.json).

## Reproduce the targeted probe

From the repository root, the existing `export_review_bundle` API can recreate
the 10-task probe with seed 1729. It hashes the full frozen fixture bytes, while
passing only the four selected cases and their associated foil, unlabeled, and
metadata rows. The generated answer key stays separate; share only the printed
reviewer directory.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib
import json
import tempfile
from pathlib import Path

from tracemem.annotation_report import build_review_report
from tracemem.annotations import export_review_bundle

source = Path("examples/synthetic")
fixture_path = source / "longitudinal-dev-v2.json"
raw = fixture_path.read_bytes()
payload = json.loads(raw)
fixture_sha256 = hashlib.sha256(raw).hexdigest()
assert fixture_sha256 == "ea6b8c5a43c5c2c022ebf10b76e12581b46b1b92ff515f5e9be72340a5c0016e"
selected = {
    "long-dev-case-03",
    "long-dev-case-04",
    "long-dev-case-07",
    "long-dev-case-10",
}
cases = [row for row in payload["cases"] if row["case_id"] in selected]
assert {row["case_id"] for row in cases} == selected
foils_all = json.loads((source / "longitudinal-dev-v2-foils.json").read_text(encoding="utf-8"))
unlabeled_all = json.loads((source / "longitudinal-dev-v2-unlabeled.json").read_text(encoding="utf-8"))
foils = {key: value for key, value in foils_all.items() if key in selected}
unlabeled = {key: value for key, value in unlabeled_all.items() if key in selected}
metadata = {key: value for key, value in payload["case_metadata"].items() if key in selected}
blind, answer_key = export_review_bundle(
    cases,
    fixture_sha256,
    foils_by_case=foils,
    seed=1729,
    case_metadata=metadata,
    unlabeled_by_case=unlabeled,
)
root = Path(tempfile.mkdtemp(prefix="longitudinal-v2-review-"))
reviewer_dir, curator_dir = root / "reviewer", root / "curator"
reviewer_dir.mkdir()
curator_dir.mkdir()
for path, value in (
    (reviewer_dir / "blind.json", blind),
    (curator_dir / "answer-key.json", answer_key),
):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
response_path = Path(
    "examples/synthetic/reviews/longitudinal-v2/agent-astra-responses.json"
)
expected_report_path = Path(
    "examples/synthetic/reviews/longitudinal-v2/agent-report.json"
)
response = json.loads(response_path.read_text(encoding="utf-8"))
expected_report = json.loads(expected_report_path.read_text(encoding="utf-8"))
report = build_review_report(blind, answer_key, [response])
assert report == expected_report
report_path = curator_dir / "report.json"
with report_path.open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
    output.write("\n")
report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
assert report_sha256 == "f629336668301965e514468ca573dbc9ae3c5e4f6f5a89277f35427357199d62"
print("Share only:", reviewer_dir)
print("Probe bundle SHA-256:", blind["bundle_sha256"])
print("Curator report SHA-256:", report_sha256)
PY
```

Expected dataset SHA-256 is
`ea6b8c5a43c5c2c022ebf10b76e12581b46b1b92ff515f5e9be72340a5c0016e`; expected
blind bundle SHA-256 is
`65ef70d47d0feb04c3760582835d50487c75e927d82f45afd6422be634cafd6d` (10 tasks).
Keep the complete fixture and historical receipts unchanged.

## Validation and next step

Source and installed replay metadata differed only in the verified
`package_origin` field; full and probe exports, rendered HTML, and report
outputs matched byte-for-byte. The report SHA-256 is
`f629336668301965e514468ca573dbc9ae3c5e4f6f5a89277f35427357199d62`; source
and installed annotation/HTML runs each passed 23 tests, and all 37 runtime
members (32 Python files and 5 data assets) matched. Raw local captures are
listed in the validation record.

The next validation step is human annotation using the full blinded packet, followed by a separate review of those labels and the dataset's readiness criteria. Do not infer Gate A approval from this agent probe or from its replay checks.
