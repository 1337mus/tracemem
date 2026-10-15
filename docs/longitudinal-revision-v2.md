# Longitudinal development slice, revision 2

Updated 2026-10-15 UTC.

## Revision and files

Revision 2 keeps the same 12 fictional development scenarios, messages, IDs, and metadata as v1. It is a dataset revision, not 12 additional or independent cases. The root JSON version stays `1` for the existing challenge schema; the `v2` filenames identify this revision.

- **v1 fixture SHA-256:** `d75b2237e739bfd8ff72be25d234f62004b6641a48a8bd9893442d64e8bc8fda`.
- **v1 foils SHA-256:** `26f65c6c80f8e1abcb4c9d7eaa1e7215dc41d114113f33f781550612d77f91be`.
- **v2 fixture:** [`examples/synthetic/longitudinal-dev-v2.json`](../examples/synthetic/longitudinal-dev-v2.json), SHA-256 `ea6b8c5a43c5c2c022ebf10b76e12581b46b1b92ff515f5e9be72340a5c0016e`.
- **v2 foils:** [`examples/synthetic/longitudinal-dev-v2-foils.json`](../examples/synthetic/longitudinal-dev-v2-foils.json), SHA-256 `4e257e21091b62a35693135020faac11a37d9aeccbea7de246d1ef0ff3f490ec`.
- **v2 unlabeled candidate:** [`examples/synthetic/longitudinal-dev-v2-unlabeled.json`](../examples/synthetic/longitudinal-dev-v2-unlabeled.json), SHA-256 `f2df24c71c19b86207b2ad74d9e2570b3c568021992b063a5bdee9787e57bea4`.
- **Counts:** 12 cases, 146 messages, 49 case-scoped sessions, 13 selected targets, 12 foils, and 1 unlabeled candidate. The review export has 26 tasks. Four gold targets cite more than one session after the revision; this citation count does not establish four distinct reasoning requirements or four independent examples.

V2 makes two annotation changes. For case `long-dev-case-10`, the second target no longer cites message `m06`; its evidence now points to `m07` and `m09`. For case `long-dev-case-07`, one candidate moves unchanged from the foil map to the unlabeled map. Its same-name reference remains unresolved, so the fixture does not force an unsupported label.

## Review status and limits

V1 fixture labels were agent-authored and provisional. The earlier blind v1
review returned 13 supported, 12 unsupported, and 1 uncertain, but it used the
same Luna model family that authored the labels; it is not independent
validation or truth. V2 now has a fresh agent review of an explicit,
nonrepresentative 10-task probe: 6 supported, 3 unsupported, and 1 uncertain,
with 9/9 agreement on decisive gold/foil tasks. The unlabeled task is excluded
from that denominator. A fresh fork-none Astra agent saw only the probe blind
JSON; earlier labels were authored/reviewed by Luna, so this does not establish
model-family or version independence. The unblinded root audit agrees on all
10 labels and checked 13 exact evidence spans; it is not human annotation or
independent ground truth. There are still zero human labels and Gate A remains
unmet. See the [review card](longitudinal-v2-review.md) and [review validation
record](validation-longitudinal-v2-review.json). V2 task IDs bind to the full
new fixture hash, so do not transfer v1 response labels to v2.

The files are newly authored fictional material under MIT. Names, settings,
organizations, and dates are fictional, not observed event truth. All cases
remain `dev`; no `train` or `test` data is included. Labels are agent-authored
and provisional. There are zero human annotations and no model-serving
extraction measurements for this v2 fixture; targeted agent annotation is
reported separately in the [review card](longitudinal-v2-review.md). Gate A
remains unmet. Integrity tests check schemas and exact evidence spans, not
semantic support or real-world truth. A reviewer may choose `uncertain` when
the conversation does not resolve identity; that label does not prove the claim
false.

## Validation

See the [v2 validation record](validation-longitudinal-v2.json) for source checks, installed v1/v2 tests, artifact hashes, deterministic export parity, and historical pin status. The initial local installed receipt contained reconstructed export stdout with doubled braces, then was edited; it is superseded and is not evidence. The raw subprocess captures are named in the validation record.

## Export separate reviewer and curator files

Run this at the repository root. It creates fresh directories and opens each output exclusively. Share only the reviewer directory; keep the key in the curator directory.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib
import json
import tempfile
from pathlib import Path

from tracemem.annotations import export_review_bundle

source = Path("examples/synthetic")
fixture_path = source / "longitudinal-dev-v2.json"
raw = fixture_path.read_bytes()
payload = json.loads(raw)
foils = json.loads((source / "longitudinal-dev-v2-foils.json").read_text(encoding="utf-8"))
unlabeled = json.loads((source / "longitudinal-dev-v2-unlabeled.json").read_text(encoding="utf-8"))
blind, answer_key = export_review_bundle(
    payload["cases"],
    hashlib.sha256(raw).hexdigest(),
    foils_by_case=foils,
    seed=1729,
    case_metadata=payload["case_metadata"],
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
print("Share only:", reviewer_dir / "blind.json")
print("Retain separately:", curator_dir / "answer-key.json")
PY
```

The exporter labels the moved candidate `membership: "unlabeled"` with a null reference label. That preserves review uncertainty without recording a new judgment.
