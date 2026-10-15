# Cross-session fictional test fixture

Updated 2026-10-15.

## Provenance and split

[`cross-session-test-v1.json`](../examples/synthetic/cross-session-test-v1.json) is a newly authored, fictional LongMemEval-shaped input licensed under MIT. It contains six assigned `test` questions, each with four dated sessions and three turns per session: 72 turns total. The first and last session dates span more than 30 days, and each question date follows its history. The exact fixture SHA-256 is `dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf`.

[`cross-session-test-v1-metadata.json`](../examples/synthetic/cross-session-test-v1-metadata.json) records provenance, provisional status, split assignment, bookkeeping IDs, and evidence dependencies. Its unique source, scenario, template, and entity IDs support exact-overlap audits only; they do not prove semantic independence. The fixture has four answerable questions with eight provisional labeled evidence turns and two `UNKNOWN` cases. It contains no human labels or independently verified ground truth.

## Evidence dependencies

Turn coordinates below use the upstream session ID and a one-based turn number, matching the metadata file.

| Question | Required evidence | Why the marked turns are separated |
| --- | --- | --- |
| `cst-moonbridge-route` | `cst-moonbridge-route-s01/t1` gives the first-edition opening stop; `cst-moonbridge-route-s03/t1` gives its second stop and identifies the later replacement. | The February draft in `s02` lies between the two historical facts. |
| `cst-hearth-reed-rule` | `cst-hearth-reed-rule-s01/t1` states the if/otherwise handling rule; `cst-hearth-reed-rule-s03/t1` gives two consecutive readings. | `s02` intervenes; neither the rule nor the measurements alone determines the action. |
| `cst-quietpine-identity` | `cst-quietpine-identity-s02/t3` maps the account alias to a registered name; `cst-quietpine-identity-s04/t3` records the sketch upload under the alias. | `s03` intervenes; the upload record alone identifies only an account. |
| `cst-blueshingle-report` | `cst-blueshingle-report-s01/t1` states the report's included and excluded work; `cst-blueshingle-report-s04/t3` maps that report to its project. | `s02` and `s03` intervene between scope and project attribution. |
| `cst-hushglass-causality_abs` | No causal evidence is labeled. Two events occur two minutes apart, without a recorded causal link. | Temporal proximity alone cannot establish cause. |
| `cst-rook-badge-identity_abs` | No identity evidence is labeled. The history names a coordinator and separately records a badge, without mapping them. | The link between person and badge is absent. |

These explanations record annotation intent; they do not by themselves establish semantic sufficiency.

## Baseline and limits

The frozen CPU-only replay uses the existing bytes API, with namespace `cross-session-test-v1`, variant `s`, and cutoffs 1, 3, and 5. Its complete per-question results, returned citations, environment, and aggregate counts are in the [replay report](cross-session-test-retrieval.json). Macro turn-label recall averages each question's fraction of explicitly labeled turns retrieved over the four scored questions; complete coverage counts the scored questions for which every labeled turn appears in the prefix.

| Mode | Macro turn-label recall at k=1 / 3 / 5 | Complete coverage at k=1 / 3 / 5 |
| --- | --- | --- |
| BM25 | 0.50 / 1.00 / 1.00 | 0/4; 4/4; 4/4 |
| Hash vector | 0.50 / 0.75 / 1.00 | 0/4; 2/4; 4/4 |
| Hybrid | 0.50 / 1.00 / 1.00 | 0/4; 4/4; 4/4 |

The ranking audit finds the later support turn (`s03/t1`) at rank 4 for both Moonbridge and Hearth in hash-vector mode. At k=3, neither question has all labeled turns in that mode's prefix. All four answerable questions have complete labeled-turn coverage by k=5 for all three modes. The two `UNKNOWN` questions are excluded from recall; retrieval returns hits for both, so this run measures no abstention accuracy. No answering model or QA evaluator ran.

An unblinded, agent-only root review provisionally supported the four answerable labels and retained both unknown labels. This is an internal review, not independent validation: there are zero human labels. The assigned `test` split is bookkeeping, not held-out evidence. Unique IDs and exact-text checks establish only exact-overlap results, not semantic independence. Turn-label recall measures explicit-label retrieval only, not answer accuracy or evidence sufficiency. The signed-token hash vectors are weak lexical features, not semantic embeddings. Do not infer generalization, model improvement, or QA quality from this slice.

The source and installed replays produced byte-identical reports (SHA-256
`34492b7c47d6fcae0042c730df14641b6062f2e47348b46d52958a263afa0fa6`). Three
focused tests passed in each environment; all 37 runtime package members and 50
historical pins matched. The final `make check` passed 602 of 603 tests with one
optional skip; Ruff, formatting across 134 files, mypy across 32 source files,
actionlint, and `git diff --check` passed. See the [validation record](validation-cross-session-test.json)
for full receipts and scope.

Reproduce the evaluation directly from fixture bytes without writing a report file:

```sh
PYTHONPATH=src:. .venv/bin/python - <<'PY'
import json
from pathlib import Path

from experiments.evaluate_longmemeval_retrieval import evaluate_retrieval_bytes

raw = Path("examples/synthetic/cross-session-test-v1.json").read_bytes()
report = evaluate_retrieval_bytes(
    raw,
    namespace="cross-session-test-v1",
    variant="s",
    cutoffs=(1, 3, 5),
)
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
PY
```
