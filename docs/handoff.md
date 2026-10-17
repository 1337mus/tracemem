# TraceMem handoff

Updated 2026-10-17 UTC. Read `.local/active-cycle.json` for exact private paths,
hashes, runner archives, receipts and live delivery state. Historical source
records and experiment receipts remain unchanged.

## Latest model result

The second private comparison completed with 50 private rows from four training
sessions and 192 fictional train-only replay rows. One epoch ran 242 microsteps
and 61 optimizer steps in 316.333 seconds of training on personal Modal.

| Development measure | Base | Prior adapter | Latest private-trained |
| --- | ---: | ---: | ---: |
| Useful memories accepted | 4/7 | 4/7 | 6/7 |
| Supported claims retained | 7/8 | 7/8 | 8/8 |
| Unsupported claims rejected | 7/8 | 8/8 | 7/8 |
| Uncertain scope correct | 2/2 | 2/2 | 0/2 |

The prospective gate failed. Useful acceptance improved by one over the first
private run, but false acceptance and uncertain scope remain unresolved. No
product model is selected. All 48 outputs were valid JSON and ended at EOS,
without offline correction. Base/prior input and output token IDs each matched
the first run on all 16 prompts. See the [latest results](private-sft-scope-results.md).

Development prompts and labels are unchanged: 16 supplied candidates from two
sessions, all seven useful references from one. Labels remain agent-provisional
with zero human reviewers. This does not measure extraction or downstream QA.
Both private data and replay coverage changed, so the gain cannot be attributed
to either one alone. The reserved real test source and synthetic test stay unused.

## First private comparison, preserved

The first private continuation completed on personal Modal: 30 reviewed private
training rows from two sessions plus 48 fictional replay rows, one epoch,
78 microsteps and 20 optimizer steps. Training took 98 seconds. All three arms
received the same 16 development prompts from two other sessions.

| Development measure | Base | Prior adapter | Private-trained |
| --- | ---: | ---: | ---: |
| Useful memories accepted | 4/7 | 4/7 | 5/7 |
| Supported claims retained | 7/8 | 7/8 | 8/8 |
| Unsupported claims rejected | 7/8 | 8/8 | 7/8 |
| Uncertain scope correct | 2/2 | 2/2 | 0/2 |

The frozen gate failed. No product model is selected. All seven useful
references come from one development session; these correlated, supplied
candidates cannot establish extraction recall or better answers. Labels remain
agent-provisional, with zero human reviewers. See [results](private-sft-results.md)
and the [aggregate machine score](private-sft-results.json).

The original runner included the rendered EOS marker after JSON. Literal raw
scoring therefore found zero valid answers in each arm. The table is a separate
content diagnostic after removing one exact terminal marker, verified against
the pinned tokenizer EOS ID and final generated token. Original token IDs,
receipt hash and literal report remain unchanged. The original runner sources
are archived locally; the future runner now uses content-only decoding. Tests
cover both the correction and false-marker rejection. No GPU rerun occurred.

## Data and next action

The local corpus now contains 33 windows / 66 candidates across six sessions:
25 training windows / 50 rows from four sessions, plus eight development
windows / 16 candidates from two sessions. An additional unlabeled test source
remains reserved. The new eight train windows share an existing personal-editing
task family; their independent review agreed on all 16 support and scope labels.
All are eligible scope. The combined training set has four uncertain-scope rows.

The old 82-row candidate was never launched. It was superseded by the completed
242-row comparison. Historical datasets and receipts remain unchanged.

Next: obtain distinct source-backed negation/deferral and writing-scope examples,
calibrate disputed scope labels with the owner, and broaden independent
evaluation before another training run. Do not repeat this job or promote its
adapter. Roughly 100 useful windows, human calibration, a separately labeled
final test cohort and matched downstream QA remain necessary research work.

The first comparison's data selection is retained here for history:
That run excluded four older writing examples. A later blind review
and root adjudication changed only their scope labels to uncertain, preserving
historical annotations and review disagreements separately. An unrun candidate
then had 34 private rows from three training sessions plus 48 fictional replay
rows. It was superseded by the second comparison without changing development
cases, labels or prompts.

## Boundaries

Private training/evaluation on personal Modal is authorized. Use profile
`reflex-personal`, workspace `rajath-61258`; verify live workspace and inventory
before launch. One GPU call at a time, exact reviewed payload and runner hashes,
source-free serialization, no model weights on this Mac. Private-trained
adapters use the separate private volume; historical prior volumes stay read-only.

Keep private text, labels, prompts, receipts, vectors and private-trained
checkpoints out of GitHub. Original sources, Context Kit and `~/.llm-context`
remain read-only. Public publication is not authorized. Reviewed code, aggregate
docs and fictional fixtures may be signed and pushed to PRIVATE
`1337mus/tracemem`; verify private visibility immediately before every push.
Automation remains PAUSED. Do not duplicate workers or completed GPU calls.

## Validation and delivery

Before the narrow EOS correction, `make check PYTHON=.venv/bin/python` passed:
1,103 tests passed, one optional skip; Ruff and strict mypy passed. After the
correction, 35 affected tests, repository Ruff/format and strict mypy across
74 source files passed. The full suite was not repeated for that focused fix.
See [validation](validation-private-sft.json).

For the second comparison, all 35 focused private SFT/runner/scorer tests passed.
No production code changed. Existing exporters validated the new annotations,
split separation, exact citations and byte-identical development prompts. The
exact runner and payload were reviewed and archived before the single call.
Raw receipt and token hashes were verified after scoring. Personal Modal reported
the app stopped with zero tasks. Recheck live state before any further call.

Read `.local/active-cycle.json` for the current signed/pushed commit and CI
state at delivery; do not treat older successful CI as validation of new code.
The earlier delivered head was `d12cc2e5c27177602ae3a4d268d644be0169a468`.

Verify live GPU/worker/delivery state before acting and by 2026-10-18.
