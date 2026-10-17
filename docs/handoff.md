# TraceMem handoff

Updated 2026-10-17 UTC. Read `.local/active-cycle.json` for exact private paths,
hashes, runner archives, receipts and live delivery state. Historical source
records and experiment receipts remain unchanged.

## Latest model result

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

The local corpus contains 25 windows / 50 candidates across five sessions.
One additional unlabeled test source from a sixth session remains reserved.
The completed run excluded four older writing examples. A later blind review
and root adjudication changed only their scope labels to uncertain, preserving
historical annotations and review disagreements separately. The next unrun
candidate has 34 private rows from three training sessions plus 48 fictional
replay rows. Its development cases, labels and prompts are unchanged.

Next: broaden realistic uncertain-scope, attribution and conditional examples
across distinct sessions before another frozen comparison. Do not launch the
82-row candidate merely because it exists. Roughly 100 useful source windows,
owner review of ambiguous policy labels, a separately labeled final test cohort
and matched downstream QA remain necessary research work. Preserve the sealed
synthetic candidate and reserved real test source.

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

Read `.local/active-cycle.json` for the current signed/pushed commit and CI
state at delivery; do not treat older successful CI as validation of new code.
The earlier delivered head was `d12cc2e5c27177602ae3a4d268d644be0169a468`.

Verify live GPU/worker/delivery state before acting and by 2026-10-18.
