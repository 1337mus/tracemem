# TraceMem handoff

Updated 2026-10-18 UTC. Start with `.local/active-cycle.json` for current
operational state, approvals and delivery. Verify the live inventory is idle
before any remote launch. Historical experiment records are immutable.

## Latest verified results

### Current downstream QA comparison

A root-adjudicated, inference-only prompt comparison on the same 24 adaptive public questions scored 13/24 grounded-correct with hybrid retrieval, 10/24 with oracle evidence, and 0/24 with empty context, versus baseline results of 5/24, 6/24, and 0/24. Unsupported additions rose from 1 to 4 for hybrid and 1 to 7 for oracle, so both answer-condition gates failed and nothing was promoted. Only the base Qwen3-4B system instructions changed; this is not trained-memory efficacy evidence. The [aggregate downstream QA report](downstream-qa-baseline.md) records the full24 and 22/2 subset counts, paired changes, runtime, and limits. Immediate next step: run the original fictional temporal/modality verifier diagnostic before any training. Broader independent conversations with human-reviewed labels and matched baseline/candidate conditions remain necessary before an efficacy claim or promotion.

The latest controlled matched-exposure run used 256 exposures, 64 optimizer
steps and 212 unique consumed rows, including all 52 reviewed fictional claims
once. It matched recovery's private exposure identities and positions, and the
frozen 32-prompt dev set remained unchanged. The candidate reproduced recovery
on all 32 predictions; its private uncertain-eligibility target still failed,
so there was no measured efficacy gain and no model was selected. See the
[aggregate-only matched-exposure report](matched-exposure-results.md) for
cohort scores, run accounting, limits and the next evaluation direction.

The first approved balanced-support launch failed with an `OSError` during
runtime setup, before training. Its planned schedule was not consumed and
produced no score; the original cause remains unestablished. A separate
CPU-only public-checkpoint cache and offline loader succeeded, staging ten
public files (about 8.06 GB) in remote-only storage.

A reviewed recovery then trained on 256 exposures: 85 supported, 85
unsupported, and 86 uncertain, with repeats. The schedule consumed 193 unique
source rows, including 37 private rows out of 50 available. Training used 64
optimizer steps and took 334.441137 seconds. On 32 dev prompts, all 96 outputs
were valid JSON and ended with EOS. Base and prior input/output token arrays
matched frozen inference on 32/32 cases; the recovery candidate's input arrays
matched on 32/32 cases.

Compared with the historical 256-row adapter, the recovery retained the same
supported claims in both cohorts and rejected one additional unsupported
private claim. It accepted one fewer useful private reference and had one fewer
exact private eligibility label; fictional eligibility improved to 16/16 and
uncertain-support labels to 2/2. Both frozen gates failed, so no model was
selected or promoted. The [aggregate report](balanced-support-results.md)
contains the full comparison. The original comparison and training-fit
diagnostic remain historical in the [uncertainty report](uncertainty-training-results.md).

A later one-call, inference-only transfer diagnostic compared frozen adapters
on 16 new fictional training claims and 16 unchanged adaptive private controls.
The latest adapter scored 4/8 on uncertain fictional eligibility, below prior
at 6/8 and base at 5/8; the prospective uncertainty target failed. It tied
prior on overall fictional eligibility (12/16), support retention (6/7),
unsupported rejection (4/4), and useful acceptance (1/2). This train-split
diagnostic does not establish held-out effectiveness. All 96 outputs were valid
JSON with EOS, and all three arms matched historical control input/output
tokens 16/16. No training or promotion occurred. See the
[eligibility transfer diagnostic](eligibility-transfer-diagnostic.md) for
aggregate results, timing and limitations.

Labels remain provisional agent judgments with zero human reviewers. These are
small adaptive development sets, not held-out evaluation; they do not support
causal or downstream QA claims. The reserved verifier test remains unused.

## Next experiment

An eligibility-focused audit found 67 uncertain-eligibility exposures. A
54-candidate, 27-scenario fictional set completed a label-hidden annotation
pass. Root review found two semantic ambiguities despite full initial agent
agreement; one label was corrected and a disputed two-claim scenario withheld.
The reviewed set contains 52 claims across 26 scenarios and passed evidence,
annotation and export checks. A bounded train-only comparison has completed
with those claims replacing 26 legacy scenarios in matching eligibility
categories. Its target passed, but the overall and original gates failed; no
model was selected or promoted. The [eligibility-transfer results](eligibility-transfer-results.md)
include the aggregate scores, run counts and limitations. The earlier
[inference-only transfer diagnostic](eligibility-transfer-diagnostic.md)
remains a separate historical comparison.

The completed schedule used 256 exposures and 64 optimizer steps, consuming
212 unique rows: 45 of the new claims and 41 private rows. The recovery used 37
private rows, so the outcome cannot be attributed solely to the fictional
replacements. All 50 private source rows and all 32 dev prompts and references
remained unchanged. Four builder tests and eleven scorer tests passed; no full
workspace suite was rerun for this experiment.

The 24-claim comparison on 12 public benchmark dialogue excerpts is complete.
The continued candidate matched recovery on supported retention (11/12) and
unsupported rejection (11/12), but had lower useful acceptance (9/12 versus
10/12) and eligibility exactness (20/24 versus 22/24). The frozen gate failed,
so no model was selected or promoted. See the [aggregate-only public-dialogue
results](public-dialogue-verifier-results.md); labels remain provisional, the
sample has no uncertainty or ineligible references, and it is not a held-out or
official benchmark evaluation. Broader real-session evaluation, human
calibration, and downstream QA remain necessary. The matched private-exposure
audit is now complete. Next, evaluate on genuinely independent conversations
with separately reviewed labels and downstream QA, and resolve the disputed
memory-scope policy before changing the data or hypothesis for any further
training. Keep the reserved verifier test sealed.

## Operational boundaries

Private training and evaluation on personal Modal are owner-authorized. Before
each launch, verify the exact payload and internal root review, confirm the live
inventory is idle, and run one GPU call at a time. A proposal here does not
authorize a launch. Keep private data and trained checkpoints out of GitHub.
Original sources and machine-local context remain read-only. Public publication
is not authorized. Keep historical receipts and adapter volumes immutable.

## Verification and delivery

The last full `make check` ran 1,143 tests: 1,142 passed and one optional test
was skipped. It ran before the narrow scorer receipt-parser fix. After that
fix, 61 focused tests passed; Ruff and format checks and configured mypy over
74 files also passed. The canonical aggregate score receipt succeeded and
agrees with the independent counts. The frozen runner remained pinned to
`181fe0ad6c68e529351e48048e5770da76031aecf7257cb49cb4e84232069ba4`. The full
workspace suite was not rerun after the fix.

CI passed for code head `9e5b151d627a09ba5d52bcf6d12dcbda278d95d4` in
[run 39089212082](https://github.com/1337mus/tracemem/actions/runs/39089212082),
including full checks and the installed-wheel smoke test. Check
`.local/active-cycle.json` for current documentation delivery and CI status.
