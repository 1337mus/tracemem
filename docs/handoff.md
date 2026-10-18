# TraceMem handoff

Updated 2026-10-18 UTC. Start with `.local/active-cycle.json` for current
operational state, approvals and delivery. Verify the live inventory is idle
before any remote launch. Historical experiment records are immutable.

## Latest verified results

### Current calendar-evidence augmentation

A calendar-day evidence addition was tested on the same 32 fictional references
with the same three frozen systems. Support-label matches were 28/32 for base,
29/32 for the historical adapter, and 28/32 for the latest adapter, versus
27/32 for each without the addition. The latest elapsed-time score rose from
4/8 to 5/8, below the fixed 6/8 minimum; it rejected only 1/4 unsupported
elapsed-time claims, below the required three. The historical adapter reached
6/8 but retained only 2/4 supported elapsed-time claims. The 3/4 retention
requirement applied only to the latest adapter, which retained 4/4; its other
elapsed-time checks failed. Base eligibility fell from 32/32 to 29/32,
including two new errors outside elapsed time. The latest
retained 15/16 supported claims versus 14/16 before, but its unsupported
rejections remained 13/16; false accepts fell from 3/16 to 1/16. Other fixed
checks also failed, so no model was promoted.

The helper adds Gregorian calendar-day intervals from cited source-date pairs
with record revisions; it does not establish clock-hour duration or event
identity. Root verified that an independent tally matched aggregate and
family-level counts on both scoring axes. The run had zero human reviewers, no
held-out set, no new training and no model selection. All outputs were valid;
the application is stopped with no active tasks. Input grew from 13,034 to
30,726 tokens per arm. Remote/client elapsed time was 201.1/216.8
seconds; billed cost is unknown. Full `make check` exited 0 (1,157 passed, one
optional skip); Ruff covered 233 files and mypy 76. No unchanged rerun is
justified. Next, improve variation in date-event and
precision examples, then evaluate independent real sessions with human review
and calibration. See the [aggregate diagnostic](temporal-grounding-diagnostic.md).

### Original temporal-grounding comparison without calendar evidence

On 32 new fictional train-only claims in 16 paired scenarios across four
families, the base, historical adapter and latest matched adapter all matched
27/32 support labels. The latest adapter rejected 13/16 unsupported claims
(versus 11/16 for base and 15/16 for the historical adapter) and retained
14/16 supported claims (versus 16/16 and 12/16); this showed a trade-off, not
an overall gain. There were zero human reviewers. The author created the claims
and supplied support labels; the second agent labeled them without seeing those
labels, and root independently labeled them before scoring against the frozen
references. The examples were templated. Eligibility was disclosed in the
protocol and every example was personal, so this was not a blind scope-calibration
test. No model was trained or selected. The [original aggregate diagnostic](temporal-grounding-diagnostic.md)
records the family results and limits.

### Earlier downstream QA comparison

Changing only the answering instructions on the same 24 adaptive public
questions scored 13/24 grounded-correct with hybrid retrieval, 10/24 with
oracle evidence, and 0/24 with empty context, versus baseline results of 5/24,
6/24 and 0/24. Unsupported additions rose from 1 to 4 for hybrid and 1 to 7
for oracle, so both answer-condition gates failed and nothing was promoted.
This remains a prompt-only base-model diagnostic, not trained-memory efficacy.
The [aggregate downstream QA report](downstream-qa-baseline.md) records the
subsets, paired changes, runtime and limits.

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

The current full `make check` exited successfully: 1,158 tests ran, 1,157
passed and one optional test was skipped. Ruff covered 233 files and mypy 76.
The GPU runner remains pinned to
`11181e73e0f99e9c4ec0e973d00396acf38985b551d68f16ec6043069e7fa586`.

The previous delivered head `ae0e489` passed [CI run
39096053992](https://github.com/1337mus/tracemem/actions/runs/39096053992).
Check `.local/active-cycle.json` for current delivery and CI status.
