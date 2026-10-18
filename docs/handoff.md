# TraceMem handoff

Updated 2026-10-18 UTC. Start with `.local/active-cycle.json` for current
operational state, approvals and delivery. Verify the live inventory is idle
before any remote launch. Historical experiment records are immutable.

## Latest verified results

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

The next evaluation batch has 24 claims from 12 metadata-selected public
benchmark dialogue excerpts. Four source/citation problems were corrected in
a preserved v2; root verified all 14 whole turns against the original source.
Blind agent labels and author labels agree on all 24 claims, and root froze
provisional references before inference: 12 supported and 12 unsupported,
all eligible. These examples add source variety but have no uncertainty-class
coverage and are not independent owner sessions. A local label-free serializer
is in progress; no GPU comparison is approved yet. Independent real-session
evaluation, downstream QA and human calibration remain necessary. Keep the
reserved verifier test sealed.

## Operational boundaries

Private training and evaluation on personal Modal are owner-authorized. Before
each launch, verify the exact payload and internal root review, confirm the live
inventory is idle, and run one GPU call at a time. A proposal here does not
authorize a launch. Keep private data and trained checkpoints out of GitHub.
Original sources and machine-local context remain read-only. Public publication
is not authorized. Keep historical receipts and adapter volumes immutable.

## Verification and delivery

The latest full check ran 1,138 tests: 1,137 passed and one optional test was
skipped. Ruff checked 229 files and configured mypy passed 74 files. Fifty-one
focused tests and nine scorer-regression tests passed. Hosted CI passed for the
delivered code commit `b837638917762a48b92d1deca091eb709d8c97d5` in
[run 39082188255](https://github.com/1337mus/tracemem/actions/runs/39082188255).
The eligibility-transfer builder and scorer suites passed four and eleven
focused tests, respectively. This experiment and the documentation updates
are outside that CI run; the full workspace suite was not repeated.
