# TraceMem handoff

Updated 2026-10-17 UTC. Start with `.local/active-cycle.json` for current
operational state, receipts, delivery and CI. Verify the live inventory is idle
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

Labels remain provisional agent judgments with zero human reviewers. These are
small adaptive development sets, not held-out evaluation; they do not support
causal or downstream QA claims. The reserved verifier test remains unused.

## Next experiment

An eligibility-focused audit found 67 uncertain-eligibility exposures. A
54-candidate, 27-scenario fictional training set with joint 3×3 balance is being
authored; blind labels are required before training. Claude metadata verification
is complete: five older leads included three already represented in development
and two unverified; no new private rows were admitted. Keep the reserved test
sealed and freeze scorers before any future predictions.

## Operational boundaries

Private training and evaluation on personal Modal are authorized; run one GPU
call at a time and confirm live state is idle before launching. Keep private data
and trained checkpoints out of GitHub. Original sources, Context Kit and
`~/.llm-context` remain read-only. Public publication is not authorized. Keep
historical receipts and adapter volumes immutable.

## Verification and delivery

The current full check ran 1,138 tests: 1,137 passed and one optional test was
skipped. Ruff checked 229 files and configured mypy passed 74 files. Fifty-one
focused tests and nine scorer-regression tests passed. Hosted CI covers the
previously published head only; CI for the current changes is pending.
