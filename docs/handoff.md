# TraceMem handoff

Updated 2026-10-17 UTC. Start with `.local/active-cycle.json` for live state,
private paths, exact hashes, receipts, signed delivery and CI. Historical
sources and experiment receipts are immutable. Verify running jobs before acting.

## Latest verified results

The 256-row candidate trained from synthetic-v2 for one epoch: 256 microsteps,
64 optimizer steps and 335.533268 seconds. On the unchanged 16 private dev
candidates it accepted 7/7 useful claims, retained 8/8 supported claims, rejected
7/8 unsupported claims and matched 0/2 uncertain-scope labels. The original
gate failed. All seven useful references come from one session; the 16
candidates span two sessions. See [contrast results](private-sft-contrast-results.md).

A separate frozen inference evaluated that adapter, synthetic-v2 and base on
16 new fictional dev candidates across eight scenarios, plus the same 16
private controls. Each arm reproduced its earlier private-control input/output
token IDs exactly. All 96 outputs were valid JSON and ended at EOS. The Modal
app was verified stopped with zero tasks; no GPU call remained active at that
verification. Recheck the local state and live inventory before any launch.

On the fictional set, the private adapter and synthetic-v2 each accepted 6/7
useful claims, retained 7/8 supported claims, rejected 6/6 unsupported claims
and matched 0/2 uncertain-support labels. Eligibility improved from 14/16 to
15/16. The base retained 8/8 supported claims and matched 1/2 uncertain-support
labels. The narrow comparative diagnostic passed; the failed private training
gate remains separate. No uncertain-eligibility references occur in this set.
See [broader fictional results](naturalistic-dev-results.md).

These are adaptive development diagnostics with provisional agent labels,
zero human reviewers and scripted fictional scenarios. The reserved test stays
unused. No extraction or downstream QA efficacy was measured; no model was
selected or promoted. Historical results remain linked in the documentation index.

## Next experiment

The 256-row training mix contains zero uncertain-support references. The next
hypothesis is that reviewed, train-only ambiguity contrasts can restore this
label without reducing useful acceptance or unsupported rejection. Preparation
is underway in the local state; do not infer launch approval. Keep all existing
dev examples out of training. Review labels independently, freeze a bounded
comparison and exact hashes, then use the existing source-free runner. Reuse
the current 256-row compute envelope; avoid another unchanged training run.

Private training/evaluation on personal Modal are authorized. Only one GPU
call may run at a time. Keep private text, labels, prompts, receipts, vectors
and private-trained checkpoints out of GitHub. Context Kit, original sources,
historical receipts and `~/.llm-context` stay read-only. Public publication is
not authorized. Historical adapter volumes stay read-only.

## Verification and delivery

The full suite passed 1,115 tests with one optional skip before final narrow
corrections. Afterwards, 35 focused tests passed; the final runner checks passed
18 tests, Ruff/format checked 227 files and configured mypy passed 74 files.
Six local inference-workflow tests passed. Exact source hashes and logs are in
`.local/active-cycle.json` and its linked validation records. Read its delivery
entry for the pushed commit and CI; an older green run does not validate a new head.
