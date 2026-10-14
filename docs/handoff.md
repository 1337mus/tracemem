# TraceMem continuation

Updated 2026-10-14 after the 267-test uncertainty checkpoint. Live jobs/workers/remote state must be checked
again before acting. Verify by: 2026-10-14. Read `.local/active-cycle.json` first.
This is a continuation checkpoint, not task completion; Raj repeatedly asked for
continuous useful progress beyond passing tests.

Workspace `/Users/raj/Projects/tracemem`. Specification
`/Users/raj/Downloads/TRACEMEM_AGENT_BRIEF.md` was read entirely. Local prototype.
Raj subsequently authorized creating a private project under his personal GitHub
profile. Root created https://github.com/1337mus/tracemem and independently checked
`visibility=PRIVATE`. Signed initial commit `309f41b1810981056b30f811383634d2953165f3`
is pushed to main. Initial hosted CI passed; see `docs/github-validation.json`.
Local signature verification passed at delivery; GitHub reports `bad_email`.
No signing-key or identity settings were changed. No PR required for this prototype.
No public visibility change is authorized; owner plans publication later.

## Boundaries

Preserve Context Kit, original transcripts and `~/.llm-context` as read-only inputs.
All private text, derived claims, vectors, identifiers and checkpoints stay local.
No destructive changes, private-data uploads or expensive training without approval.
Keep weights/Torch/Transformers/model inference off this Mac. All six prior fictional
Modal calls completed; latest Qwen/Phi apps stopped with zero tasks. Do not relaunch.
Current work needs no GPU. The nominal six-call estimate $0.65066 is not billing or
a provider-enforced cap. No model selected; no training started.

Context Kit `/Users/raj/Projects/ctxkit` remains clean at
`20096bcb1095883262c77a37bbbb458732d9cf3a` (root rechecked this cycle).
It has no ChatGPT importer/conversation/vector database. Reuse config/discovery,
read source transcripts separately. Inventory is aggregate-only in
`docs/context-kit-inventory.md`; private manifests remain ignored under `.local/`.
No persistent private conversation DB was created. Do not run write-capable ctx
commands or update the original memory corpus. Durable context stays here.

## Current engineering

- Phase0/1 runs offline: read-only Claude/Codex adapters, immutable exact-text
  evidence revisions, mutable last-imported heads, SQLite FTS5 BM25, signed-hash
  vectors/RRF, optional offline pretrained vectors, revision citations, UTF8-byte
  context packing. No runtime dependencies for default demo.
- Root approved schema3 indexed FTS maintenance and complete source/tests.
  Version2 rejected through read-only preflight, no migration. Identical10k/50k
  corpus/ranking hashes; setup4.058199→0.260205s and129.551996→1.575206s.
  See `docs/lexical-scale-results.md`. Synthetic timings, Python allocations not RSS.
- Root reviewed all ChatGPT reader source/tests and found/fixed three edge cases:
  escaped lone surrogate evidence, huge JSON integer errors, symlink-loop path
  leaks. Adapter now approved, fixture-only compatibility; no actual export read.
  See `docs/chatgpt-export.md`. Selected-parent-chain text only,64MiB input cap.
- Fresh-store ChatGPT API/CLI completed and root-reviewed.11 fictional integration
  tests cover branch independence, exact Unicode citations, output/sidecar
  preservation, malformed imports and exclusive-publication collisions. See
  `docs/chatgpt-snapshot.md`. No actual-export compatibility claim. New fictional
  export fixture demonstrates selected/excluded branches without private data.
- GitHub CI completed and root-reviewed: one Python3.12 Ubuntu job, official
  pinned actions, read-only permissions, locked dev/Modal SDK dependencies, no
  secrets/model calls/artifact uploads. Installed smoke uses its Python tempfile
  root, fixing a RUNNER_TEMP mismatch found in review. actionlint/shell checks
  passed; first hosted run passed on the initial commit. See `docs/ci.md`.
- Fresh offline wheel passed all9commands and34installed ChatGPT tests outside
  checkout.31package files match approved source bytes; schema3/import paths,
  demo hashes and stable benchmark fields match. Source stayed unchanged.
  WheelSHA256 63325e2fa0a4e32079938fdd07d8ae9c8364593256013ee4d1a20f393b628342.
  Receipt `.local/package-smoke-fts-v3-chatgpt-20261014T190150652005Z.json`.
  Extra installed checks reproduce the full budget report and byte-identical
  84-task final HTML; `.local/installed-review-budget-new-apis.json`.
  An initial probe caught a stale preview; failure log preserved, final probe passed.
- Fixed UTF8-byte budget diagnostic completed:14answerable/3unknown queries,
  three retrieval modes,six ceilings. Root checked306old-packer combinations and
  resolved728exact fictional citations. At2048bytes authored evidence sufficiency
  is BM25=14/14,hash=10/14,hybrid=13/14. Not QA accuracy or tokenizer counts.
  All unknowns receive nonempty packs at512bytes+. See `docs/context-budget.md`.
- Offline blind-review HTML completed: source/candidate display, Unicode citation
  checks, explicit drafts/restore and complete exports. Nine focused tests include
  Node VM DOM harness and Python importer roundtrip. Root reproduced/fixed citation
  end truncation, whitespace parity and literal NUL/CSP hash faults. Checked all
  1,114,112Unicode code points. Browser rendering unverified: CUA rejected file URL;
  no workaround attempted. See `docs/offline-review.md`. Final fictional page
  `.local/scenario-human-review-final-20261014.html`,SHA256
  07521f971f5554e5476d15a1299eb75cf0b5b20fbc6854c79b6053a1aecc474e.
- Pinned benchmark-contract source audit complete and root-reviewed.13 public
  README/license/evaluator/tree records, no benchmark data/upstream execution.
  `docs/benchmark-contracts.md/json`; exact hashes verified. Root added precise
  LongMemEval DCG/projection conventions and LoCoMo empty-context/rounding caveats.

## Frozen synthetic extraction comparison

Dataset `src/tracemem/data/scenarios_v1.json` SHA256
`5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214`.
48 fictional multi-session cases,24dev/24test,eight no-memory,43 provisional
nonexhaustive authored targets; no training split or human labels. Gate A unmet.
Frozen prompts/runner/payload hashes retained in `.local/active-cycle.json`.
No post-test tuning, salvage or model selection.

Qwen raw receipt `examples/synthetic/qwen3-4b-scenarios-remote.json` SHA256
`2035de5bfeceaf3d4be4117dd47a515ef67df1cb7899a56d5a2eacefa349410f`.
Dev1schema failure/41claims; test2schema failures/42claims. Integrity raw1/41,
1/42; anchored41/41,40/42. No-memory3/4dev,1/4test. No truncation.

Phi raw receipt `examples/synthetic/phi-3.5-mini-scenarios-remote.json` SHA256
`d5eff2e56a827a76a7426b47ccd26e88601bd85f2f57f178c4e0f62725a2648d`.
Dev1parse+23schema failures; test23schema failures+one valid claim with invented
end date. No-memory0/4both. No truncation. Main faults date/enum schema misuse.
Both exact structured F1=0 measures reference agreement, not semantic precision.

All84 schema-valid candidates reviewed blind by Luna:50supported/31unsupported/
3uncertain. Root read all84 full candidates/conversations/rationales and made nine
separate adjudications:41supported/33unsupported/10uncertain. Root nonblind,
nonindependent, no humans or auto-acceptance. Reports preserve parse/schema
exclusions and per-model/split/reviewer denominators. Original reviews unchanged.
Separate8positive+8simple-negative dev-derived controls:agent16/16 agreement,
no calibration/independent truth. Historical43-positive-only review unchanged.
Reproducible fictional bundles/keys/responses under `examples/synthetic/reviews/`.

Candidate ledger, strict claim/citation validators, anchoring, blind review tools
and exact revision fetch implemented. Ledger stores claim quotes; callers retain
EvidenceStore separately. Supersession records proposal lineage, not current-fact
truth. NLI diagnostic had high-confidence false positives; no verifier policy.

## Validation and next actions

Current full suite:267tests,zero skips,4.514seconds including actual Context Kit
integration and Node VM. Ruff71files,source mypy26,scoped39. Exact commands/exit
codes `.local/final-uncertainty-commands.json`; `docs/validation.json` records
source/test hashes. Historical257test checkpoint in `docs/validation-review-budget.json`.
Runtime package unchanged; all31source hashes still match the installed wheel.

Scenario-cluster uncertainty experiment completed and root-reviewed. All24cases
per split,including excluded/empty outputs,remain in resampling population.2000draws,
seed1729. Root independently checked8cells/24metrics via case-origin counts and
statistics.quantiles,receipt `.local/review-uncertainty-root-verification.json`.
Qwen blind-agent supported/all:dev26/41,interval[.474,.794],test24/42[.388,.778].
Root nonblind dev20/41[.326,.659],test21/42[.306,.718]. Phi dev undefined;test0/1,
746/2000draws undefined,degenerate[0,0]conditional on remaining draws. These are
resampling diagnostics,not independent truth,precision guarantees or model selection.
See `docs/review-uncertainty.md` and `docs/scenarios-review-uncertainty.json`.

Three earlier unmapped Graphiti hashes now match exact pinned source files and
Git blob IDs. Root inspected cited schema/prompt,episode mapping and temporal
normalization code. `docs/graphiti-source-audit.json` and `docs/related-work.md`.
No upstream code execution or benchmark conversations fetched. No new GPU calls.

1. Private repository verified live on2026-10-14. Reviewed budget/offline source
   pushed as signed664587cd5689d44a6572a978efd32ecffcd20de4. HostedCI38620827280
   passed257tests(one expected Context Kit skip),Node VM and34installed ChatGPT
   tests. GitHub signature reason remainsbad_email; local verify-commit passes.
   New267test uncertainty/doc slice prepared for signed private delivery.
2. No active worker at this checkpoint. Next bounded slice is
   `docs/installed-new-apis-brief.md`: permanently exercise installed budget/review
   APIs in existing CI smoke; preserve all existing checks. Root owns integration,
   exact source/test hashes and final validation. Check `.local/active-cycle.json`
   before starting so a resumed session does not duplicate a newer worker.
3. Continue useful work beyond CI. No PR/Linear required for prototype bootstrap.

Open: independent human labels/Gate A, calibrated semantic verification, source
lifecycle/erasure, temporal truth and matched-budget downstream QA. No trained
model, full private-corpus deployment, public release or novelty claim.

Automation `continue-tracemem-prototype` remains ACTIVE,every30minutes, same chat;
root verified saved config this cycle. Quiet unless meaningful progress/failure/
action. Avoid duplicate work. Original memory context stays read-only.
