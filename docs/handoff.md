# TraceMem continuation

Updated 2026-10-14, 11:39 Pacific. Live jobs/workers/remote state must be checked
again before acting. Verify by: 2026-10-14. Read `.local/active-cycle.json` first.
This is a continuation checkpoint, not task completion; Raj repeatedly asked for
continuous useful progress beyond passing tests.

Workspace `/Users/raj/Projects/tracemem`. Specification
`/Users/raj/Downloads/TRACEMEM_AGENT_BRIEF.md` was read entirely. Local prototype.
Raj subsequently authorized creating a private project under his personal GitHub
profile. Root created https://github.com/1337mus/tracemem and independently checked
`visibility=PRIVATE`; it is still empty. Unborn local main; origin is the private HTTPS repository; no commit/PR yet.
Next: verified signed initial commit and push of code/fictional fixtures only.
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
  passed; first hosted run awaits push. See `docs/ci.md`.
- Fresh offline wheel passed all9commands and34installed ChatGPT tests outside
  checkout.29package files match approved source bytes; schema3/import paths,
  demo hashes and stable benchmark fields match. Source stayed unchanged.
  WheelSHA256 d8e66f6a8554138100a639374501e55c9dc33bdbb5e26666c86defddbe7bee0c.
  Receipt `.local/package-smoke-fts-v3-chatgpt-20261014T183745147324Z.json`.
  Historical storage-only465d58... wheel remains preserved. No new model calls.
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

Current full suite:242tests,zero skips,2.808seconds with actual Context Kit and
Modal SDK. Ruff63files,source mypy24,scoped34; workflow actionlint passed.
`docs/validation.json` now records this checkpoint and all source/test/artifact
hashes. Historical136test receipt stays in `docs/validation-support-ledger.json`.
Root independently checked package receipt/hashes and ran fictional CLI import→
BM25→exact citation; excluded sibling query returned zero hits.

1. Refresh and inspect exact staged-upload manifest. Only code, docs, configs and
   fictional fixtures; never .local or original sources. Scanner's only hits were
   deliberately synthetic privacy test strings. Signed initial commit and push
   to private1337mus/tracemem. Verify remoteHEAD and private visibility.
2. Watch first GitHub Actions outcome and fix any real CI/platform/dependency
   failures. Do not claim hosted CI passed before a completed successful run.
3. Record exact GitHub delivery and CI state; then continue the next useful
   bounded prototype slice. No PR/Linear required for this prototype bootstrap.

Open: independent human labels/Gate A, calibrated semantic verification, source
lifecycle/erasure, temporal truth and matched-budget downstream QA. No trained
model, full private-corpus deployment, public release or novelty claim.

Automation `continue-tracemem-prototype` remains ACTIVE,every30minutes, same chat;
root verified saved config this cycle. Quiet unless meaningful progress/failure/
action. Avoid duplicate work. Original memory context stays read-only.
