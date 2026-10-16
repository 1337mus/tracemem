# Data governance

Durable evidence and retrieval run locally. Training experiments have
used newly authored fictional fixtures. The owner has also authorized selected
private conversations for training and evaluation on personal Modal; that
authorization does not permit putting them or adapters trained on them in GitHub.
Use restricted private experiment paths and explicit private provenance. See
[realistic data](realistic-data.md) for the collection and review workflow.
Four extractor training experiments and two verifier training experiments have
completed on fiction. Their results are mixed; none selected a product model.
The latest [verifier result](verifier-v2-results.md) includes separate development
and frozen test reports. Reviews have zero human reviewers.
The first [private inference pilot](private-pilot-results.md) evaluated seven
real user messages and 14 candidate claims on personal Modal; no private
training or private checkpoint creation has occurred. Raw inputs and receipts
remain restricted and outside GitHub. No public release occurred, and no model
weights were downloaded to this Mac. Check
`.local/active-cycle.json` and live Modal state before any new call. Code
licensing does not grant rights to private conversations.

Context Kit inspection included a bounded local parse of original Claude/Codex
logs for format, role, timestamp and privacy-indicator checks. It did not run
the trained extractor over those conversations or add them to training.
No actual ChatGPT export was read. The adapters and retrieval implementation
have synthetic tests; those tests do not establish quality on the private corpus.

The [LongMemEval dataset metadata audit](longmemeval-rights.md) pins the official
cleaned dataset's public card and tree. The card declares MIT; constituent provenance
and third-party permissions remain unresolved. Only metadata was fetched. The format
adapter uses newly authored fictional conversations, with no benchmark data imported.

## Two data lanes

- **Synthetic:** newly authored fictional fixtures, MIT licensed with this project.
  These contain no copied private conversation excerpts. They are diagnostic examples,
  not representative population samples or an audited training dataset.
- **Private:** configured Context Kit transcripts and derived notes. Read-only inspection
  and local evaluation are allowed. Owner-authorized private training and evaluation
  may also run on personal Modal after minimizing and reviewing the exact batch.
  Treat all text, claims, vectors, checkpoints,
  source locators, identifiers and generated context as private. Keep them in `.local/`,
  which is ignored by git, with directory mode 0700 and file mode 0600.

## Dataset readiness evidence

The bounded [dataset readiness report](dataset-readiness.json) describes the historical
24-case challenge only. All 24 cases are in the `diagnostic` split, with zero train, dev,
test, or held-out cases. It found no exact duplicate pairs. Cross-split near-duplicate
checks are vacuous because no evaluation splits are assigned. This audit does not prove
entity or template independence, legal permission, annotation quality, sample adequacy,
or Gate A approval.

The separate frozen [scenario fixture](scenarios-data-card.md) contains 48 fictional,
agent-authored cases, split 24 dev and 24 test, with 43 provisional target claims. Qwen
and Phi completed the same frozen comparison. One blind agent reviewer agreed with all
43 positive reference targets; that review had no negative controls or human reviewers.
A separate simple dev-derived control check had 8 positive and 8 negative controls with
16/16 agreement from one agent reviewer and no human reviewers. The controls show
discrimination on those examples only, not semantic calibration, independent truth,
held-out performance, or Gate A. See the [historical reference review](scenarios-reference-review.json)
and [control receipt](reviewer-controls.json).

The 84 generated candidates were reviewed separately after explicit parse/schema
exclusions. The original blind agent responses totaled 50 supported, 31 unsupported and
3 uncertain. A subsequent nonblind root adjudication changed nine labels, resulting in
41 supported, 33 unsupported and 10 uncertain. These are separate descriptive agent
judgments, not human truth, consensus, model acceptance, or Gate A. The raw candidate
support accounting and adjudication are described in the
[scenario report](experiments-scenarios.md).

The distinct [NLI support challenge data card](support-challenge-data-card.md) describes
48 premise/hypothesis pairs across 24 new fictional scenarios. It is also diagnostic-only
and has no independent human labels or held-out split. The [support audit](support-audit.md)
records the measured per-threshold errors and the limitations of those results.

## Model license evidence

The pinned [Qwen3-4B LICENSE text](https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/LICENSE)
was verified as Apache-2.0; its SHA-256 is
`832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`. The pinned
[all-MiniLM-L6-v2 README](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)
has SHA-256
`dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0` and its license
card says Apache-2.0; that pinned snapshot has no standalone `LICENSE` file. These are
bounded checks of the pinned repository artifacts, not a complete rights review of model
ancestry, training data, dependencies, use restrictions, or all redistribution conditions.
Do not infer dataset rights, training authorization, or permission to release weights
from these model-license artifacts. The Phi-3.5-mini model metadata reports MIT; this
metadata observation is not a certificate of rights or a complete review of licensing,
model ancestry, training data, dependencies, usage restrictions, or redistribution terms.

The separate verifier experiment uses
[cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
at revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`. Its public model card declares
Apache-2.0, and the pinned snapshot has no standalone `LICENSE` file. This metadata check
does not complete a rights review. The verifier call and offline replay are complete.
Integrated checks and fresh installed-wheel smoke passed. See the
[verification brief](verification-brief.md) for the source-free payload boundary and the
[support audit](support-audit.md) for the measured results. The separate three-call
baseline receipt does not cover this experiment.

Do not write exports into Context Kit or provider directories. Do not publish this lane
or send it to a hosted API without explicit owner approval. Local permissions are access
controls, not encryption. Disk encryption and access by the same OS user remain external
controls. An ignored path is not a substitute for reviewing the staged files before release.

## Source and trust boundaries

Original provider JSONL files and explicitly supplied ChatGPT JSON exports remain
the source of truth. The optional [ChatGPT reader](chatgpt-export.md) is verified
against fictional fixtures only; no actual export was located or read. It follows
one selected parent chain and rejects malformed conversations before emitting
their records. Reimporting changed branches into an existing append-only store
does not remove older sibling heads.

TraceMem preserves decoded text blocks
without concatenating roles. Character offsets count Python Unicode codepoints in those
blocks, not bytes, UTF-16 units, or offsets in the escaped JSON file. IDs and revisions
bind citations to specific source content. Native provider IDs are preferred; fallback
IDs depend on original line order. Moving a source can change its opaque locator.

Text from a user message may include quoted documents, pasted third-party claims and
embedded instructions. Its outer role alone does not prove direct user authorship of
every sentence. Such text is untrusted. Tool output and assistant suggestions cannot
establish user decisions. TraceMem never executes text from transcripts.

Attachments and unsupported blocks require explicit future adapters. Do not call a
partial text import a complete attachment archive. Derived markdown notes are useful
navigation material but cannot substitute for original conversational evidence.

## Deletion and publication gates

The first prototype does not ingest the whole private corpus and does not implement
source erasure. The [source-retention impact diagnostic](source-retention.md) traces
explicit snapshot dependencies without opening or modifying a store. It distinguishes
exact citations from conservative ambiguous span matches and follows correction
lineage. It does not discover external copies or establish semantic information flow. It has no automatic destructive cleanup. Before persistent private
corpus deployment, implement owner-authorized erasure of source-linked evidence,
FTS rows, vectors, claims, exports and caches, including SQLite free-page/WAL handling
and backup retention. Source deletion and claim supersession are different operations.

Before any dataset or weight release, review source licenses, consent, ownership,
split leakage, secret/PII detector misses, and memorization attacks. Automatic redaction
cannot certify a conversation as safe to share. Sensitive text must not appear in
test failure messages, repository fixtures or benchmark logs.
