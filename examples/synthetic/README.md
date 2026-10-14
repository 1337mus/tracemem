# Synthetic retrieval fixture

The canonical authored corpus and relevance labels live in the packaged resources
[`corpus.json`](../../src/tracemem/data/corpus.json) and
[`queries.json`](../../src/tracemem/data/queries.json). Keeping one packaged copy lets
`tracemem demo` and `tracemem benchmark` work after installation as well as from the
repository.

Run `python -m tracemem demo` to inspect synthetic evidence, or
`python -m tracemem benchmark` to compare BM25, signed-hash vectors, and reciprocal rank
fusion. Benchmark output contains IDs and aggregate retrieval diagnostics, not fixture text.
The labels measure retrieval and curated evidence-set coverage; they do not score generated
answers or claim fidelity.

## Synthetic-only remote receipts

The following receipts contain authored synthetic inputs and remote model outputs only:

- [`minilm-remote.json`](minilm-remote.json): pinned MiniLM vectors for the retrieval
  comparison.
- [`qwen3-4b-remote.json`](qwen3-4b-remote.json): Qwen3-4B V1 extraction outputs.
- [`qwen3-4b-explicit-v2-remote.json`](qwen3-4b-explicit-v2-remote.json): explicit-v2
  prompt outputs. Its SHA-256 is
  `e624dcab7ff7bb7c7214e4245ff9659509be7331459b9193f2c76730c8fc9dcd`.
- [`nli-deberta-remote.json`](nli-deberta-remote.json): raw probabilities and logits for
  the source-free synthetic support diagnostic. Its SHA-256 is
  `16be7f4c542bd174e49f975bfc3d8baa7ef9995fdd5d8cfcf2e3df03893f3871`.
- [`qwen3-4b-scenarios-remote.json`](qwen3-4b-scenarios-remote.json): Qwen3-4B output
  for the frozen 48-case scenario comparison. Its SHA-256 is
  `2035de5bfeceaf3d4be4117dd47a515ef67df1cb7899a56d5a2eacefa349410f`.
- [`phi-3.5-mini-scenarios-remote.json`](phi-3.5-mini-scenarios-remote.json):
  Phi-3.5-mini output for the same comparison. Its SHA-256 is
  `d5eff2e56a827a76a7426b47ccd26e88601bd85f2f57f178c4e0f62725a2648d`.

Offline evaluations and offset-anchoring replays are linked from the
[Phase 2 report](../../docs/experiments-phase2.md). The bounded [dataset readiness
report](../../docs/dataset-readiness.json) records that the 24 cases remain in one
diagnostic split. V1 and explicit-v2 use that same historical challenge, so their
comparison is prompt development only. The newer frozen comparison is a distinct
48-case fictional fixture with 24 dev and 24 test cases and 43 provisional
agent-authored targets. Its models use the same frozen prompt and decoding ceilings;
the outputs and limitations are in the [scenario report](../../docs/experiments-scenarios.md)
and [data card](../../docs/scenarios-data-card.md). Neither comparison establishes
held-out quality or authorizes training.

The NLI receipt corresponds to the separate 48-pair, 24-scenario fixture
[`support_challenge.json`](../../src/tracemem/data/support_challenge.json), described in
the [support-challenge data card](../../docs/support-challenge-data-card.md). Its
[computed report](../../docs/baseline-nli-support.json) and
[error audit](../../docs/support-audit.md) are diagnostic only. The fixture labels were
agent-reviewed and are not independent human ground truth; no held-out split is assigned.

The combined generated-candidate review and simple control artifacts are stored under
[`reviews/`](reviews/README.md), with hashes in the [manifest](reviews/manifest.json).
They contain fictional scenario text and agent judgments only. Parse/schema exclusions
remain explicit; the blind agent responses and subsequent nonblind root adjudication are
reported separately and are not human consensus.

`chatgpt-export.json` is a newly authored fictional ChatGPT-export-shaped fixture
with an explicitly selected branch, an excluded sibling, and Unicode text. It
contains no copied export data and is covered by the project's MIT license.
