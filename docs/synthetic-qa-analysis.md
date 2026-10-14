# Synthetic QA receipt analysis

This tool analyzes a receipt from the frozen 72-request diagnostic in
[`synthetic-qa-answer-protocol.md`](synthetic-qa-answer-protocol.md). It calls no
model and does not change the prepared requests, references, prompt, retrieval,
or payload. The analyzer's implementation tests use fictional constructed
receipts. A separate successful Qwen3-4B synthetic answer-generation run is now
recorded in the [split results](synthetic-qa-qwen3-results.md), with its raw
receipt and full analysis preserved separately.

Run it with the two reviewed hash approvals and a new output path:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.analyze_synthetic_qa \
  --receipt .local/synthetic-qa-receipt.json \
  --approve-runner-sha256 <reviewed-runner-sha256> \
  --approve-template-sha256 <reviewed-template-sha256> \
  --output .local/synthetic-qa-analysis.json
```

The receipt is limited to 2 MiB. JSON with duplicate object keys or non-finite
numbers is rejected. Analysis requires the exact prepared request and reference
hashes, fixture hash, and payload hash. The receipt must identify the same
payload, approved runner, and approved tokenizer template; the pinned model,
revision, config, package versions, request count, and output set are checked as
well. These are reviewed provenance checks, not remote attestation. Other
runtime metadata remains in the input receipt and is not treated as attestation.

The output file must not already exist. The CLI creates it with mode `0600`,
rejects symlink targets, and returns a sanitized nonzero error for invalid input
or output paths.

For the completed run, root independently replayed all 18 cells, 72 literal
outcomes, and token/finish/empty counts. The nonzero rendered context for each
question was identical across retrieval modes and 512/2048-byte ceilings, so
the run cannot compare retrieval modes or context capacity. See the
[run results and limitations](synthetic-qa-qwen3-results.md). The output is
provisional agent-only evidence; it is not semantic accuracy, training-ready
data, or authorization for another inference run.

The report contains 18 separately labeled cells: dev and test, each crossed with
the three retrieval modes and byte ceilings 0, 512, and 2048. Every cell has four
requests: three answerable questions and one question whose reference is exactly
`UNKNOWN`. Literal agreement is reported as numerator and denominator for all
four questions, answerable questions, and unknown questions. Zero-byte cells
also count literal `UNKNOWN` outputs across all four questions; the corresponding
denominator is zero in other cells. Each cell includes finish reason, hit-limit,
truncation, and empty-output counts, plus min/max/sum input and generated token
counts. Empty and length-stopped outputs stay in agreement denominators. A stop
reason of `other` stays in every denominator and appears in the finish counts.

Normalization is exactly `' '.join(text.casefold().split())`. Punctuation,
substrings, aliases, and semantic equivalence receive no special handling. The
report preserves all 72 request outcomes in fixed request order so cell counts
can be recomputed; it omits raw answer text, which remains in the immutable input
receipt. Shuffling output rows leaves cells and outcomes unchanged, while the
canonical source-receipt hash changes because it hashes the supplied array order.

Provenance includes canonical hashes for the input receipt, request and
reference bundles, fixture, payload, runner, template, model config, and analyzer
source. The report is labeled **provisional agent-only** and states its limits:
there is no semantic judging, matched-token budget, or independent scenario
population. Byte ceilings are UTF-8 bytes, and the three ceilings repeat the
same eight questions. Dev and test remain separate; there is no pooled score or
ranking. These fictional references are not independently reviewed and the
diagnostic says nothing about benchmark compatibility, model superiority,
calibration, semantic correctness, cost per correct answer, or training
readiness.

Validation chronology: the first CLI test attempt invoked the script directly
and only exposed a harness import failure (`red-15.log`). The corrected module
invocation exposed a report-creation failure (`red-15-fixed-invocation.log`),
which subsequently passed (`green-15-module.log`). The oversized timing-integer
regression was reproduced and fixed in a later focused run; those standalone
RED/GREEN outputs were observed when run but were not saved as raw logs. Persisted
final validation includes that regression in `synthetic-qa-inference-final-check.log`
(445 tests) and `synthetic-qa-inference-installed.log` (48 tests).
