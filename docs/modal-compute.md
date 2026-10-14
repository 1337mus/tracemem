# Modal compute preference

On 2026-10-13 the owner selected Modal for GPU rental and requested low local model/token
use. Keep this Mac on lightweight CPU inventory, tests and retrieval. Do not download
model weights, install Torch or run LLM inference on the Mac by default.

## Completed synthetic-only calls

The one-shot runner completed six bounded Modal calls using authored synthetic fixtures
only: MiniLM embeddings, Qwen3-4B V1 and explicit-v2 on the historical 24-case challenge,
the NLI diagnostic, and Qwen3-4B plus Phi-3.5-mini on the frozen 48-case comparison. All
six Modal apps were verified stopped with zero active tasks. The two scenario apps stopped
at 10:53:34 and 11:11:16 Pacific on 2026-10-14. No private conversation data was uploaded.
No training ran. Model weights were fetched in remote containers; no model weights were
downloaded to this Mac.

| Run | GPU compute | Model load | Remote function | Client wall, including image/startup | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM embedding | 0.752113 s | 9.295326 s | 13.315121 s | 129.854275 s | 59,023,872 bytes |
| Qwen3-4B V1 extraction | 304.772849 s | 73.579751 s | 382.079294 s | 399.468317 s | 8,822,848,512 bytes |
| Qwen3-4B explicit-v2 extraction | 220.447001 s | 60.141208 s | 283.622759 s | 291.010439 s | 8,822,848,512 bytes |
| NLI support diagnostic | 1.558278 s | 9.804156 s | 11.563326 s | 24.168136 s | 826,340,864 bytes |
| Qwen3-4B frozen 48-case comparison | 919.587205 s | 75.284621 s | 995.338163 s | 1,069.663869 s | 8,822,848,512 bytes |
| Phi-3.5-mini frozen 48-case comparison | 899.763988 s | 56.772880 s | 956.932029 s | 968.419187 s | 9,715,433,984 bytes |

All three Qwen runs used a Tesla T4 and the pinned revision
`1cfa9a7208912126459214e8b04321603b3df60c`. V1 generated 5,978 tokens from 8,560 input
tokens without truncation. Explicit-v2 generated 4,231 tokens from 23,632 input tokens,
with one output truncated at 512 tokens. The MiniLM revision is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. See the raw receipts in
[`examples/synthetic`](../examples/synthetic/README.md) and the [Phase 2 report](experiments-phase2.md).

At the nominal allocation rate of $0.00022572 per second, applying the rate to the sum
of all six full client wall times (2,882.584223 seconds) gives an estimated $0.65066.
The two new comparison calls account for 2,038.083056 client-wall seconds and an
estimated $0.46004 at that same nominal rate. Client wall includes image and startup
time. These calculations are estimates, not billing receipts or actual provider charges,
and the $2 operational target is not a hard spend cap.

The MiniLM receipt records core package versions but not a full environment lock or the
runner Git hash. The V1 Qwen receipt records a broader package manifest, Python and CUDA
versions, and GPU name, but no runner Git hash. The explicit-v2 receipt also records a
runner file SHA-256, `6624bb9281438cdb22173afaea2066667c350cd9fabee98233f1825c5eb3c066`,
which is not a Git commit. Treat the receipts as input/output and runtime evidence, not a
complete immutable environment build record.

## Account and runner controls

Modal SDK 1.6.1 was installed in TraceMem's separate virtual environment. A captured,
read-only `modal token info` call matched the previously used personal workspace.
Credential output was neither printed nor saved. The sanitized local receipt is
`.local/modal-status.json`, mode 0600. Reverify account access before another run.

The one-experiment runner pins the model revision and synthetic input hash, uses one T4
container, min containers 0, max containers 1, no retries, no detached job, no schedule,
and no implicit source mount. It sets startup and runtime limits. Only explicit synthetic
payloads are uploaded. These controls limit execution scope; they are not a provider-
enforced dollar cap. See the [Phase 2 implementation brief](phase2-brief.md) for runner
constraints.

## Completed comparison and validation state

The Qwen and Phi comparison used one frozen explicit-v2 prompt, the same scenario inputs,
and pinned model revisions; it was not an independent held-out study. Both raw receipts
and separate computed reports are linked from the [scenario report](experiments-scenarios.md).
The bounded dataset readiness audit is complete; see the [dataset readiness report](dataset-readiness.json).
The source-free NLI verifier call is also complete, and corrected offline replay
reproduced its computed report; it remains separate from the historical three-call
baseline receipt. The integrated suite passed 242 tests with zero skips after review of the ChatGPT
reader and fresh-snapshot importer. Ruff and strict source/scoped type checks passed,
as did the fresh offline wheel smoke and 34 installed ChatGPT tests. See
[current validation](validation.json) for commands, hashes, and scope.

Token costs for coding agents are separate from Modal GPU charges and were not measured.
The deterministic inventory and offline evaluators need no LLM tokens.
