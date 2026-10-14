# Bounded synthetic QA inference runner

Root next-slice brief, 2026-10-14. Implement a reviewed source-free Modal runner for
the new fictional QA protocol. Existing request preparation is validated; actual
inference has not run. Do not alter the frozen extraction experiments or raw receipts.
Keep private sources, Context Kit and llm-context read-only. No local tokenizer/model
weights, Torch or Transformers imports/execution. No launch during implementation.

Start by reading docs/synthetic-qa.md, docs/validation-synthetic-qa.json,
experiments/prepare_synthetic_qa.py, experiments/modal_scenarios.py and
experiments/scenario_payload.py. Re-derive their failure semantics; do not copy
historical defaults wholesale. Root owns exact payload review and any later launch.
One active worker and one GPU call at a time. Check active-cycle and live jobs first.

Proposed files: experiments/modal_synthetic_qa.py, tests/test_modal_synthetic_qa.py,
docs/synthetic-qa-modal.md, ignored .local/synthetic-qa-modal-worker-* logs.
Target <=350 production LOC; propose a file-level partition before exceeding500.
Use genuine RED/GREEN per behavior, save raw test output as it occurs. Do not stage,
commit, push, run full suite, create Modal apps or call model/tokenizer services.

Build a deterministic payload only by regenerating prepare_synthetic_qa() from the
fixed reviewed fixture. Do not accept arbitrary request files or references. Pin
fixture SHA256 a4d94c8e727c71e8c8b723e97c9674b1d0008ac9f35ba675485a7ea796349d78,
preparer SHA256 161a45011807f08c98dd56b588a82907d6caa1519fa0691c8f71f7cbc8c4209b,
and canonical request SHA256
85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084.
Remote payload contains only exact reviewed request IDs and two role/content messages,
source/prompt/payload hashes and pinned generation config; no answer references,
labels, local paths, split or answerability flags. Keep dev/test mapping locally.

Use only Qwen/Qwen3-4B at revision1cfa9a7208912126459214e8b04321603b3df60c,
float16, trust_remote_code=False, safetensors, eager attention, seed0, greedy decoding,
num_beams1, enable_thinking=False, max_new_tokens128, max_prompt_tokens2048.
Pin remote torch2.7.1/transformers4.51.3/accelerate1.6.0 as in reviewed historical
runner. No retries/fallback models/truncation or automatic recovery. Apply the same
system/user chat template to every request. Tokenize all72 prompts remotely before
generation and refuse any over limit. Record exact input tokens; byte ceilings are
not tokenizer-matched budgets and must never be reported as such.

Modal options: one T4, bounded2CPU and16GiB RAM, min0/max1/buffer0, retries0,
startup_timeout300, runtime timeout600, scaledown_window2, single_use_containers,
serialized=True, include_source=False, no source mount, volumes, secrets, schedule
or detached jobs. Body must be source-free, all imports local to remote body.
Validate remote payload shape/hash/exact allowlist before loading model. Reuse SDK
serialization-only test pattern to prove there are no module-global dependencies.
Remote result records every request ID, raw output, input/generated token counts,
eOS versus length stop/truncation flags, model/config/template hash, runtime/package/
GPU identities, elapsed/load/GPU time, memory and Modal call ID. No answer scoring.

Default CLI is a local dry run. --show-payload prints only the fixed fictional
payload; --launch requires exact --approve-payload-sha256 and runner-file hash,
plus fresh personal workspace verification using existing helper semantics. Root
must verify account/workspace, no active GPU call, exact source/payload hashes and
save launch intent before launching once. Persist raw receipt to a NEW0600 local
file and client elapsed time; never overwrite historical receipts. Failures are
sanitized and recorded, not retried. No provider-enforced dollar cap claim: limits
bound resource/time exposure, nominal rate estimates are not billing evidence.

Tests use real payload preparation and narrow mocked launch/SDK seams: default no
network, wrong/missing approval refused before Modal build, payload contains only
allowlisted reviewed messages, references cannot enter serialization, source changes
refused, bounded resource options, source-free SDK serialization, malformed remote
payload refused before importing heavy inference libraries (arrange validation first),
new-output safety and no accidental second call. Focused unittest/Ruff/strict mypy
only. Release exact files/hashes/raw RED/GREEN log; root independently reviews,
validates and decides whether to make one already-authorized bounded synthetic call.
