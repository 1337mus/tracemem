# First comparison on private conversation evidence

The first private pilot uses seven complete user messages from one personal
project conversation. Each message has a faithful candidate claim and an
authored contrast claim. This gives 14 verifier prompts, not 14 naturally
extracted memories or independent conversations.

Two agent annotation passes agreed on all 14 support labels and all 14
eligibility labels before model inference. Labels remain provisional and have
zero human reviewers. All candidates are eligible under the research policy;
this slice cannot measure eligibility discrimination. The original transcript
anchors, text hashes and source revisions were checked against the source file,
which remained unchanged. Secrets, employer-confidential material and identifying
information about other people are excluded from this batch.

The comparison sends identical prompts to the pinned Qwen3-4B base model, the
first verifier adapter and the continued verifier adapter. One personal Modal
A10G call is allowed, with a 900-second execution limit and zero retries.
There is no training in this call. Prompts omit annotation labels and local
source paths. Private prompts, labels, token IDs and raw receipts stay outside
GitHub, under restricted ignored directories.

The reviewed payload is 28,557 canonical JSON bytes. Its SHA-256 is
`bb9124b2690d81f60abad9353898ad78a06bb6825e2346e62c96932f41c38ef1`.
The inference runner SHA-256 is
`daba95fbbf27ae4c48066f3d29a6c0948205698db4908c9df66a0be36673e9ea`.
Local launch approval binds the exact labels and source records before
inference. Code and aggregate results can be shared; private inputs and
outputs cannot be reconstructed from this repository alone.

## Results

| Measure | Base | Prior adapter | Continued adapter |
| --- | --- | --- | --- |
| Supported claims retained | 6/7 | 7/7 | 6/7 |
| Unsupported claims rejected | 6/7 | 7/7 | 7/7 |
| Eligible labels correct | 4/14 | 0/14 | 7/14 |
| Supported, eligible memories accepted | 4/7 | 0/7 | 4/7 |
| Unsupported claims accepted as eligible | 0/7 | 0/7 | 0/7 |
| Valid, complete outputs | 14/14 | 14/14 | 14/14 |

All three arms received identical input token IDs. The base model abstained on
one supported and one unsupported candidate; those are not counted as correct
support decisions. The prior adapter returned uncertain eligibility for every
candidate. The continued adapter improved that axis, but rejected one faithful
project question and left eligibility uncertain for several supported facts.
Root checked every disagreement against the frozen labels. No labels changed
after inference, and no model was selected.

**The continued adapter did not improve useful-memory acceptance over the
base model on this pilot.** Zero false accepts among seven contrast claims is
too little evidence for a reliability claim. These are supplied candidates;
we did not measure extraction recall or downstream answer quality.

The call used one NVIDIA A10G. Remote time was 240.87 seconds, including
181.75 seconds loading the model; the three inference arms took 58.86 seconds
combined. Client time was 263.98 seconds. Billing was not measured. No private
training or private checkpoint creation occurred. See [aggregate metrics](private-pilot-results.json).

## What this changes about data collection

Collect more verified human-authored project dialogue, preserving the adapters'
actual role and content-kind metadata. The current user records carry
`mixed_untrusted`: this conservative source classification also occurs for
machine-injected text. It does not mean the message is fictional or ineligible.
A metadata shortcut is a hypothesis to test, not an established cause of the
scope errors.

Include questions without decisions, completed setup facts, reusable
preferences, task-local instructions and genuine no-memory windows. Annotate
scope and retention duration explicitly: a temporary instruction should not
silently become a permanent preference. Extend to distinct primary sessions
before creating a private training set. Keep this pilot in development.

## Interpretation

This is a selected, balanced development challenge from one session. It is
not a held-out benchmark, a population error estimate, an extraction-recall
measurement, or a downstream memory-QA result. Its purpose is to find specific
transfer errors before collecting a separate private training set. With one
conversation cluster, a conversation-level confidence interval would provide
no useful evidence of generalization.

See [the data collection plan](realistic-data.md), [private annotation
contract](private-verifier-contract.md) and [code validation](validation-private-verifier.json).
