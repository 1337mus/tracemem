# Fresh scenarios and annotation workflow

2026-10-14. Continue autonomously after the 136-test checkpoint. Preserve the
old fixtures, prompts, runners, reports and validation receipt. Do not train,
upload private data, run local model inference or publish.

Build 48 newly authored fictional extraction cases, each with 3-5 messages across
at least two sessions. Use varied domains and natural short exchanges, not a
repeated 'fictional record says' wrapper. Cover attribution, tentative language,
negation, temporal changes, corrections, cause direction, ownership/location,
coreference, quoted instructions and ambiguity. Include 8 no-memory cases. Gold
claims should be conservative and atomic; all offsets must resolve exactly.
Agent-authored labels remain provisional and are never independent human labels.

Artifact src/tracemem/data/scenarios_v1.json has exact root keys:
version=1, license='MIT', annotation_status='agent_authored', cases, case_metadata.
Each case uses the existing challenge case shape: case_id, scenario_id, split,
tags, messages, expected. Assign 24 dev and 24 test cases before inference;
no training cases. Cases are distinct whole scenarios, not name substitutions.
Each case_metadata entry (keyed by case_id) has exact fields:
scenario_group, template_group, entity_ids (nonempty list of declared fictional
entity identifiers). No scenario/template/entity group crosses dev/test.
These declared groups and lexical audit do not prove natural-language independence.
The test partition is reserved for this new comparison, with provisional authored
labels; it is not independently human validated or population-representative.

src/tracemem/scenarios.py exposes:
load_scenarios() -> (cases, manifest, raw_sha256)
validate_scenarios(payload) -> (cases, manifest)
Manifest contains version, license, annotation_status, case_metadata only.
Validate strict root/metadata keys, duplicate keys/IDs, group separation, split
counts, two sessions/3-5 messages, timestamp format/order, expected spans/schema
using the existing challenge validator. Do not change old fixture loaders. Reject
invalid content, do not silently repair it. Production module roughly250lines,
tests meaningful and focused. Dataset-only raw literals excluded from line budget.
Root audits every scenario and all references before remote payload approval.

Annotation workflow is a separate module src/tracemem/annotations.py, tests and
experiments/review_annotations.py. Export shuffled blind extraction review tasks
(messages + candidate records) with opaque task IDs and no gold labels, split,
case IDs or group names. Candidate set is the gold snapshot plus caller-supplied
foil records; every candidate must have valid schema, but semantic support and
span integrity are annotation questions. Track the exact dataset SHA and task
bundle SHA. Gold or foil membership stays in a separate key, never the blind file.
No automatic generated human attestations. Import reviewer responses with task ID,
label (supported/unsupported/uncertain), rationale, reviewer ID, reviewer_kind
(human/agent), and timestamp. Require exact one-to-one task coverage; reject
unknown/duplicate/missing tasks, invalid hashes and role metadata. Human is a
self-attested input, not a verified identity claim. Agreement/coverage reports must
separate human and agent responses, never promote authored labels automatically.
CLI exports and reports locally, refuses any existing output and input/output
aliases; use exclusive creation. No remote calls, private-source discovery or
mutation. Production roughly300lines plus a small CLI; propose partition if larger.
See [annotation-workflow.md](annotation-workflow.md) for runnable commands, the
response schema, and the support-versus-integrity review policy.

Compare Qwen3-4B and Phi-3.5-mini using the frozen explicit-v2 prompt. New runner
experiments/modal_scenarios.py and pure evaluator experiments/evaluate_scenarios.py;
old runners remain unchanged. Read only the packaged new fixture, no arbitrary
input path for remote text. New payload contains schema version, fixture SHA,
model settings and only case_id/prompt items. No expected claims/metadata. Root
reviews the actual full payload before launch. Evaluate dev and test separately,
raw and unique-quote anchoring, retaining all raw outputs and failures.

Use pinned model revisions in configs/model-candidates.json and fresh public
metadata/license checks before launch. One model per call on T4 FP16, greedy,
max1024generatedtokens, max4096inputtokens, thinking disabled for Qwen. Reject
oversized prompts, never silently truncate. Max48cases, max1container, min/buffer0,
no retries/detach/source mounts, serialized __main__ remote function, single-use,
startup300s/runtime1200s, scaledown2s, CPU2/RAM16GiB. Two planned synthetic calls,
additional target<=US$2 (not enforced billing cap); no training. Record runner,
payload, prompt hashes, full versions, GPU/call/app IDs, timing, memory and output
length/finish reasons. Explicit hash approval and fresh personal-workspace check.
No remote calls by workers. Raw outputs immutable; local evaluator writes a
separate exclusively-created report. Root owns launch, receipt acceptance and
final adversarial review. Separate runner/evaluator/helpers to avoid >400lines
per file; source-free remote function must remain __main__ by value.
The by-value serialization proof is an optional Modal SDK integration check:
the focused test suite skips only that proof when importing the pinned SDK raises
ModuleNotFoundError. The standard-library test suite otherwise remains runnable
without the Modal extra.
