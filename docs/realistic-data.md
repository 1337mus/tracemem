# Getting better training data

The current collection has 33 annotated conversation windows and 66 candidate
judgments across six real sessions. Training uses 25 windows and 50 rows from
four sessions; development uses eight windows and 16 rows from two sessions.
The separately reserved source remains unlabeled and unused. All annotations
are provisional agent judgments. No human has reviewed them.

For the completed private SFT run, we used:

| Use | Sessions | Windows | Candidate judgments |
| --- | ---: | ---: | ---: |
| Training | 4 | 25 | 50 |
| Development | 2 | 8 | 16 |

The 25 training windows include 15 earlier windows from two sessions, two
older writing-task windows, and eight newly labeled windows from one session
in the same editing-task family. All 16 eligibility labels on the new windows
are eligible.
The older four writing-task judgments retain their adjudicated labels: two
supported, two unsupported, with eligibility uncertain on all four.

Training also includes 192 fictional replay rows. Synthetic development and
test examples are excluded. The completed 242-row private continuation and its
development results are in the [latest scope report](private-sft-scope-results.md).

## What the real examples add

People give permission without saying an action happened, correct earlier
requests, relay someone else's feedback, and describe preferences for one
project rather than for their whole life. The new training examples cover
those distinctions using complete original source blocks and exact citations.
They are a balanced challenge of source-derived claims and deliberately wrong
variants. They do not estimate how often mistakes occur in ordinary use.

Automatic browser context and file wrappers are preserved for source integrity
but distinguished from human requests. Citations select the explicit human
span. Source IDs, revisions, quotes, offsets and text hashes are checked against
the original read-only adapter records. A saved summary can help find a source;
it cannot replace the original words as evidence.

The first discovery attempt selected subagent launch material rather than
human dialogue. Those receipts remain as a failed selection. Later screening
also rejected complete personal messages too broadly because they contained
file context. Individual source and privacy review recovered useful examples.
A provider's `user` role alone does not establish human authorship.

## How labels are checked

Each candidate has two separate labels:

- **Support:** does the source entail the claim with the same actor, reporting
  source, conditions, time and certainty?
- **Eligibility:** assuming it were supported, does the subject belong in the
  stated personal-memory policy?

Both axes allow uncertainty. Temporary project context and permanent retention
are separate decisions. The frozen prompt treats real user-stated or reported
facts as eligible, fictional-character facts as ineligible, and writing or
exercise metadata as uncertain.

For an earlier set of 20 candidates, an independent pass agreed with initial support
labels on 20/20 and eligibility on 16/20. Root review nevertheless caught a
segment-versus-duration inference and qualifiers absent from standalone source
windows. Five claims were corrected and independently checked again. Agreement
between agents is useful evidence, but it did not catch every error.

Seven initial eligibility labels changed during adjudication. Three final scope
labels are root-only decisions that disagree with the blind pass; the originals
and disagreements remain in separate local receipts. None is human gold.

A later blind review and root adjudication revisited four older writing-task
examples. Their support labels stayed at two supported and two unsupported;
eligibility is uncertain for all four under the existing prompt. These labels
remain provisional agent judgments, not human gold. A separate 82-row candidate
was proposed from those reviews but was not run; it was superseded by the
completed 242-row continuation. The completed run used the current 50 private
training rows and 192 fictional replay rows, with the same development prompts
and labels as before.

## What the experiment can establish

The first private inference pilot accepted 4/7 useful supported memories with
the continued adapter, matching the base model. It established no overall gain.
See the [private pilot results](private-pilot-results.md).

The first private run compared the pinned base model, the existing synthetic
adapter and one private continuation on identical development prompts. It
accepted 5/7 useful memories, rejected 7/8 unsupported claims and retained 8/8
supported claims. The later 242-row comparison accepted 6/7 useful memories,
retained 8/8 supported claims, rejected 7/8 unsupported claims and got 0/2
uncertain-scope cases correct. See the [242-row results](private-sft-scope-results.md)
and [first private run report](private-sft-results.md).

The newest comparison used 256 rows: the prior 242-row mix plus 14 new fictional
rows, starting from synthetic-v2. It accepted 7/7 useful candidates and retained
8/8 supported claims, while rejecting 7/8 unsupported claims and getting 0/2
uncertain-scope labels right. The frozen gate failed. Data order and optimizer
steps changed too, so the result does not isolate the 14 new rows. See the
[contrast report](private-sft-contrast-results.md).

A separate inference-only comparison used 16 fictional candidates across eight
scripted scenarios. Its narrow dev gate passed, but it does not change the
failed private-training gate or establish product quality. See the
[naturalistic dev results](naturalistic-dev-results.md).

This is a small development experiment on supplied candidates, with previously
inspected examples and correlated conversation windows. It cannot establish
extraction recall, natural error prevalence, durable memory quality or better
answers. Development remains development. The reserved test source and sealed
synthetic candidate stay untouched.

Ten bounded source scans yielded no new real-source admissions; this local zero
yield does not establish that a larger archive is unusable. Keep the reserved
test source untouched. Before further training, target source-backed negation
and deferral examples, clarify scope labels with the owner, and plan a broader
separate evaluation with a distinct test cohort and downstream QA comparison
against transcript retrieval. See the [contrast results](private-sft-contrast-results.md)
for the latest development gate and its limits.

## Privacy and release

The owner authorizes private training and evaluation on personal Modal. Original
conversations, Context Kit and the local context archive remain read-only.
Exclude secrets, employer-confidential information and identifying details about
other people from remote inputs. Keep private transcripts, labels, prompts,
receipts, vectors and private-trained adapters out of GitHub and public releases.

Public benchmarks such as LongMemEval and LoCoMo require a separate license and
version review before acquisition. Evaluation conversations must stay out of
training. New shareable fiction should come from general error patterns, not
renamed private conversations.
