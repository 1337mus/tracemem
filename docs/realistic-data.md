# Getting better training data

We now have 25 annotated conversation windows from five real sessions: 50
candidate judgments. One additional session has a source window reserved for
later testing; it has no labels and is excluded from training and development.
All annotations are provisional agent judgments. No human has reviewed them.

For the completed private SFT run, we used:

| Use | Sessions | Windows | Candidate judgments |
| --- | ---: | ---: | ---: |
| Training | 2 | 15 | 30 |
| Development | 2 | 8 | 16 |
| Excluded older writing examples | 1 | 2 | 4 |

Training also includes 48 fictional training examples through the existing
synthetic exporter. Synthetic development and test examples are excluded. The
one-epoch private continuation completed; its aggregate run and development
results are in the [private SFT report](private-sft-results.md).

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

For the newest 20 candidates, an independent pass agreed with initial support
labels on 20/20 and eligibility on 16/20. Root review nevertheless caught a
segment-versus-duration inference and qualifiers absent from standalone source
windows. Five claims were corrected and independently checked again. Agreement
between agents is useful evidence, but it did not catch every error.

Seven initial eligibility labels changed during adjudication. Three final scope
labels are root-only decisions that disagree with the blind pass; the originals
and disagreements remain in separate local receipts. None is human gold.

A later blind review and root adjudication revisited four older writing-task
examples for a separate candidate. Their support labels stayed at two supported
and two unsupported; eligibility is uncertain for all four under the existing
prompt. These examples were not in the completed run, and its records remain
unchanged. The separate unrun candidate has 34 private rows from three sessions
plus the same 48 fictional replay rows, 82 total. Its development cases, labels,
and prompts are byte-identical to those in the completed comparison. It has not
been trained, and no additional GPU call was made.

## What the experiment can establish

The first private inference pilot accepted 4/7 useful supported memories with
the continued adapter, matching the base model. It established no overall gain.
See the [private pilot results](private-pilot-results.md).

The completed run compared the same pinned base model, the existing synthetic
adapter and one private continuation on identical development prompts. Its
frozen criterion requires at least six of seven useful memories accepted, at
least two more than the same-call prior adapter, all eight unsupported claims
rejected, and no loss in supported retention. The trained arm reached 5/7 useful
acceptances (one more than prior), 7/8 unsupported rejections (one fewer than
prior), and 8/8 supported retention. It failed the criterion. The strict-literal
report counted 0/16 valid outputs per arm because decoded text included terminal
EOS. The offline diagnostic removed exactly one verified trailing EOS token from
a copy before content validation; the original receipt and strict report remain
unchanged. See the results report for the full table and limits.

This is a small development experiment on supplied candidates, with previously
inspected examples and correlated conversation windows. It cannot establish
extraction recall, natural error prevalence, durable memory quality or better
answers. Development remains development. The reserved test source and sealed
synthetic candidate stay untouched.

The next data target is roughly 100 useful windows from distinct sessions,
including no-memory examples and naturally occurring model mistakes. Review
policy disagreements and a random sample with the owner, then compare increasing
training subsets on fixed development data. A final candidate needs a separately
labeled test cohort and a downstream QA comparison against transcript retrieval.

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
