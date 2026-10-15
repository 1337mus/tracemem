# SFT pilot data card

This folder contains a fiction-only, agent-authored dataset for an initial technical SFT pilot. The training slice has 32 authored scenario families expanded into 16 variants each (512 rows total), including 128 explicit abstentions across 8 families.

The variants share scenario templates and labels, so they are correlated examples rather than 512 independent observations. Labels are provisional and have not been validated as human truth; neither an asserted claim nor an empty target establishes ground truth or evidence sufficiency.

This is an initial technical pilot only. A 128-row sample has been used in a bounded SFT run, but the pilot has not passed Gate A and does not establish model quality, annotation validity, or readiness for production use. See the [results report](../../docs/sft-pilot-results.md).

Offline preparation prints only source/module hashes and row counts. The completed run used a pinned job hash and unique run ID. Any future remote execution would require separate authorization and a new local receipt path.

Pilot status: the bounded remote training run completed; see the [pilot scope and execution record](../../docs/sft-pilot.md) and [results report](../../docs/sft-pilot-results.md).
