# Local private development review

The [first private model pilot](private-pilot-results.md) is complete: seven real
user messages, 14 reviewed claims and one inference-only comparison on personal
Modal. It found no useful-memory acceptance gain over the base model. No private
training has run. All labels remain provisional agent judgments.

The local preparation API below creates review snapshots; it does not itself
run a model. Its first selection failed to provide suitable human dialogue.
Keep those historical artifacts separate from the later primary-session pilot.

To prepare a snapshot, create an explicit manifest that names the sources you
choose and call the API with a new output directory under `.local`:

Keep the manifest in `.local` too. Its format is:

```json
{
  "version": 1,
  "provenance": "private_source",
  "sources": [
    {"provider": "codex", "path": "/path/to/session.jsonl", "project_group": "personal-project"}
  ]
}
```

```python
from pathlib import Path
from experiments.private_eval import prepare_private_eval

result = prepare_private_eval(
    manifest_path=Path(".local/private-source-manifest.json"),
    output_dir=Path(".local/private-eval-review"),
)
```

Preparation accepts at most 24 source files, each no larger than 8 MiB. It
selects up to two six-record windows per source, with an 8,000-character limit
per window. Oversized windows are skipped. It checks exact whole-window
duplicates and normalized whole-message overlap of at least 160 characters.

In the initial preparation, a selection contained 24 closed Codex
sources from one verified personal project. The first preparation failed the
duplicate-window check. A derived manifest retained 22 sources and produced 36
windows with 183 record appearances, which may overlap. All 36 are dev; test is
empty, so held-out review is unavailable. Nineteen oversized window candidates
were skipped. Source hashes and integrity were rechecked; output directories
use mode 0700 and files mode 0600.

The local artifacts are in
`.local/private-eval-20261016/prepared-v1/`. `review.html` and
`annotation-template.json` are pending review, with zero human reviewers. Two
email indicators were detected; these are heuristic markers, not privacy
clearance. At that preparation checkpoint, no source text went to a hosted model,
and no private inference, training, or upload had occurred. Review embedded quotes, instructions, and
authorship locally. To annotate, edit a copy of the template and keep its
original frozen. Leave claim and QA labels pending until a reviewer checks the
evidence; the current artifacts do not provide a completed annotation interface
or completed review.

For a smaller first pass, a separate local shortlist contains one window from
each of four sessions, chosen in frozen snapshot order without reading their
meaning. It lives in `.local/private-eval-20261016/review-shortlist-v1/` and
has its own review page and pending annotation template. The full snapshot is
unchanged. This shortlist is development material, not an independent test.

## Selection review and remote authorization update

The owner now permits selected private conversations for training and evaluation
on personal Modal, while forbidding their upload to GitHub. The later primary
pilot used one reviewed private remote inference call. Private data and any adapter trained on it
must remain outside release artifacts.

Root and an annotation agent inspected the four-window shortlist. It contains
machine environment context, assistant plans, and a tool-transcript delta;
none provides an unambiguous direct human-authored positive memory claim.
A metadata audit found all 22 selected sources are subagent sessions. This is
a source-selection limitation, not a completed labeled dataset. The originals,
prepared snapshot and pending template remain intact. The next selection must
use substantive user dialogue from primary personal-project conversations.
See [getting realistic data](realistic-data.md) for the execution workflow.
