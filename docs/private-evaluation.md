# Local private development review

Private evaluation is a bounded local review of real conversations. It does not
run a model, measure model quality, or train one. The first benefit is realism:
reviewers can see whether the project’s guidance handles ordinary conversation,
quoted text, and instructions in context. Durable learning should improve
interpretation skills and procedures, rather than store private facts in a model.
This preparation makes no claim of a quality gain.

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

In the current preparation, an initial selection contained 24 closed Codex
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
clearance. No source text went to a hosted model, and no private inference,
training, or upload occurred. Review embedded quotes, instructions, and
authorship locally. To annotate, edit a copy of the template and keep its
original frozen. Leave claim and QA labels pending until a reviewer checks the
evidence; the current artifacts do not provide a completed annotation interface
or completed review.

For a smaller first pass, a separate local shortlist contains one window from
each of four sessions, chosen in frozen snapshot order without reading their
meaning. It lives in `.local/private-eval-20261016/review-shortlist-v1/` and
has its own review page and pending annotation template. The full snapshot is
unchanged. This shortlist is development material, not an independent test.
