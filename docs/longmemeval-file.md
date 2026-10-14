# Bounded LongMemEval-format file scan

`tracemem.adapters.longmemeval_file.scan_longmemeval_file` reads a local,
fictional LongMemEval-format JSON array without loading the complete file into
memory. It delegates item JSON, duplicate-key, Unicode, schema, and evidence-ID
validation to the existing bytes adapter. This scanner only frames one object
at a time and retains selected questions after validating the full file.

```python
from tracemem.adapters.longmemeval_file import scan_longmemeval_file

scan = scan_longmemeval_file(
    "fictional-longmemeval.json",
    namespace="local-study",
    variant="s",
    question_ids=("fictional-question-1",),
)
```

The result is a frozen, slotted `LongMemEvalFileScan` containing the exact raw
file SHA-256 and byte count, total question count, and requested questions in
source order. An empty `question_ids` tuple validates the whole file and returns
no question conversations. Every requested ID must occur in the input. Selectors
must be distinct, nonblank scalar-Unicode strings; at most five are accepted.

Reads use 64 KiB chunks. The input is limited to 4 GiB, each raw question to
64 MiB minus the two bytes used by the adapter's array wrapper, the file to
10,000 questions, and JSON nesting to 128 levels. Each question ID may use at
most 4 KiB of UTF-8, and all distinct IDs together may use at most 1 MiB. These
ID budgets bound the retained cross-file duplicate-check set. Only selected
questions are retained; there is no persistent spool or output file. These
limits bound raw input buffers and selection size, not an exact peak-RAM figure
for JSON decoding or conversation objects.

The path must resolve to a regular, non-symlink file and is opened read-only.
The scanner compares descriptor and resolved-path identity, size, modification
time, and change time before and after reading. Supply a quiescent file: these
checks reject ordinary concurrent changes but do not provide an atomic snapshot
or an adversarial ABA guarantee. Errors use static messages without paths,
input content, or requested IDs.

This is verified only against the repository's fictional-format fixtures. It
does not establish compatibility with an upstream release, data-use rights,
quality against upstream scoring, or completion of a large-dataset evaluation.
No network access, model inference, retrieval policy change, CLI, or GPU work is
part of this module.
