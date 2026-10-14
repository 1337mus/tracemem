# Continuous integration

The private repository workflow runs on pushes, pull requests, and manual dispatches. It uses one Ubuntu job with Python 3.12 and read-only repository access. The official checkout and setup-python actions are pinned to full commit SHAs. The job uses no repository secrets, caches, or artifact uploads.

The job installs both locked requirement files into `.venv` and runs `make check`: the unit suite, Ruff checks and formatting, and mypy. Context Kit is intentionally absent from the locked environment, so its one optional integration test skips. The Modal SDK serialization test runs locally with the installed SDK. It does not call Modal, use credentials, or require a GPU.

The job builds a wheel with the installed setuptools backend, installs it without dependencies into a fresh virtual environment, and runs both demo entry points from outside the checkout. It also runs `experiments/smoke_installed.py` with an absolute source root and a fresh work directory under the system temporary directory. The ChatGPT adapter, ChatGPT snapshot, LongMemEval adapter, turn-retrieval diagnostic, and source-retention impact test modules are copied outside the checkout and run against the installed wheel. The diagnostic experiment and authored LongMemEval format fixture are copied with their relative directory structure; no benchmark conversations are downloaded. Tests run from that copied root with `PYTHONPATH` and `PYTHONHOME` unset, so package imports resolve from the installed wheel while the experiment runs from its isolated copy.

The installed smoke also runs the fixed context-budget diagnostic and offline review
renderer through the installed package. It binds the recorded budget report and 84-task
blind bundle by their raw SHA-256 hashes, compares all budget results and configuration,
and checks the HTML against the reviewed final hash. Python and SQLite version values
may differ; their field names and types must match and both environments are recorded.
Metrics, citations, retrieval settings, and all other fields must match. The module-path
inventory is checked after these calls, so the new APIs must resolve outside the checkout
under installed `site-packages` too. All existing ledger, citation and NLI checks remain.

The frozen hashes identify reviewed fictional inputs and rendered bytes. An intentional
fixture or renderer change requires a reviewed new reference and matching validation;
a changed hash is a failure to investigate. Browser rendering remains unverified in
this environment; the Node VM tests and installed HTML hash check cover different layers.

The workflow makes no cloud inference or training calls. Test, build, and smoke failures fail the job; it has no allowed-to-fail step. A hosted CI pass cannot be claimed until the workflow has run after a push.

The initial hosted run [38617328181](https://github.com/1337mus/tracemem/actions/runs/38617328181) passed on commit `309f41b1810981056b30f811383634d2953165f3`: 242 tests with one expected Context Kit skip, all lint/type checks, and 34 installed-wheel ChatGPT tests. See [delivery evidence](github-validation.json).
