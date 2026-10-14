# Source-free Modal tokenizer counting

`experiments/modal_token_count.py` regenerates the reviewed 28-prompt catalog
and can count each complete system/user message with the pinned Qwen tokenizer.
The fixed payload contains authored fictional prompts only. It carries no
answers or private conversation data.

The local runner verifies the catalog SHA-256
`85682edbcdd728103e6540c86106de953f400e217a9341ffbd82420d0429e478` and the
canonical payload SHA-256
`125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88` before any
launch setup. The payload pins `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Transformers 4.51.3, fast
tokenization, `trust_remote_code=False`, the reviewed chat-template hash, and
the chat-template options. The remote function checks the payload hash,
configuration, 28 sorted prompt IDs, and each ID's message hash before it
loads `AutoTokenizer`. It rejects a different chat template and token ID
results that are not lists of nonnegative integers.

The worker runs in Python 3.12 on bounded CPU resources. It uses one container,
no GPU, no mounts, no secrets, no forwarded local environment, no retries, and
no detached run. It is serialized with `include_source=False` and only imports
the pinned tokenizer inside the remote function. No model weights, model
generation, Torch, or inference are used.

## Offline preparation and explicit launch

The default command is offline and prints payload and runner hashes:

```sh
PYTHONPATH=src:. python3 experiments/modal_token_count.py
```

To inspect the exact reviewed payload without launching:

```sh
PYTHONPATH=src:. python3 experiments/modal_token_count.py --show-payload
```

An explicit `--launch` also requires both reviewed SHA-256 approvals and a new
receipt path. Before it creates the Modal app, the runner requires local Python
3.12, regenerates the fixed payload, checks the current runner file hash, and
allocates the receipt with exclusive creation, no-follow, and mode `0600`. It
then checks the personal Modal workspace. A remote error leaves a sanitized
failure receipt and is not retried. This implementation slice and its tests do
not invoke the launch path.

The remote result contains `count_receipt`, which matches the nested receipt
shape accepted by `experiments.token_catalog.pack_from_counts`, plus separate
tokenizer provenance. Provenance records the pinned revision and template
hash, package and Python versions, template options, elapsed time, the backend
tokenizer hash when available, and the payload and runner hashes. A future
replay must independently inspect this envelope and its counts before passing
the nested receipt to the offline packer.

Fake-tokenizer tests exercise validation and receipt handling. Their fixture
integers do not establish real token counts. No tokenizer download or Modal
call was made while implementing this runner. Real counts remain unmeasured
until the separately reviewed bounded launch and receipt replay are completed.
