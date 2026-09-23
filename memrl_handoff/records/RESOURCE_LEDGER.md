# Resource ledger

Initial project envelope: 1,500 allocated GPU-hours. No training or evaluation study has been charged against it.

Engineering probes on 2026-09-23, wall clock from the command transcripts, not a leased study:

| Probe | Devices | Wall clock | Result |
|---|---|---|---|
| hardware inventory, bf16 matmul, NCCL all-reduce | GPU 0 and GPU 1 for the all-reduce | about 12 s | sum of the two rank vectors is 3 |
| HF logprob and LoRA on/off | GPU 0 | about 12 s | mae about 6e-6 nats |
| vLLM 0.25.1 engine start | GPU 0 | about 55 s | failed, driver/runtime mismatch |
| vLLM 0.10.2 greedy tokens vs HF | GPU 0, conda env memrl | about 43 s | token ids `[562, 151645]` matched |
| one 7B LoRA Adam step | GPU 1 | about 25 s | finite loss and gradient |
| 0.5B two-rank vs one-rank gradient | GPU 2 and GPU 3 | about 35 s | max absolute sample difference 0 |

Peak study-budget HBM, reader tokens, and retry costs are unknown. Remaining envelope is still 1,500 hours because these probes were inventory, not allocated study jobs.

