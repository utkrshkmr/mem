# Assumptions to verify on the actual machine

| Item | Status | Required evidence |
|---|---|---|
| H100 memory variant and interconnect | measured 2026-09-23 | Four H100 80GB HBM3, NV6 links, driver 570.211.01. reports/hardware.json |
| BF16 matmul and two-rank all-reduce | measured | hardware.json bf16_matmul_finite true; ddp sum0 is 3.0 |
| BF16 7B LoRA memory fit at exact 8k budgets | unverified | One short 7B LoRA step fit on one GPU. Steady-state 8k update is not measured |
| Compatible CUDA/torch/vLLM/PEFT/tokenizer stack | split stacks | Python 3.13 torch 2.11.0+cu128 peft 0.21.0 runs HF. Its vLLM 0.25.1 engine fails. Env memrl has torch 2.8.0+cu128 and vLLM 0.10.2, which matched one greedy completion |
| Base-model revision in the local cache | observed, not frozen as the study lock | Qwen/Qwen2.5-7B-Instruct ref a09a35458c702b33eeacc393d103063234e8bc28 |
| Frozen reader can solve tasks when feasible evidence is present | unverified | Gold-support development calibration |
| C1024 creates a useful scarce-memory regime | unverified | Strong representation occupancy and capacity curve |
| Need for shared format SFT | unverified | Raw-base tool error pilot; same decision for all learned arms |
| 200 iterations and 3 seeds fit resources and power needs | planning only | Throughput and variance pilot |
| F03 improves beyond flat continuation augmentation | unverified hypothesis | Matched primary comparison |
| Independent natural chronological source available | unverified | Source license, split plan, labels and reviewer receipts |

Resolve practical API choices autonomously, record evidence, and retain uncertainty for research questions.
