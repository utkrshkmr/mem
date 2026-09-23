# Implementation status

Production code lives in `src/memrl`. The CPU reference in `src/memrl_contract` is unchanged. No scientific claim is established.

| Component | implemented | CPU_validated | GPU_validated | scientifically_validated |
|---|---|---|---|---|
| Contracts, config launch gate, CLI refusals | yes | yes | n/a | no |
| Synthetic ledger data, two interpreters, public/private split | yes | yes | n/a | no |
| Bounded store, tool parser, writer environment | yes | yes | n/a | no |
| Fork/flat loss terms and checkpoint selector guard | yes | yes | n/a | no |
| Scripted ledger baseline and scripted fork fixture | yes | yes | n/a | no |
| Leases, atomic jobs/adapters, resume index | yes | yes | n/a | no |
| HF greedy logprob self-check and adapter on/off, Qwen2.5-7B | yes | n/a | partial | no |
| vLLM vs HF greedy token IDs, one short prompt | yes | n/a | partial, separate environment | no |
| One 7B LoRA Adam step | yes | n/a | partial | no |
| Two-rank gradient match | yes | n/a | 0.5B only | no |
| Four-GPU pilot, frozen reader benchmark, protocol freeze | no | no | no | no |

CPU commands that passed on 2026-09-23: `PYTHONPATH=src python3 -m unittest discover -s tests -q` (70 tests), `PYTHONPATH=src python3 -m unittest discover -s checks -q` (11 tests), `PYTHONPATH=scripts python3 scripts/verify_handoff.py`.

GPU evidence is under `reports/hardware.json` and `reports/api_probe/`. The default Python 3.13 stack's vLLM 0.25.1 engine did not start. The token-id comparison used conda env `memrl` (Python 3.12, torch 2.8.0+cu128, vLLM 0.10.2).
