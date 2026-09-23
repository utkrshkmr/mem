# Validation performed for this handoff

Date: 2026-09-23. Scope: local CPU reference, planning helpers and document/config consistency. No models were downloaded and no H100 training, inference, benchmark evaluation or throughput measurement was performed.

| Check | Observed result |
|---|---|
| `PYTHONPATH=src python3 -m unittest discover -s tests -v` | 40 tests passed |
| `python3 -m unittest discover -s checks -v` | 11 tests passed |
| `python3 scripts/run_cpu_demo.py` | Equal current totals 10/10; the same cancellation produces 8/3; explicitly a CPU fixture |
| Draft resolver, F03 profile | Valid draft emitted; production_ready=false and unresolved launch blockers listed |
| Planner, pilot suite | Four planned jobs; unknown costs remain unknown; no jobs launched |
| Planner, core suite | Nine planned jobs; three arms by three seeds; no jobs launched |
| Production-mode negative check | Rejected without writing a success artifact |
| Handoff validator | Required files, profiles, links, Markdown fences and Python source syntax pass |
| All document code fences | 33 Python snippets parsed; 7 JSON snippets parsed; 4 YAML snippets parsed; 13 shell snippets pass bash syntax check |
| Main spec consistency | Sections 0–35 sequential; navigation anchors resolve; embedded YAML equals configs/base.json; end marker occurs after revision3 |
| Schema files | JSON syntax checked; full JSON Schema validation was not run because jsonschema is not installed in this validation environment |

The test suite checks immutable public/private projections, strict revision identities, canonical whole-store charge, rejection atomicity, non-additive token-counter cases, deletion increasing token charge, clone isolation, ledger event semantics, branch advantage weighting, prefix counted once, branch permutation, K=1 reduction and uniform call-sampling expectation. A tiny finite-policy enumeration checks the expected gradient against the derivative of its exact expected objective. Planning tests cover bad types/keys, information leaks, wrong objectives, device overlap, draft production rejection, missing costs, seed multiplicity and budget overrun.

These tests establish only the exercised reference behavior. They do not establish production tokenizer accounting, real-model competence, vLLM/HF numerical parity, LoRA synchronization, GPU memory fit, DDP gradients, checkpoint recovery, absence of leakage in a future implementation, statistical power or scientific benefit.

The final archive includes a SHA256SUMS manifest for file integrity. This is not a cryptographic attestation of research results. Production gates G01–G11 remain uncompleted.
