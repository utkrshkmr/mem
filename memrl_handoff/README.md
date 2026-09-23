# MEMRL research rebuild kit

This is a detailed implementation handoff and a small, tested CPU reference core. It is **not a completed GPU training system or evidence of publication readiness**. No GPU experiments, accuracy improvements, throughput measurements, or acceptance guarantees are supplied.

The research candidate is **learning bounded memories that remain useful after unknown future updates**. Closely related work already establishes the failure mode; the proposed continuation-training method must earn its contribution against strong matched controls. Keep `memrl` as the internal package name; choose a publication name only after a name and literature check.

## Start here

1. Preserve this entire directory outside any implementation directory you intend to delete. It contains the specification, contracts, experiments, and reference code needed for a fresh build.
2. Read [MEMRL_SPEC.md](MEMRL_SPEC.md), especially revision-3 Sections 30 onward, then [NOVELTY_AND_VENUE.md](docs/NOVELTY_AND_VENUE.md).
3. Run the CPU checks below. Read [CONTINUATION_METHOD.md](docs/CONTINUATION_METHOD.md), [INTERFACES.md](docs/INTERFACES.md), and [PARALLEL_BUILD.md](docs/PARALLEL_BUILD.md).
4. Open this directory as the new project in Cursor. Give the coordinating agent [COORDINATOR.md](prompts/COORDINATOR.md). Start the separate agent task packets only in the permitted dependency waves.
5. Implement the production package under `src/memrl/`. The supplied `src/memrl_contract/` remains the small reference implementation against which production code is checked. The two names intentionally distinguish the deliverables.
6. Run the hardware and API probes on the actual four-H100 machine; replace template locks with measured, verified values. Then complete the gates in [ACCEPTANCE_GATES.md](docs/ACCEPTANCE_GATES.md).

## Runnable now, without models or GPUs

Use Python 3.11 or newer from the project root. These commands do not download models, access test benchmarks, start GPU jobs, or invent research results.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m unittest discover -s checks -v
python3 scripts/resolve_config.py --profile configs/profiles/F03_fork.json --output /tmp/memrl-fork-draft.json
python3 scripts/plan_experiments.py --suite pilot --output /tmp/memrl-pilot-plan.json
python3 scripts/verify_handoff.py
```

The resolver emits a **draft** configuration with `production_ready=false` because it is a planning tool. `--production` deliberately rejects launch certification; the future production runtime must verify resolved model revisions, tokenizer identities, dataset manifests, runtime locks and gate evidence. The planner emits jobs and dependency gates; it never launches them. References to `python -m memrl...` elsewhere describe production entry points the coding agents must implement, not commands that currently work.

## What is provided

| Component | Purpose |
|---|---|
| `MEMRL_SPEC.md` | Full revised research and engineering specification |
| `docs/CONTINUATION_METHOD.md` | Objective, correct shared-prefix gradient, data construction and falsifiers |
| `docs/EXPERIMENTS.md` | Core experiments, comparison rules, statistics and stop criteria |
| `docs/FOUR_H100_RUNBOOK.md` | Profiling, GPU topology, scheduling, launch and recovery contracts |
| `docs/RUNTIME_WIRING.md` | Production services, ownership, messages and iteration lifecycle |
| `docs/INTERFACES.md` | Cross-module types, invariants and evidence artifacts |
| `docs/PARALLEL_BUILD.md` | Agent ownership, dependency waves and integration process |
| `prompts/agents/` | Complete task packets for independent coding sessions |
| `.cursor/rules/`, `AGENTS.md` | Persistent coding instructions and scientific constraints |
| `configs/` | Machine-readable base, arm overlays, experiment registry and gates |
| `src/memrl_contract/`, `tests/` | CPU reference semantics and mathematical checks |
| `scripts/`, `checks/` | Draft configuration resolution, experiment planning and handoff validation |
| `locks/`, `records/` | Honest empty templates for runtime locks and research evidence |
| `paper/` | Claim-to-evidence requirements and a result table template |

## Completion boundary

A passing CPU test suite establishes only the properties that it tests. The production trainer, benchmark adapters, complete synthetic generator, model inference, distributed parity, and all research experiments remain implementation work. Do not rename these missing components to “done” after creating stubs. Every coding task has a separately reviewable acceptance gate.

Four H100s are a resource constraint, not an experimental result. Aim for useful completed work per GPU-hour, with measured scheduling decisions. The initial synchronous 2-training/2-rollout design necessarily leaves some devices idle during phase barriers. Never introduce uncorrected stale-policy rollouts merely to show 100% utilization.
