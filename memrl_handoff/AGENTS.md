# Instructions for rebuilding this research project

This file applies to work inside this project. The user has authorized a from-scratch implementation with parallel coding agents. Preserve the specification, fixtures, contracts, and evidence records while replacing application code.

## Read and follow

Read `README.md`, `MEMRL_SPEC.md`, `docs/CONTINUATION_METHOD.md`, `docs/INTERFACES.md`, and your task packet in `prompts/agents/`. Revision-3 explicit overrides in MEMRL_SPEC Sections 30 onward take precedence over its earlier defaults. Executable reference code is a checked example, not permission to change scientific semantics. Resolve discrepancies in `records/DECISIONS.md` and an interface version before proceeding.

Only the coordinator owns shared interfaces, the dependency lock, the experiment registry, and final integration. Work within the paths assigned in `docs/PARALLEL_BUILD.md`. Submit a concrete interface change request when another module needs a change; do not silently fork a schema or duplicate another owner's module.

## Scientific constraints

- Gold labels, hidden future suffixes and questions remain outside writer inputs. An evaluator may supply scalar rewards only after writer completion.
- The primary reader is frozen, deterministic, has no writer adapter, and sees all retained memory. Charge every persistent model-visible channel under the exact pinned reference tokenizer.
- The writer never retrieves discarded raw history. A branching simulator may clone audit state for gold computation but must not expose it to the writer.
- Train log probabilities on the exact per-call token IDs and exact compressed context used during collection. Never reconstruct a full-history prompt at training time.
- One adapter version per complete RL iteration. Prefix calls appear once with mean branch advantage; each suffix has its own advantage divided by K. Maintain the fixed H*G*Z denominator.
- Fail closed on unknown configuration keys, missing revisions, incompatible cache keys, and missing production modules. Do not return mock success, synthetic benchmark scores, or fallback random predictions.
- Separate smoke data, development data, and untouched final tests. Never select checkpoints, prompts, budgets, or hyperparameters using final benchmark results.
- Do not change a protocol, denominator, inference budget, or timeout failure rule to improve a result. Record deviations and retain failed runs.
- No acceptance guarantees, invented novelty, unmeasured speedups, or figures containing fabricated results. Close prior methods and continuation-data augmentation are mandatory controls.

## Implementation and verification

Use typed public/private boundaries, immutable snapshot objects, exact serialization, atomic artifact writes, deterministic named RNG streams, and explicit failure classes. GPU leases are acquired by the coordinator/runtime scheduler only. API probes must precede dependency pinning; inspect installed signatures and use official documentation.

Run the CPU reference and assigned contract tests before requesting integration. GPU acceptance requires measurements on the target node; passing CPU tests does not substitute. Add meaningful tests for leakage, estimator expectation, accounting, state transitions and replay, rather than tests that merely repeat implementation expressions.

Completion reports must state changed files, exact commands and outcomes, evidence paths, unresolved items, and which gates remain blocked. “Implemented” without exercised behavior is not “validated.” Continue routine implementation and fixes without asking the user to reapprove already authorized work.
