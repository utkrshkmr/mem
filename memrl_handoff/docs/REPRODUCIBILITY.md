# Reproducibility and evidence records

## Environment lock

The kit deliberately does not invent a tested torch/vLLM/CUDA combination. `locks/runtime_lock.template.json` is unverified. Agent03 probes a compatible environment on the actual node, agent00 reviews it, and the resulting `locks/runtime_lock.json` plus an exact dependency lock and container digest become mandatory run inputs. Record Python, driver/runtime CUDA, torch, vLLM, transformers, tokenizers, PEFT, attention implementation, model revisions and chat-template hash. Save complete successful API and parity probe outputs. A `pip freeze` without a working test is not compatibility evidence.

The CPU reference has no runtime dependencies beyond Python 3.11+. Its build-system declaration is only for installing the reference package; it is not the production GPU environment lock.

## Immutable lineage

For each run save code commit and dirty diff, full resolved configuration, public/private data manifest hashes, split ancestry, model/tokenizer/template identities, initialization adapter, sampler seeds, inference sampling parameters, per-call exact token IDs, behavior log probabilities, adapter versions, snapshots, selected call IDs/probabilities, scalar rewards, checkpoints and resource records. Keep private gold inaccessible to writer workers, even when both ultimately execute on the same researcher-owned node.

Hash files from bytes, not filenames. Atomic publication writes data first and a final manifest last. A consumer validates both expected schema and checksum. A run's local path is not a reproducibility identity. Store large artifacts in an immutable artifact location chosen by the researcher; keep manifests in git. Credentials and private data never go into git or a paper archive.

## Planner and cost templates

The included `scripts/plan_experiments.py` expands only the selected staged suite. It does not launch training. `--require-budgeted` rejects a plan without measured projections or over its cap. A valid measured cost file looks like this **shape**, with actual numeric values supplied after profiling:

```json
{
  "status": "measured",
  "units": "allocated_gpu_hours_per_complete_run",
  "projection_basis": "sha256 of actual profile artifact",
  "safety_factor": 1.25,
  "costs": {}
}
```

Populate one key per `ARM:C<capacity>:I<iterations>`; for example `F03_fork:C1024:I200`. Each value is a measured projection for one complete run at that setting, before the listed reserve. Baseline evaluation uses the same suite iteration label for lookup but performs zero optimizer steps. The planner multiplies by every seed/job and the safety factor. Missing entries remain unknown, never zero. Actual launch additionally checks the remaining project budget and current runtime/gate receipts; a draft planner output cannot authorize GPU work by itself.

Allocated GPU-hours are the sum over devices of leased duration, including idle phases; active engine time is a separate diagnostic. Show data generation, dev tuning, failed/retried runs, checkpoint overhead and reader/judge work in the ledger. A 1,500 GPU-hour plan is a configurable research envelope, not a measured runtime estimate.

## Freeze and final testing

Before final access, freeze hypotheses, primary contrast, endpoints, capacities, exact data populations, world/seed counts, checkpoint rule, inference budgets, prompt/renderer versions, baseline tuning choices, multiple-comparison family, missingness handling and statistical script hash. Record who reviewed the freeze and when. A locally written manifest is not an externally timestamped preregistration; use that term only if actually registered.

Final data may be prepared and hashed without inspecting model outcomes. Selection scripts cannot load their result paths. Unlocking test execution requires a valid G09 receipt, not interactive user permission for every job. If a later error invalidates a result, retain the invalid artifact, record the cause, issue a new protocol version and rerun the affected comparisons consistently.

## Human scientific review

The researcher should inspect at least a predeclared stratified sample of raw trajectories and judge outcomes, including every failure family and representative baseline wins. Natural chronological labels need two independent annotators or reviewers with adjudication; record disagreement and evidence spans. AI-generated annotations may assist drafting but are not independent human validation. Do not hide AI assistance; check the chosen venue's actual policy at submission time.

## Rebuild procedure

1. Fresh checkout; verify file checksums and CPU tests.
2. Recreate the pinned environment and run model/API parity probes.
3. Regenerate synthetic data from frozen seeds; match public and private manifest hashes.
4. Reproduce the small scripted trajectory, one update, DDP equivalence and restart behavior.
5. Reproduce a fixed development run and compare declared deterministic artifacts or numerical tolerances.
6. Rebuild tables/figures from immutable result rows without rerunning models; verify row counts, denominators and source hashes.
7. Release source, licensed data/derived-data instructions, locks, configs, logs sufficient for the stated claims and a clearly scoped artifact README.

Do not claim bitwise deterministic GPU generation across different kernels or devices unless tested. Record reproducible statistical behavior and numerical tolerances separately from byte-identical data generation.
