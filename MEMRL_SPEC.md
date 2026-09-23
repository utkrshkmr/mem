# MEMRL_SPEC.md — Detailed research and implementation specification, revision 3

**Working project name:** STREAMCAP — learned retention under constrained storage and changing future queries  
**Existing repository/package name:** memrl (keep for continuity; do not publish under “MemRL”)  
**Hardware available:** one machine with 4 NVIDIA H100 GPUs; plan assumes 80 GB per GPU, subject to hardware inventory  
**Document date:** 2026-09-23  
**Audience:** a coding agent capable of Python engineering, with explicit guidance for experimental design, RL, distributed execution, and analysis  
**Status:** research and implementation contract plus a separate CPU reference kit; the production GPU system and scientific results are not yet implemented or validated  
**Supersedes:** the previous version of this file in full

Revision 2 replaced a broad memory-tool benchmark with a focused study of whether reinforcement learning improves retention when useful information exceeds storage capacity, and whether that retention survives changes in future question demand. Revision 3 adds a specific continuation-training hypothesis, a from-scratch parallel-agent build kit, and explicit evidence gates. It preserves reproducibility, exact accounting, inspectable traces, and phased implementation. Read Section 30 before applying earlier defaults.

**Revision-3 navigation:** [Overrides and readiness](#30-revision-3-authority-and-research-readiness) · [Continuation method](#31-continuation-training-method-contract) · [From-scratch handoff](#32-from-scratch-project-and-parallel-implementation) · [Four-H100 execution](#33-four-h100-execution-and-budget-control) · [Evidence gates](#34-publication-evidence-and-claim-gates) · [New references](#35-additional-primary-references)

**Core specification navigation:** [Research questions](#1-research-objective-hypotheses-and-claim-boundaries) · [Four-H100 plan](#3-hardware-and-execution-on-four-h100s) · [Configuration](#5-configuration-identities-and-reference-budgets) · [Synthetic data](#7-synthetic-benchmark-independent-controls-for-length-useful-information-and-demand) · [Baselines](#12-baselines-attribution-arms-and-fair-comparisons) · [RL mathematics](#14-writer-only-rl-objective-and-estimator) · [Diagnostics](#17-retention-measurement-visibility-and-intervention-diagnostics) · [Statistics](#21-statistical-analysis-estimands-and-power-planning) · [Study schedule](#22-study-schedule-and-compute-prioritization) · [Implementation gates](#25-phase-by-phase-implementation-with-acceptance-gates)

Use the research-facing working name STREAMCAP only provisionally. Verify its availability before publication. “MemRL” already names another published research project [R5]. This document does not assert that STREAMCAP is a novel algorithm or that any planned experiment will yield a positive result.

---

## 0. Non-negotiable instructions for the coding agent

1. Read this document before implementing, including revision-3 Sections 30–35 and the companion rebuild kit. Follow the dependency gates in Section 32 and docs/ACCEPTANCE_GATES.md; Section 25 remains the baseline implementation breakdown. A working trainer does not excuse an invalid experiment.
2. Implement the primary protocol first. Optional studies are explicitly labeled. Do not implement the KV-cache backend, three-optimizer sweep, persistent cross-user streams, or large-model training before the primary gates pass.
3. Treat gold answers, support annotations, query pools, hidden future questions, and evaluator state as a separate private data plane. They must never enter an ordinary writer observation, tool response, memory key, exception, filename shown to the policy, or retrieval index.
4. The primary trainable component is the memory writer. The reader is frozen and deterministic. Reader completions have no policy-gradient loss.
5. Every method receives the same observation chunks in the same order. Capacity or storage-entry size must not change observation boundaries.
6. A generated memory is reused for all question cohorts about that history. Never construct a different memory for each test question in the question-blind protocol.
7. Capacity includes all persistent text exposed to a model: keys, values, visible IDs, and visible stored metadata. Audit-only fields are inaccessible to all policies and readers.
8. Enforce prompt and memory budgets using exact tokenization of actual strings. Empirical tokenizer ratios are planning estimates, never correctness guarantees.
9. Do not silently truncate histories, evidence, memory values, questions, or answers. Explicit benchmark-defined context truncation is allowed only in a named truncation baseline with logged boundaries. An overlong model completion is a modeled failure, not something to repair using gold data.
10. Distinguish a model/environment failure, which remains in the denominator, from an infrastructure failure, which requires deterministic rerun. Never drop difficult examples because generation, parsing, or retrieval failed.
11. Freeze experimental choices on development data. LongMemEval and LoCoMo are final external tests in the primary protocol; they are not checkpoint-selection data.
12. Use one clear RL estimator. The reference estimator is on-policy REINFORCE with a leave-one-out baseline, called rloo_pg. Do not call it clipped PPO, GRPO, or Dr. GRPO.
13. Every comparison records backbone, writer initialization, reader identity, memory capacity, observation protocol, retrieval budget, and generation cost. A shared context limit alone does not establish equal compute.
14. Keep independent histories, users, stories, books, question families, and training seeds identifiable. Statistical resampling must respect these units.
15. Use official library documentation and inspect installed signatures. Mark external API uncertainty as VERIFY, resolve it in a probe, and pin the working version. Do not build against an imagined API.
16. No fixed training-time or GPU-memory guarantee is made here. Measure the proposed workload on the actual 4-H100 machine before expanding a study.
17. Keep credentials out of logs, prompts, traces, manifests, screenshots, and git. If an hf-tok file exists, only the authentication module may read it.
18. Never invoke eval-only gold-informed policies during primary training. Enforce this in configuration and process boundaries.
19. Use atomic publication of manifests, jobs, checkpoints, and results. Never let another process read an incompletely written artifact.
20. Code sketches below specify semantics. Complete imports, error handling, interfaces, and tests before treating a sketch as production code.

Create these decision records immediately:

| File | Required contents |
|---|---|
| docs/DECISIONS.md | Date, issue, alternatives, chosen resolution, affected protocol IDs |
| docs/ASSUMPTIONS_TO_VERIFY.md | Every VERIFY item and its pass/fail evidence |
| docs/CLAIM_REGISTRY.md | Each research claim, required contrast, eligible experiment, limitations |
| docs/PROTOCOL_FREEZE.md | Data and configuration hashes frozen before final testing |
| docs/RESOURCE_LEDGER.md | Measured GPU hours, CPU time, storage, projections, completed stages |
| docs/DEVIATIONS.md | Any departure from this specification, including negative or failed studies |

Changes to the research question, target population, test splits, metric, training budget, or comparison rules require a new protocol version and a documented explanation. Routine API fixes do not require asking the user for permission.

---

## 1. Research objective, hypotheses, and claim boundaries

### 1.1 Primary task

A writer observes a chronological history once, in fixed chunks. It can maintain a bounded text memory. It cannot revisit the raw history. After ingestion finishes, the memory is frozen. A separate frozen reader answers questions using a specified view of that memory.

The writer sees the task family and, in designated experiments, a prior description of expected query demand. It does not see the realized questions or their answers.

For a history x, query distribution p, memory capacity C, writer policy pi, fixed retrieval function f, and fixed reader g:

~~~text
M = Write_pi(x; C)
y(q) = g(q, f(M, q))
V(pi; p, C) = E_x E_q~p(.|x) [score(y(q), answer(x,q))]
~~~

The experiment tests retention decisions, representation, and updates. It does not establish autonomous planning ability, online improvement of model weights at deployment, or transfer of a LoRA adapter to an incompatible backbone.

### 1.2 Preregistered questions

**RQ1 — Binding capacity.** Does a learned writer outperform strong non-RL memory constructors when the quantity of independently queryable information exceeds the capacity of practical compact representations?

**RQ2 — Attribution.** How much of any improvement is due to memory construction, retrieval, answer generation, or merely learning the tool-output format?

**RQ3 — Demand shift.** Does a writer trained for skewed query demand lose disproportionate accuracy on tail facts, historical questions, or changed query distributions?

**RQ4 — Mechanism.** When the system fails, is sufficient information demonstrably retained, exposed to the reader, and usable by that reader? Which failures are corrected by controlled restoration?

**RQ5 — Transfer and cost.** Do the effects replicate on untouched conversational benchmarks and another backbone, and what are their ingestion, storage, retrieval, and answering costs?

### 1.3 Hypotheses, including outcomes that would weaken the paper

| ID | Hypothesis or diagnostic contrast | What would weaken it |
|---|---|---|
| H1 | RL improves fixed-reader accuracy over the strongest locked non-RL baseline under calibrated binding capacity | Gains disappear with a strong summary/fact baseline or occur only when all facts fit |
| H2 | Writer-only improvements survive a common frozen reader and common memory serialization | Gains appear only with a trained reader or different retrieval budget |
| H3 | Training demand affects which facts are retained; the effect is exposed by reweighting questions on the same frozen memory | Apparent shift effects come from rebuilding memories or changing history difficulty |
| H4 | A predeclared multi-cohort training variant improves tail/shift performance at a measurable cost, or improves the frontier | Gains arise only from more questions, more generated tokens, or more training samples |
| H5 | A mechanistic explanation supported by interventions replicates beyond the constructed synthetic environment | Results depend on one template, one source history, or one scoring shortcut |

H4 is conditional: implement it only after H1–H3 expose a concrete weakness worth addressing. Multi-query training, RL memory tools, and restoration diagnostics already have related precedents. Do not claim their general concepts as new.

### 1.4 Contribution routes

Choose one route after development pilots and before final testing:

- **Empirical mechanism paper:** a reproducible finding about how limited memory and future query demand interact, with strong controls and external confirmation.
- **Method paper:** an explicitly specified retention-learning intervention that improves a declared trade-off over close prior methods under matched resources.
- **Negative-result/measurement paper:** a substantial finding that apparent memory improvements vanish under controlled readers, realistic capacity pressure, or complete visibility accounting. This needs breadth and a useful released protocol; an unsuccessful implementation alone is not a contribution.

### 1.5 Claims forbidden without additional evidence

- “State of the art” from custom subsets or changed judges.
- “Causal failure attribution” from a conditional funnel rate alone.
- “Optimal oracle ceiling” from greedy evidence selection or truncated support.
- “Zero-shot dataset transfer” when target-dataset scores selected checkpoints.
- “Information-theoretic compression pressure” from raw history length divided by text capacity.
- “Cross-model policy transfer” when a separate adapter was trained for each model.
- “Efficient” from fewer calls while ignoring prompt tokens, ingestion work, or storage.
- “All evidence absent” from failed string matching against free-form notes.

---

## 2. Changes from revision 1

| Previous design | Required revision |
|---|---|
| Joint writer and answer training as main result | Writer-only training with frozen reader; joint training is an attribution arm |
| C=8192 with a few needles | C in {512,1024,2048,4096} plus independently varied useful information |
| Different chunks for raw and note memory | One frozen observation chunk manifest shared by all arms |
| Question bootstrap | History/conversation/stream cluster inference with seed uncertainty |
| LongMemEval/LoCoMo used during checkpoint selection | Untouched external tests; separate development distributions |
| Content overlap called strict provenance | Exact certification where possible, semantic audits otherwise, uncertainty retained |
| READ events define evidence visibility | Actual reader-input ledger covering every visible channel |
| Greedy gold-informed arms called ceilings | Privileged references with explicit feasibility and budget disclosures |
| GRPO/RLOO/Dr. GRPO sweep | One validated rloo_pg estimator, then only justified optimization ablations |
| FIFO and LRU treated as distinct default arms | One FIFO arm; LRU only in settings with intervening reads |
| Per-session wipes as central contribution | Matched corruption and retention interventions; wipes are optional controls |
| Always occupy all four GPUs | Use all four when beneficial; CPU phases and debugging need not reserve GPUs |
| KV backend mixed into the plan | Deferred independent project |

---

## 3. Hardware and execution on four H100s

### 3.1 Inventory before downloading large models

Run scripts/probe_hardware.py and save reports/hardware.json with:

- GPU UUIDs, names, total/free memory, driver, CUDA runtime, compute capability.
- GPU interconnect topology from nvidia-smi topo -m.
- CPU count, RAM, free scratch/NVMe space, filesystem type.
- Python and package versions; operating system.
- Current competing GPU processes, without collecting unrelated process environments.
- bf16 support and a small two-rank DDP all-reduce test.

Assume 4 x H100 80 GB only after the inventory confirms it. A 94 GB H100 variant, PCIe rather than SXM, or limited host RAM changes throughput planning. Do not alter scientific budgets to hide an out-of-memory problem.

### 3.2 Default writer-only training allocation

| Physical GPU | Process | Role |
|---|---|---|
| 0 | trainer rank 0 | LoRA gradient accumulation; orchestration |
| 1 | trainer rank 1 | LoRA gradient accumulation |
| 2 | rollout worker 0 | Writer trajectories and frozen-reader scoring |
| 3 | rollout worker 1 | Writer trajectories and frozen-reader scoring |

The reference writer and reader share the same pretrained backbone. Each rollout engine serves writer requests with the current writer LoRA and reader requests with the base model, or an explicitly different reader adapter for an attribution arm. The frozen-reader requests must never accidentally use the writer adapter.

Generation is phase-batched: ingest writer requests, freeze snapshots, then batch reader questions. Sharing base weights avoids a separate full reader model on each rollout GPU. It does not make reader computation free; log it separately.

The trainers keep a bf16 base model plus LoRA and gradients. Start with microbatch=1, gradient checkpointing, LoRA rank 16, and 8192 maximum prompt-plus-completion tokens. Do not begin with 14B or 32B training.

### 3.3 Evaluation allocation

- Memory construction: four independent single-GPU workers for 7B/8B models.
- Frozen-reader evaluation: four workers, with the same fixed reader model and settings.
- Judge audit: stop policy workers first; use two TP=2 replicas of a 32B judge across {0,1} and {2,3}, if the inventory and measured KV requirements permit.
- Larger reader sensitivity analysis: a separate phase with explicit TP and memory measurements.
- Full-history retrieval indices: CPU by default for the small text-memory experiment; dense embedding jobs can be batched separately.
- Pure synthetic generation, metrics, and bootstrap analyses run on CPU.

### 3.4 Memory planning

Approximate bf16 parameter storage as 2 * parameter_count bytes before runtime buffers, activations, KV cache, optimizer state for trainable parameters, and fragmentation.

For a decoder with n_layers, n_kv_heads, head_dim, and bf16 KV:

~~~python
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, element_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * element_bytes
~~~

Read these quantities from the actual model configuration. Multiply by simultaneously resident token sequences only as a rough planning estimate; vLLM scheduling and prefix reuse affect the realized allocation.

Start rollout gpu_memory_utilization at 0.80, max concurrent sequences at 8, and raise concurrency only after measuring peak allocation and successful long-length batches. Do not reserve 85% for one engine and then load another engine assuming their allocations will coexist.

### 3.5 GPU launcher

Prefer a Python supervisor over background shell commands with fragile cleanup:

~~~python
# scripts/launch_train.py — core pattern, add CLI/config/log routing.
import os
import signal
import subprocess
import sys
from pathlib import Path

def spawn(argv, visible_devices, log_path, extra_env=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_devices))
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONHASHSEED"] = "0"  # stable hashes still use hashlib, not hash()
    env.update(extra_env or {})
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        argv, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle

# Start workers on [2] and [3], then:
# python -m torch.distributed.run --standalone --nproc_per_node=2
#        -m memrl.rl.trainer --config resolved.yaml
# with CUDA_VISIBLE_DEVICES=0,1.
# Inside each worker, its assigned physical GPU is logical cuda:0.
# Inside DDP, use LOCAL_RANK, not physical device IDs.

def stop_child(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
~~~

Handle SIGINT/SIGTERM, preserve exit codes, and close log handles. Never kill arbitrary GPU processes.

### 3.6 Throughput calibration

Measure at 8k, 16k, and 32k history length; at low/high capacity; and with the actual query count per history. A timing estimate from the first short curriculum stage is insufficient.

For each probe report:

- Writer calls, generated tokens, prefill tokens, wall time.
- Reader calls and tokens.
- Forward/backward training tokens and wall time.
- CPU retrieval/indexing time and serialization overhead.
- Adapter publication/loading time.
- Peak allocated/reserved GPU memory per process.
- Node-hours and allocated GPU-hours: sum over device-allocation intervals.
- Median and p90 across at least three post-warmup iterations.

The planner computes full-study cost from a length/capacity/task mixture, including validation and final evaluation. It must print uncertainty bands from observed timing variation. Full sweeps require an explicit GPU-hour budget argument; this is a scheduler limit, not permission to change the experiment after observing test results.

---

## 4. Repository layout and installation

~~~text
memrl/
  pyproject.toml
  requirements.lock
  README.md
  hf-tok                         # optional, gitignored
  configs/
    base.yaml
    models/
    protocols/
    data/
    arms/
    studies/
  docs/
    DECISIONS.md
    ASSUMPTIONS_TO_VERIFY.md
    CLAIM_REGISTRY.md
    PROTOCOL_FREEZE.md
    DEVIATIONS.md
    RESOURCE_LEDGER.md
    PRIOR_WORK_MATRIX.md
  memrl/
    auth.py
    config.py
    hashing.py
    manifests.py
    budget.py
    tokenizers.py
    models/compat.py
    data/
      schema.py
      generate_worlds.py
      renderers.py
      query_cohorts.py
      background.py
      chunker.py
      longmemeval.py
      locomo.py
      splits.py
      leakage.py
      stats.py
    memory/
      schema.py
      serialize.py
      store.py
      retrieval.py
      baseline_summary.py
      baseline_facts.py
      baseline_raw.py
      prior_adapters.py
    env/
      observations.py
      tools.py
      parser.py
      writer_env.py
      reader_env.py
      visibility.py
      interventions.py
    rollout/
      worker.py
      jobs.py
      adapters.py
      scoring.py
    rl/
      formatter_sft.py
      rewards.py
      advantages.py
      samples.py
      estimator.py
      trainer.py
      checkpoints.py
      reader_only.py
    eval/
      freeze.py
      build_memories.py
      answer_snapshots.py
      scorers.py
      judges.py
      audit.py
      run.py
    logging/
      events.py
      parquet.py
      snapshots.py
      traces.py
    cli/
      data.py
      validate.py
      train.py
      eval.py
      sweep.py
  analysis/
    collect.py
    contrasts.py
    bootstrap.py
    retention.py
    interventions.py
    costs.py
    plots.py
    report.py
  scripts/
    probe_hardware.py
    probe_dependencies.py
    probe_models.py
    launch_train.py
    launch_eval.py
    smoke_test.py
    plan_compute.py
  tests/
    unit/
    property/
    gpu/
    research_contracts/
  data_cache/                    # immutable/versioned, usually not git
  artifacts/                     # adapters and protocol freezes
  runs/                          # append-only run outputs, not git
  reports/
~~~

### 4.1 Dependency policy

Use a clean virtual environment. Install a supported stable vLLM build first, resolve PyTorch compatibility, then install remaining dependencies. Never upgrade an active experiment environment in place.

Required packages: torch, vllm, transformers, peft, accelerate, datasets, huggingface_hub, safetensors, numpy, scipy, pandas, pyarrow, duckdb, pydantic or typed dataclasses, omegaconf, orjson, jinja2, matplotlib, pytest, hypothesis. Optional secondary retrieval: sentence-transformers and rank_bm25.

Use PyTorch SDPA initially if FlashAttention installation is problematic. If switching attention implementations changes log probabilities beyond the declared tolerance, resolve the mismatch before training.

Freeze the working environment, including exact package versions and model/tokenizer revision SHAs. The checked-in installation instructions reproduce the lock, not whatever “latest” resolves to later.

Bootstrap commands for the initial compatibility probe (run inside the project root):

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install vllm
python -m pip install transformers peft accelerate datasets huggingface_hub \
  safetensors numpy scipy pandas pyarrow duckdb pydantic omegaconf orjson \
  jinja2 matplotlib pytest hypothesis rank_bm25
python -m pip check
python -m pip freeze > requirements.discovery.txt
~~~

Choose a Python version supported by the selected vLLM release before creating the environment. These discovery commands intentionally resolve versions once; after the probes pass, create requirements.lock from the tested environment and reproduce subsequent runs from that lock. Do not reuse requirements.discovery.txt as evidence of compatibility without running the probes. Install sentence-transformers only for the dense/hybrid retrieval phase, resolve its dependencies against the lock, and version that environment separately if it changes the stack.

### 4.2 Authentication

~~~python
# memrl/auth.py
import os
from pathlib import Path

def configure_hf_auth(repo_root: Path):
    if os.environ.get("HF_TOKEN"):
        return
    token_path = repo_root / "hf-tok"
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("hf-tok is empty")
        os.environ["HF_TOKEN"] = token
    # Public model/dataset access must work without forcing a token.
~~~

Do not print the environment or serialize all environment variables. A run manifest uses an allowlist of nonsensitive settings.

### 4.3 Model registry

| Role | Initial model | Policy |
|---|---|---|
| CPU/GPU smoke model | Qwen/Qwen2.5-1.5B-Instruct | Debugging only |
| Main writer/base reader | Qwen/Qwen2.5-7B-Instruct | Stable controlled baseline; no claim it is the newest model |
| Second-backbone replication | Qwen/Qwen3-8B | Disable thinking if the pinned tokenizer supports that option; verify |
| Different-family sensitivity | meta-llama/Llama-3.1-8B-Instruct | Optional; gated access and common observations required |
| Offline judge | Qwen/Qwen2.5-32B-Instruct | Audit and conversational scoring, not synthetic training reward |

Read actual native context limits from configs. Do not silently enable RoPE scaling. All controlled main arms use the same 8192 per-call limit even if the native model supports more.

A second backbone may use its own frozen base reader during training for the shared-weight serving layout. This yields a within-backbone replication. To compare memory portability, rerun exported snapshots through a common reader in a separate evaluation phase.

### 4.4 Mandatory API probes

scripts/probe_dependencies.py MUST check and save:

- LLM.generate argument names and acceptance of token-ID prompts.
- Per-request SamplingParams and LoRA selection behavior.
- Sampled-token log-probability availability, interpretation, and alignment.
- Stop-string behavior versus completion token IDs.
- Base-model requests after adapter requests really disable adapters.
- PEFT save/load and adapter-disable behavior.
- Transformer forward support for logits_to_keep or an equivalent.
- Gradient checkpointing with LoRA inputs and DDP.
- Chat-template tokenization type and generation-prompt behavior.
- Prefix caching isolation across adapter identities.
- Actual bf16 memory requirements at the planned context length.

If an optimized interface is unavailable, use the tested slower equivalent. Record the change; never change loss semantics to accommodate an API.

---

## 5. Configuration, identities, and reference budgets

### 5.1 Merge order

base -> model -> protocol -> dataset -> arm -> study -> CLI.

Convert to typed configuration, reject unknown keys, run validation, and save the fully resolved YAML plus canonical JSON.

Separate IDs:

- protocol_id: hash of scientific settings, data manifests, prompt templates, scoring and comparison definitions.
- execution_id: hash of protocol_id, seed, model revisions, code commit, package lock, hardware profile, and execution settings affecting numerics.
- memory_id: hash of history_id, writer identity, writer seed, memory budget, observation-manifest hash, and resulting canonical snapshot.
- reader_input_id: hash of the exact token IDs, reader revision/adapter, decoding settings, and intervention specification.

Never exclude a setting from caching merely because it is called “hardware”; dtype, quantization, and batching can affect outputs.

### 5.2 Reference configuration

~~~yaml
spec_version: 3
run:
  study: F2
  name: fork_development
  seed: 101
  protocol: continuation_blind_v3
  development_only: true
  arm: F03_fork
  mode: train
model:
  writer: Qwen/Qwen2.5-7B-Instruct
  reader: Qwen/Qwen2.5-7B-Instruct
  writer_revision: RESOLVE_AND_PIN
  reader_revision: RESOLVE_AND_PIN
  dtype: bfloat16
  attention: sdpa
  chat_template_kwargs: {}
  native_window: FROM_PINNED_CONFIG
  quantization: null
initialization:
  writer_adapter: null
  writer_adapter_sha256: null
  format_training_manifest: null
  raw_base_format_error_rate: null
  use_shared_format_initialization: false
tokenization:
  memory_reference: Qwen/Qwen2.5-7B-Instruct
  reference_revision: RESOLVE_AND_PIN
  token_count_add_special_tokens: false
observation:
  reference_chunk_cap: 1024
  policy_chunk_cap: 1280
  pack_within_session: true
  split_long_turns: true
  manifest: BUILD_ONCE_AND_FREEZE
writer_budget:
  total: 8192
  system: 1536
  chunk: 1280
  index: 1536
  loaded: 2048
  last_result: 256
  framing_slack: 256
  generation: 768
reader_budget:
  total: 8192
  system: 768
  question: 512
  memory: 6144
  framing_slack: 256
  generation: 256
memory:
  capacity_ref_tokens: 1024
  max_entries: 64
  max_key_ref_tokens: 24
  max_value_ref_tokens: 256
  serializer_version: canonical_v1
  include_visible_metadata_in_capacity: true
  eviction: reject
  oversize: reject
  writer_tools: full
  expose_raw_history_search: false
  accounting_version: whole_store_v3
  primary_index_overflow: reject
environment:
  max_writer_calls_per_chunk: 3
  writer_question_visibility: blind
  writer_prior: declared_task_prior
  carry_loaded_across_chunks: false
  carry_tool_result_across_chunks: false
  mutable_reader_state: false
  user_scope: one_identity_per_history
  persistence: within_history_only
reader:
  trainable: false
  decoding: greedy
  memory_view: all_retained
  order: creation_order
  tools: none
  expose_writer_index_separately: false
  output_format: json_answer
  copies_per_query: 1
retrieval:
  enabled: false
  method: bm25
  max_results: 8
  candidate_unit: entry
  snippet_policy: none
  dense_token_limit: 512
  dense_long_entry_policy: overlapping_windows
data:
  train: continuation_train_v3
  validation: continuation_dev_v3
  final_external:
  - longmemeval_s_cleaned
  - locomo10
  train_histories: 4096
  dev_histories: 256
  locked_synthetic_test_histories_per_cell: 128
  train_lengths_ref:
  - 8000
  - 16000
  - 32000
  train_candidate_counts:
  - 16
  - 32
  - 64
  - 128
  - 256
  main_test_length_ref: 32000
  main_test_candidate_counts:
  - 32
  - 128
  - 512
  training_query_cohort: skew_80_20
  training_questions_per_group: 8
  questions_hidden_until_writer_finished: true
  allow_external_test_for_selection: false
  background_split_by_book: true
  renderer_split_by_template_family: true
  conditional_sampling_rule: reject_infeasible_pair_and_log
rl:
  estimator: rloo_pg
  component: writer_only
  groups_per_iteration: 4
  trajectories_per_group: 4
  iterations: 200
  writer_temperature: 1.0
  top_p: 1.0
  top_k: -1
  repetition_penalty: 1.0
  learning_rate: 5.0e-06
  warmup_iterations: 10
  gradient_clip: 1.0
  weight_decay: 0.0
  microbatch_calls: 1
  selected_calls_per_iteration: 256
  call_sampling: uniform_without_replacement
  estimator_scale: 256
  normalize_advantages: false
  kl_beta: 0.0
  reward:
    task_weight: 1.0
    call_penalty: 0.0
    format_penalty: 0.0
    gold_retention_shaping: 0.0
  lora:
    rank: 16
    alpha: 32
    dropout: 0.0
    target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  validation_interval: 20
  save_interval: 20
evaluation:
  writer_temperature: 0.0
  writer_samples_per_history: 1
  training_seeds:
  - 101
  - 202
  - 303
  capacities:
  - 512
  - 1024
  - 2048
  - 4096
  locked_primary_capacity: 1024
  cohort_questions_per_history: 16
  cluster_unit: source_world
  bootstrap_replicates: 10000
  checkpoint_rule: highest_dev_task_score_then_earliest
  final_external_access: locked_until_protocol_freeze
hardware:
  share_writer_reader_backbone: true
  trainer_physical_gpus:
  - 0
  - 1
  rollout_physical_gpus:
  - 2
  - 3
  rollout_gpu_memory_utilization: 0.8
  max_active_writer_sequences_per_worker: 8
  max_active_reader_sequences_per_worker: 16
  embedding_device: cpu
  schedule: split_2train_2rollout
  gpu_count: 4
logging:
  retain_all_evaluation_events: true
  full_prompt_fraction_train: 0.02
  full_prompt_fraction_eval: 0.05
  retain_exact_training_token_ids: true
  storage_compression: zstd
continuation:
  enabled: true
  mode: fork
  futures_per_prefix: 2
  reuse_prefix: true
  reward_endpoint: future_mean
  prefix_boundary_event_fractions:
  - 0.5
  - 0.75
  kernel_version: ledger_future_v1
  target_age_mixture:
    recent: 0.4
    old: 0.2
    uniform: 0.4
  future_rng_independent_of_writer_actions: true
  share_futures_across_g: true
  share_queries_across_g: true
  reveal_future_before_fork: false
  raw_history_archive_access: false
  allocator:
    enabled: false
    pool_size: 8
    draws: 2
    epsilon: 0.25
resources:
  project_gpu_hour_cap: 1500.0
  run_gpu_hour_cap: null
  require_measured_projection: true
  charge_allocated_idle_devices: true
artifacts:
  runtime_lock: null
  data_manifest: null
  gate_receipts: null
  protocol_freeze: null
~~~

Revision 3 trains and tests the primary experiment at capacity 1024. Evaluation at 512/2048/4096 without retraining is capacity transfer; matched-capacity retraining is a separately costed study. The YAML above mirrors configs/base.json in the rebuild kit and selects the F03 development candidate; it is a draft, not a launch-ready production configuration. Model/data/runtime placeholders must be resolved through the real API and data gates. Section 31 defines its continuation objective.

The 128 histories per synthetic cell and 16 questions per cohort are planning defaults, not a power guarantee. Section 21 specifies pilot-based sample-size decisions before final-test generation/access.

### 5.3 Exact budget arithmetic

Writer upper bound:

~~~text
1536 system + 1280 chunk + 1536 index + 2048 loaded
+ 256 last result + 256 framing + 768 generation
= 7680 <= 8192.
~~~

Reader upper bound:

~~~text
768 system + 512 question + 6144 memory + 256 framing + 256 generation
= 7936 <= 8192.
~~~

These section sums provide design headroom. Final acceptance always checks the exact chat-template token IDs, because concatenation and tokenizer boundaries need not be additive.

### 5.4 Validators

Implement config validation before model loading:

| Check | Required behavior |
|---|---|
| Section sums plus generation fit total | Reject if false |
| Total <= verified native context | Reject if false |
| C > 0, K > 0, key/value caps > 0 | Reject if false |
| Primary C in supported all-retained range | Validate actual serialized snapshots; reject incompatible reader views |
| Fixed-reader primary has trainable reader=false | Reject if false |
| Shared-engine training has identical writer/reader base revisions and compatible chat templates | Reject if false; use a separate exported-snapshot evaluation phase for a different reader |
| Query-blind writer receives no question field | Schema-level guarantee plus leak test |
| Ordinary training has no privileged reference or gold shaping | Reject if false |
| Same observation manifest across matched arms | Reject comparison if hashes differ |
| Retrieval disabled for all-retained primary | Reject conflicting configuration |
| No benchmark test source in selection lineage | Reject final-transfer label if violated |
| RLOO group size >= 2 | Reject if false |
| Sampling temperature=1, top_p=1, no generation penalties for reference estimator | Reject unsupported sampling distribution |
| Mutable read counters absent from frozen-reader inputs | Reject if false |
| LoRA dropout=0 for trainer/rollout parity | Reject if false |
| Dense entries fully indexed via windows | Reject silent truncation |
| Study cost within explicit scheduler budget | Refuse to launch excess jobs; leave scientific config unchanged |

At runtime assert exact memory charges, entry caps, prompt length, canonical snapshot hashes, and reader-adapter identity. Runtime budget exceptions caused by model actions return a bounded error and consume the action; impossible prompts or corrupt state are implementation bugs that stop the run.

---

## 6. Formal information boundary and data schema

### 6.1 Separate public observations from private evaluation state

Use different classes, serialization files, and constructors. A writer worker may access public history chunks and its memory only. A trusted evaluator loads private answers and support labels after snapshots are frozen.

It is acceptable for both services to run on the same physical machine. The separation is an engineering boundary preventing accidental leakage, not a claim of adversarial security isolation.

~~~python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class PublicTurn:
    turn_id: str
    speaker: str
    text: str
    timestamp: str | None
    user_id: str

@dataclass(frozen=True)
class PublicChunk:
    history_id: str
    chunk_id: str
    session_id: str
    user_id: str
    timestamp: str | None
    text: str
    # Source ranges are for deterministic reconstruction, not gold relevance.
    source_spans: tuple[tuple[str, int, int], ...]

@dataclass(frozen=True)
class PublicHistory:
    history_id: str
    user_id: str
    turns: tuple[PublicTurn, ...]
    task_prior_text: str
    manifest_id: str

@dataclass(frozen=True)
class FactAtom:
    atom_id: str
    entity_id: str
    relation: str
    value: str
    valid_from: int
    valid_to: int | None
    source_turn_ids: tuple[str, ...]
    source_char_spans: tuple[tuple[str, int, int], ...]

@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    history_id: str
    text: str
    answer_type: str
    gold: Any
    answerable: bool
    cohort: str
    query_weight: float
    # Multiple alternative sufficient support sets are allowed.
    sufficient_support_sets: tuple[tuple[str, ...], ...]
    support_granularity: str
    question_time: str | None
    source_cluster_id: str
    private_meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PrivateEvalBundle:
    history_id: str
    atoms: tuple[FactAtom, ...]
    questions: tuple[EvalQuestion, ...]
    generator_seed: int
    lineage: dict[str, str]
~~~

For datasets lacking atom-level structure, atoms may be empty. Support labels then refer to source turns or sessions and retain that granularity.

### 6.2 Dataset files

~~~text
data_cache/<dataset>/<version>/
  manifest.json
  public/<split>/histories.jsonl
  public/<split>/chunks.jsonl
  private/<split>/eval_bundles.jsonl
  splits.json
  source_hashes.parquet
  stats.json
~~~

Do not put question type, “evidence,” answer text, or support flags in public chunk IDs. Source dataset IDs that encode such labels must be mapped to opaque stable identifiers; retain the mapping in private metadata.

### 6.3 Deterministic IDs and random-number streams

Do not use Python hash(), which is process-dependent. Allocate independent RNG streams for world generation, rendering, background choice, insertion, query selection, rollout sampling, call subsampling, bootstrap, and intervention shams.

~~~python
import hashlib
import json

def stable_seed(*parts, bits=63):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << bits) - 1)
~~~

A new rollout seed must not change the underlying world or its questions. A different query cohort must not change the memory-construction seed or source history.

### 6.4 Information-leak contract tests

Implement tests that:

1. Insert random sentinel secrets into private answers and evidence labels.
2. Render every writer prompt and tool response for scripted/random trajectories.
3. Assert the sentinels, question text, private IDs, and support labels are absent.
4. Verify no policy-facing exception prints a private object repr.
5. Verify retrieval indexes contain current memory entries only.
6. Verify changing the hidden question cohort with a fixed public history and writer seed produces byte-identical writer requests and memory.
7. Verify public observations remain identical across C, K, slot size, and memory method.
8. Verify all calls to the evaluator occur after a snapshot is immutable.

## 7. Synthetic benchmark: independent controls for length, useful information, and demand

### 7.1 Purpose and separation of suites

Build three suites with separate result tables:

1. **Sanity suite:** simple exact retrieval, updates, and two-hop questions. Used to debug tools and training. It may resemble BABILong/RULER, but is labeled custom.
2. **Capacity suite:** many independently queryable facts, high-entropy values, matched style distractors, and a frozen question pool. This supports the main capacity experiment.
3. **Dynamic suite:** updates, reversals, historical questions, temporal comparisons, and short derived-state tasks. This tests whether retention handles more than salient-value extraction.

Official BABILong/RULER evaluations are optional external additions using their official versions and protocols. Do not put custom-generator scores under an official benchmark name.

### 7.2 World construction

Each history describes one synthetic user or organization with randomly generated entities and relations. Use fresh pseudonyms/IDs and high-entropy values so the pretrained model cannot know the answers.

Initial task families:

| Family | State and query | Important control |
|---|---|---|
| independent_kv | Entity/relation -> random value; ask one mapping | Many candidate mappings have identical stylistic salience |
| current_state | Reassign a relation over time; ask current value | Latest assignment is gold; earlier assignments remain history |
| historical_state | Ask the value at a specified earlier time | Blindly overwriting old facts is insufficient |
| update_reversal | A -> B -> A or A -> B -> C | Distinguish current state, transition, and duration questions |
| set_membership | Add/remove items from a collection | Gold computed from operations, not substring containment |
| relation_chain | Entity -> project -> location/code | Alternative sufficient paths represented explicitly if present |
| bounded_aggregate | Sum/count within a time interval | Allow a sufficient statistic to replace raw events |
| unanswerable | Query an unobserved relation or unresolved reference | Unknown is an explicit gold label, not a text-substring heuristic |

Main attribution and capacity tests start with independent_kv and current_state. Add complexity only after the frozen base reader succeeds when given sufficient evidence.

A “candidate fact” means a distinct information item that has nonzero probability under the declared query pool. It is not the same as a turn, occurrence, update, or support sentence.

### 7.3 High-entropy payloads and plausibility

Use two renderings of the same abstract world:

- **Exact stress rendering:** random 12- to 16-digit identifiers or 16-character hexadecimal values assigned to entities. This prevents guessing and compressing repeated values.
- **Conversational rendering:** natural-language paraphrases describing comparable independent attributes, appointments, projects, preferences, and event histories.

Report both. Random codes make storage pressure controllable but are not a substitute for external conversational validation.

Never label only useful lines as “fact,” “important,” “needle,” or “special magic number” while background appears as narration. All candidate facts use matched formatting. Include nonqueried facts from the same rendering process as hard distractors.

### 7.4 Abstract world generator

~~~python
# memrl/data/generate_worlds.py — runnable core; rendering is separate.
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Assignment:
    event_id: str
    entity: str
    relation: str
    value: str
    t: int

def make_assignments(seed, n_entities, updates_per_entity=0):
    rng = random.Random(seed)
    events = []
    used_values = set()
    t = 0
    for update_round in range(updates_per_entity + 1):
        order = list(range(n_entities))
        rng.shuffle(order)
        for i in order:
            value = f"{rng.getrandbits(64):016x}"
            while value in used_values:
                value = f"{rng.getrandbits(64):016x}"
            used_values.add(value)
            events.append(Assignment(
                event_id=f"event_{len(events):06d}",
                entity=f"entity_{i:05d}",
                relation="access_code",
                value=value,
                t=t,
            ))
            t += 1
    return events

def state_at(events, entity, relation, time):
    candidates = [e for e in events
                  if e.entity == entity and e.relation == relation and e.t <= time]
    return max(candidates, key=lambda e: e.t).value if candidates else None
~~~

For realistic mixed relation families, entities can have several independently generated relations. Generate valid_from/valid_to intervals from the event log; never ask the language model to establish ground truth.

Store an explicit truth table for every generated question. Assert answer consistency using two independent simple implementations for the temporal/aggregate tasks.

### 7.5 Rendering and train/test separation

Use at least:

- Four training template families.
- A held-out paraphrase/template family for development.
- A separate final held-out family.
- Multiple speaker styles and sentence orders within a family.
- Varied session lengths and fact positions.

Split by template family, not just individual surface strings. A test containing the same template with new names is entity generalization, not linguistic generalization.

If an LLM produces paraphrases:

1. Generate the underlying world first.
2. Supply only the facts for that public text segment.
3. Require preservation of entities, relations, values, and temporal ordering.
4. Validate exact identifiers and state consistency automatically.
5. Audit a stratified development sample manually.
6. Store the paraphrase model, revision, prompt, seed, and output hash.
7. Reject ambiguous paraphrases without consulting downstream model accuracy.
8. Keep paraphrasing data generation outside reported inference/training costs, but disclose its total cost separately.

Do not use test performance to choose or repair renderers. Repair generator bugs globally by versioning the dataset and rerunning all affected arms.

### 7.6 Background construction and length control

Background can include PG-19 text in the sanity suite. The capacity/dynamic suites require a substantial matched-style background so that the task is not solved solely by detecting unusual formatting.

Partition books and source documents by identity before sampling windows. Hash normalized overlapping windows to detect near-duplicate background reuse. Keep source provenance in the manifest.

Target history length is measured in the reference tokenizer. Use a tolerance of +/- 2% around the requested target after adding public turn/session formatting.

Generation procedure:

~~~text
1. Generate all candidate events and any matched-style distractor events.
2. Render them using the selected template family.
3. Measure their total tokens including public formatting.
4. If they exceed the requested target, reject this (length, candidate_count)
   combination and log infeasible_length_fact_pair.
5. Sample enough background from allowed sources.
6. Choose insertion positions independently of query cohort.
7. Preserve causal event order for temporal families.
8. Build sessions and turns; preserve exact source-to-character mappings.
9. Add/cut background only until the length tolerance is met.
10. Never remove or shorten facts to hit the target.
11. Create question pools from the world, independent of rendering positions.
12. Freeze both public and private manifests.
~~~

For length-only experiments, use the same world, assignments, query pool, fact ordering, and source IDs. Add background at fixed relative gaps and record how fact depth changes. Include a separate evidence-depth-controlled version if the mechanism concerns recency; raw length and distance-to-answer must not be conflated.

### 7.7 Capacity calibration

Define and report:

~~~text
L_raw       = reference tokens in the complete public history
U_canonical = accounted size of a canonical store containing every atom needed
              to answer the declared complete query pool
U_compact   = size of a practical deterministic compact representation for that
              task (e.g., entity -> current value, or event ledger)
rho_raw     = L_raw / C
rho_canon   = U_canonical / C
rho_compact = U_compact / C
~~~

U_canonical and U_compact are representation-specific workload descriptors. They are not information-theoretic lower bounds.

Calibration requires:

- Some nonbinding cells where the compact store fits.
- Some cells where it does not fit at any declared primary capacity.
- Near-capacity cells where meaningful trade-offs emerge.
- A sufficient-evidence reader check demonstrating that wrong answers are not primarily task incomprehension.
- An optional atomic-slot control: restrict entries to one validated immutable fact each, forbid compression/packing, and choose candidate_count > max_entries. This creates a transparent item-capacity constraint, but is a separate restricted interface from free-form memory.

For the main free-form experiment, verify that high-entropy random values, compact-map size, observed retention, and capacity saturation jointly support the “binding” interpretation. Do not assert that no possible text code can compress the data further.

### 7.8 Initial grids

**Capacity grid:** history length 32k; candidate counts {32,128,512}; capacities {512,1024,2048,4096}; primary backbone; independent_kv and current_state.

**Length grid:** history lengths {8k,32k,128k}; hold one calibrated world/query pool fixed per paired family; use only combinations where facts fit before background addition.

**Dynamic grid:** update rates {0,1,3} additional assignments per entity; current/historical question mixes; capacities chosen from the calibrated binding cells.

Do not fully cross every axis. First generate a feasibility table and select a preregistered subset with a stated rationale. Candidate counts are initial values; adjust them using development calibration to obtain actual pressure, then freeze.

### 7.9 Query cohorts and distribution shift

Within each history define a fixed partition into head and tail entities, e.g., 20% head and 80% tail. The partition comes from observable entity/topic categories or a declared prior, not from hidden evidence flags.

Construct cohorts from the same query pool:

| Cohort | Sampling |
|---|---|
| uniform | Uniform across eligible candidate facts |
| skew_80_20 | 80% probability mass on head, 20% on tail |
| reversed_20_80 | 20% head, 80% tail |
| tail_only | Eligible tail facts only |
| historical | Questions about earlier valid states |
| composition | Held-out valid combinations of supported relations |

State whether the writer was told the training prior. The default tells it the general demand prior but never the actual questions. At test, keep that writer instruction fixed for the main demand-shift contrast. A separate informed-adaptation arm may change the prior instruction; do not combine it with the frozen-policy shift result.

To make comparisons paired, construct every cohort for the same underlying histories and writer snapshots. Use equal question counts per cohort. If fewer distinct eligible questions exist, evaluate all and record the count; do not fabricate independence by duplicate sampling.

~~~python
# Weighted query selection with independent RNG, no writer access.
import random

def draw_query_ids(question_ids, weights, n, seed):
    if len(question_ids) != len(weights):
        raise ValueError("mismatched query arrays")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid query weights")
    rng = random.Random(seed)
    # Training permits replacement: this is an unbiased Monte Carlo sample
    # from the specified query distribution. Keep multiplicities as weights.
    return rng.choices(question_ids, weights=weights, k=n)
~~~

For final evaluation, prefer a fixed stratified question sample with explicit population weights, or exhaustive evaluation for small worlds. Do not treat an unweighted mean of a stratified sample as the target distribution unless allocation matches that distribution.

### 7.10 Training question selection

For one training group:

1. Choose one public history and capacity.
2. Generate G writer trajectories with different action seeds and the same public observations.
3. Freeze all G snapshots.
4. Draw Q questions once using a query RNG independent of writer actions.
5. Evaluate those exact Q questions for every snapshot with the same reader.
6. Compute each trajectory's mean task score.

The delayed draw enforces a clean interface; the essential statistical requirement is that question sampling is independent of the writer's sampled actions. Never choose questions based on which facts a particular rollout retained.

Multiple hidden questions reduce the chance that training only rewards one salient fact. All variants compared as training methods must match Q or explicitly report the extra supervision/computation.

---

## 8. External benchmarks, development data, and leakage prevention

### 8.1 Primary external protocol

Use LongMemEval-S cleaned and all available LoCoMo10 conversations as final external tests. Do not train on them, inspect their scores during development, select prompts on them, or use their labels for checkpoint selection.

The released data/schema may be inspected to implement a correct loader. Loader debugging uses a tiny clearly designated fixture; if real benchmark examples are examined with model outputs for substantive tuning, remove those examples from a separately labeled evaluation and disclose the deviation.

Official repositories and dataset references are listed in Section 28. Pin their source commit/version. Dataset revisions matter.

### 8.2 LongMemEval mapping

VERIFY source files in xiaowu0162/longmemeval-cleaned. The cleaned S filename may differ from the historical longmemeval_s.json. Save the exact downloaded filename, dataset revision, and SHA-256.

Expected fields, verified against an actual record:

~~~text
question_id, question_type, question, answer, question_date,
haystack_session_ids, haystack_dates, haystack_sessions, answer_session_ids
~~~

Turns may include has_answer. Official documentation distinguishes turn-level and session-level evidence [R7].

Mapping requirements:

- One benchmark record -> one history with one official question.
- Preserve official chronology and timestamp semantics.
- Parse dates robustly; log missing/ambiguous dates.
- Preserve the original benchmark ID privately; assign public opaque IDs.
- Treat IDs ending in _abs as abstention only after verifying the pinned schema.
- has_answer labels belong only in the private bundle.
- If only session-level support is available, mark granularity=session.
- Do not assert that every turn in an evidence session is necessary evidence.
- If a support set is longer than the reader budget, report reference_infeasible rather than truncating it and calling the result an oracle ceiling.
- Keep official haystacks unmodified in final evaluation.

Use official per-type QA evaluation prompts with attribution and preserve their semantics. Replacing the official judge model with a local judge is a modified evaluation configuration; identify it in every table and avoid direct claims of identical leaderboard comparability.

### 8.3 LoCoMo mapping

VERIFY data/locomo10.json at the pinned snap-research/locomo commit.

Map sample_id and the complete conversation into one history. Keep all questions associated with that history, including category and available evidence IDs in the private bundle.

- Render speaker names consistently.
- Preserve timestamps and supplied image captions as text, with the text-only setting disclosed.
- Normalize evidence references and log unmatched IDs.
- Missing evidence annotations do not remove a question from QA scoring.
- Such questions are ineligible for unsupported evidence-level claims.
- Include adversarial/unanswerable questions in a separately reported stratum.
- Verify category meanings against the official release; do not infer them from third-party summaries.
- Snapshot memory once per writer/history/capacity/seed.
- Every question uses the same immutable snapshot and independent reader context.
- Do not let read counts, last-access metadata, or earlier answers modify subsequent question prompts.

Report per-conversation accuracy and a conversation-weighted mean in addition to the official/question-weighted metric. With only ten conversations, discuss the limited independent population coverage. A three-conversation test split is not the primary protocol.

### 8.4 Development data

The minimum reproducible setup needs no external conversational training set: use procedurally generated worlds with conversational renderers, separate books, and separate template families.

Optional conversational development data must have:

- A documented source/license and immutable version.
- Disjoint user/conversation IDs from final benchmarks.
- Near-duplicate checks against final benchmark text where legally/technically feasible.
- Public text and private labels separated.
- An explicit scorer aligned with the intended answers.
- A provenance statement describing annotation quality.

If development only uses synthetic data, external results measure transfer from that defined source. Do not call the transfer fully distribution-independent.

### 8.5 Optional in-domain adaptation protocol

If later training on LongMemEval or LoCoMo:

- Give the protocol a distinct name, dataset version, and result table.
- Split by connected source/provenance groups before any question-level stratification.
- Group exact and near-duplicate sessions, conversations, and question variants.
- Do not train on a held-out question's evidence embedded as a distractor.
- Keep test histories intact.
- Prefer dropping/resampling contaminated training histories over replacing individual sessions in a way that creates artificial chronology.
- Report realized independent group counts and class balance.
- If source connectivity creates a giant component, report it and redesign the split; do not pretend an arbitrary question split is independent.

Never combine adaptation results with untouched-benchmark transfer results.

### 8.6 Generalization labels

| Label | Required separation |
|---|---|
| Entity generalization | New underlying entity/value assignments |
| Template generalization | New rendering families |
| Length extrapolation | Longer histories with controlled information content and depth |
| Density generalization | More candidate facts at fixed history length |
| Task generalization | Held-out task/operation families, not only new surface wording |
| Query-distribution shift | Same histories/memories, changed question weights or cohorts |
| Dataset transfer | No target benchmark used for fitting or selection |
| Backbone replication | Independently trained/evaluated model under comparable protocol |
| Memory portability | Same saved memory evaluated with a different frozen reader |
| Adapter transfer | Only if the actual learned parameters are transferred compatibly |

---

## 9. Fixed observations, chunking, and token accounting

### 9.1 Observation chunks are data, not memory policy choices

Build chunks once per observation protocol and save their manifest. All arms use the same chunk IDs, text, order, and boundaries.

The primary reference cap is 1024 reference tokens, including the rendered speaker/timestamp/turn formatting. Additionally every selected backbone must fit the same chunk into its configured policy_chunk_cap.

Adding a new tokenizer that requires different chunks creates a new cross-model observation protocol. Rerun compared arms under that protocol or label the conditions unmatched.

### 9.2 Chunk construction

Pack complete turns within a session. Do not mix sessions solely to fill a chunk. Split long turns using text spans with both reference/policy token counts checked.

A robust implementation uses character-boundary binary search for the largest prefix that fits all configured tokenizer caps; a more efficient token-based proposal may be used if the decoded span round-trips exactly. Always preserve the actual original text bytes or a documented normalized text representation.

~~~text
for each session in chronology:
    current = []
    for each public turn:
        if rendered turn fits an empty chunk:
            append if whole rendered chunk fits all caps
            otherwise flush and start a new chunk
        else:
            flush
            split the turn into ordered exact text spans that each fit
            emit spans with (turn_id, start_char, end_char)
    flush
~~~

Evidence annotations stay attached to private source-character ranges. Do not copy the full evidence label to every fragment of a split turn.

Acceptance:

- Reconstructing source spans reproduces normalized session text exactly.
- No span is duplicated or omitted.
- All arms receive byte-identical observations.
- Changing C, max_entries, or value size leaves chunk hashes unchanged.
- Long-turn split counts and lengths are reported.

### 9.3 Storage fragments for raw baselines

A raw-memory baseline may divide one observation chunk into smaller storage entries. This is a storage transformation, not a new observation.

All fragments are exact substrings of the currently visible chunk. They have their own charged key/header overhead. Preserve range mappings for audits.

A scripted raw baseline does not incur model generation for fragmentation; report this computational advantage honestly. The learned writer may use a store_span tool if that arm enables it, but gets no ability to revisit past chunks.

### 9.4 Capacity accounting

The accounted persistent text size is:

~~~text
C_used = ref_token_count(serialize_memory(all_live_entries))
~~~

canonical_entry includes its visible ID, key, value, and any visible persistent timestamps or metadata. Revision 3 charges the exact complete concatenated serialization, including delimiters, under the pinned reference tokenizer. This explicitly replaces revision 2's additive per-entry convention. Log additive entry counts only as a diagnostic and never use them as the enforced total. Tokenizer merges across boundaries can change the result. Revalidate every proposed mutation, including deletion; do not assume token count is monotone under removal of text. All arms use the same accounting version.

Audit fields, source ranges, embeddings, and database overhead are not part of the semantic text budget unless visible. They are nevertheless included in physical-storage cost reports. No audit field may influence ordinary retention/retrieval actions.

Ordinary controllers may use the current stored text, its charged entry-ID ordering, the current public observation, and fixed public instructions. Any additional content-dependent persistent feature used by a heuristic, such as per-entity mention counts or last-seen dates, must be serialized into charged live-record fields. Do not keep an unbounded side dictionary of facts, aliases, counts, or deleted entries outside C. Retrieval embeddings may be deterministic functions of current stored entries only and must be removed when those entries are removed. Bookkeeping needed solely to reconstruct human audit traces is isolated from ordinary controllers.

### 9.5 Canonical serialization

~~~python
# memrl/memory/serialize.py
import json

VISIBLE_FIELDS = ("id", "key", "value")

def canonical_entry(entry):
    obj = {"id": entry.slot_id, "key": entry.key, "value": entry.value}
    # Temporal facts must be in key/value if the writer wants to retain them.
    # The primary environment does not attach free historical facts.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"

def reference_charge(entries, tokenizer):
    return len(tokenizer.encode(serialize_memory(entries), add_special_tokens=False))

def serialize_memory(entries):
    return "".join(canonical_entry(e) for e in sorted(entries, key=lambda e: e.slot_id))
~~~

The JSON encoding overhead is deliberately charged. Rolling summaries and fact stores export into this common representation; preserve their text exactly rather than expanding a compact store with evaluator knowledge.

Visible IDs are locally assigned entry IDs, never gold atom IDs. If an alternative arm exposes source dates or source IDs, serialize and charge them explicitly.

### 9.6 Exact prompt assembly

Render the final chat template to IDs and verify:

~~~python
def ensure_prompt_fits(prompt_ids, max_new_tokens, total):
    required = len(prompt_ids) + max_new_tokens
    if required > total:
        raise RuntimeError(f"prompt budget bug: {required} > {total}")
~~~

Per-section limits are checked before assembly. The revision-3 primary store rejects a mutation whose complete visible index exceeds its index region; it does not truncate or paginate the index. Pagination requires a separately versioned protocol with charged page state and notices. For loaded entries, include complete canonical records in requested order; report any entries not admitted. Never show a partial value as if it were complete.

The all-retained reader must fit the complete serialized snapshot. If a cross-tokenizer snapshot does not fit, label that configuration incompatible for the all-retained comparison. Do not silently switch it to retrieval.

### 9.7 Index/loaded region lifetimes

Writer prompts carry only:

- Current public chunk.
- Current memory index page.
- Values explicitly read in the current chunk.
- The immediately previous tool result within that chunk.
- Fixed public instructions and numeric memory status.

Clear loaded values and last_result on every chunk transition. No growing conversation transcript or hidden summary is carried forward.

Deleting/updating an entry invalidates or refreshes every loaded copy of that entry immediately. Otherwise the model could retain deleted content through a stale loaded-value cache. Such same-chunk caches are bounded by writer_budget.loaded and are still cleared at the next chunk.

---

## 10. Memory store, tools, parser, and environment

### 10.1 Entry schema

~~~python
from dataclasses import dataclass, field

@dataclass
class MemoryEntry:
    slot_id: int
    key: str
    value: str
    created_step: int
    updated_step: int
    # Private trace provenance only, never part of reader/store decisions.
    observed_source_ranges: list[tuple[str, int, int]] = field(default_factory=list)
    origin: str = "policy"

@dataclass(frozen=True)
class OpResult:
    ok: bool
    code: str
    message: str
    affected_ids: tuple[int, ...] = ()
~~~

IDs are monotonic within a history, never recycled, and their visible serialization is charged. A large ID can increase the charge. Hash the complete store after every successful mutation.

### 10.2 Primary tools

| Tool | Arguments | Semantics |
|---|---|---|
| memory_write | key, value | Add one entry if all caps permit |
| memory_update | slot_id, key?, value? | Atomically replace specified fields |
| memory_delete | slot_ids | Delete live entries; report missing IDs |
| memory_read | slot_ids | Load current values within writer loaded budget |
| memory_search | query, top_k | Search current memory only; return bounded charged/visible content |
| memory_list | page | Change index page |
| next_chunk | none | Advance ingestion |
| memory_store_span | start_char, end_char, key | Optional exact substring storage from current chunk |

The primary reader has no tools. Question visibility=question_first and tool-using readers are separately labeled extensions.

### 10.3 Mutation transaction

For write/update:

1. Validate arguments and UTF-8/string constraints.
2. Create a candidate entry without modifying live state.
3. Tokenize its key/value and complete canonical record.
4. Validate key/value caps, K, and total C.
5. If the primary reject policy cannot fit it, return MEMORY_FULL with numerical limits.
6. If an explicitly configured automatic eviction baseline is active, plan removals first.
7. Verify the complete proposed store fits.
8. Commit the operation and all removals atomically.
9. Refresh/invalidate loaded values and indexes.
10. Emit a complete event with before/after snapshot hashes and token charges.

Failure must leave the live store unchanged. An update cannot evict itself to create space. If eviction cannot produce a valid store, do not partially evict entries before returning an error.

### 10.4 Eviction policies

Primary learned policy: reject; the writer must delete, update, or compress.

Scripted controls:

- fifo: earliest creation first.
- random: uniformly selected current entries using a dedicated seeded RNG.
- deterministic utility heuristic: a preregistered score based on observable recency, redundancy, declared query prior, and explicit fact structure. No future question labels.
- lru: optional only when reads occur before subsequent writes.

For the default write-all-then-answer raw baseline, assert FIFO and LRU produce identical snapshots if neither has intervening reads. Do not spend a full evaluation budget on both equivalent arms.

### 10.5 Parser

Use one strict call per completion. Accept an optional short reasoning prefix, then exactly one tagged JSON object. Do not silently execute only the last of several calls.

~~~python
import json
import re

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

def parse_call(text):
    matches = list(CALL_RE.finditer(text))
    if len(matches) != 1:
        return None, "PARSE_ERROR"
    trailing = text[matches[0].end():].strip()
    if trailing:
        return None, "TRAILING_CONTENT"
    try:
        obj = json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return None, "BAD_JSON"
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None, "BAD_SCHEMA"
    if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        return None, "BAD_SCHEMA"
    return obj, None
~~~

Validate tool names, exact argument types, ranges, and phase after parsing. Reject booleans where integers are required. Do not repair quotes, infer missing gold values, or coerce arbitrary strings into lists in the primary protocol.

A permissive-parser sensitivity study is allowed if applied to every relevant arm.

### 10.6 Writer prompt

System instruction template, with exact rendered version stored in the run:

~~~text
You will read a history once, one part at a time. Later a separate assistant
will answer questions using only the memory you leave. You will not see the
actual questions while reading.

Memory limits: {C} accounted tokens, {K} entries, at most {key_cap} tokens
per key and {value_cap} tokens per value. The displayed entry format and IDs
also count toward memory capacity.

Preserve accurate facts, relationships, and dates. Questions may ask current
or earlier states, so preserve history when it is relevant to the stated task.
Use the task prior below; do not assume all questions ask only the newest value.

You can read/search only the current memory. You cannot revisit past history.
Use one tool call per response:
<tool_call>{"name":"...","arguments":{...}}</tool_call>

{tool_schemas}
Task prior: {public_task_prior}
~~~

Public user message:

~~~text
Memory status: {used}/{C} tokens; {n_entries}/{K} entries
Current index page: {page}
{complete_index_rows}

Loaded current values:
{loaded_canonical_entries}

Previous tool result:
{bounded_last_result}

Current history part:
{current_public_chunk}
~~~

The actual question, final answer, hidden support annotations, query cohort ID, and private IDs are absent.

### 10.7 Writer state machine

~~~text
START
  empty memory, loaded region, last_result
  show chunk 0
INGEST
  render prompt -> sample completion -> parse/execute/log
  every completion consumes one call, including errors
  next_chunk or call cap -> clear ephemeral regions -> next chunk
  final chunk completed -> freeze canonical snapshot -> DONE
DONE
  no further writer mutation; hand immutable snapshot ID to evaluator
~~~

At the call cap, advance automatically and record AUTO_ADVANCE. Do not insert a synthetic model action or assign loss to that environment transition.

If the writer never stores anything, it still produces a valid empty snapshot and receives the reader's resulting score.

### 10.8 Minimal tool ablation

Minimal tools: write, delete, read, list, next_chunk. Full tools add update and search. Keep framing and observation caps fixed. Retrain for a tool-interface comparison if making a claim about learned capability; simply removing tools at evaluation measures robustness to an interface change.

### 10.9 Store tests

Required examples include:

- Failed writes and updates leave byte-identical stores.
- Every successful operation satisfies C, K, key and value limits.
- Overwriting a value can remove previously retained information.
- Current and historical values can coexist if the writer stores them.
- Deletion refreshes all index/loaded caches.
- No deleted entry can be retrieved by a stale embedding.
- Memory search cannot reach the public history archive.
- Exact same input/action sequence gives the same memory hash.
- 5,000 randomized operation sequences preserve all invariants.
- Changes to hidden evaluation labels cannot change tool results.

---

## 11. Frozen reader and retrieval protocols

### 11.1 Primary: complete retained-memory view

For primary C <= 4096, pass the complete canonical memory snapshot to the frozen reader within the 6144-token memory region. Validate the exact rendered input.

This deliberately removes retrieval selection as a source of failure. Every retained visible key/value is available to the reader. Consequently, this setting cannot support a claim that retrieval caused an error.

Reader request fields:

~~~text
SYSTEM
Answer using the provided memory. If the required information is absent or
insufficient, return {"answer": null}. Do not infer a personal/random fact from
general knowledge. Follow the required answer type exactly.

USER
Question: {question_text}
Required answer type: {answer_type}
Question date: {question_time_or_none}
Memory:
{complete_canonical_snapshot}

Return only one JSON object: {"answer": ...}
~~~

For official conversational tasks, use a separately versioned natural-language response prompt compatible with the benchmark scorer. All arms in that table use it.

Freeze model revision, chat template, decoding, prompt, answer cap, and memory order. Use temperature=0. Do not assume output is bitwise invariant across arbitrary hardware/batching; test and log reproducibility.

### 11.2 Secondary: retrieval stress protocol

To test retrieval, choose C in {4096,8192,16384} and reader memory budgets in {1024,2048}. Keep writer observation protocol unchanged. Train matched-capacity writers only for claims about that capacity; transfer-only evaluations remain labeled.

Fixed retrieval pipeline:

1. Index canonical keys and values from the frozen snapshot.
2. Retrieve candidates with BM25 or a pinned hybrid retriever.
3. Use deterministic tie-breaking by slot_id.
4. Fill the reader budget with complete canonical entries in retrieval order.
5. Charge headers and delimiters.
6. If the next entry does not fit, continue checking later candidates under a declared skip-large policy, or stop under a declared prefix policy. Freeze one policy and apply it everywhere.
7. Save selected IDs, excluded IDs, scores, and exact reader text.

Default: budgeted scan in rank order, skip entries that do not fit, continue until all candidates have been considered. This avoids treating one large entry as a reason to discard every smaller lower-ranked entry. It is a protocol choice, not a guarantee of optimal retrieval.

### 11.3 Dense/hybrid implementation

BM25 over lowercase word tokens is the first reproducible baseline.

If dense retrieval is enabled, use a pinned model and explicit query instructions from its model card. Do not silently embed only the first 512 tokens of a long entry.

Split entries into overlapping embedder-token windows, e.g., length 384 and stride 256 after reserving model special tokens. Embed every window. Score an entry by the maximum window-query cosine similarity, then use reciprocal rank fusion with BM25 for a hybrid arm.

Log vectors, index bytes, embedding time, and query time as physical/computational costs. Maintain immutable evaluation indexes keyed by memory_id and embedder revision.

### 11.4 Keys, snippets, and auxiliary visibility

All-retained readers receive keys as part of the canonical snapshot. There is no separate uncharged index.

If a tool-reader extension is enabled, log every key, search snippet, tool error, timestamp, and loaded value actually visible at each call. A snippet containing an answer counts as exposed information even if no READ occurred.

Reader inputs for different questions must not share mutable tool history, read-count fields, scratchpads, or previous answers.

### 11.5 Memory order sensitivity

Use creation-order serialization for the primary all-retained view. On a prespecified diagnostic subset, rerun reverse order and one seeded shuffled order with the same memory and reader budget.

If the claimed memory improvement disappears under reasonable order changes, report the interaction. Do not select the best memory order separately for each method on test data.

## 12. Baselines, attribution arms, and fair comparisons

### 12.1 Arm registry

Every arm exports the same snapshot format or explicitly declares why it is a separate resource regime.

| Arm | Writer/memory construction | Reader | Purpose |
|---|---|---|---|
| B00_empty | Empty store | R0 | Guessing/parametric-answer control |
| B01_recent_context | Most recent raw history that fits the reader input | R0 | Truncation reference; no ingestion model |
| B02_full_context | Complete raw history if it fits native/context policy | R0 or labeled long-context reader | Separate resource reference |
| B03_raw_fifo | Exact raw storage fragments with FIFO | R0 | Simple bounded retention |
| B04_raw_random | Same fragments with random eviction | R0 | Recency-free bounded retention |
| B05_rolling_summary | Incrementally rewrite a compact summary | R0 | Strong text compression baseline |
| B05b_summary_bank | Fixed-size summary blocks with deterministic merging | R0 | Strong summary control when full rewrites do not fit |
| B06_structured_facts | Extract/update a compact fact/event store | R0 | Strong structured heuristic |
| B07_prompted_tools | Untrained/common-format writer W0 | R0 | Same tools, no task RL |
| B08_format_only | Format SFT initialization without task RL | R0 | Tool-output learning control, if separate from W0 |
| B09_rft | Rejection-sampling fine-tuned writer from training-only successful trajectories | R0 | Supervised outcome-filtered baseline |
| B10_prior_memory | At least one close implemented learned-memory method | R0 when compatible | Direct prior-work comparison |
| M11_writer_rl | Writer-only RL, W1 | R0 | Primary learned-memory result |
| A12_reader_only | Fixed W0 memory, trained reader R1 | R1 | Answer-generation attribution |
| A13_crossed_writer | Frozen W1 memory, same R1 as A12 | R1 | Crossed writer/reader attribution |
| A14_joint | Jointly trained writer/reader | Joint reader | Optional integrated-system result |
| P15_full_archive_rag | Full raw history retained and indexed | R0 | Practical, larger-storage reference |
| P16_gold_support | Gold sufficient support only if it fits | R0 | Privileged reader-competence reference |

B01 and B02 are not bounded-write-memory methods; their storage/ingestion settings must be visible in tables. B02 is skipped only for declared context incompatibility, not because it scored poorly.

### 12.2 Rolling summary implementation

Maintain one summary string. For each fixed observation chunk:

1. Prompt the baseline model with previous summary and current chunk.
2. Ask for an updated factual summary preserving entity bindings, current state, relevant history, and uncertainty.
3. Use the same task-prior information given to the RL writer.
4. Require output within a declared token cap.
5. Split the summary into storage records if necessary, charging keys and JSON overhead.
6. If it exceeds capacity, allow a single explicit compression pass under the same per-chunk call budget. If still oversized, mark the action failed and keep the previous valid summary; do not truncate it secretly.
7. Log all prefill/generation tokens.

The baseline should use the same base model or a declared stronger model sensitivity. Tune its prompt and summary allocation on development data. Do not leave it with an obviously generic or incapable prompt.

A single-entry summary may have a larger per-entry cap than the main slot interface. This is an architectural baseline under matched total C, not a same-tool ablation. Also include a common-slot-cap version if that distinction changes the conclusion.

**Generation-budget feasibility is mandatory.** Do not force a whole-store rewrite to use a 768-token completion when the intended summary has 2048 or 4096 tokens and then call it a strong baseline. Before evaluation, compute whether old summary + current chunk + instructions + complete new summary fit the per-call context. A full rewrite may be infeasible at large C under an 8192-token context. Report its realized summary size and include B05b below; do not conceal a smaller effective capacity.

For B05b_summary_bank, implement this fixed procedure:

1. Summarize the current observation into one or more records of at most 256 value tokens, keeping entity bindings and relevant temporal qualifiers.
2. Append proposed records if they fit C and K.
3. If capacity is insufficient, combine the two oldest summary blocks in one compression call. Both source blocks and the current chunk must fit the prompt; otherwise compress the blocks without the chunk, which remains available for the next call in this observation.
4. Require the replacement to be smaller by at least the amount needed, or report compression_failed and leave the store unchanged.
5. Allow at most three generation calls per observation and at most 3 * 768 generated tokens in the matched-compute profile. Every call also fits the 8192-token context.
6. If no feasible append/merge remains, apply a declared deterministic oldest-block removal rule, then append the valid new block. Log that information was discarded.
7. Use no gold labels or future realized questions to choose blocks or rewrite text.

The baseline prompt must explicitly preserve uncertainty, historical validity, and entity relationships. Tune the compression instruction and a small set of block-size options on development data. Include the stronger locked summary variant in the main baseline comparison. A relaxed-compute summary profile may additionally use larger generation allowances, but belongs on the cost frontier.

### 12.3 Structured fact baseline

Use the base model to extract candidate records from the current chunk and a bounded view of the existing store.

Schema:

~~~json
{
  "entity": "name or local ID",
  "relation": "attribute or event type",
  "value": "text",
  "valid_from": "time or null",
  "valid_to": "time or null",
  "supersedes": "existing local record ID or null"
}
~~~

Implementation details:

- Normalize relation names using a fixed development-defined mapping.
- Resolve entity aliases only using observed text.
- Keep historical records if the task family includes historical questions.
- Do not equate “superseded” with “irrelevant.”
- Deduplicate exact semantic records when normalization makes this certain.
- Use a deterministic retention score chosen on development data.
- Candidate score features can include declared prior weight, observable recency, repetition, and redundancy. Gold support or future realized query membership is prohibited.
- Charge the full rendered record, including temporal fields.
- If retention uses mention counts or last-seen metadata, include those values in the charged serialized record; do not retain them in an unbounded hidden history table.
- Enforce capacity transactionally.
- Report extraction errors separately from retention decisions.

Suggested deterministic score form:

~~~text
score(entry) =
    w_prior * log(1 + declared_expected_query_weight)
  + w_recency * exp(-age / tau)
  + w_frequency * log(1 + observed_mentions)
  - w_redundancy * max_similarity_to_another_entry
~~~

Tune a small fixed grid on development data, with the same development-access budget disclosed for RL and baseline methods. Retention by score-per-token is a reasonable declared heuristic; it is not optimal under multi-fact/compositional questions.

### 12.4 Rejection fine-tuning baseline

Generate writer trajectories from W0 on training histories with the same Q-query frozen-reader evaluation. Keep trajectories meeting a development-fixed task-score threshold, e.g., >=0.75, not necessarily perfect histories under impossible capacity.

Train the writer on its own accepted tool completions only. Do not include reader answers. Match or report:

- Number of sampled trajectories.
- Number of frozen-reader question evaluations.
- Generated writer tokens.
- Gradient-update tokens.
- Total GPU hours.

An RFT baseline given a larger teacher must be labeled teacher-assisted. A fair principal RFT control uses the same W0.

### 12.5 Close prior method requirement

Implement at least one of Memory-R1 or Mem-α using the official code/checkpoint when available [R1,R3]. Create docs/PRIOR_WORK_MATRIX.md before choosing.

Required columns:

~~~text
method, paper_version, code_commit, checkpoint_revision, training_data_disclosure,
writer_model, reader_model, question_visibility, persistent_budget,
retrieval_budget, tool_interface, changes_for_our_protocol, reproduction_status
~~~

There are two different comparisons:

1. **Faithful original configuration:** reproduce a documented setting sufficiently to validate the integration; report original resource requirements.
2. **Budget-constrained adaptation:** adapt its persistent memory to the common C and reader view, describing every change.

Never claim that a heavily changed wrapper is an exact reproduction. Do not compare numbers copied from papers to your reruns under a different reader, judge, budget, or dataset subset.

If a public checkpoint trained on the final external benchmark, label that provenance. It may remain a practical comparator but cannot be presented as satisfying the project's untouched-target training protocol.

The adapter interface should be:

~~~python
class MemoryConstructor:
    def initialize(self, public_task_prior, capacity, seed): ...
    def observe(self, public_chunk): ...
    def finish(self): ...  # returns immutable exported text + complete cost record
~~~

An adapter cannot receive EvalQuestion objects. Exported memory conversion must not use gold labels or add information not present in the method's memory.

### 12.6 Crossed writer/reader experiment

Use a common set of saved W0 and W1 snapshots for the same histories.

Train R1 on training-only frozen memories generated by W0, with a reader-only objective. For a clean crossed experiment, reuse exactly that R1 for both W0 and W1 memories. Also report whether an alternative R1 trained on a balanced mixture of W0/W1 memories changes results.

Primary 2x2:

~~~text
(W0,R0), (W1,R0), (W0,R1), (W1,R1)
~~~

Keep reader prompts and retrieval identical. Differences have these interpretations:

~~~text
writer_effect_at_R0 = score(W1,R0) - score(W0,R0)
reader_effect_at_W0 = score(W0,R1) - score(W0,R0)
interaction = score(W1,R1) - score(W1,R0) - score(W0,R1) + score(W0,R0)
~~~

These are controlled component comparisons, not universal mediation estimates. They depend on the selected writers/readers and training distributions.

**Reader-only training recipe for R1:** build a fixed training-only W0 snapshot bank before reader training, including the same capacity/task distribution used for writer training. A reader training group is one (snapshot, hidden question) pair with four independently sampled answer completions. Use the same strict task scorer, leave-one-out baseline, one on-policy update, completion-only loss, and fixed normalization convention as Section 14. Initialize a new reader LoRA from R0; do not initialize it from the learned writer adapter. The writer and its snapshots are never updated. An initial setting of 32 question groups * 4 answer samples gives 128 answer generations per reader-training iteration, matching the reference writer iteration's number of frozen-reader answer generations, while total compute is still different and must be reported. Use training-only questions, reader-development selection, and 200 planned updates subject to the pilot. Both GPUs 2–3 now generate reader-policy samples with the current R1 adapter; GPUs 0–1 train that reader adapter. Preserve the R0 base reader for the crossed evaluation.

If all answer samples are wrong because the fixed memory lacks the required random fact, the group has zero task gradient. Log this and do not fabricate a gold-containing memory to make the control train. A separate support-available reader-training sensitivity can be reported with its eligibility rule and selection bias disclosed.

Joint training A14 is separate; it does not replace the 2x2 because its components co-adapt. Do not infer writer-only gains by subtracting unrelated end-to-end systems.

### 12.7 Resource matching

Maintain two comparison tables:

- **Matched information resources:** same C, observation chunks, reader, reader context, and question visibility; record differing construction compute.
- **Compute frontier:** vary allowed writer calls/generation budgets and report accuracy versus measured total tokens and GPU time.

A scripted extractor or raw FIFO baseline need not waste LLM calls to match RL. Its lower compute is part of the result. Do not describe a higher-accuracy, much-higher-cost system as uniformly superior.

---

## 13. Initialization and a shared format-learning control

### 13.1 Why format initialization is separate

A small instruct model may fail the strict tool parser before it can explore retention. Task RL could then appear to improve memory merely by learning valid JSON/tool names.

Evaluate raw-base tool validity on a held-out development fixture. If the failure rate exceeds a preregistered threshold of 5%, train a small formatting adapter and use it as the common W0 initialization for tool-based learned/control arms. Report raw-base B07 as an additional reference.

Choose this branch once on development data, before final tests. Save the decision and the same W0 adapter hash in every relevant run.

### 13.2 Formatting data

Generate 1,000–3,000 short examples covering:

- Each tool and valid argument type.
- Reading and updating existing IDs.
- Correct responses to MEMORY_FULL, NO_SUCH_SLOT, and oversize errors.
- next_chunk.
- Strict tagged JSON without trailing calls.
- Short values and escaped characters.

Use random synthetic strings and tiny stores unrelated to final benchmark histories. No long-horizon task reward, future query relevance, or gold retention policy is supplied.

This teaches interface syntax and local preconditions; it is not claimed to teach an optimal memory policy.

### 13.3 Format SFT

- Base: same writer backbone.
- LoRA: same target modules/rank as RL, unless a separately documented composition is used.
- Loss: teacher-forced cross entropy on completion tokens only.
- One epoch initially; early stop on formatting validation.
- Learning rate initial value 1e-5.
- Maximum 2048 total tokens per formatting example unless a longer example is necessary.
- Save W0 and a report of per-tool validity.

Initialize task RL by loading W0 as the trainable adapter; do not stack an untracked second adapter. The frozen reader R0 remains the base model without W0.

### 13.4 Initialization identity

Add an initialization block to resolved configuration:

~~~yaml
initialization:
  writer_adapter: null  # set to the exact W0 path after the format gate
  writer_adapter_sha256: null
  format_training_manifest: null
  raw_base_format_error_rate: null
  use_shared_format_initialization: false
~~~

All checkpoint manifests name the base and initialization revision. A0/base-reader requests cannot infer their adapter from a global “latest adapter” setting.

---

## 14. Writer-only RL objective and estimator

This section defines the K=1 complete-trajectory baseline. The continuation study uses the branch-aware extension and explicit prefix/suffix weights in Section 31. Both use the same on-policy score-function objective; do not apply the K=1 denominator to flattened K-future groups.

### 14.1 Reward

For each writer trajectory i, evaluate Q common hidden questions using the frozen reader:

~~~text
R_i_task = (1/Q) * sum_q score(reader(snapshot_i, q), gold_q)
R_i = R_i_task
~~~

The primary reward has no call penalty, format penalty, or evidence-survival shaping. Hard call, generation, and memory budgets provide resource limits. This prevents a group of uniformly incorrect rollouts from generating a large normalized learning signal solely from tiny differences in call penalties.

Optional cost-regularized reward, in a separate experiment:

~~~text
R_i = R_i_task - lambda_token * writer_generated_tokens_i / writer_token_allowance
~~~

Use a fixed allowance computed from the common observation/call budget. Report the full trade-off and tune lambda on development data. Do not normalize its advantages by a small empirical standard deviation.

If the query sample is drawn directly from p, the sample mean estimates expected accuracy under p. If stratified sampling uses proposal p_tilde, use declared importance weights p/p_tilde with common support; record effective sample size and avoid unstable tail weighting. The reference implementation uses direct draws or exact cohort weights.

### 14.2 Group construction

Each iteration contains H distinct sampled histories/capacity/task conditions. For each group h, sample G writer trajectories with independent writer-action RNG.

Reference H=4, G=4, Q=8. This creates 16 writer trajectories and 128 frozen-reader answer requests per iteration before caching.

Reuse identical questions across the G trajectories. This reduces comparison noise while maintaining independence from each sampled writer action.

### 14.3 Leave-one-out baseline

~~~python
import numpy as np

def rloo_advantages(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("expected >=2 finite rewards")
    return r - (r.sum() - r) / (len(r) - 1)
~~~

Every completion token of every writer call in trajectory i receives A_i. Prompts, tool outputs, environment transitions, and frozen-reader completions receive no loss.

A_i is not divided by the group standard deviation. There is no clipping objective in the reference estimator.

If all task rewards in a group are equal, the advantages are zero. Omitting its zero-gradient calls saves computation; the normalization still refers to the original H*G trajectories.

### 14.4 Objective and normalization

Let N=H*G be the number of sampled trajectories. Let J_i denote writer calls in trajectory i. The full sampled loss is:

~~~text
L_full = -1/(N * Z) *
         sum_i A_i *
         sum_{j in J_i} sum_t log pi_theta(y_ijt | prompt_ij, y_ij,<t)
~~~

Z is a fixed positive scale, reference 256. It does not depend on sampled action lengths, rewards, or selected-call counts. Scaling by Z changes gradient magnitude, not the optimum. Learning-rate/gradient-clipping behavior still depends on that scale, so freeze and report it.

This estimates the gradient of expected trajectory reward, up to the fixed scale and numerical approximation between inference/training kernels. It is not a per-call or per-token mean objective. Do not silently divide by each trajectory's token count.

### 14.5 Uniform call subsampling with correct inclusion weights

Long trajectories make full backpropagation expensive. Let M be the count of eligible nonzero-advantage writer calls across the complete iteration, and choose m=min(M, configured_cap) calls uniformly without replacement. Every eligible call has inclusion probability p=m/M.

Use:

~~~text
L_sample = -1/(N*Z) *
           sum_{selected calls j} (1/p_j) * A_owner(j) * sum_t log pi_theta(...)
~~~

Conditional on the sampled trajectories and advantages, this is an unbiased estimator of L_full and its unclipped gradient. The denominator is the known N*Z, not the random number of selected completion tokens. Subsequent norm clipping and adaptive optimization are nonlinear operations; do not claim their final parameter update is an unbiased full-batch update.

~~~python
from dataclasses import dataclass
import random

@dataclass
class CallSample:
    rollout_id: str
    group_id: str
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float
    behavior_logprobs: list[float]
    adapter_sha256: str
    inclusion_probability: float = 1.0

def select_calls(calls, cap, seed):
    eligible = [c for c in calls
                if c.completion_ids and c.advantage != 0.0]
    if not eligible:
        return []
    m = min(cap, len(eligible))
    chosen = random.Random(seed).sample(eligible, m)
    p = m / len(eligible)
    return [
        CallSample(**{**c.__dict__, "inclusion_probability": p})
        for c in chosen
    ]
~~~

If stratifying by early/middle/late chunks, sample within every nonempty stratum with m_h>0 and weight by p_h=m_h/M_h. Never force selection of “important-looking” calls based on gold or observed loss without accounting for the changed inclusion probabilities.

A short-history full-gradient versus repeated-subsample experiment is a required estimator test. Compare both gradient expectation and variance, not one random subsample.

### 14.6 On-policy requirements

The reference estimator takes exactly one optimizer step after collecting the current iteration's trajectories.

- Rollout and trainer weights must have the same base and adapter hashes.
- Temperature is 1, top_p=1, unrestricted top_k, min_p=0, repetition penalty=1, frequency/presence penalties=0.
- No constrained JSON decoder or token masks in the reference training policy.
- Disable generation-config defaults that would silently alter sampling.
- LoRA dropout is zero. Verify the model's other dropout paths are zero for the chosen training mode.
- Every generated token ID, including closing tool tags and relevant stop/EOS tokens, is retained with its sampled log probability.
- Never decode and re-tokenize completions for training.
- Do not perform a second gradient step on the same rollout batch.

If adding top-p, temperature other than 1, constraints, or multiple optimizer epochs, derive and implement the behavior-policy likelihood and any necessary importance correction. This is outside the initial reference implementation.

### 14.7 Completion log probabilities

~~~python
import torch

def completion_logprobs(model, prompt_ids, completion_ids, device,
                        supports_logits_to_keep=True):
    if not prompt_ids or not completion_ids:
        raise ValueError("nonempty prompt and completion required")
    ids = torch.tensor([prompt_ids + completion_ids],
                       dtype=torch.long, device=device)
    k = len(completion_ids)
    if supports_logits_to_keep:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=k + 1)
        logits = out.logits[:, :-1, :].float()
    else:
        out = model(input_ids=ids, use_cache=False)
        # Logits at positions P-1 ... P+K-2 predict the completion tokens.
        p = len(prompt_ids)
        logits = out.logits[:, p - 1:p + k - 1, :].float()
    targets = ids[:, -k:]
    return torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(0).squeeze(-1)
~~~

VERIFY the installed model forwards logits_to_keep through DDP. Some models use a different keyword or do not support this optimization. The fallback is correct but can use much more memory.

The code performs the log-softmax in fp32. Do not compute prompt-token losses.

### 14.8 DDP loss scaling

PyTorch DDP averages gradients across W ranks. For each selected call:

~~~python
def writer_call_loss(logp, sample, n_trajectories, fixed_scale, world_size):
    if not (0 < sample.inclusion_probability <= 1):
        raise ValueError("invalid sampling probability")
    return (
        -sample.advantage
        * logp.sum()
        * world_size
        / (sample.inclusion_probability * n_trajectories * fixed_scale)
    )
~~~

After DDP's average, accumulated gradients equal the globally weighted sampled gradient.

Ensure every rank executes the same number of backward calls. If the selected-call count is odd, choose a deterministic even count before sampling, recompute inclusion probabilities using that actual count, and keep all ranks synchronized. If only one eligible call exists, duplicate it onto both ranks with half weight per copy or use a tested single-active-rank scheme; the initial implementation should use explicit weighted duplication and record it.

Balance sequence lengths between ranks without changing sample weights. Use no_sync for all but the last backward call on each rank.

### 14.9 Trainer skeleton

~~~python
import contextlib
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

def build_train_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.writer,
        revision=cfg.model.writer_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attention,
    )
    if cfg.initialization.writer_adapter:
        model = PeftModel.from_pretrained(
            model, cfg.initialization.writer_adapter, is_trainable=True
        )
    else:
        model = get_peft_model(model, LoraConfig(
            r=cfg.rl.lora.rank,
            lora_alpha=cfg.rl.lora.alpha,
            lora_dropout=0.0,
            target_modules=list(cfg.rl.lora.target_modules),
            task_type="CAUSAL_LM",
        ))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model

def train_step(ddp, optimizer, samples, n_trajectories, fixed_scale,
               device, clip_norm, sample_weight_overrides=None):
    ddp.train()
    # Registry/probe must verify stochastic dropout is disabled.
    optimizer.zero_grad(set_to_none=True)
    W = dist.get_world_size()
    for index, sample in enumerate(samples):
        sync_context = (
            contextlib.nullcontext()
            if index == len(samples) - 1 else ddp.no_sync()
        )
        with sync_context:
            logp = completion_logprobs(
                ddp, sample.prompt_ids, sample.completion_ids, device
            )
            loss = writer_call_loss(
                logp, sample, n_trajectories, fixed_scale, W
            )
            if sample_weight_overrides is not None:
                loss = loss * sample_weight_overrides[index]
            loss.backward()
    params = [p for p in ddp.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("nonfinite gradient")
    optimizer.step()
    return float(grad_norm)
~~~

The orchestration layer checks empty batches and skips optimizer.step on all ranks when every advantage is zero. Do not step Adam merely to advance its state in an empty iteration.

Call ddp(...) for trainable forwards so DDP hooks run. The initialization's gradient-checkpointing/dropout behavior is model-version-dependent and must be verified.

### 14.10 Why the old optimizer sweep is removed

At fixed group size, RLOO and unnormalized mean-centered advantages are proportional. Under one on-policy step, an importance ratio set to exp(logp-logp.detach()) is one and clipping is inactive.

This document therefore names and tests the actual estimator. An optimizer paper would need separately justified algorithmic differences, learning-rate tuning, comparable training budgets, and multiple update regimes. That is not a primary contribution here.

---

## 15. Rollout workers, frozen scoring, adapters, and checkpoints

### 15.1 Worker ownership

Each rollout worker owns:

- One vLLM engine.
- Current writer adapter identity and bounded adapter cache.
- Independent writer environments.
- Immutable snapshot serialization.
- A local deterministic reader request queue.
- No optimizer.
- No policy-accessible private evaluation labels.

The trusted scoring component attaches answers after reader generation. It may share a process initially if public/private interfaces are enforced and tested; separate subprocesses are preferred once the pipeline works.

### 15.2 Engine sketch

~~~python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def make_engine(cfg):
    return LLM(
        model=cfg.model.writer,
        revision=cfg.model.writer_revision,
        dtype="bfloat16",
        max_model_len=cfg.writer_budget.total,
        gpu_memory_utilization=cfg.hardware.rollout_gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=cfg.rl.lora.rank,
        max_loras=2,
        enable_prefix_caching=True,
        seed=cfg.run.seed,
        # VERIFY the pinned version's generation_config override.
    )

def writer_sampling(seed, max_tokens):
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, presence_penalty=0.0,
        frequency_penalty=0.0,
        max_tokens=max_tokens,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
        logprobs=1,
        seed=seed,
    )

def reader_sampling(seed, max_tokens):
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
    )
~~~

Verify top_k=-1 and generation-config semantics in the installed vLLM version. If the API differs, adapt through memrl/models/compat.py and save the effective sampling settings.

Training writer calls use a LoRARequest naming the current writer adapter. Frozen R0 calls explicitly pass lora_request=None. Never rely on a previous request's adapter state.

### 15.3 Lockstep writer loop

~~~text
while unfinished writer environments remain:
    choose at most max_active_writer_sequences_per_worker
    build exact prompt token IDs
    build per-environment sampling seeds
    generate using the same current adapter
    save prompt IDs, completion IDs, aligned behavior logprobs
    parse and execute one action in each environment
    finalize and hash completed memories
    admit waiting environments
~~~

After all G snapshots for a group exist, request the shared Q questions and score the GxQ reader outputs. Batching reader requests with identical prompts is allowed; caching must preserve semantic identity and decoding settings.

Store scripted baseline actions with origin=script and exclude them from RL samples.

### 15.4 Log-probability and adapter parity tests

Before task RL:

1. Use a tiny formatting adapter with an easily verified output change.
2. Load it into vLLM and confirm changed behavior.
3. Immediately issue a base-model reader request and confirm it matches the no-adapter reference.
4. Save a second adapter and verify no stale adapter is served.
5. Compare trainer and vLLM sampled-token log probabilities on exact identical tokens for base and adapted models.
6. Record mean absolute difference, p99 absolute difference, maximum difference, and per-position differences.
7. Initial gates: MAE <=0.02 nats and p99 <=0.10 nats on a development probe, with no systematic adapter or temperature mismatch. These are engineering thresholds, not a mathematical proof of identical policies.
8. If thresholds fail, debug model revision, token IDs, sampling transforms, dropout, dtype, attention backend, and adapter identity before training.

A signed mean difference can cancel errors and is insufficient. Do not loosen thresholds because a run already looks successful.

### 15.5 Job protocol

~~~text
runs/<execution_id>/queue/iter_00020/
  writer_job_0.json
  writer_job_1.json
  worker_0_results.jsonl.zst
  worker_1_results.jsonl.zst
  worker_0_DONE.json
  worker_1_DONE.json
  train_batch.pt
  train_batch_READY.json
~~~

Each job contains:

~~~json
{
  "job_schema": 2,
  "execution_id": "...",
  "iteration": 20,
  "kind": "writer_rollout_and_frozen_reader",
  "writer_adapter_path": "...",
  "writer_adapter_sha256": "...",
  "writer_adapter_version": 21,
  "reader_base_revision": "...",
  "reader_adapter": null,
  "public_history_manifest": "...",
  "group_specs": [],
  "query_selection_seed": 123,
  "expected_protocol_id": "..."
}
~~~

Workers validate hashes before generation. DONE files include output hashes, counts, and status. A DONE marker without a matching validated result is not completion.

Heartbeat every 30 seconds from a separate thread/process, with phase, completed environments, and last progress timestamp. Startup/model-load timeouts and inference timeouts are separate. Handle a stalled worker by terminating the affected job and rerunning its exact manifest; do not drop its histories.

### 15.6 Atomic files and directories

~~~python
import json
import os
import tempfile
from pathlib import Path

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                               dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
~~~

For adapters/checkpoints, write a new uniquely named temporary directory, verify expected files/checksums, write its manifest last, then rename to a previously nonexistent final directory. Never overwrite an existing nonempty checkpoint directory.

### 15.7 Adapter lifecycle

- Version IDs are unique positive integers within an engine lifetime.
- Every version points to immutable saved weights.
- Limit resident adapters; use the supported unload/cache API or restart a worker between jobs if necessary.
- Never reuse a version ID for different weights.
- Reader adapter identity is specified per request.
- Prefix-cache correctness across adapters is tested; performance can change after a swap.
- Record writer/reader adapter IDs in every request log.

### 15.8 Iteration orchestration

~~~text
rank 0:
  sample H histories from train only
  construct G trajectories per group
  save jobs with current adapter hash
all trainer ranks:
  wait on READY files using CPU filesystem polling, not a long NCCL collective
rank 0:
  collect all writer snapshots and frozen-reader scores
  compute per-trajectory rewards and RLOO advantages
  build eligible writer CallSamples
  sample calls with recorded inclusion probabilities
  shard equal backward-call counts across ranks
  publish trusted train_batch.pt and READY
all trainer ranks:
  quick barrier after READY
  validate common batch/adaptor identities
  accumulate weighted gradients
  one optimizer step, or common skip if zero gradient batch
rank 0:
  publish new adapter
  log metrics and enqueue development validation when due
all ranks:
  short final barrier and next iteration
~~~

Only load internally generated trusted batch/checkpoint files. Prefer dictionaries/tensors compatible with safe torch loading rather than arbitrary pickled class instances.

### 15.9 Checkpoint/resume

Save:

- Base model/tokenizer revisions and adapter.
- Optimizer and scheduler state.
- Completed optimizer-step count and sampled-iteration count separately.
- Data sampler position, curriculum state, and all RNG states.
- Query sampler and call-sampler state.
- Per-rank torch/CUDA RNG states.
- Last complete job IDs and resource ledger.
- Protocol/execution IDs and configuration hashes.

Resume from the newest complete manifest. Reuse completed immutable rollout results only if adapter, seeds, and protocol hashes match. Incomplete jobs are rerun; partially applied optimizer steps are never guessed.

Test that interrupted and uninterrupted runs select the same histories, query IDs, and writer seeds. Bitwise identical floating-point results are a stronger separate property and must not be promised without measurement.

---

## 16. Reward scoring, reader calibration, and training health

### 16.1 Typed synthetic answers

Use strict JSON parsing and task-specific validators:

| Answer type | Accepted value | Correctness |
|---|---|---|
| code | String in the generated code alphabet | Exact normalized code equality |
| entity | String or approved alias | Exact canonical ID match |
| integer | JSON integer, excluding bool | Exact equality |
| set | JSON list with no duplicate canonical members | Set equality |
| boolean | JSON bool | Exact equality |
| unknown | null | Correct only when gold answerable=false |

Do not strip arbitrary digits/punctuation from codes. Do not accept a substring of a longer answer. A prediction containing both a correct and contradictory value is invalid.

~~~python
import json

def parse_json_answer(text):
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(obj, dict) or set(obj) != {"answer"}:
        return None, False
    return obj["answer"], True

def score_code(text, gold, answerable=True):
    value, valid = parse_json_answer(text)
    if not valid:
        return 0.0
    if not answerable:
        return float(value is None)
    if not isinstance(value, str):
        return 0.0
    return float(value.strip().lower() == str(gold).strip().lower())

def score_integer(text, gold, answerable=True):
    value, valid = parse_json_answer(text)
    if not valid:
        return 0.0
    if not answerable:
        return float(value is None)
    return float(type(value) is int and value == gold)
~~~

Store format validity, answerability decision, and task accuracy separately. Test “I don't know but X,” candidate lists, negation, numeric substrings, repeated members, and empty responses as adversarial scorer fixtures.

### 16.2 Frozen-reader competence calibration

On development data, compare R0 with:

- Empty memory.
- A canonical compact store with all necessary information.
- Exact sufficient support.
- Support with irrelevant records added.
- Alternate valid memory serialization/order.

Calibrate task difficulty before training. A target of at least 95% exact accuracy on simple sufficient-support synthetic cases is a useful engineering gate. More complex tasks may have lower ceilings; report them and do not interpret reader incapacity as memory loss.

Choose easier question formulations or a stronger frozen reader on development data if necessary, then freeze that decision for all compared arms.

### 16.3 Conversational scores

For LongMemEval, store the official-style per-type judge score plus any secondary exact/F1 metrics. For LoCoMo, store official-compatible metrics and a declared semantic judge score.

Do not train the primary system using “gold substring appears” or “F1 >=0.5.” If optional conversational training requires a judge reward, validate that reward on a separate labeled development set and account for its cost. It is a distinct training setting from exact synthetic rewards.

### 16.4 Judge validation

Before final evaluation, use development responses spanning all relevant baseline styles, errors, abstentions, verbosity, and temporal answers.

- Two blinded human labels where feasible; adjudicate disagreements.
- Randomize response order and conceal method labels.
- Report confusion matrix, agreement, class-specific errors, and kappa.
- Check whether one method's response style systematically benefits.
- Include a different-model-family judge sensitivity if available.
- Freeze prompts, judge revision, and parsing before final tests.
- A final-test audit can assess reliability but cannot silently retune the judge and preserve an “untouched test” claim.

Judge cache key includes exact question, gold/rubric, response, prompt hash, model/revision, decoding, and dataset version. question_id alone is insufficient.

### 16.5 Health metrics

Log per sampled iteration and per optimizer step:

~~~text
task_reward_mean/std
fraction_groups_with_task_variation
fraction_zero_advantage_groups
reward_by_task/length/capacity
writer_format_error_rate
reader_format_error_rate
memory_full_rate
capacity_and_entry_occupancy
writes/updates/deletes/reads/searches/auto_advances
writer_generated_tokens and prefill_tokens
reader_generated_tokens and prefill_tokens
eligible_calls, selected_calls, inclusion_probability_distribution
per-phase/position sampled-call coverage
gradient_norm_before_clip, gradient_clipped
learning_rate
adapter_hash and base_reader_request_count
trainer_vllm_logprob_MAE/p99
rollout_seconds, reader_seconds, train_seconds, adapter_seconds
allocated_GPU_hours and peak_memory
~~~

Track actual task-reward variation separately from optional cost/format components.

### 16.6 Failure handling and curriculum

If >80% of groups have zero task-reward variation for ten development iterations:

1. Determine whether every trajectory succeeds or fails.
2. Check scorer/reader competence and format validity.
3. Inspect a preregistered random trace sample.
4. If too easy, increase useful-information pressure, not only background length.
5. If too hard, reduce candidate count/updates or begin with shorter histories.
6. Version any curriculum change and rerun comparison arms affected by it.

Reference curriculum:

~~~yaml
curriculum:
  - start_iteration: 0
    max_history_ref_tokens: 8000
    candidate_counts: [16, 32, 64]
  - start_iteration: 40
    max_history_ref_tokens: 16000
    candidate_counts: [32, 64, 128]
  - start_iteration: 100
    max_history_ref_tokens: 32000
    candidate_counts: [64, 128, 256]
~~~

Sample a mix of solvable and pressured examples within each stage. Preserve some earlier easy examples, e.g., 20%, to detect regression. Do not advance purely because a fixed iteration count has passed if basic validity gates fail.

### 16.7 Checkpoint selection

Every 20 iterations, evaluate a fixed development set using the frozen reader and exact task metric. Select the highest declared development score, breaking ties by earliest checkpoint.

Define the development objective before training: e.g., equal-weight mean across capacity/task cells under the training query prior. Do not select one checkpoint per external test cohort.

Report best-development and final checkpoints as separate sensitivity results. Do not choose whichever is better on test.

## 17. Retention measurement, visibility, and intervention diagnostics

### 17.1 Three distinct questions

Keep these separate:

1. **Storage:** Is the information represented in the saved memory?
2. **Exposure:** What information is actually present in the reader's input?
3. **Use:** Can the specified frozen reader answer from that input?

No single lexical match answers all three. The outcome metric remains question accuracy; retention diagnostics explain it with explicitly stated uncertainty.

### 17.2 Retention statuses

For every question or atom, record one of:

- certified_present: a validated structured representation or human/semantic audit establishes the relevant entity-relation-value-time fact, or a sufficient derived representation.
- certified_absent: the eligible representation can be exhaustively interpreted and the required information is demonstrably absent.
- uncertain: free-form compression, paraphrase, alternative derivation, or annotation incompleteness prevents a reliable decision.
- not_applicable: unanswerable question or no usable evidence annotation.

For synthetic canonical JSON/fact schemas, exact certification checks the complete tuple, including entity, relation, time, polarity, and value. The occurrence of a gold value anywhere in the memory is not certification.

For free-form notes, failure of the structured decoder yields uncertain, not absent. A semantic judge can provide an additional estimated status; report its audit agreement and do not relabel it “exact.”

### 17.3 Alternative sufficient information

A question can be answerable from:

- Any one of several support sets.
- A stored aggregate or relation closure.
- A correctly derived answer with its entity/question binding.
- A compact map omitting the original wording.

Use sufficient_support_sets rather than requiring every labeled turn to survive. For derived-state tasks, evaluate whether the stored statistic supports the entire declared query family; one precomputed answer may not support a different historical window.

### 17.4 Visibility ledger

For every reader call record an immutable ledger:

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class VisibleSegment:
    segment_id: str
    channel: str  # memory_entry, index_key, search_snippet, tool_result, metadata
    source_memory_id: str
    source_entry_id: int | None
    text: str
    prompt_char_start: int
    prompt_char_end: int
    policy_token_count: int
    audit_support_status: str | None  # populated after generation, not shown
~~~

For the primary all-retained reader, every canonical entry appears as a segment. For tool-based extensions, include all auxiliary channels.

Hash and save the exact rendered input or an exact reproducible content-addressed reconstruction. Every final diagnostic must be reconstructible from this ledger, not inferred from READ events.

### 17.5 Descriptive metrics

Report:

- Certified retained fraction, with uncertain fraction separately.
- Exact synthetic key/value/time reconstruction where supported.
- Head and tail retention.
- Current and historical retention.
- Duplicate/stale/contradictory record rates.
- Capacity occupancy and active binding constraint.
- Information exposure under the actual reader view.
- Accuracy conditional on certified support, explicitly descriptive.
- Accuracy with empty memory.
- Accuracy when no support is certified, without calling it “parametric leakage.”

All-retained settings have no retrieval-selection failure by construction. Reader errors can still arise from distractors, contradictions, serialization, or model limitations.

### 17.6 Snapshot replay interventions

Generate interventions after ordinary predictions and immutable snapshots are saved. The ordinary policy must never see intervention results.

Use the same reader revision, question, decoding, and per-call context limit for paired variants unless a budget relaxation is explicitly named.

| ID | Reader input | Question answered |
|---|---|---|
| I0_observed | Ordinary memory view | Original performance |
| I1_retained_gold_selection | Select sufficient information already in memory, with the same read budget | Can better selection of retained information help? |
| I2_support_only | Gold source support, if it fits | Can this reader solve the question with privileged support? |
| I3_trimmed_control | Original input with space reserved by a fixed removal rule | What does making room alone change? |
| I4_restore | I3 plus missing gold support | Does restoring information change the answer? |
| I5_sham | I3 plus unrelated, matched-size content | Is the effect specific to relevant information? |
| I6_all_retained | Entire retained memory, if an expanded read budget is needed | Sensitivity to read budget; not a matched-budget retrieval effect |

For I1, do not add missing source text. It may select only content actually retained. If a semantic conversion is needed, treat that transformation as a separate intervention and disclose it.

### 17.7 Restoration construction

For each question:

1. Determine an eligible sufficient support set from private annotations.
2. Render it without adding the gold answer unless the source support itself contains it.
3. Check that it fits the reader memory budget with the question/system prompt.
4. If not, record intervention_infeasible and keep the ordinary result.
5. Choose a deterministic removal order for original memory entries, e.g., reverse retrieval rank, without using the original correctness result.
6. Remove complete entries until enough space remains for the restoration.
7. Freeze that common trimmed context I3.
8. Add support to obtain I4.
9. Add unrelated matched-size content to obtain I5, using a dedicated sham RNG.
10. Recheck exact token counts, then run all reader variants.

The sham must not contain the target answer or support accidentally. Check its generated world identity and content. Match policy-token length as closely as feasible and record the residual difference; do not claim exact matching when tokenization prevents it.

Report I4-I5 as a relevance-specific intervention contrast and I4-I0 as the overall restoration change. The latter also includes any effect of removing original entries.

### 17.8 Diagnostic interpretation

For exact synthetic cases:

- Ordinary wrong, I1 correct, and support certified retained: selection-recoverable under the specified intervention.
- Ordinary wrong, I4 correct, support certified absent: restoration-responsive information loss.
- Support was stored earlier and later removed: a retention-loss subtype, with the actual update/delete/eviction event identified.
- Support was never certified stored: a construction-loss subtype.
- I2 wrong: this reader did not solve the support-only reference; do not assign a pure memory cause.
- Uncertain representation or incomplete support labels: unresolved.

These categories are operational, not proofs of one unique causal mechanism. Interventions change context and may interact with reasoning. Report overlaps or unresolved cases rather than forcing every error into a single category.

### 17.9 Timeline loss accounting

Do not classify “evicted” merely because an evidence-associated entry was deleted once. The same information may have been rewritten elsewhere before question time.

Track representation state over snapshots:

~~~text
first_certified_present
last_certified_present
present_at_question
events_between_last_present_and_question
alternative_representation_present
audit_uncertainty
~~~

If an update preserves an aggregate but removes raw events, evaluate sufficiency for the current query before calling it loss.

### 17.10 Gold-informed references

Privileged policies are evaluation-only:

- Perfect atomic extraction from known synthetic state.
- Budgeted selection of atomic records using hidden query weights.
- Gold retrieval restricted to retained records.
- Support-only reader input.

For the restricted independent-atom setting with additive record costs and additive query utility, an offline 0/1 knapsack solver can compute an optimal selected set within that restricted class. It is not an upper bound over arbitrary language compression, derived facts, or reader behavior.

~~~python
# Exact value-only DP for the restricted additive-utility reference.
def knapsack_value(costs, values, capacity):
    if len(costs) != len(values):
        raise ValueError("mismatched items")
    if capacity < 0 or any(c <= 0 for c in costs):
        raise ValueError("invalid capacity/costs")
    dp = [0.0] * (capacity + 1)
    for cost, value in zip(costs, values):
        for budget in range(capacity, cost - 1, -1):
            dp[budget] = max(dp[budget], dp[budget - cost] + value)
    return max(dp)
~~~

Production code also reconstructs selected item IDs and rechecks the resulting canonical capacity. In revision 3, the whole-string tokenizer charge is generally nonadditive, so this toy additive-cost DP is not an exact oracle for the primary storage budget. Use exhaustive enumeration on small actual serialized candidate sets or another certified solver for any exact primary-budget claim. Multiple-source/compositional utility is also nonadditive; label approximate selectors as heuristics.

The oracle's selected set is never supplied as an ordinary training target in the main protocol.

---

## 18. Retention perturbations, persistence, and optional extensions

### 18.1 Replace the broad wipe sweep

After the main effect is established, test at most a few hypothesis-driven interventions:

- Remove a fixed fraction of accounted memory tokens, choosing complete entries by a declared rule.
- Inject stale or contradictory facts under the same total capacity.
- Change topic frequency while preserving candidate information.
- Change query demand on the same snapshot.
- Delay questions while inserting additional potentially useful information.

Record actual tokens/items removed; deleting 20% of entries is not necessarily deleting 20% of information or tokens.

### 18.2 Wipe controls

Full wipe before answering should reduce the reader to the empty-memory condition. Treat it as a correctness/control test.

Per-session wiping with end-of-history questions mostly tests removal of earlier sessions. It is not a major research contribution by itself. If reported, preserve identical streams and distinguish information that was never eligible to survive from policy mistakes.

### 18.3 Persistent single-user streams

Only implement after the primary protocol:

- Same stable user_id across episodes.
- Chronological timestamps across the entire stream.
- New facts, updates, and questions interleaved at prespecified points.
- No repeated benchmark question/evidence package masquerading as new experience.
- No change to memory capacity across boundaries.
- Cluster inference at stream/user level.
- Separate read-only answering from any allowed answer-phase writes.

Interleaving unrelated users is a distinct interference/isolation experiment. It needs user namespaces, cross-user leakage metrics, and a clear application interpretation. Do not present it as ordinary long-term memory.

### 18.4 Optional balanced-cohort training

If development pilots show a head/tail trade-off, compare two objectives using the same hidden questions, training histories, Q, generation allowance, and gradient budget:

~~~text
Sample Q/2 head and Q/2 tail questions for every group.
R_skew     = 0.8 * mean(head scores) + 0.2 * mean(tail scores)
R_balanced = 0.5 * mean(head scores) + 0.5 * mean(tail scores)
~~~

Use even Q and identical question-sampling rules in both runs. These estimate different declared objectives; the balanced version is ordinary distribution reweighting, not automatically a novel algorithm.

Evaluate both across the same frozen cohort matrix. Report gains and sacrifices in ID, tail, and shifted demand. A further proposed method must be explicitly defined and compared against this simple reweighting baseline.

### 18.5 Deferred topics

The following are outside the initial submission scope:

- KV-cache memory backend.
- Large-model FSDP training.
- Three-optimizer leaderboards.
- Every tool/retriever/eviction combination.
- User-defined semantic themes based on hidden future question labels.
- Unbounded cross-user persistence.
- Deployment claims about real-time latency or privacy without corresponding evaluation.

---

## 19. Logging, content-addressed artifacts, and trace inspection

### 19.1 Run manifest

Every run manifest contains:

~~~text
spec_version, protocol_id, execution_id, run_status
git_commit and dirty_diff_hash
resolved_config_hash
model/tokenizer/adapter revisions and hashes
dataset and observation manifests
prompt/scorer/judge versions
seeds for each RNG namespace
hardware/software inventory
start/end times
completed sampled iterations and optimizer steps
resource ledger
test-access/protocol-freeze identity
deviations
~~~

Use UTC timestamps and monotonically increasing local event counters. Runtime failures must leave an informative FAILED manifest rather than a misleading DONE file.

### 19.2 Tables

**histories.parquet**

~~~text
history_id, source_cluster_id, dataset, split, world_id, user_id,
renderer_family, background_source_ids, L_raw, candidate_count,
update_count, U_canonical, U_compact, chunk_count, observation_manifest
~~~

**memories.parquet**

~~~text
memory_id, execution_id, arm, training_seed, writer_seed, history_id,
writer_base_revision, writer_adapter_hash, C, K, key_cap, value_cap,
snapshot_hash, snapshot_blob, accounted_tokens, serialized_ref_tokens,
serialized_policy_tokens, utf8_bytes, index_bytes, active_constraints,
writer_call_count, writer_prefill_tokens, writer_generated_tokens,
construction_wall_seconds, construction_GPU_seconds
~~~

**questions.parquet**

~~~text
question_id, history_id, source_cluster_id, cohort, task, answer_type,
answerable, population_weight, sampling_probability, question_hash,
private_gold_reference, support_granularity
~~~

**answers.parquet**

~~~text
memory_id, question_id, reader_input_id, reader_base_revision,
reader_adapter_hash, reader_view, read_budget, intervention_id,
prediction, format_valid, abstention, exact_score, f1, judge_score,
termination_reason, reader_prefill_tokens, reader_generated_tokens,
reader_wall_seconds, infrastructure_status
~~~

**calls.parquet**

~~~text
call_id, execution_id, rollout_id, group_id, phase, step, chunk_id,
adapter_hash, prompt_blob_id, completion_blob_id, prompt_token_count,
completion_token_count, sampled_logprob_blob_id, tool_name,
parser_status, environment_status, latency_seconds, origin
~~~

**memory_events.parquet**

~~~text
event_id, rollout_id, step, op, entry_ids, before_hash, after_hash,
old_entry_blob_ids, new_entry_blob_ids, accounted_tokens_before/after,
eviction_reason, current_public_chunk_id, private_source_range_refs
~~~

**visibility.parquet**

~~~text
reader_input_id, segment_id, channel, memory_id, entry_id,
text_blob_id, prompt_offsets, policy_token_count
~~~

**diagnostics.parquet**

~~~text
memory_id, question_id, retention_status, certification_method,
exposure_status, alternative_support_status, I0/I1/I2/I3/I4/I5/I6_scores,
intervention_feasible, unresolved_reason, first/last_present_steps,
loss_event_ids, audit_label_version
~~~

**training_iterations.parquet**

All health metrics from Section 16.5, plus complete reward matrices by group/trajectory/query and call-sampling inclusion probabilities.

### 19.3 Content-addressed storage

Deduplicate repeated prompts, chunks, memories, and question text by SHA-256. Use zstd compression. A blob reference includes content type, uncompressed length, and hash.

Always retain exact training token IDs used in the gradient step. Retain enough exact text/template/tokenizer metadata to reconstruct every evaluation reader input and verify its token-ID hash. Full rendered-prompt sampling is an additional convenience, not the only basis of reconstruction.

Audit logs can grow with history length; they are not part of the deployed memory store and must never be queried by the policy.

### 19.4 Snapshot frequency

- Initial empty store.
- Every successful mutation as a diff with before/after hashes.
- Full checkpoint snapshot every 20 chunks.
- Mandatory full frozen snapshot at history end.
- Before/after every diagnostic intervention.

Reconstruction from diffs must reproduce the exact final canonical bytes. A diff chain with a missing event fails validation.

### 19.5 HTML trace

Provide a self-contained HTML trace with:

1. History, arm, budgets, source cluster, and model identities.
2. Exact public chunk sequence.
3. Writer actions, errors, and calls consumed.
4. Memory occupancy and active-constraint plot.
5. Entry lifecycle table.
6. Final memory.
7. Question cohorts and reader inputs.
8. All visible information channels.
9. Retention certification and uncertainty.
10. Intervention variants and paired outcomes.
11. Costs and metadata.
12. Links to local blobs/manifests for reproducibility.

Gold labels can appear in a human-facing audit trace only after the rollout; label the trace “private evaluation artifact.” Do not feed it back into writer training.

Select qualitative examples with a frozen rule: random sample, largest gains, largest regressions, and each diagnostic category. Include failures; do not curate only successful stories.

### 19.6 Disk planning and retention

Before a full study, measure bytes per writer call, history, memory, and reader answer on a pilot. Project total storage, including checkpoints and duplicated token IDs.

Keep all final evaluation artifacts and final/best adapters. Intermediate training logs may be compressed/pruned according to a preregistered retention policy after verification, but preserve manifests, sampled traces, reward matrices, and estimator inputs needed to audit the main run.

---

## 20. Evaluation execution and cache correctness

### 20.1 Separate construction from answering

Evaluation has two commands:

~~~bash
python -m memrl.cli.eval build-memories \
  --config configs/studies/E4_primary.yaml \
  --split locked_synthetic_test \
  --output runs/E4/memories

python -m memrl.cli.eval answer-snapshots \
  --memory-manifest runs/E4/memories/manifest.json \
  --question-manifest data_cache/synthetic/v2/private/test/eval_bundles.jsonl \
  --reader-config configs/protocols/reader_all_retained.yaml \
  --output runs/E4/answers
~~~

The CLI must resolve exact paths from its project root and validate all hashes. These command names are required interfaces to implement, not claims that code already exists.

A second query cohort reuses build-memories output. A different reader reuses the same output. A different C generally requires rebuilding the writer memory unless the experiment explicitly studies post-hoc pruning.

### 20.2 Cache keys

Memory cache includes history/chunk manifest, writer model/adapter, writer prompt, sampling seed/parameters, C/K/entry caps, tool semantics, and observation protocol.

Reader cache includes exact prompt token IDs, reader model/adapter/revision, decoding, and intervention. Gold scores are cached separately and include scorer/gold/rubric versions.

Never cache by question_id and answer string alone. Never let a scoring cache from a changed gold reference survive unnoticed.

### 20.3 Evaluation failures

- Invalid writer calls: consume calls, keep valid previous memory.
- Empty memory: evaluate normally.
- Reader malformed output or generation-length exhaustion: score according to the declared scorer, usually wrong.
- Context-incompatible reference arm: record ineligibility reason and coverage.
- Infrastructure failure: rerun exact job; report unresolved failures and block a headline comparison if selective missingness remains.
- Judge invalid output: one prespecified retry, then unresolved. Report coverage and sensitivity; do not silently remove unresolved responses.

Paired tables must use the same eligible history/question set or explicitly show coverage differences.

### 20.4 Final test access

Implement a freeze manifest containing:

~~~json
{
  "protocol_id": "...",
  "code_commit": "...",
  "checkpoint_selection_rule": "...",
  "selected_checkpoint_hashes": [],
  "baseline_prompt_hashes": [],
  "scorer_and_judge_hashes": [],
  "primary_contrasts": [],
  "test_manifest_hashes": [],
  "freeze_timestamp_utc": "..."
}
~~~

This is a reproducibility record, not a user approval flow. The coding agent creates it after development gates pass and before launching final evaluations. Post-freeze changes require a new version and disclosed reruns.

---

## 21. Statistical analysis, estimands, and power planning

### 21.1 Define the target mean

For a history h and cohort c, compute a weighted question mean using that cohort's declared population distribution.

Then report:

- Primary synthetic metric: equal-weight mean across independent histories in the locked primary cell/cohort.
- LoCoMo: official/question-weighted score and equal-weight conversation mean.
- LongMemEval: official item-weighted score, plus dependence-aware sensitivity where shared-source grouping is available.
- Persistent streams: stream/user-level mean.
- Training-seed mean and all individual seed results.

Questions inside one history share a memory. They are not independent replications of memory construction.

### 21.2 Preregister primary contrasts

Suggested initial registry:

~~~text
C1: writer-only RL vs strongest locked non-RL baseline at matched C=1024,
    L=32k, calibrated high candidate count, training-prior cohort.

C2: the same comparison under reversed/tail demand using the same memories.

C3: change in the RL-vs-baseline gap between ID and shifted cohorts.

C4: writer effect at R0 in the crossed writer/reader experiment.

C5: external transfer difference on LongMemEval-S and LoCoMo,
    separately reported and not averaged into one opaque score.

C6: writer-only RL vs the locked closest learned-memory comparator,
    at the matched information-resource setting, with adaptation differences
    and construction-compute costs disclosed.
~~~

Choose the strongest non-RL baseline on development data once. Also report all prespecified baselines so the selection is visible.

If C=1024/high-count is too hard for every feasible method or too easy for all methods on development data, select a different scientifically meaningful cell before freezing. Record the change; do not select a winning test cell after evaluation.

### 21.3 Paired cluster bootstrap

For each method and training seed, first aggregate questions within history/cohort. Align history IDs across methods. Resample independent source clusters, using the same selected clusters for both methods.

For multiple training seeds, treat seed and history as crossed uncertainty dimensions. When both methods have deliberately paired seed replications, resample those seed pairs. Otherwise resample each method's training seeds independently while preserving paired histories. A deterministic baseline is reused and must not be treated as additional independent trained runs.

~~~python
import numpy as np

def crossed_paired_bootstrap(delta_seed_history, n_boot=10000, seed=0):
    """
    delta_seed_history: [S,H], already aggregated within independent histories.
    Applicable when seed pairings across arms are meaningful by design.
    """
    delta = np.asarray(delta_seed_history, dtype=float)
    if delta.ndim != 2 or not np.isfinite(delta).all():
        raise ValueError("expected finite [seed,history] differences")
    s, h = delta.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        sampled_seeds = rng.integers(0, s, size=s)
        sampled_histories = rng.integers(0, h, size=h)
        estimates[b] = delta[np.ix_(sampled_seeds, sampled_histories)].mean()
    return {
        "difference": float(delta.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "per_seed": delta.mean(axis=1).tolist(),
    }
~~~

If several length/rendering instances share one underlying world, aggregate or resample at world_id/source_cluster_id, not at their separate history IDs.

With only three training seeds, bootstrap tails are uncertain. Show per-seed effects and ranges, and avoid implying that a narrow conditional interval establishes broad training reproducibility.

### 21.4 Small number of conversations

Ten LoCoMo conversations support descriptive external validation with limited population inference. Show all conversation-level differences.

Do not run ordinary question-level McNemar tests and interpret their p-values as if hundreds of QA pairs were independent. Exact McNemar is appropriate only for genuinely independent paired binary units, such as one independently generated question per independently generated world.

### 21.5 Power/sample-size planning

Use development-pilot cluster-level paired differences to estimate their standard deviation. Choose a smallest practically meaningful improvement before final testing, initially 0.03 absolute accuracy as a planning example.

~~~python
import math
from statistics import NormalDist

def approximate_required_clusters(sd_paired_difference, min_effect=0.03,
                                  alpha=0.05, power=0.80):
    z = NormalDist()
    z_alpha = z.inv_cdf(1 - alpha / 2)
    z_power = z.inv_cdf(power)
    return math.ceil(
        ((z_alpha + z_power) * sd_paired_difference / min_effect) ** 2
    )
~~~

This normal approximation is only planning guidance. Validate with a simulation using observed cluster sizes and seed variation where practical.

The initial 128 histories per cell may be insufficient. Increase the number of independent synthetic worlds before test access if the estimated power requires it, or disclose the detectable effect at the affordable sample size. Do not repeatedly inspect test results and add samples until significance appears.

### 21.6 Multiple comparisons and selection

- One small declared primary contrast family: report effect sizes and confidence intervals, with Holm correction for confirmatory p-values if used.
- Secondary sweeps: label exploratory; report all tested cells.
- Development-selected baseline/checkpoint: freeze before tests.
- “Best of many test checkpoints” is prohibited.
- Judge sensitivity, memory order, and decoding sensitivity are robustness analyses, not additional opportunities to select the headline.

### 21.7 Cost and frontier metrics

Report raw components rather than a single arbitrary efficiency ratio:

~~~text
accuracy
construction input/output tokens
construction GPU seconds
answering input/output tokens per query
CPU retrieval/index seconds per query
persistent text bytes
embedding/index bytes
peak GPU memory
allocated GPU-hours for training
one-time data-generation and baseline-tuning costs
~~~

For Q future queries:

~~~text
total_serving_cost(Q) = construction_cost + Q * mean_answer_cost
~~~

Show at Q in {1,10,100} when meaningful. Report complete frontiers and identify nondominated methods; do not infer superiority from accuracy per million tokens alone.

A full-history archive consumes more storage but may be cheap operationally. Explain the intended setting for a hard memory cap rather than assuming text storage is inherently expensive.

---

## 22. Study schedule and compute prioritization

Revision 3 uses the F0–F9 schedule in Sections 30–32 and the companion docs/EXPERIMENTS.md. The E-series below describes reusable baseline components; it is not a second mandatory full sweep.

### 22.1 Studies

| Study | Purpose | Essential arms/settings | Gate |
|---|---|---|---|
| E0 | Data, reader, budget, throughput calibration | Scripted fixtures; empty and support-only readers | All correctness contracts pass |
| E1 | Strong baseline and capacity calibration | B03,B05/B05b,B06,B07; C grid; candidate-count grid | Useful pressure and room for improvement identified |
| E2 | RL feasibility pilot | One seed, 1.5B smoke then 7B, 30–50 iterations | Valid estimator, task-reward variation, stable execution |
| E3 | Main writer-only learning | M11, 3 seeds, locked training schedule | Compare to locked strong baselines |
| E4 | Matched-capacity and demand-shift result | Matched C=1024 plus declared capacity-transfer curves | Same snapshots across cohorts |
| E5 | Attribution | W0/W1 x R0/R1 and format/RFT controls | Writer contribution quantified |
| E6 | Mechanism | Visibility audit and limited paired interventions | No unsupported unique-cause claims |
| E7 | External tests | Untouched LongMemEval-S and LoCoMo | Protocol freeze complete |
| E8 | Replication and cost | Second backbone, common-reader snapshot sensitivity | Expand seeds if making a robust cross-backbone claim |
| E9 | Conditional objective improvement | Balanced-cohort variant vs same-question skew objective | Only if a concrete retention trade-off was found |

Revision 3 uses C=1024 for primary training and evaluation. Treat these E-series studies as component descriptions reused by the authoritative F-series schedule in Sections 30–32; do not launch both complete schedules. Matched-capacity retraining at other capacities is optional and distinct from transfer.

### 22.2 Minimum credible package

Prioritize:

1. One main writer backbone with three independent training seeds.
2. Strong summary, fact, prompted-tool, and at least one close prior comparator.
3. One informative capacity experiment with independently varied useful information.
4. A fixed-reader result and crossed-component attribution.
5. Paired query-demand shift using the same snapshots.
6. Untouched external evaluation.
7. A small intervention/audit study.
8. A second backbone replication proportional to the intended claim.

A broad one-seed sweep over six models is lower priority.

### 22.3 Development pass/fail decisions

After E1:

- If every compact baseline fits all useful information, increase candidate information density.
- If support-only R0 is weak, fix task/reader calibration first.
- If empty memory scores unexpectedly well, inspect answer predictability/contamination.
- If one simple baseline saturates accuracy under actual pressure, that may eliminate the proposed headroom.

After E2:

- Require stable adapter/reader identity and no unresolved likelihood mismatch.
- Require learning signal from task accuracy rather than format/cost changes alone.
- Compare early RL to its same-initialization prompted baseline.
- If no advantage appears, investigate a declared small diagnostic set; do not launch the full sweep automatically.

After E3/E4:

- If writer-only gains vanish against strong baselines, report that and reconsider the contribution.
- If gains appear only under the training query prior, investigate the trade-off explicitly.
- If gains depend on larger compute, characterize the frontier rather than hiding it.

No fixed accuracy-gain threshold guarantees publication. The acceptance gates are about validity and usefulness of the next experiment.

### 22.4 Cost planner interface

~~~bash
python scripts/plan_compute.py \
  --study configs/studies/E3_main.yaml \
  --profile reports/throughput_profile.json \
  --output reports/E3_compute_plan.json

python -m memrl.cli.sweep \
  --study configs/studies/E3_main.yaml \
  --dry-run

python -m memrl.cli.sweep \
  --study configs/studies/E3_main.yaml \
  --gpu-hour-budget 120
~~~

The last value is an illustrative scheduler allocation, not an estimate that E3 will finish within 120 GPU-hours. The planner must compare the measured projection with the supplied allocation before launch.

If projected cost exceeds the allocation, list the unfunded jobs in priority order. Do not shrink test sets, remove seeds, or change primary endpoints automatically.

### 22.5 Initial experiment manifest example

~~~yaml
study: E4_primary
protocol: primary_question_blind
from_training_study: E3_or_matched_C1024
checkpoints: [best_dev]
training_seeds: [101, 202, 303]
writer_eval_seeds: [0]
arms: [B03_raw_fifo, B05_rolling_summary, B05b_summary_bank, B06_structured_facts,
       B07_prompted_tools, B10_prior_memory, M11_writer_rl]
data:
  split: locked_synthetic_test
  history_ref_tokens: [32000]
  candidate_counts: [128, 512]
memory_capacities: [1024, 2048]
cohorts: [skew_80_20, uniform, reversed_20_80, tail_only]
reader:
  model: Qwen/Qwen2.5-7B-Instruct
  view: all_retained
comparisons:
  primary:
    capacity: 1024
    candidate_count: CALIBRATE_AND_FREEZE
    baseline: SELECT_ON_DEV_AND_FREEZE
  secondary:
    - capacity_transfer
    - demand_shift_interaction
question_samples:
  per_history_per_cohort: 16
  reuse_memory_across_cohorts: true
~~~

Resolve every placeholder before launch. A sweep with unresolved baseline/cell/revision names is invalid.

## 23. Analysis pipeline and required paper artifacts

### 23.1 Collection and joins

analysis/collect.py scans completed manifests and builds a DuckDB database. It must validate:

- Unique memory_id and reader_input_id identities.
- Unique answer row per (memory_id,question_id,reader,intervention,decoding sample).
- One snapshot reused across all question cohorts for a writer/history condition.
- Matching observation/protocol IDs for declared paired comparisons.
- Expected counts, including explicit failure and ineligibility rows.
- No final-test score appears in checkpoint-selection lineage.
- Scorer, judge, and model revisions are consistent within each table.

Do not join on question_id alone: IDs can be reused across dataset versions. Use dataset/version/history/question identity or stable content hashes.

### 23.2 Required tables

1. **Protocol/resource table:** each arm's writer, reader, persistent C, read budget, question visibility, training data, and construction compute.
2. **Main fixed-reader result:** paired accuracy differences at the locked primary condition, per seed and aggregate interval.
3. **Capacity table:** useful-information descriptors, occupancy, active constraints, and accuracy.
4. **Demand-shift table:** ID, uniform, reversed, tail, and historical cohorts from the same memories.
5. **Attribution 2x2:** W0/W1 by R0/R1 with interaction.
6. **External results:** LongMemEval and LoCoMo separately, with benchmark/judge versions and per-type/conversation breakdowns.
7. **Diagnostic table:** certified/uncertain retention, intervention feasibility, and paired restoration effects.
8. **Cost table:** construction, answering, storage/index, and training costs.
9. **Robustness table:** memory ordering, second reader/backbone, and scorer sensitivity.
10. **Coverage/failure table:** malformed outputs, no answers, infeasible references, and infrastructure reruns.

### 23.3 Required figures

Use ordinary plotting tools; export PDF/SVG and 300-dpi PNG, with source CSVs.

- Accuracy versus C at fixed candidate information.
- Accuracy versus candidate information at fixed history length and C.
- Length-only curve with useful information held fixed.
- Head/tail or ID/shift performance using paired memory snapshots.
- Training task accuracy, format errors, and zero-advantage groups on separate axes/panels.
- Writer/reader attribution plot.
- Accuracy–construction-cost and accuracy–total-serving-cost frontiers.
- Diagnostic intervention effects with uncertainty and unresolved categories.
- Representative occupancy/retention traces.

Do not use an unlabeled rho_raw heatmap as the only evidence of memory pressure. Do not draw bars of mutually exclusive failure causes if the underlying categories overlap or contain uncertainty.

### 23.4 Predeclared narrative questions

The report generator should answer, using actual numbers:

- Which useful-information pressures were truly binding for practical representations?
- Did RL beat the strongest locked baseline with a common reader?
- Did the result survive matched capacity and accounting for compute?
- Which future-query cohorts improved or deteriorated?
- How much of the integrated result came from the reader?
- What evidence supports the proposed failure mechanism?
- How much diagnostic uncertainty remains?
- Which result replicated externally and across seeds/backbones?
- Which comparison failed or was underpowered?

It must not automatically turn “best observed score” into a scientific claim.

### 23.5 Report outputs

~~~text
reports/<study>/
  report.md
  protocol.json
  primary_contrasts.csv
  all_results.csv
  coverage.csv
  costs.csv
  figures/
  traces/
  limitations.md
~~~

Every figure/table states independent cluster count, question count, training seed count, and the precise metric/uncertainty definition.

---

## 24. Verification suite with detailed acceptance tests

### 24.1 Pure-Python/unit tests

| Test | Required assertion |
|---|---|
| test_seed.py | Same namespace/inputs -> same seed across subprocesses; namespaces differ |
| test_schema_boundary.py | Writer serialization excludes every private field |
| test_truth_engine.py | Independent temporal/aggregate implementations agree |
| test_query_sampling.py | Empirical frequencies match declared weights; query selection is action-independent |
| test_chunker.py | Exact source reconstruction and all tokenizer caps |
| test_fixed_observations.py | C/K/value-cap changes never alter chunks |
| test_capacity.py | Full canonical visible record charge, including IDs/metadata |
| test_store_transactions.py | Failed mutation has no partial state change |
| test_cache_invalidation.py | Updates/deletes invalidate loaded values and retrieval vectors |
| test_parser.py | Multiple calls, trailing text, bad types, missing tags rejected |
| test_scorers.py | Substrings/contradictions/false abstention phrases do not earn reward |
| test_rloo.py | Correct numerical advantages and zero-sum property |
| test_sample_weights.py | Correct inclusion probabilities and normalization |
| test_snapshot_hash.py | Exact deterministic canonical bytes |
| test_visibility.py | Keys/snippets/results represented in actual-input ledger |
| test_retention_status.py | Missing lexical match yields uncertain when appropriate |
| test_intervention.py | I3 shared between I4/I5; budgets and support-only feasibility verified |
| test_cache_keys.py | Gold/scorer/model/prompt/revision changes invalidate appropriate cache |
| test_paired_stats.py | Pairings preserve world/conversation clusters |

### 24.2 Numerical advantage examples

~~~python
import numpy as np

assert np.allclose(
    rloo_advantages([1, 0, 0, 0]),
    [1, -1/3, -1/3, -1/3]
)
assert np.allclose(rloo_advantages([0.5, 0.5, 0.5, 0.5]), 0)
assert abs(rloo_advantages([0.2, 0.4, 0.6, 0.8]).sum()) < 1e-12
~~~

### 24.3 Exact subsampling test without a language model

Enumerate every subset for a tiny call population. Use vector-valued synthetic call gradients so the test checks each parameter coordinate.

~~~python
from itertools import combinations
import numpy as np

def exact_subsample_gradient_check():
    # Each row already includes its trajectory advantage and token-gradient sum.
    call_gradients = np.array([
        [1.0, 2.0],
        [-0.5, 0.25],
        [3.0, -1.0],
        [0.0, 4.0],
    ])
    M, m = 4, 2
    N, Z = 2, 256
    full = call_gradients.sum(axis=0) / (N * Z)
    estimates = []
    for subset in combinations(range(M), m):
        estimates.append(
            call_gradients[list(subset)].sum(axis=0)
            * (M / m) / (N * Z)
        )
    assert np.allclose(np.mean(estimates, axis=0), full)
~~~

Add a counterexample demonstrating that division by the sampled number of completion tokens generally does not recover the intended trajectory-sum objective.

### 24.4 Property tests

Use Hypothesis or equivalent to generate:

- Random sequences of write/update/delete/read/search actions.
- Small capacities, long IDs, Unicode strings, escaped JSON, and empty values.
- Updates that grow/shrink entries.
- Automatic eviction transactions that cannot satisfy protected-entry constraints.
- Read-cache state followed by deletion/update.
- Chunk splits near tokenizer/character boundaries.

At least 5,000 stateful sequences must preserve all store/budget invariants before GPU experiments.

### 24.5 GPU tests

Run on the small model before the main model:

1. Completion log probabilities from optimized slicing match the naive full-logit implementation on the same model.
2. Completion-only loss/gradients match an explicit masked-loss implementation.
3. No gradient is assigned to reader completions.
4. Base reader output remains unchanged after a writer adapter request under fixed batching.
5. Adapter hash/version changes are observed in generation.
6. Trainer/vLLM likelihood parity meets Section 15.4 gates.
7. Two-rank weighted DDP gradients match single-process accumulation on identical samples within declared numerical tolerance.
8. Even/odd sample-count handling preserves the estimator weights.
9. Zero-advantage iteration skips optimizer.step on every rank.
10. Interrupted/resumed run preserves job/history/question/action seed sequence.
11. Longest planned prompt fits with measured memory margin.
12. Mixed writer/read-only serving does not leak the writer adapter.

The required tolerances should reflect dtype. Use fp32 tiny-model tests for tight mathematical comparisons, then report bf16 production discrepancies separately.

### 24.6 Research-contract tests

These are as important as training tests:

~~~text
test_query_cohort_changes_do_not_change_memory
test_hidden_gold_changes_do_not_change_writer_requests
test_all_retained_reader_sees_every_counted_key_and_value
test_no_audit_log_available_to_policy_search
test_external_test_not_used_for_checkpoint_selection
test_question_reordering_does_not_mutate_snapshot
test_raw_and_note_arms_share_observation_manifest
test_fifo_equals_lru_when_no_intermediate_reads
test_historical_questions_require_correct_validity_interval
test_source_world_variants_share_bootstrap_cluster
test_infrastructure_failures_cannot_reduce_score_denominator_silently
test_gold_support_reference_is_labeled_and_feasibility_checked
test_different_gold_rubric_invalidates_judge_cache
~~~

### 24.7 End-to-end scripted fixtures

Create five tiny worlds, each with explicit expected tool traces:

- Store/read a fact under nonbinding capacity.
- Choose two of three independent facts under an atomic two-slot cap.
- Update current state while retaining a historical value.
- Aggregate several events into one valid sufficient statistic.
- Lose a fact, then restore it only in an evaluation intervention.

Use a deterministic fixture reader that reads only the serialized memory, not the private world state. It must get all answerable nonbinding fixture questions correct.

Then run the frozen small LLM reader on the same fixtures. Its score is a competence measurement; do not rewrite the deterministic expected answers to make the model pass.

### 24.8 Analysis tests

Construct a tiny synthetic result database with:

- Two methods.
- Two training seeds.
- Three independent histories.
- Multiple correlated questions per history.
- One shared-world length variant.
- One uncertain retention label.
- One infeasible restoration reference.
- One infrastructure failure followed by a successful rerun.

Check every aggregate against hand-computed expected values. Assert that duplicating questions within one history does not increase the independent cluster count or create artificially narrow history-level confidence intervals.

---

## 25. Phase-by-phase implementation with acceptance gates

| Phase | Deliverables | Gate to advance |
|---|---|---|
| P0 — Inventory/protocol | Hardware/dependency/model probes, typed configs, IDs, initial claim registry | Exact model/API versions pinned; budget examples valid |
| P1 — Data | Synthetic truth engine, renderers, query pools, fixed chunks, benchmark loaders | Leakage and truth tests; pressure/feasibility report |
| P2 — Store/environment | Charged canonical memory, transactions, tools, parser, state machine | Property tests and scripted fixtures pass |
| P3 — Frozen reader | All-retained reader, strict scorers, visibility ledger, caching | Reader competence calibrated; no adapter contamination |
| P4 — Baselines | Raw, summary, structured facts, prompted tools, prior-method integration | Baselines tuned on development data; costs recorded |
| P5 — Format gate | Raw-base validity report; optional common W0 adapter | Tool validity adequate; initialization common and frozen |
| P6 — RL math/distributed | RLOO estimator, weighted call sampling, DDP, adapter sync, resume | Numerical/GPU/research-contract tests pass |
| P7 — Feasibility pilot | 30–50 main-model iterations; realistic timing profile | Stable task signal and interpretable baseline comparison |
| P8 — Main learning | Locked studies with three training seeds | Complete logs, budget/compute adherence, no selective failures |
| P9 — Attribution/mechanism | Crossed components, retention audit, interventions | Effects computed from shared immutable snapshots |
| P10 — Freeze/external | Protocol freeze; LongMemEval/LoCoMo final evaluation | No test-based selection; score coverage and judge audit |
| P11 — Replication/report | Second backbone/reader checks, statistics, figures, traces | Claims match evidence and uncertainty |

Do not wait until all infrastructure exists to learn whether the task is scientifically useful. P4 must produce a baseline/pressure report before P6 becomes a large engineering investment.

### 25.1 Required CLI sequence

The coding agent implements these commands, then runs them in this order:

~~~bash
python scripts/probe_hardware.py --output reports/hardware.json

python scripts/probe_dependencies.py \
  --output reports/dependency_probe.json

python scripts/probe_models.py \
  --config configs/models/qwen25_1p5b.yaml \
  --output reports/model_probe.json

python -m memrl.cli.data generate \
  --config configs/data/synthetic_dev.yaml

python -m memrl.cli.data build-chunks \
  --config configs/protocols/observations_v2.yaml

python -m memrl.cli.validate \
  --config configs/base.yaml \
  --resolve-model-revisions

pytest tests/unit tests/property tests/research_contracts

python scripts/smoke_test.py \
  --config configs/models/qwen25_1p5b.yaml

python -m memrl.cli.sweep \
  --study configs/studies/E1_baselines.yaml \
  --dry-run

python scripts/launch_train.py \
  --config configs/studies/E2_pilot.yaml

python scripts/plan_compute.py \
  --study configs/studies/E3_main.yaml \
  --profile reports/throughput_profile.json \
  --output reports/E3_compute_plan.json
~~~

Actual study launching uses the measured resource plan. Do not run a full E3 sweep before the pilot and baseline gates pass.

### 25.2 Smoke-test scope

The smoke test uses:

- 1.5B model.
- Eight tiny public histories with known truth.
- Four writer trajectories per group for two groups.
- Four frozen-reader questions per memory.
- A few iterations sufficient to verify end-to-end gradients and adapter publication.
- A resume boundary.
- One paired diagnostic replay.
- CPU analysis and one HTML trace.

Do not advertise a fixed runtime until measured. A smoke pass verifies mechanics, not research quality.

### 25.3 First main-model pilot

Use one training seed, 30–50 sampled iterations, and a development-only subset. Save the complete pilot manifest.

Required pilot report:

- Baseline and support-only competence.
- Training reward and independent validation accuracy.
- Tool validity before/after common formatting.
- Percentage of groups with task-reward variation.
- Retention/occupancy examples from a random sample.
- Exact reader identity audit.
- Likelihood parity and gradient statistics.
- Time at each curriculum length, including a forced 32k probe.
- Projected cost of all proposed main studies.
- A decision to continue, narrow, or redesign, tied to evidence.

### 25.4 Scientific completion checklist

Before calling the project submission-ready, the report must answer:

- What precise finding or method is new relative to close memory-RL work?
- Which primary contrast supports it?
- Was useful information actually capacity-constrained?
- Did the effect survive a frozen reader and strong baselines?
- Was the question distribution hidden during writing?
- Were test cohorts evaluated on the same saved memories?
- Were compute/storage differences disclosed?
- Are claims about retention supported beyond lexical matching?
- Are confidence intervals based on independent units?
- Are negative results and limitations visible?
- Can another researcher reproduce the principal table from immutable manifests?

---

## 26. Detailed edge cases and implementation decisions

### 26.1 Overlong questions

Do not silently trim a benchmark question. If it exceeds the declared question region but the complete prompt fits, a predeclared flexible section-allocation protocol may admit it for every arm. Otherwise record incompatibility and report coverage.

A protocol permitting such reallocation must not depend on answer correctness or method identity.

### 26.2 Overlong individual memory entries

Primary learned writes reject values exceeding the cap. A summary baseline with a different per-entry cap is an explicitly different architecture under matched total C.

For retrieval, never truncate the value and still claim its complete support is visible. Either admit the whole canonical entry, use a separately defined fragment-retrieval protocol, or report it not selected.

### 26.3 Dates and conflicting statements

Store both event time and statement time in public text when the source distinguishes them. Historical questions use event validity, not simply ingestion order.

A later statement can correct an earlier report about the past. The truth engine should represent this explicitly in a separate correction task; do not mix it into simple latest-assignment tasks without a clear semantic rule.

Ambiguous conversational annotations remain ambiguous. Do not invent exact temporal gold where the benchmark provides only an approximate rubric.

### 26.4 Model reasoning prefixes

All generated reasoning tokens count toward writer compute and generation cap, and receive policy-gradient loss as sampled actions. They are not carried to the next chunk unless explicitly written to memory.

A model may reason before the tool tag, but only one call is executed. Long reasoning that prevents a complete call is a modeled failure.

### 26.5 Automatic metadata side channels

Visible memory metadata must be charged and specified. Private source IDs, future support flags, oracle relevance, and hidden question counts cannot appear in index rows.

Numeric status such as current used tokens and chunk progress is a fixed environment observation. Do not permit the policy to write arbitrary uncharged data into counters or tool error fields. Record any metadata-bearing extension as part of the protocol.

### 26.6 Memory size versus slot count

At every snapshot report:

~~~text
C_used / C
entry_count / K
largest_value / value_cap
number_of_rejected_writes_by_reason
fraction_of_episodes_where_C_binds
fraction_where_K_binds
fraction_where_value_cap_binds
reader_view_fit_status
~~~

A capacity sweep can have no effect if K or value size binds first. Do not attribute such results to C alone.

### 26.7 Model comparisons and tokenizers

The persistent-memory charge uses one pinned reference tokenizer for every model. Actual writer/reader prompts use their own tokenizers and exact caps.

This equalizes one declared semantic storage accounting convention, not all notions of information capacity. Also report bytes and actual prompt tokens. A model that emits a different representation may achieve different compression; that is part of the result.

### 26.8 More than one question per history

Multiple questions share one memory but must have independent reader contexts. Greedy answers cannot modify the snapshot or future question order.

If a reader learns from earlier answers, that is a separate interactive learning protocol with different information access.

### 26.9 Handling missing source evidence

QA remains scored when evidence labels are missing. Retention diagnostics are ineligible or uncertain. Do not exclude such questions from the headline QA score to make diagnostic coverage look complete.

### 26.10 Benchmark-trained public checkpoints

Document known training data for every borrowed checkpoint. Unknown pretraining contamination cannot generally be ruled out; report what is verifiable and use fresh random synthetic worlds as an additional control.

A checkpoint explicitly trained on LoCoMo is a labeled comparator, not evidence of your project's untouched-dataset transfer.

### 26.11 Training termination and unstable runs

Define divergence before final runs: nonfinite gradients, repeated unexplained likelihood mismatch, or failed integrity checks terminate the run.

An ordinary low-reward seed is not an infrastructure failure. Do not discard and replace it until a favorable seed appears. Report unstable seeds and any prespecified rerun policy.

### 26.12 Negative results

If RL fails to outperform the strongest baseline under the controlled protocol, retain the artifacts. Check whether a meaningful negative finding remains:

- Earlier gains were reader or format effects.
- Apparent memory scarcity was nonbinding.
- Query-prior optimization traded away tail retention.
- Retention methods converge under all-retained access but differ under retrieval.
- Computational overhead exceeds any practical accuracy benefit.

Do not turn an implementation bug into a scientific negative result. Establish correctness and baseline competence first.

---

## 27. Source verification and publication positioning

### 27.1 Prior-work matrix tasks

Before writing a novelty claim, read the full current versions of [R1]–[R6] and [R9], and inspect their released code where available.

Record:

- Whether memory writing is query-blind.
- Whether persistent storage is hard-capped or merely regularized.
- Whether the reader is frozen.
- Whether the benchmark is trained on or held out.
- Whether useful information, background length, and retrieval budgets are separated.
- Whether future query-distribution shift is evaluated.
- Whether evidence restoration or related diagnostics are used.
- Whether their claimed improvements hold under comparable models and costs.

Novelty must be grounded in a specific difference plus empirical value. A larger implementation matrix does not establish it.

### 27.2 Citation policy in the project

Use normal bibliographic citations in the eventual paper. The URLs below are source locators for the coding agent. Pin access/version dates and paper revisions in docs/PRIOR_WORK_MATRIX.md.

Do not copy long prose passages from sources into project documentation. Short official evaluator prompts may have specific reuse terms; preserve attribution/license requirements and use the official evaluation implementation where appropriate.

### 27.3 API uncertainty is not a research uncertainty

Resolve package signatures with installed-code probes. Resolve scientific uncertainty with experiments. Do not leave a silent API workaround that changes the scientific comparison.

---

## 28. Primary references and official implementation sources

The following were used to position this revision. Paper status and repository contents can change; pin the versions actually used.

- **[R1] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning.** ACL 2026 paper: https://aclanthology.org/2026.acl-long.583/ ; arXiv: https://arxiv.org/abs/2508.19828 ; author repository: https://github.com/yansikuan/memory-r1
- **[R2] MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent.** https://arxiv.org/abs/2507.02259 ; author repository: https://github.com/BytedTsinghua-SIA/MemAgent
- **[R3] Mem-α: Learning Memory Construction via Reinforcement Learning.** https://arxiv.org/abs/2509.25911 ; author repository: https://github.com/wangyu-ustc/Mem-alpha
- **[R4] Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning (UMA).** https://arxiv.org/abs/2602.18493
- **[R5] MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory.** https://arxiv.org/abs/2601.03192 ; author repository: https://github.com/MemTensor/MemRL
- **[R6] What Eviction Destroys: A Restore-Counterfactual Audit of Forgetting in Agent Memory.** September 8, 2026 preprint: https://arxiv.org/abs/2609.08279
- **[R7] LongMemEval official repository and dataset.** https://github.com/xiaowu0162/LongMemEval ; https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **[R8] LoCoMo official repository.** https://github.com/snap-research/locomo
- **[R9] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory (BudgetMem).** https://arxiv.org/abs/2602.06025
- **[R10] vLLM LoRA documentation.** https://docs.vllm.ai/en/latest/features/lora/
- **[R11] Hugging Face PEFT LoRA documentation.** https://huggingface.co/docs/peft/en/developer_guides/lora
- **[R12] PyTorch DistributedDataParallel documentation.** https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

Reference links are not evidence that a comparison has been implemented or a result reproduced. The final paper must distinguish verified reproductions, adapted implementations, and conceptual related work.

---

## 29. Definition of done for the coding agent

The implementation is complete when:

1. A fresh environment can reproduce the hardware/dependency/model probes and unit/property tests.
2. Public observations and private evaluator state are separated and leakage tests pass.
3. The data generator produces calibrated nonbinding and binding cases with fixed observation manifests.
4. At least the required strong baselines and one close prior method run under documented conditions.
5. The writer-only RL estimator passes numerical, gradient, adapter, and resume tests.
6. Every final memory can be replayed with different question cohorts/readers without rebuilding it.
7. The principal comparisons use immutable protocol and test manifests.
8. All ordinary model failures remain accounted for.
9. Confidence intervals respect independent histories/conversations/streams and training seeds.
10. Every reported figure/table can be regenerated from saved results.
11. The main claim is supported, weakened, or rejected explicitly by the results.
12. The README gives exact setup and launch commands, realistic measured resource requirements, and known limitations.

The coding agent should finish each phase with a short report: what was implemented, which gate passed, what remains unresolved, and the next scientifically necessary step. Do not substitute a growing list of optional features for the experiments needed to answer the primary research question.



---

## 30. Revision 3 authority and research readiness

**Direct assessment:** the improved experimental setup is a credible basis for research, not an ICLR/MLSys-ready contribution. No idea, file structure, GPU count or experiment plan guarantees publication. Scientific merit depends on the actual increment over prior work and evidence that survives strong controls.

A direct September 17, 2026 preprint, *Correct Now, Insufficient Later* [R21], already studies present-answer agreement followed by divergent update requirements. Therefore that observation and its elementary collision argument are not this project's novelty. Revision 3 investigates a candidate learned remedy and its cost/quality trade-off. The strongest competing explanation is ordinary continuation-data augmentation; it is now a mandatory comparison.

### 30.1 Precedence and explicit changes

| Topic | Revision-3 authoritative choice |
|---|---|
| Primary scientific question | Does F03_fork improve the accuracy–training-cost frontier beyond F02_flat and strong bounded non-RL constructors? |
| Primary protocol ID | continuation_blind_v3 |
| Primary train/test capacity | 1024 reference tokens; other capacities are labeled transfer or matched retraining |
| Persistent charge | Exact whole canonical serialization, replacing additive per-entry charge |
| Index overflow | Reject the proposed mutation; no primary index pagination |
| Memory process | One bounded prefix memory per writer sample, then independent clones process hidden future observations |
| Primary future reward | Mean hidden future-question accuracy with frozen reader; current accuracy reported separately |
| Learning defaults | H=4, G=4, K=2, Q=8; prefix loss once with average branch advantage; suffix loss divided by K |
| Crucial comparator | Flat K-future data augmentation, with correct H*K*G normalization for independent full trajectories |
| Shared initialization | Raw base unless a development format gate requires common format-only SFT; every learned arm shares the choice |
| Study schedule | F0–F9 supersedes running the entire earlier E0–E9 schedule separately; reuse its components |
| Optional allocation method | Section 31.8; disabled until core method and cost gates pass |
| Build strategy | Fresh production src/memrl plus tested src/memrl_contract reference; isolated ownership and integration gates |
| Resource planning | Initial 1500 allocated GPU-hour envelope; actual feasibility determined by profile data |
| Publication status | Unvalidated hypothesis; no guarantee or claim of measured improvement |

Other invariants from Sections 0–29 remain in force: fixed observations, gold isolation, frozen primary reader, strong baselines, exact prompt accounting, controlled attribution, untouched final benchmarks and cluster-aware statistics. A specific revision-3 override wins over an older example. Record any additional change and update all machine-readable profiles before execution.

### 30.2 Research directions, ranked

1. **Primary candidate: future-continuation training for bounded retention.** Learn one prefix memory against multiple possible later event streams, with the exact score-function gradient below. The potential contribution is demonstrated finite-compute learning benefit and a supported account of when it occurs, not the existence of rollout branches.
2. **Conditional method extension: allocate continuation evaluations under an unchanged target objective.** Learn which suffix groups justify computation and use explicit positive-support importance weights. This is a research idea with a complete sampling contract below, not a novelty guarantee. Compare to uniform allocation and charge allocator overhead.
3. **Conditional MLSys route: accelerate the validated memory-learning workload.** Optimize the measured mix of shared ingestion, bounded-state clones, frozen-reader batches and adapter synchronization. Existing prefix caching and tree training are prior work. An MLSys claim needs a meaningful remaining bottleneck, correctness equivalence, strong system baselines and end-to-end time-to-quality gains.

Do not implement all three merely to make the paper sound innovative. Finish the small falsification pilot, then choose the evidence-supported route. The companion docs/NOVELTY_AND_VENUE.md records close prior work and explicit claim boundaries.


## 31. Continuation-training method contract

Protocol: `continuation_blind_v3`. Status: **candidate method, unverified novelty and unmeasured benefit**. This document specifies the experiment precisely enough to implement and falsify it. It does not promise that the method will beat ordinary data augmentation. Read the direct overlap analysis in `NOVELTY_AND_VENUE.md` before writing a paper introduction.

### 31.1 The question worth testing

Can training across several possible future updates improve the information a fixed-capacity writer preserves, relative to training on the same updates as independent complete histories, at a declared resource budget?

The setting differs from asking many questions about a final frozen memory: **the memory must first process previously unseen future events**. A correction can depend on an old value that was unnecessary for a current answer. All methods know the public event semantics and the training distribution. No method sees its realized future before creating the prefix memory.

An illustrative toy example has two independent histories. One contains transactions `p=2,q=8`; another `p=7,q=3`. Both currently total 10. A subsequent `cancel(p)` makes their correct totals 8 and 3. Saving the present total alone loses something required for the update. This is a test fixture and prior-motivated example, not a new theorem. A strong baseline that stores transaction identities and amounts may solve it; the real experiment must impose calibrated scarcity and uncertainty over which identities will be referenced.

Do not label future usefulness unknowable in all senses: its **distribution** is declared. The realized suffix and questions are hidden. No bounded store can preserve all independently random values for all possible future questions. Scope every claim to the tested distributions and capacities.

### 31.2 Exact random variables and information boundary

- `h`: a raw prefix world and its rendered chronological chunks. It is sampled independently of the writer's actions.
- `u[k]`: a future event sequence sampled from a declared training kernel `P_train(u | h)`. The simulator may inspect the complete world to generate valid events and gold answers. The writer receives only the eventual event text, in order, after the fork.
- `q[k,r]`: hidden reader questions for suffix k, drawn before writer collection or by an independent named RNG stream. Gold answers are private.
- `g`: one of G independent writer trajectories through the same prefix. Each produces a possibly different bounded memory `m[g]`.
- `s[g,k]`: a suffix writer trajectory starting from a deep clone of `m[g]` and processing `u[k]`.
- `R[g,k]`: mean frozen-reader score over the Q questions for the final memory of `(g,k)`. Rewards lie in [0,1]. No auxiliary shaping is used in the primary study.

Primary defaults are H=4 prefix worlds per iteration, G=4 prefix writer samples, K=2 futures, Q=8 questions per future. This is 16 prefix trajectories, 32 suffix trajectories and 256 reader requests before exact-request caching. A cache hit does not create another statistically independent observation.

All G writers for the same world use the same sampled suffixes and questions. Writer-action RNG is independent across g and across suffix trajectories. All questions remain hidden until the relevant writer has finished. The writer sees the same task-prior instruction in every matched arm; it never sees an operator label, hidden dependency annotation or suffix count unless explicitly present in the common public task description.

The simulator's raw prefix is **private audit state after ingestion**. It is not a writer-accessible archive, tool database, language-model KV cache, filesystem path, exception string or prompt attachment. Branches clone only the permitted charged memory and legitimate cursor/budget state for writing. Auditor state may be cloned in a separately inaccessible process to compute scores.

### 31.3 Objective and estimator

Let P[g] be the sum of writer completion log probabilities in prefix trajectory g, and S[g,k] the corresponding sum in suffix trajectory (g,k). Every term is computed on the exact per-call context seen at generation time. Prompt and reader tokens receive no loss.

For each world and branch k, leave-one-out operates over independent prefix trajectories:

```text
A[g,k] = R[g,k] - sum_{j != g} R[j,k] / (G-1)
A_prefix[g] = sum_k A[g,k] / K

L = -1/(H*G*Z) * sum_worlds sum_g (
        A_prefix[g] * P[g]
        + sum_k A[g,k] * S[g,k] / K
    )
Z = 256  # fixed gradient scale, not a random token denominator
```

Advantages are detached. There is no group-standard-deviation normalization, clipping ratio, task-conditioned postselection or extra gold-survival reward. The continuation method uses the same REINFORCE score-function identity as the single-future arm; it is not a fundamentally new policy-gradient estimator.

To see the weighting, differentiate the expectation of `(1/K) sum_k R[g,k]`. A prefix action changes all its descendants, so its score term multiplies the average reward. A suffix action affects only its branch, so its term multiplies that branch's reward divided by K. The other g trajectories are independent of the current g actions conditional on the shared exogenous world, futures and questions, making their leave-one-out return a valid baseline. Sharing model parameters is not dependence between independently sampled actions.

This reasoning assumes:

1. The suffix kernel and hidden questions are exogenous to current writer actions. No selecting a suffix because it defeats the sampled memory in the primary protocol.
2. Each call is sampled from the same adapter and sampling distribution whose log probability is differentiated. Use temperature=1, top_p=1, top_k disabled and no repetition penalty for this reference.
3. Shared prefix calls are included **once**, not once per descendant. A prefix average multiplied by K would change its gradient relative to suffix gradients.
4. The reader and evaluator are frozen. No gradient passes through generated memories or discrete environment transitions; score-function terms carry the learning signal.
5. Truncation, invalid tool actions and exhausted call budgets have specified terminal behavior and stay in the reward denominator.

For selected writer calls j, use the coefficient above multiplied by `1 / inclusion_probability[j]`. Sample eligible calls uniformly without replacement after branch advantages are known. Preserve the fixed denominator `H*G*Z`, including zero-advantage groups. Correct an entire per-call log-probability sum, not a per-token mean. Gradient clipping and Adam are nonlinear; unbiased unclipped gradient estimates do not imply unbiased optimizer updates.

The CPU reference tests enumerate a tiny finite policy and compare the expected estimator with the derivative of its exact expected return. GPU acceptance separately checks completion masks, DDP scaling, inference/training log probabilities and adapter parity.

### 31.4 Collection pseudocode

The following is a target algorithm, not an installed GPU function:

```python
def collect_iteration(policy_version, prefix_batch, cfg, services):
    rows = []
    for world in prefix_batch:  # vectorize calls across worlds in production
        futures = services.private_sampler.sample_futures(world, cfg.K)
        questions = services.private_sampler.sample_questions(world, futures, cfg.Q)
        prefixes = [
            services.writer.ingest_public_prefix(
                world.public_chunks, policy_version,
                action_seed=services.seeds.prefix(world.id, g),
            )
            for g in range(cfg.G)
        ]
        branch_rows = []
        for g, prefix in enumerate(prefixes):
            for k, future in enumerate(futures):
                writer_state = prefix.public_state.deep_clone()
                assert writer_state.memory_hash == prefix.memory_hash
                suffix = services.writer.ingest_public_suffix(
                    writer_state, future.public_chunks, policy_version,
                    action_seed=services.seeds.suffix(world.id, g, k),
                )
                reward = services.frozen_evaluator.score(
                    suffix.snapshot, questions.ticket(k)
                )
                branch_rows.append((g, k, suffix, reward))
        rows.append(services.assemble(prefixes, branch_rows))
    services.verify_all_branches_complete(rows)
    return services.attach_coefficients_and_select_calls(rows)
```

The literal serial loops communicate dependencies. Runtime scheduling batches eligible calls across worlds and branches. It must not expose future chunks early or feed one branch's tool result into another. Current-state evaluation, if requested for a diagnostic, reads an immutable prefix snapshot in an isolated reader operation and never mutates a writer state or reveals its question to the writer.

### 31.5 Required controls

| Arm | Prefix collection | Future processing | Objective and purpose |
|---|---|---|---|
| `F00_static` | G samples | none | Current-state QA training; isolates objective difference |
| `F01_singlefuture` | G samples | one hidden suffix each | Standard complete-trajectory RLOO reference |
| `F02_flat` | independently regenerate prefix for every k | same K sampled suffix worlds and same questions as F03 | Ordinary continuation-data augmentation; principal novelty control |
| `F03_fork` | G samples reused across k | K different hidden futures from each prefix memory | Candidate method |
| `F04_event_ledger` | deterministic public-event parser and bounded state | same public updates | Strong domain-aware non-RL baseline; all state charged |
| `F05_duplicate_future` | G samples reused across k | duplicate a single hidden suffix K times, independent suffix-action RNG | Separates future diversity from extra policy sampling |

`F02_flat` has H*K groups, each with G independent complete trajectories. Its ordinary RLOO denominator is `H*K*G*Z`, not the fork denominator. Use the same future/query material but independently sample each complete prefix. Both F02 and F03 estimate the same expected future-return objective under their respective sampling topology. A benefit is therefore a potential finite-compute/sample-efficiency result, not a superior asymptotic objective by definition.

Run two distinct comparisons: equal unique prefix/suffix/query material and equal total measured training GPU-hours. Report generated and prefetched tokens, reader requests, selected backward calls and optimizer steps for both. Equal steps alone is not equal compute. Do not give F03 a longer run and silently compare it to a smaller F02 run.

The event-ledger baseline must process rendered public events, not private generator objects. In the symbolic track, all methods get the same explicit event syntax and semantics. In natural-text tracks, either evaluate extraction as part of the baseline or give every method the same extraction front end and label the track accordingly. A full hidden-state interpreter is an unbounded audit oracle, not F04.

### 31.6 Data construction in detail

Keep four separate layers: abstract event generator, deterministic world interpreter, public renderer, and hidden query/continuation generator. A separately implemented interpreter validates all gold labels before model collection.

**Core public operations.** Start with `create(id, amount)`, `correction(id, new_amount)` and `cancel(id)`. Core values are nonnegative integers or exact nonnegative integer minor units; never binary floats. Signed amounts require a separately versioned extension. IDs are unique within a world and independent of amount, salience and future relevance. A correction replaces the active value of an existing uncancelled ID. Cancellation deactivates that ID. Repeated cancellation, correction of an inactive/unknown ID and reused creation IDs are deterministic invalid events with no state change; render the public invalid-event semantics consistently. The supplied toy interpreter is a correctness fixture for this subset. Do not silently extend it into version rollback.

**Additional production families.** Version replacement/revocation; temporal expiry; reference/alias dependency changes; and mixed two-step updates are separate interpreters with explicit semantics and tests. Implement only after the core accounting gate passes. A revocation family needs tombstones, replay policy and version ordering; an alias family needs cycle and missing-target behavior. Register every operation by version, including invalid-event rules. Never reuse a dataset name after changing semantics.

**Prefix generation.** Sample world seeds from distinct train/dev/test namespaces. Generate 16–512 independently useful records, with candidate counts and raw length varied separately. Values should resist reconstruction from entity names and priors. Render 8k/16k/32k reference-token training histories and controlled 32k main evaluations using the frozen chunker. Use background that is genuinely disjoint by source and renderer-family split. Record rejected infeasible count/length pairs; do not truncate useful events to make a grid fit.

**Fork boundary.** Select a boundary from `{0.50, 0.75}` of event progress using an exogenous seed before policy sampling. Snap to a legal frozen chunk boundary and record the actual token/event position. Both prefix and suffix must contain at least one chunk. Each future has at least one meaningful update and matched background; keep suffix-length strata comparable. Do not branch where the policy happens to make a high-loss action in the primary method.

**Future sampling.** For in-distribution training, draw target IDs from a declared mix of recently used, old and uniformly sampled valid IDs, with probabilities `(0.4,0.2,0.4)` initially. Resolve overlapping candidate sets through a documented mixture sampler, not by assuming disjoint sets. The mixture component is not public; only actual event text becomes public at its turn. Draw K futures independently conditional on the same complete prefix world. They may coincide naturally; log this. F05 explicitly duplicates one future. Lock the kernel and selection rule on development data before final use.

**Queries.** Sample questions from totals, current individual values, active-set membership and historical values where the family permits them. A family requiring historical answers must make the semantics and representation cost explicit. Use Q=8 shared questions across g per k during training, Q=16 per evaluation cohort. `unknown` is a typed answer, not an empty response. Ensure unknown-answer cases cannot be solved by a renderer marker or exceptional question length. Evaluate only declared questions; do not quietly make every fact relevant at every moment.

**Held-out shifts.** Independently vary target-age distribution, update count, operation composition, candidate count, capacity and renderer family. Report the single-axis shifts before intersections. Hold out at least one multi-operation composition while keeping its individual operators observed in training. Rename entity and event IDs, permute legally commuting independent events, and vary distractor placement. These are metamorphic checks within a source world, not new independent test worlds.

**Natural validation.** Synthetic continuation edits to conversation text are synthetic interventions. Real chronological update examples require source-licensed histories, a defensible timestamp cut, evidence that the later correction actually occurred, two independent human label checks with adjudication, and separation by person/document. A model paraphrase is not a new natural task. LongMemEval and LoCoMo remain untouched external transfer evaluations and cannot alone prove this particular mechanism.

### 31.7 Failure accounting and mechanism measurements

Evaluate final accuracy on every planned world/branch/question. Also report unconditional current accuracy, invalid action rate, insufficient memory errors, charged occupancy, whole-record retention in the symbolic track, and cost. A method that only improves future accuracy by breaking ordinary current tasks has a trade-off, not dominance.

Use `certified_present`, `certified_absent` and `uncertain` for sufficient information. Free-form summaries may encode derived sufficient state. Do not declare absence because the original amount string disappeared. A typed symbolic-memory track can support exact interpretation; a general text-memory track requires independent semantic audits and reports their uncertainty.

On preselected evaluation worlds, intervene with equal-token replacement of genuinely necessary old information versus matched irrelevant information, under the same frozen reader. Existing restoration methods are diagnostics and must be cited. Do not count any gold-informed restoration as ordinary test accuracy or train-time capability.

For branch statistics, aggregate questions within branch and branches within source world before the primary paired analysis. Variant pairs, paraphrases and ID renamings stay inside their source-world cluster. Report all training seeds, not only the best one.

### 31.8 Optional second method: allocate continuation work with explicit inclusion weights

This is a separately gated research idea, not part of the primary F03 configuration and not an asserted first use of importance sampling. Investigate it only if profiling shows continuation evaluation dominates and the core F03 result survives F02.

Generate a fixed exogenous pool of `K_pool=8` suffixes per world. A lightweight allocator trained only on completed **earlier development/training batches** predicts which suffix groups have high learning signal per unit cost. Before sampling current policy actions, form a mixture proposal `p[k] = epsilon/K_pool + (1-epsilon)*softmax(z[k])`, with epsilon=0.25, using public prefix features and prior statistics. No current rewards or final-test labels. Sample m=2 suffix indices with replacement; evaluate every g for each sampled index to preserve the leave-one-out comparison. For duplicate draws, use fresh suffix policy RNG and log the draws separately.

Replace every branch contribution `X[k]/K_pool` in the full-pool estimator by:

```text
(1/m) * sum_draws X[k_draw] / (K_pool * p[k_draw])
```

Apply this correction to both the prefix reward contribution and suffix loss contribution. Freeze and log p before collection; detach it in the writer loss. This estimator targets the same finite-pool average in expectation under positive support. The allocator has its own separately specified supervised objective; it is not implicitly trained through detached sampling probabilities. Check exact expectation in a small enumerated pool before any GPU use. Uniform allocation, unweighted biased allocation and the full-pool method are mandatory diagnostic controls; only the first and full-pool methods are fair objective-preserving comparators.

Candidate value: learning when future diversity is worth spending computation on while preserving the target objective. Required evidence: lower end-to-end time to fixed development quality and unchanged target estimator expectation, with overhead charged. This can still be incremental; do not add it merely to increase the number of components in the paper.

### 31.9 Stop criteria

Stop the method-novelty narrative if gains vanish against F02 with equal compute, against the strongest budgeted event/summary baseline, or after removing renderer and ID shortcuts. If improvement is entirely explained by more reader queries, report that rather than relabeling it as a retention algorithm. If evidence stays synthetic, state that limit. Publishability depends on a meaningful supported contribution, not the number of experiments or files produced.


## 32. From-scratch project and parallel implementation

The companion `memrl_handoff.zip` contains a complete instruction and reference-code tree. It is self-contained and depends on no old application implementation. Preserve it before removing any existing code. Start a fresh repository or isolated branch; deletion is unnecessary to begin.

The kit contains runnable CPU semantics and planning tools, not a finished GPU trainer. Production code belongs under `src/memrl/`. Its reference package `src/memrl_contract/` contains immutable public/private contracts, bounded memory, a ledger fixture and the branch estimator. Never pass toy UTF-8-byte capacities off as model-token budgets.

### 32.1 File map

| File or directory | What the coding agent must use it for |
|---|---|
| README.md | Read order, runnable CPU commands and implementation boundary |
| AGENTS.md and .cursor/rules/*.mdc | Persistent scientific and ownership constraints |
| prompts/COORDINATOR.md | Initial coordinating-agent prompt |
| prompts/agents/00_COORDINATOR.md through 06_ANALYSIS.md | Complete module tasks, inputs, ownership and acceptance |
| docs/INTERFACES.md | Types, exact memory/prompt/branch contracts and required artifacts |
| docs/PARALLEL_BUILD.md | Worktree setup, dependency waves and integration reports |
| docs/CONTINUATION_METHOD.md | Standalone version of Section 31 |
| docs/EXPERIMENTS.md | F0–F9 schedule, exact arms, metrics, resource matching, statistics and stop rules |
| docs/NOVELTY_AND_VENUE.md | Prior-work collisions, plausible incremental claim and venue criteria |
| docs/FOUR_H100_RUNBOOK.md | Hardware inventory, APIs, profiling, topology, OOM/restart and cost gates |
| docs/RUNTIME_WIRING.md | Controller, feeder, inference, scorer and trainer messages and lifecycle |
| docs/ACCEPTANCE_GATES.md | G00–G11 evidence gates and negative tests |
| docs/REPRODUCIBILITY.md | Locks, artifact lineage, final-test freeze and paper rebuild |
| configs/base.json and configs/profiles/*.json | Draft base and six exact arm overlays |
| configs/experiment_registry.json | Staged pilot/core/ablation plans; no launch or result records |
| scripts/resolve_config.py | Strict draft resolution; rejects production certification |
| scripts/plan_experiments.py | Plan expansion and measured-cost budget check; never launches |
| scripts/verify_handoff.py | Source/config/link/structure checks |
| src/memrl_contract/ and tests/ | Runnable mathematical and state-contract reference |
| checks/ | Configuration leakage/type/budget and planner negative tests |
| schemas/ and examples/ | Public/wire schema examples and honest cost template |
| locks/runtime_lock.template.json | Values to fill only after actual hardware/API probes |
| records/ | Decisions, assumptions, deviations, gates, status, freeze and cost ledger |
| paper/ | Claim-to-evidence mapping and an empty results-table schema |

### 32.2 Dependency waves

Coordinator first freezes production types and common fixtures. Data, Environment, Runtime, Learning, Evaluation and Analysis then implement their owned modules in separate worktrees. They can do CPU work concurrently against fixtures; they cannot independently invent incompatible schemas or launch engines on the same GPUs.

Integrate the data/environment CPU path first, then baselines/scorers, runtime job interfaces, learning coefficients and analysis. Next run the real inference/API gate, one-update test, two-rank parity and fault-injected resume. Only then profile all four GPUs and start development training. Final tests remain locked until the protocol and analysis are frozen.

The six ownership domains are `src/memrl/data/`; `memory/` plus `env/`; `runtime/` plus `models/`; `rl/`; `eval/` plus `baselines/`; and `analysis/`. Coordinator alone owns `contracts/`, configuration, schemas, locks and cross-owner integration tests. Every change request includes callers, migration and a reproducing fixture. Every completion report lists implemented behavior, executed checks, evidence and unverified work.

### 32.3 Commands that are available in the kit

Run from the extracted kit directory:

~~~bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m unittest discover -s checks -v
python3 scripts/run_cpu_demo.py
python3 scripts/resolve_config.py --profile configs/profiles/F03_fork.json --output /tmp/memrl-fork-draft.json
python3 scripts/plan_experiments.py --suite pilot --output /tmp/memrl-pilot-plan.json
python3 scripts/verify_handoff.py
~~~

Production `python -m memrl...` commands in this specification/runbook are interfaces for the coding agent to implement. They are not silently backed by a dummy trainer. The draft resolver always reports production_ready=false; the future launcher verifies actual lock contents, gate evidence, data lineage, available resources and exact tokenizer constraints.

### 32.4 Core study to implement first

F0 validates semantics; F1 calibrates reader/strong baselines/capacity; F2 performs a 20-iteration smoke then 30–50-iteration one-seed learning pilot. If warranted, F3 trains F01/F02/F03 with three seeds, 200 iterations initially. F4 evaluates supervision-matched and allocated-GPU-hour-matched frontiers. F5 tests unfamiliar continuation distributions; F6 contains F00/F05 and mechanism controls. F7 distinguishes independent natural chronological evidence from untouched general-memory transfer. F8 systems and F9 another backbone are conditional expansions.

The initial core matrix contains nine learned runs (three arms by three seeds) at C1024, not a Cartesian product of every task, capacity, length and model. Strong baseline construction, shared initialization, evaluations, failed runs and profiling consume the same project budget. All counts remain subject to development throughput/power calibration before final freeze.


## 33. Four-H100 execution and budget control

Inventory actual VRAM, SXM/PCIe topology, peer connectivity, driver/runtime compatibility, CPU RAM and fast disk before selecting processes. The 7B BF16/LoRA16 plan assumes each device can independently host inference with the declared context; this must be measured. No H100 is assumed available inside the environment where this document was prepared.

The first correct implementation uses GPUs0/1 for two-rank DDP training and GPUs2/3 for two inference workers, sharing the frozen 7B base between writer-LoRA calls and reader calls with no adapter. Every iteration completes ingestion, suffix branches and frozen-reader scores under one policy version before updating weights. Publish a new adapter atomically and wait for worker acknowledgements.

This synchronous plan has idle allocation during phase barriers. Measure it rather than claim four devices are continuously busy. A four-GPU time-multiplex schedule can collect with four inference workers and then use all four for training, but engine/model/optimizer residency and transition costs must fit memory and improve measured wall time. Do not introduce stale on-policy data or reuse model KV states across adapter changes to increase utilization.

For final evaluation use four independent replicas when the model fits one device. A larger local judge may use two tensor-parallel pairs after writer training is stopped. CPU data generation and analysis do not need reserved GPU leases. One scheduler owns leases; parallel coding agents cannot allocate GPUs independently.

Let P, S and QR be measured prefix, suffix and reader work per writer sample. At fixed H/G/K, naively repeated prefixes cost proportional to `H*G*K*(P+S+QR)`; reusing sampled prefix trajectories costs `H*G*(P+K*(S+QR))`. This is workload accounting, not a wall-clock prediction, and the flat control has independently sampled prefixes. Report actual prefill/decode tokens and latency separately.

Start with a 1500 allocated-GPU-hour planning cap. It is an editable research envelope, not a budget supplied by the user or a promise that all experiments fit. Four exclusively allocated devices convert this to 375 node-hours before downtime. Profiling must produce per-run projections with reserve, and the scheduler must account for every allocated device during idle phases. Reduce optional scope before reducing already-frozen core comparisons or seeds.

OOM recovery first reduces active sequences, batch tokens and backward microbatch size; it preserves fixed scientific contexts, storage and trajectory counts. Context/budget changes require a protocol decision. Save complete optimizer/adapter/RNG/sampler state at a declared boundary. Incomplete optimizer commits roll back to the last complete checkpoint and replay the intended iteration, rather than continuing from mixed weights.

The detailed runbook includes launch targets, API probes, cost tables, leases, health checks and recovery requirements. No runtime or speedup number has been measured for this project yet.


## 34. Publication evidence and claim gates

A credible primary contrast compares F03 to F02 on future accuracy under both supervision-matched and equal allocated-GPU-hour schedules, plus the strongest development-locked non-RL constructor under equal information/storage access. Use the same frozen reader and all retained memory. An advantage over current-only F00 cannot establish a new learning method.

Report current and future accuracy without selecting only currently correct examples. Questions and future branches are nested within source worlds; renamed IDs, paraphrases and matched variants stay in their source cluster. Show individual training seeds and cluster-aware paired intervals. Use development variance to set independent world and seed counts before final testing. Three seeds are an initial plan, not a statistical guarantee.

Natural chronological evidence requires real permitted source records and independent label checks. Synthetic corrections to natural prose are a semisynthetic stress test. LongMemEval/LoCoMo remain untouched final transfer checks and do not independently establish the specific future-update mechanism. If the natural mechanism dependency cannot be completed, narrow the practical claim visibly.

For an ICLR-style method claim, require a useful increment beyond ordinary continuation augmentation and close learned-memory methods, a supported mechanism, robust held-out behavior and complete resource accounting. For an MLSys-style claim, require a measured bottleneck, a substantive systems improvement beyond ordinary caching/batching, numerical equivalence, realistic workloads and end-to-end time to quality. These are research judgments, not official acceptance criteria.

Stop or revise the headline if the event ledger dominates, F02 explains the gain, capacity does not bind, renderer shortcuts explain performance, or the core study cannot be completed credibly. Do not add unrelated mechanisms or selectively reported results to make the project appear novel.

To avoid a generic agent-written paper, make one falsifiable claim, cite direct predecessors prominently, retain baseline wins and failed runs, inspect raw evidence, obtain real independent annotation where required and disclose assistance according to the selected venue's actual policy. Human scientific ownership is expressed through defensible decisions and audited evidence, not stylistic disguise.

The rebuild kit's G00–G11 gates separate CPU correctness, real GPU validation, scientific evidence and artifact reproducibility. Passing the first is not passing the last. The delivered reference tests and plans do not include trained checkpoints or paper results.


## 35. Additional primary references

These sources materially affect revision-3 positioning. Recheck versions and publication status before submission; preprints are relevant prior work even without peer review. The companion novelty matrix explains overlap. Do not copy published score tables into this project's results.

- [R21] Zhang. *Correct Now, Insufficient Later: Auditing Update Sufficiency in Context Compression*, 2026-09-17. https://arxiv.org/abs/2609.20045
- [R22] *ChronoMem: Version Control and Semantic Rollback for Large Language Model Agent Memory*, v2. https://arxiv.org/abs/2607.27773v2
- [R23] *MEMAUDIT: An Exact Package-Oracle Evaluation Protocol for Budgeted Long-Term LLM Memory Writing*. https://arxiv.org/abs/2605.02199
- [R24] *Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning*. https://arxiv.org/abs/2607.14171
- [R25] *RTMC: Step-Level Credit Assignment via Rollout Trees*. https://arxiv.org/abs/2604.11037
- [R26] *MemoryWalker: Stop Training Agents on Contexts They Never Saw*. https://arxiv.org/abs/2609.00865
- [R27] UMA, *Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning*, v2. https://arxiv.org/abs/2602.18493v2
- [R28] *OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search*, v3; live title differs from stale PRAISE search metadata. https://arxiv.org/abs/2604.03675v3
- [R29] *Schedule-Level Shared-Prefix Reuse for LLM RL Training*, version to pin in reproduction. https://arxiv.org/abs/2606.01143v3
- [R30] Official Cursor project-rule format and AGENTS.md support. https://cursor.com/docs/rules

The complete handoff is an implementation-ready research plan with tested reference components. A coding agent must still implement and validate the production system; the experiments must still determine whether the candidate contribution is real.

<!-- END_OF_SPEC -->
