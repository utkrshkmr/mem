# Four-H100 execution runbook

This is an implementation and measurement plan, not a report of completed GPU experiments. The CPU starter included with this handoff does not implement the GPU trainer, establish inference/training numerical parity, or demonstrate H100 throughput. Commands marked **REQUIRED TARGET INTERFACE** must be implemented and verified by the assigned coding agents before use. The repository's root specification and frozen resolved configuration define the scientific protocol; this document defines its execution.

The starting experiment uses `Qwen/Qwen2.5-7B-Instruct`, a pinned model/tokenizer revision, bf16, writer LoRA rank 16, and a frozen base-model reader. The primary training and evaluation capacity is **1,024 charged reference-tokenizer tokens**. Capacity 2,048 is a separately named arm, not an unnoticed inherited default. Each model call has a maximum of 8,192 prompt-plus-completion tokens; a 32k history is processed through multiple bounded calls.

## 1. The allocation to implement first

| Physical devices | Owner | Resident model | Work |
|---|---|---|---|
| GPU 0 | DDP trainer rank 0 | Frozen bf16 base plus trainable LoRA | Gradient accumulation, optimizer; CPU controller supervises |
| GPU 1 | DDP trainer rank 1 | Same base and LoRA | Gradient accumulation, optimizer |
| GPU 2 | Rollout worker 0 | One vLLM base engine | Writer with current LoRA; reader with `lora_request=None` |
| GPU 3 | Rollout worker 1 | One vLLM base engine | Writer with current LoRA; reader with `lora_request=None` |

Use GPU UUIDs in the resource ledger, with the physical-index mapping recorded for convenience. Set `CUDA_VISIBLE_DEVICES` before importing torch/vLLM in a child process. A worker assigned physical GPU 2 normally sees logical `cuda:0`. DDP processes use `LOCAL_RANK`, not physical indices.

The baseline is synchronous and strictly on-policy: rollout all required trajectories and reader responses at adapter version v, compute a complete loss batch, execute one optimizer step, then publish v+1. Trainer GPUs wait during rollout; rollout GPUs wait during backward. **Four allocated GPUs do not imply four continuously busy GPUs.** Report both allocated GPU-hours and measured utilization. Do not hide these gaps by collecting next-iteration trajectories at stale weights.

CPU generation, manifests, tokenization, retrieval, aggregation, and plots may overlap independent GPU work. A CPU queue must never reveal held-out questions to the writer or change sampling order according to observed reward.

## 2. Inventory and compatibility gate

Before model downloads, create `reports/hardware.json`, `reports/nvidia-smi.txt`, and `reports/topology.txt`. Record:

- Four device UUIDs, product names, total/free VRAM, compute capability, driver version, clocks/power settings if readable, and whether MIG is enabled.
- Actual 80 GB versus 94 GB configuration, PCIe/SXM/NVL topology, and `nvidia-smi topo -m`. Do not assume NVLink exists because the cards are H100s.
- CPU model/core count, host RAM, filesystem and available space for model cache, run artifacts, checkpoints, and temporary results. Reserve headroom based on a measured pilot trace size, not only model weights.
- OS, Python executable/version, container image digest if applicable, torch build, `torch.version.cuda`, CUDA runtime loaded by vLLM, vLLM/Transformers/PEFT/NCCL package versions.
- Current GPU process ownership. Do not kill unrelated jobs or modify node drivers as a convenience.
- One single-GPU bf16 matrix multiplication and one two-rank DDP all-reduce. Save timings and correctness outputs. Repeat the all-reduce on the intended pair if GPU placement changes.

Read-only inventory commands:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi -q
python -m pip freeze --all
```

NVIDIA's driver compatibility guide and the chosen container's support matrix are the authority for driver/runtime compatibility. The CUDA version displayed by `nvidia-smi` is not a substitute for recording the torch/vLLM runtime builds. A container still depends on a compatible host NVIDIA driver. Select one supported software stack; do not combine independently selected latest torch and vLLM wheels and assume binary compatibility.

Create these locks **after** the probes succeed:

| Artifact | Required content |
|---|---|
| `locks/environment.lock.json` | Package versions and wheel/source hashes; Python; OS/container digest; CUDA/driver/NCCL; successful probe IDs |
| `locks/models.lock.json` | Repository, immutable revision, weight/config/tokenizer hashes, chat-template hash, licenses, dtype |
| `locks/runtime_api.lock.json` | Tested generation options, LoRA loading/unloading behavior, generated-token/logprob alignment, forward keyword support, attention implementation |
| `locks/hardware.lock.json` | Hardware report hash and allowed UUID set |

The source manifest can contain unresolved fields before provisioning; a training launcher must reject unresolved model revisions, empty hashes, or an absent successful API-probe report. Never put invented version pins into a paper or reproducibility manifest.

## 3. API, adapter, and policy parity gate

Implement the following **REQUIRED TARGET INTERFACES**. These names are an interface contract for future runtime code; they are not an assertion that the modules already exist.

```bash
python -m memrl.runtime.preflight --config configs/resolved/primary.json --output reports/preflight
python -m memrl.runtime.probe_api --config configs/resolved/primary.json --devices 2 --output reports/api_probe
```

The probe must exercise exact token IDs from the production prompt builder and complete the following sequence:

1. Generate base-model outputs; save token IDs and aligned sampled-token log probabilities.
2. Load a tiny nonzero formatting adapter; verify an intended output/logit change and correct adapted log probabilities. A zero-initialized adapter cannot prove successful loading.
3. Issue base reader requests with `lora_request=None`; compare against step 1 under identical decoding and batching conditions.
4. Publish a distinct immutable second adapter under a new positive integer ID; verify the engine uses it and cannot silently return the first adapter's result.
5. Compare inference and trainer completion-token log probabilities on identical tokens for base and both adapters. Report MAE, p99 and maximum absolute difference, signed bias, and per-position errors. Initial gates: MAE at most 0.02 nats and p99 at most 0.10 nats; investigate systematic deviations even inside those bounds.
6. Verify the sampled policy is temperature 1, `top_p=1`, unrestricted `top_k`, `min_p=0`, no repetition/frequency/presence penalty, no grammar mask, and no inherited generation-config transform. Verify dropout paths are disabled. Retain relevant stop/EOS token IDs and their log probabilities.
7. Probe base-to-adapter-to-base request order with prefix caching enabled, disabled, and after swapping adapter version. Cached state must not cross incompatible model/adapter identities.
8. Test the selected model's completion-logit slicing optimization and its fallback. A compatible function signature is insufficient: compare actual extracted log probabilities.
9. Run one two-rank update and compare its unclipped global gradient to the corresponding single-process accumulated gradient on a tiny fixed batch. Include odd sample counts, weighted duplication, and the all-zero-advantage skip.

Passing CPU contracts does not satisfy these GPU gates. Passing engineering logprob thresholds supports a measured approximation to on-policy execution; it is not a proof that different inference kernels sample mathematically identical distributions.

## 4. Memory planning and OOM policy

Use a single base engine per rollout GPU. It serves writer LoRA and the frozen reader in separate request phases. Do not load a second full reader while reserving 80% of the device for the first engine.

Initial engineering defaults:

| Setting | Initial value | May change during performance pilot? |
|---|---:|---|
| Writer LoRA rank/alpha/dropout | 16 / 32 / 0 | Scientific/model setting; new declared arm if changed |
| Total sequence budget | 8,192 | Protocol setting; never shorten silently |
| Charged memory capacity | 1,024 | Protocol setting; never shrink silently |
| Trainer microbatch | 1 call | Yes, retaining exact loss normalization |
| Trainer gradient checkpointing | On | Yes, with correctness checks |
| Rollout GPU memory utilization | 0.80 | Yes, after measured headroom |
| Active writer sequences per worker | 8 | Yes |
| Active reader sequences per worker | 16 | Yes |
| Tensor parallelism for 7B reference | 1 per rollout engine | Change only through a profiled deployment mode |
| BF16 base, adapters and compute | Probe and record exact realized dtypes | A numerical protocol decision, not an OOM shortcut |

Approximate parameter bytes as `parameter_count * element_bytes`. Activation memory, logits, gradients, optimizer states for trainable parameters, graph capture, and temporary buffers are additional. Read architecture quantities from the pinned model config; do not assume KV-head count from model size.

```python
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, element_bytes=2):
    # Key and value arrays; an estimate before engine/block overhead.
    return 2 * n_layers * n_kv_heads * head_dim * element_bytes
```

Measure both framework allocated/reserved memory and NVML process/device memory. Warm up engine graphs and adapter paths before recording peaks; then test the actual worst admissible prompt-plus-completion length and concurrency. Reserve measured margin for fragmentation and the longest scheduled batch.

OOM recovery order is: reduce generation concurrency; reduce prefill batching; reduce trainer microbatch while preserving global accumulation/weights; verify gradient checkpointing and completion-only logits; release unused adapter/cache objects through tested APIs; restart the affected worker. Reprofile and freeze the revised runtime settings. Changing context, capacity, generation allowance, precision, model, LoRA rank, query count, or branch count changes the experiment and requires a new configuration/protocol identity. Do not discard the offending long histories.

## 5. Continuation-aware training workload

The continuation-aware candidate method uses H=4 independent training prefix worlds, G=4 independently sampled writer prefix trajectories for each world, and K=2 hidden future environment continuations per prefix. The continuations are shared across G within a world according to the declared data protocol; writer action seeds are independent. Q=8 common hidden questions per continuation gives `H*G*K*Q = 256` reader requests per iteration before exact caching, versus 128 for the K=1 reference.

Generate each `(h,g)` prefix once. Freeze and hash its environment snapshot at the branch point. Clone semantic environment state for each continuation; do not rerun prefix generation or copy prefix loss K times. Each branch then processes its public suffix chunks with the same adapter version and finishes before scoring or the optimizer update. The controller keeps future events and questions inaccessible until their scheduled observation/read phase.

For training, one prefix trace carries the mean of its K valid branch advantages. Each suffix trace carries its own advantage scaled by `1/K`. The global trajectory denominator is `H*G`, not `H*G*K`, when these explicit coefficients are used. Use the exact estimator in the method specification; do not combine a `/K` suffix coefficient with a denominator that divides by K again. Call subsampling uses the eligible prefix-once and suffix-call records with their actual inclusion probabilities.

The expected computational saving is relative to naively repeating the same prefix K times, not necessarily relative to K=1 training. Let P be measured prefix inference cost per prefix trajectory, S suffix inference cost per branch, and Q times R the reader cost per branch. Then:

```text
naive K-branch inference work = H*G*K*(P + S + Q*R)
shared-prefix inference work = H*G*(P + K*S + K*Q*R)
nominal saved prefix work    = H*G*(K - 1)*P
```

This is a work accounting identity under identical trajectories, not a wall-clock speedup prediction. Serialization, imbalance, batching, attention-cache hits, and reader cost can dominate. Report separate prefill/decode counts and engine-seconds, not only a single total token count. Compare branch-aware learning to K=1 under both equal training examples/updates and equal measured end-to-end GPU-hour budgets.

## 6. Performance pilot and allocation selection

Implement the **REQUIRED TARGET INTERFACE**:

```bash
python -m memrl.runtime.profile --config configs/resolved/primary.json --mode split_2train_2rollout --warmup-iterations 2 --measure-iterations 3 --output reports/profile_split
python -m memrl.runtime.profile --config configs/resolved/primary.json --mode phase_4gpu --warmup-iterations 2 --measure-iterations 3 --output reports/profile_phase
```

Probe representative 8k, 16k and 32k histories and both low/high planned capacity, using the real question and branch counts. Add the slowest/longest admissible development cell. Warmup/measurement counts above are a minimum smoke profile; repeat the noisy cases until a scheduling decision is stable. Include cold startup and adapter publication separately.

| Mode | Execution | Advantages and costs to measure |
|---|---|---|
| `split_2train_2rollout` | Two resident trainers and two resident rollout engines, strict phases | Simple, low process/model reload cost; half the allocated devices often wait |
| `phase_4gpu` | Four rollout replicas, release them, four DDP trainers for the single update, release, repeat | Can reduce phase duration; initialization, graph capture, weight loading and optimizer reload can outweigh savings |
| `colocated_phase_4gpu` | Four devices host trainer/rollout resources, one role active at a time | Advanced opt-in only after measured memory feasibility and tested release/wake behavior; no unverified assumption that both allocations fit |

Implement and validate the split mode first. The phase mode is a performance experiment, not a required optimization before the first scientific pilot. Do not introduce asynchronous off-policy rollouts to make utilization look better. Independent seeds may run concurrently only through exclusive leases and only if the resulting per-run allocation is separately profiled and recorded; assigning four full 2+2 runs to four GPUs is impossible.

For the split mode:

```text
iteration wall time = writer rollout on 2 GPUs
                    + reader scoring on 2 GPUs
                    + CPU finalize/serialization/queue overhead not overlapped
                    + backward/update on 2 GPUs
                    + adapter publication/loading
                    + required synchronization
allocated GPU-hours = 4 * reserved wall hours
```

For phase-4 mode, replace each phase by its measured four-device duration and add teardown, model reload, graph recapture, optimizer state transfer, and cache-cold costs. Use actual device-allocation intervals if devices are genuinely released to another scheduled job. Idle devices held by the job still count as allocated.

Save median/p90 wall time, device-active seconds, utilization time series, generated/prefill/training token counts, queue delay, stage overlap, peak host/VRAM, adapter bytes/load time, retries, cold/warm cache indicators, and protocol hashes. Build a full-study projection by weighting measured cells according to the planned curriculum/evaluation matrix. Show an uncertainty interval from observed timings and charge validation/checkpoint overhead. Choose the deployment before final training and record the choice; scientific settings remain identical across compared modes.

## 7. A finite research budget for four cards

The following **1,500 allocated GPU-hour envelope is a planning cap, not a throughput prediction or a guarantee that every proposed study fits**. At uninterrupted exclusive use of four cards it equals 375 node-hours, about 15.6 days, before unscheduled downtime. The pilot must replace these envelopes with measured per-run estimates. Reduce optional scope before shrinking locked scientific budgets.

| Stage | Planning allocation, GPU-hours | Gate before spending |
|---|---:|---|
| Environment/API/DDP/parity probes | 30 | CPU contracts pass; immutable model pins |
| Throughput and reader calibration | 50 | Adapter and logprob parity pass |
| Baselines, first task-reward pilot, capacity calibration | 120 | Binding-capacity behavior and informative nonzero reward signal |
| Main training across methods and three seeds | 650 | Development advantage over strongest baseline; fixed selection rule and budget |
| Attribution and matched-budget ablations | 250 | Primary result interpretable; arm matrix frozen |
| Held-out evaluation and cross-domain checks | 180 | Test manifest and analysis plan locked |
| Systems profiling, interrupted-run audit, numerical controls | 100 | Identical workloads and quality criteria |
| Explicit retry/diagnostic contingency | 120 | Failure identified and logged; no fishing on test |
| **Total ceiling** | **1,500** | Stop scheduler at cap; report actual allocation |

Training methods × seeds × updates must fit the 650-hour envelope after profiling. For example, if a planned method/seed run measures 70 GPU-hours and there are three methods and three seeds, those nine runs consume 630 GPU-hours before any additional uncounted validation; include such overhead in the 70-hour estimate. This example is arithmetic only, not a performance claim. Do not schedule a huge grid with all cells labeled mandatory.

A continuation method that costs twice as much per update must be compared at the same overall cost as an unbranched method. Do not give the new method twice the query/read compute and claim an algorithmic advantage from equal optimizer-step counts alone. Retain both equal-update and equal-cost results to explain the trade-off.

## 8. Training launch, leases, and live observability

**REQUIRED TARGET INTERFACE**, after gates succeed:

```bash
python -m memrl.runtime.launch --config configs/resolved/primary.json --mode split_2train_2rollout --gpu-uuids-file locks/allocated_gpu_uuids.json --gpu-hour-cap 1500 --run-root runs
```

Only the supervisor creates/releases leases and starts GPU children. A lease contains execution ID, GPU UUIDs, host, supervisor PID, creation/heartbeat times, allowed process groups, and planned role. Acquire the entire required set atomically with filesystem locking or a scheduler allocation. A stale timestamp is evidence to investigate, not automatic authority to kill another process. Under Slurm, honor the scheduler-provided allocation; do not independently reserve devices outside it.

Direct all child stdout/stderr to role-specific append-only logs. Write a structured event stream as well; logs are not the source of scientific metrics. The dashboard shows current iteration/phase, queue length, completed prefixes/branches/read requests, adapter hash/version, wall time, VRAM, utilization, allocated GPU-hours, projected remaining cost, invalid-action rate, reward histogram, and zero-advantage-group fraction. Avoid uploading benchmark private text or labels to external logging services by default.

Initial engineering timeout policy: heartbeat every 30 seconds; model startup deadline 20 minutes; unhealthy after 3 missed heartbeats; stalled progress timeout `max(10 minutes, 5 * measured p99 unit time)` once a pilot exists. These are runtime defaults, not fixed scientific parameters. A live heartbeat with no completed work still needs a progress deadline. Set DDP collective timeouts for actual collective phases; do not put ranks into a long NCCL barrier while inference runs.

The supervisor gracefully stops only its own process groups, waits up to 30 seconds, then force-stops those groups if required. A writer failure before a complete result reruns the identical manifest/seeds/adapter. A failed GPU batch is never replaced by an easier history, silently dropped, or given reward zero. Quarantine partial outputs and retain failure records. Stop after a configured finite retry count (initially two retries) and surface the failure.

## 9. Checkpoint and resume contract

Publish a checkpoint only after the completed optimizer step and all rank states have been verified. Include model/adapter hashes, optimizer/scheduler state, successful optimizer-step count, sampled iteration count, all RNG states, sampler/curriculum position, query/call sampling states, per-rank CUDA RNG, configuration/protocol identities, last completed job IDs, and cost ledger. Use immutable temporary directories, checksums, and atomic final rename. Manifest publication is last. Do not overwrite an existing checkpoint in place.

**REQUIRED TARGET INTERFACE**:

```bash
python -m memrl.runtime.resume --run runs/EXECUTION_ID --checkpoint latest-complete --verify-hashes --replay-incomplete-iteration
```

If an interruption happens after `optimizer.step()` but before a committed checkpoint, restore the previous committed optimizer/model state and replay the incomplete iteration. Never guess whether an update occurred. Completed immutable rollout results may be reused only if all semantic identities, seed manifests, and the restored behavior adapter match. The resource ledger must still count failed and replayed work. Restart tests must compare selected histories, branch identities, queries, action seeds, and update count; claim bitwise numerical reproducibility only if separately measured.

## 10. Validation and final evaluation

Stop training and rollout processes, release their leases, and acquire an exclusive four-GPU evaluation lease. Run four independent single-GPU replicas of the 7B reference. Shard by whole history/prefix group rather than individual questions when memory construction is shared. Deterministic stable-ID partitioning prevents duplicated/omitted histories. Verify every expected `(method, seed, protocol, history, continuation, question)` result before aggregation.

**REQUIRED TARGET INTERFACE**:

```bash
python -m memrl.eval.run --manifest manifests/evaluation_locked.json --workers 4 --devices 0,1,2,3 --output runs/EVAL_EXECUTION_ID --require-complete
```

Each replica constructs memory, freezes it, then answers with the same frozen reader, all retained primary memory exposed, and exactly declared decoding. Question ordering must not mutate memory. Cache only exact identities. Include all failed generations and invalid actions according to the locked scoring rule. The final test is not a model-selection service.

Optional 32B judge audits run in a separate phase after policy engines stop, using two TP=2 replicas only if measured memory and API probes pass. A larger reader or judge changes the evaluation setting and has its own result namespace. Synthetic exact scoring and bootstrap analysis run on CPU; there is no scientific benefit to reserving four GPUs while those CPU tasks finish.

## 11. Evidence required for the paper

Save a hardware/software table, complete protocol and model locks, per-method total allocated GPU-hours including tuning, throughput-by-phase distributions, memory peaks, cache hit/cold-start rates, retry/replay overhead, and measured quality/cost frontiers. For an MLSys argument, compare systems variants on the same model, token IDs/workload manifest, output-quality criterion and scientific settings, with cold/warm behavior separated. A four-card launcher alone is not a systems contribution.

Before claiming the execution is ready, require: GPU adapter/base isolation; numerical parity report; exact branch-prefix accounting; single-rank versus DDP gradient test; complete one-iteration training trace; interruption/replay test; one baseline and one learned development evaluation through the production pipeline; projected study cost inside the cap. None of these gates is replaced by a passing CPU-only test suite.

## 12. Official runtime references

Sources were checked while preparing this handoff; pin the versions actually installed and save the corresponding documentation/source commit with the API-probe report. These references define supported APIs and compatibility, not a claim that this project has been tested on them.

- NVIDIA CUDA compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/latest/>
- NVIDIA Container Toolkit installation: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- NVIDIA framework support matrix: <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/>
- vLLM LoRA serving and per-request adapters: <https://docs.vllm.ai/en/latest/features/lora/>
- vLLM prefix-cache design: <https://docs.vllm.ai/en/latest/design/prefix_caching/>
- PyTorch DDP reference: <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
