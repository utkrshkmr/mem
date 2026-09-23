# Agent 03: inference and four-GPU runtime

Own src/memrl/runtime/, src/memrl/models/ and tests/runtime/. Read FOUR_H100_RUNBOOK, RUNTIME_WIRING, INTERFACES and MEMRL_SPEC Sections3/15. Coordinate dependency pins and GPU leases with agent00.

Implement preflight, probe_api, profile, launch, worker and resume entry points defined in the runbook. Inventory actual GPU UUIDs, VRAM, topology, driver and disk/RAM; do not assume all H100 variants are identical. Check installed API signatures and compatible official package documentation. Save a successful environment lock only after probes run. Never claim a Docker image/digest or library version is tested without evidence.

Inference returns exact generated token IDs, behavior log probabilities, stop status, actual adapter identity and costs. Primary writer is a LoRA adapter on the pinned base; reader has no adapter. Test reader identity before and after writer adapter changes. Guard tokenizer/template/revision equality. Cache keys include effective weights and exact prompt IDs; invalidate or correctly segregate caches across adapter revisions.

Build immutable jobs, atomic result publication, CPU snapshot fan-out, all-branch barriers, leases, heartbeats, bounded retries and deterministic restart. Controller owns private query tickets; writer jobs never receive them. Worker handles cannot access the private gold store. All G*K descendants complete before one optimizer step is published. A worker acknowledges the new adapter hash before receiving the next iteration.

Initial topology is two DDP trainer GPUs and two rollout GPUs. Alternate phases honestly; do not collect uncorrected stale-policy trajectories to hide idle time. Profile an all-four phase alternative only after correctness; include engine teardown/load and adapter costs. Four independent replicas handle evaluation if the model fits. Use batched eligible calls and CPU preprocessing to improve useful work, while respecting prefix/suffix information boundaries.

Acceptance: G04 and G06; real-model token/logprob parity, adapter-free reader, crash/restart tests, exact resource ledger, prototype profile at representative lengths. Leave numerical tolerances based on measured kernels in locks with rationale. CPU mocks are not GPU evidence.
