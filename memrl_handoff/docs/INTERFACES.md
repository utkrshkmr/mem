# Production interfaces and integration contract

Schema version: 3. These are required production interfaces, to be implemented under `src/memrl/`. The CPU reference package under `src/memrl_contract/` exercises selected semantics and may use smaller types. Do not mistake it for the production service layer.

## Ownership and dependency direction

`config -> contracts -> data/env -> inference -> rollout -> loss/trainer -> evaluation -> analysis` describes conceptual dependencies; data gold interpretation and score services belong to a private evaluation plane. A writer module must not import private query loaders, gold interpreters or final-test manifests. Enforce this with an import-boundary test and a subprocess integration fixture, not just a naming convention.

The coordinator owns `src/memrl/contracts/`, `src/memrl/config.py`, `schemas/`, resolved config format, model locks and this file. Changes to a signature require a schema revision or a compatible migration plus fixture updates. Other owners may propose changes in `records/INTERFACE_REQUESTS.md` and continue independent work.

## 1. Configuration

`resolve(base, profile, runtime_lock, CLI_overrides) -> ResolvedConfig` has this merge order. Lists replace; they never concatenate implicitly. Unknown keys fail. One normalized JSON serialization produces a SHA-256 fingerprint. Every job records the resolved config hash. The shipped resolver supports base plus profile for planning; runtime lock validation is a required production addition.

`validate_for_launch(cfg, lock, manifests, gate_records) -> LaunchReceipt` checks scientific constraints, exact pinned revisions, compatible model/template/tokenizer identities, completed prerequisites and available GPU-hour budget. The CPU draft resolver is deliberately insufficient to authorize a real launch. Paths in serialized jobs are relative artifact references inside an approved run directory or content hashes, never arbitrary user-controlled paths to execute.

The production receipt contains `protocol_id`, `config_sha256`, `code_commit`, `code_dirty_diff_sha256`, `environment_lock_sha256`, `data_manifest_sha256`, `gate_record_sha256`, `resource_lease_id` and `created_at_utc`. A receipt is evidence of checks, not a source of model-visible text.

## 2. Public observations and private tasks

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class PublicChunk:
    text: str
    sequence_index: int
    is_last_in_visible_phase: bool

@dataclass(frozen=True)
class WriterObservation:
    task_instruction: str
    chunk: PublicChunk
    memory_index: str
    loaded_text: str
    last_tool_result: str

@dataclass(frozen=True)
class TokenizedCall:
    prompt_ids: tuple[int, ...]
    generation_cap: int
    adapter_sha256: str | None
    action_seed: int

@dataclass(frozen=True)
class Generation:
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    text: str
    stop_reason: Literal['eos', 'stop', 'length', 'error']
    adapter_sha256: str | None
    request_id: str

class InferenceBackend(Protocol):
    def generate(self, calls: list[TokenizedCall]) -> list[Generation]: ...
```

A `PublicChunk` has no `world_id`, `family`, `gold`, `future`, `answer`, `support`, `split`, or filesystem path in the model serialization. The last-chunk flag above refers only to the currently exposed ingestion phase and must be identical across matched arms; by default it is consumed by the controller and excluded from prompt text. Do not reveal the eventual total number of future branches. Opaque routing IDs live in envelopes outside model messages and are never converted with generic `asdict` into prompts.

The private task store maps a private `query_ticket` to source world, suffix identity, hidden questions, gold types, supports, weights and scorer version. Only the frozen evaluator process resolves it. Requesting the ticket from writer code is an error. Canary tests should change private labels, branch counts, filenames and gold answers while leaving public observations fixed; prompt bytes must remain identical.

## 3. Bounded store

```python
class MemoryStore(Protocol):
    def public_snapshot(self) -> bytes: ...
    def snapshot_sha256(self) -> str: ...
    def charged_tokens(self) -> int: ...
    def preview(self, action_json: str) -> 'TransactionPreview': ...
    def commit(self, preview: 'TransactionPreview') -> 'ToolResult': ...
    def clone(self) -> 'MemoryStore': ...
```

`public_snapshot()` is the canonical UTF-8 serialization of the entire retained model-visible state, including IDs, keys, values and persistent metadata. Token count is computed on that **entire string**, with the pinned memory-reference tokenizer and no added special tokens. The sum of separately tokenized fields is not a correct substitute.

`preview` parses one strictly allowed tool action, builds a proposed immutable state, and verifies entry, key, value, total memory and visible index bounds. No mutation occurs during preview. `commit` verifies the source snapshot hash is still current and atomically applies the proposal or returns a bounded public failure. An invalid action consumes the call allowance and preserves the previous state. No private exception detail appears in the tool result.

Index overflow is a declared rejection in the primary protocol; do not truncate the index or reveal hidden entries through an uncharged auxiliary list. If pagination is subsequently added, define and charge its prompts and mutable state in a separately versioned protocol. IDs must not become an uncharged arbitrary-bit channel; policy-supplied strings are charged and environment-generated IDs follow a fixed scheme. Creation order, read counters or timestamps that are model-visible count as persistent state.

Audit-only source ranges, ancestry, full raw logs and gold-support links are stored separately and cannot be loaded by any policy tool. A branch clone shares immutable bytes safely but has independent mutable transactional state. Writer-loaded buffers and last tool results clear at each new observation chunk, including the prefix-to-suffix boundary.

## 4. Environment

`reset(public_manifest, capacity, rng) -> EnvState`; `observe(state) -> WriterObservation`; `step(state, action_text) -> StepResult`; `advance_chunk(state) -> EnvState`; `freeze(state) -> SnapshotRef`.

The environment owns chunk cursor, per-chunk call allowance, bounded memory and temporary loaded/result buffers. It never owns the complete raw prefix in an accessible policy-facing object after the relevant chunk has passed. The rollout controller may read the public data file sequentially to supply the next chunk; it cannot satisfy writer search requests against earlier chunks.

Each step yields `event_id`, `public_tool_result`, `previous_snapshot_sha256`, `next_snapshot_sha256`, `action_status`, `call_budget_remaining`, and private audit fields in a separate record. Terminal conditions distinguish `normal_finish`, `budget_exhausted`, `model_invalid_action`, `model_generation_limit`, `infrastructure_failure` and `implementation_invariant_failure`.

Infrastructure failure has no scientific score until a deterministic retry completes. Model failures remain in the planned denominator. An invariant failure stops the run for repair and creates a deviation record; it is never relabeled as model difficulty.

## 5. Inference and adapter identity

`InferenceBackend.generate` preserves input order or returns an explicit request-ID mapping. Completion tokens and per-token behavior log probabilities refer to the exact sampled distribution. Check stop/EOS inclusion, prompt boundary and whitespace/token normalization with a real model. A retokenized completion string can differ from generated IDs; training must use the IDs returned by inference.

Reader requests use `adapter_sha256=None`; test both before and after a writer-adapter swap. The base model, tokenizer, chat template, rope/context settings and numerical dtype have immutable identities. A writer adapter generated for one revision cannot silently be attached to another.

The controller publishes a new adapter only after every rollout and optimizer shard completes. Weight files are written into a temporary directory, fsynced as required, hashed, then renamed with a final manifest. Workers acknowledge the hash after loading. Cache keys include model and adapter identity. Worker restarts invalidate ephemeral handles; numeric LoRA IDs alone are not sufficient identities.

## 6. Rollout artifacts

One append-only call row has:

```json
{
  "schema_version": 3,
  "run_id": "routing-only",
  "iteration": 0,
  "world_id": "private-routing-only",
  "prefix_sample": 0,
  "branch_index": null,
  "phase": "prefix",
  "call_index": 0,
  "adapter_sha256": "64-hex-digits",
  "prompt_ids_ref": "sha256:artifact",
  "completion_ids_ref": "sha256:artifact",
  "behavior_logprobs_ref": "sha256:artifact",
  "before_snapshot_ref": "sha256:artifact",
  "after_snapshot_ref": "sha256:artifact",
  "stop_reason": "eos",
  "generated_tokens": 0,
  "prefill_tokens": 0,
  "wall_seconds": 0.0
}
```

The strings above describe field shapes, not valid production hashes or measured values. The actual JSON Schema and fixtures must reject unresolved hash strings in production. Prefix rows have a null branch index and occur exactly once. Suffix rows have k in `[0,K)`; duplicate-future experiments still assign different branch indices and action RNG streams.

`assemble_group(prefixes, branches, reward_receipts) -> CompletedGroup` validates exactly G prefix outcomes, exactly G*K suffix outcomes, one policy version and the expected question identities. It rejects duplicate IDs, missing branches and mismatched memory ancestry. A reward receipt includes question/scorer/reader hashes and is signed logically by its immutable content hash, not fabricated cryptographic provenance.

`attach_advantages(group) -> WeightedCalls` implements the method equation. A weighted call records `raw_advantage`, `branch_factor`, `inclusion_probability`, `group_denominator`, `fixed_scale` and `call_logprob_sum`. Avoid storing only a precombined scalar: separate factors make a factor-of-K bug auditable.

## 7. Training

`completion_logprobs(model, prompt_ids, completion_ids) -> tensor[T]` must include the log probability of the first generated token at the last prompt position, the final sampled EOS/stop tokens according to the locked policy, and no prompt or padding positions. Empty/malformed prompts are rejected. Batch padding uses explicit masks; no prompt-length heuristic based on decoded characters.

`loss(weighted_calls, global_H, G, Z) -> scalar` sums per-call completion log probabilities. For DDP, each rank's local sum is multiplied by world_size over the global denominator before DDP averaging, or an algebraically equivalent method is tested. Ranks with no selected real calls still participate through a masked zero-loss forward/backward with matching collectives; do not skip a rank or change the scientific denominator.

Gradient accumulation is based on the desired global summed objective, not independent microbatch averages. Clip once after accumulation and synchronization. Log preclip gradient norm, finite checks, total calls/tokens, zero-advantage groups and adapter hash before/after update. Save optimizer, scheduler, scaler if used, every named RNG state, selected call IDs, sampler progress and unfinished job status for exact resume at a declared boundary.

## 8. Evaluation and analysis

`construct_memory(arm, history, capacity, writer_seed) -> SnapshotRef` runs once per history and writer sample. `answer_snapshot(snapshot, question_ticket, reader_config) -> AnswerReceipt` reuses that memory for all question cohorts. Current and future endpoints use explicitly different snapshot refs. Selecting a memory after seeing its question is forbidden in the primary protocol.

`score(receipt, private_answer, scorer_version) -> ScoreRow` returns success, typed answer, failure class and evidence hashes. Keep model timeout/truncation and malformed-answer failures. Treat infrastructure missingness with rerun or reported bounds, never silent deletion. Do not score unknown answer strings using substring containment.

`aggregate(rows, frozen_analysis_plan) -> tables` first validates completeness and uniqueness against the frozen experiment manifest, then aggregates to source-world clusters and training seeds. The data/model/prompt/adapter/config hashes required for any comparison must match or appear as explicit experimental factors. Output table cells include n_worlds, n_seeds, estimate, interval, metric, contrast ID and source artifact hashes. Analysis functions may not read mutable development results when computing locked final comparisons.

## 9. Artifacts and schemas that each owner must deliver

| Owner | Required schemas/receipts | Acceptance evidence |
|---|---|---|
| Coordinator | ResolvedConfig, LaunchReceipt, interface registry | Unknown-key failures; reproducible hash |
| Data | PublicManifest, PrivateTask, SplitLineage | Independent gold interpreter; no train/test ancestor overlap |
| Environment | Snapshot, ToolAction, ToolResult, StepEvent | Exact charge; rejection atomicity; branch isolation |
| Inference | TokenizedCall, Generation, AdapterReceipt | Real token/logprob/adapter parity |
| RL | CompletedGroup, WeightedCall, Checkpoint | Exhaustive estimator; DDP parity; resume |
| Evaluation | AnswerReceipt, ScoreRow, ExternalManifest | Reader isolation; exact cache keys; denominator audit |
| Analysis | ContrastPlan, TableCell, RunCost | Paired cluster resampling; completeness; honest costs |

Do not build a universal “metadata” dictionary into model-visible payloads. It defeats the boundary these interfaces are intended to provide.
