# Experiment registry and execution contract

This document specifies the continuation-learning extension to [MEMRL_SPEC.md](../MEMRL_SPEC.md). It is authoritative for the `F0`–`F9` study schedule and the six `F00_static`–`F05_duplicate_future` arms below. The original specification's `E0`–`E9` studies remain useful background modules; they are not an instruction to run two complete independent sweeps. Reuse their validated loaders, baselines, frozen-reader contracts, and analysis code.

**No results are supplied.** All numerical sizes below are initial development allocations or predeclared planning examples. The coding agent must calibrate costs and statistical power before locking final experiments. None is a prediction of an improvement or a promise of publication.

## 1. The one primary question

Does `F03_fork` improve the future-update accuracy versus training-cost frontier over `F02_flat`, under identical one-pass access, storage limits, frozen reader, continuation distribution, and initialization?

The second essential comparison is against the strongest development-selected non-RL constructor, including `F04_event_ledger`. Beating a deliberately incomplete present-state summary is not sufficient.

Read [NOVELTY_AND_VENUE.md](NOVELTY_AND_VENUE.md) before interpreting these results. The update-sufficiency problem already has direct prior work. The purpose of this experiment is to test a candidate solution, not to relabel an existing observation as a discovery.

## 2. Fixed information and model protocol

| Item | Primary setting |
|---|---|
| Writer | Pinned Qwen/Qwen2.5-7B-Instruct with identical initialization for all learned arms; shared format-only SFT if the development format gate requires it |
| Reader | Same pinned 7B base model, no writer adapter, deterministic decoding, frozen throughout |
| Persistent memory | Canonical serialized store with all model-visible keys, values, IDs, and metadata charged |
| Primary capacity | 1,024 reference-tokenizer tokens; training and primary testing at the same capacity |
| Capacity curve | 512, 1,024, 2,048, 4,096; distinguish matched-capacity training from capacity transfer |
| Reader view | All retained memory fits in the reader input; retrieval is not a primary bottleneck |
| Observation chunks | Frozen once from the raw stream, independent of arm and capacity |
| Writer access | Previous bounded memory plus current public chunk and fixed public task semantics |
| Writer exclusions | Hidden questions, answers, future branch text, branch labels, source cluster IDs, generator seeds, raw-prefix archive, privileged state |
| Branch start | Immutable bounded-memory snapshot at the predeclared prefix boundary |
| Branch update | Same writer adapter processes each newly observed suffix; no weight update within a rollout group |
| Primary reward | Future-question mean only; no auxiliary format, retention, present-answer, cost, or oracle reward |
| Training group | `H=4` source prefixes, `G=4` independently sampled writer prefixes per source, `K=2` future continuations, `Q=8` questions per continuation |
| Checkpoint evaluation | Frozen development manifests; greedy writer and reader; choose best declared score with earliest checkpoint tie break |
| Initial independent seeds | 101, 202, 303; consider five before final freeze if development seed variation warrants it |
| Final writer sampling | One declared deterministic rollout per history and checkpoint for the primary endpoint |

The original specification used C=2,048 in a reference training configuration and C=1,024 in one proposed primary contrast. This extension resolves that mismatch: train the primary experiment at C=1,024. If development demonstrates floor or ceiling behavior there, change the primary capacity before freezing all arms and document the reason. A different capacity is not selected from final results.

The reference tokenizer defines storage comparability. Validate the exact rendered prompts with each model's native tokenizer as well. If a cross-backbone reader cannot consume the whole charged store within its native input budget, fix the protocol before evaluation; do not silently truncate one model's memory.

## 3. Arm definitions

Use these strings exactly in configs, run IDs, result tables, and plots. Config filenames are `configs/profiles/<arm_id>.json`.

| Arm ID | Training construction | Purpose |
|---|---|---|
| `F00_static` | Writer optimized for questions at the end of the prefix, before any future continuation | Establish what current-answer supervision learns; this is not the strongest method baseline. |
| `F01_singlefuture` | One independently drawn continuation per source prefix, `G=4` writer candidates | Ordinary future-aware on-policy RL without branching across different continuations. |
| `F02_flat` | The same `K=2` continuation draws used by the fork arm, expanded into separate training examples; independently sample `G=4` writer prefixes for each continuation | Main method baseline; controls access to continuation data and hidden-question supervision. |
| `F03_fork` | Sample `G=4` writer prefixes once, clone each bounded store into both future continuations, and optimize the correctly weighted mean branch return | Candidate method; tests sharing one memory decision across possible futures. |
| `F04_event_ledger` | Non-RL event-aware representation with exact persistence accounting, deterministic bounded selection, and the same future-hidden access | Strong representation baseline. It sees public input semantics, never hidden future identities or gold supports. |
| `F05_duplicate_future` | Same shape and number of suffix rollouts as `F03_fork`, but both continuations contain the same sampled future; suffix policy random streams remain independent | Tests whether diverse future observations matter beyond repeated suffix-policy sampling. |

`F02_flat` has `H*K*G` independently sampled prefix trajectories; `F03_fork` has `H*G`. Both have `H*K*G` suffix trajectories and `H*K*G*Q` reader requests before deterministic request-cache reuse. Therefore equal outer iteration counts do **not** match prefix compute or number of distinct retained memories. Report those differences explicitly.

At H=4, G=4, K=2, Q=8, one F03 iteration contains 16 prefix trajectories, 32 suffix trajectories, and 256 logical reader requests. F02 contains 32 prefix trajectories, the same 32 suffix trajectories, and the same 256 reader requests. F01 at K=1 contains 16 prefix trajectories, 16 suffix trajectories, and 128 reader requests. These counts exclude development evaluation, retries, and format initialization. Writer tool calls depend on the number of chunks and are logged separately; one prefix trajectory is not one generation call.

`F05_duplicate_future` must use the same suffix bytes, question IDs, and answer definitions in both copies. Do not force the writer's sampled suffix actions to be identical: that would confound duplicate future information with eliminating policy sampling. A separate deterministic replay fixture may intentionally force identical draws to test loss normalization.

### 3.1 Required supporting baselines

Retain the original specification's empty memory, raw FIFO, rolling summary, summary bank, structured facts, prompted tools, format-only writer, and a close prior learned-memory comparator. Use their exact baseline registry IDs rather than inventing silent aliases.

The minimally useful F1 screening set is empty memory, raw FIFO, summary bank, structured facts, prompted tools, and `F04_event_ledger`. Rolling summary is useful if the summary bank does not dominate it across development budgets. Keep prespecified results even when a baseline is dominated.

For the learned prior comparator, UMA is especially relevant. Reproduce an author implementation where available and pin its revision. Its native information access and our strict bounded-memory adaptation are different protocols. Report both separately if feasible. Do not attach the published method's name to a loosely related reimplementation without disclosing the adaptation.

### 3.2 Event-ledger baseline details

For machine-readable synthetic events, `F04_event_ledger` may use a deterministic parser because the same public event semantics are available to every arm. Preserve enough event identity and dependency information to implement cancellation or correction when it fits. Charge this representation's full serialized text, including any index or summary that is exposed later.

At every chunk boundary, the ledger must select from only its previous charged state plus the new chunk. It cannot reconstruct dropped events from an uncharged Python dictionary. CPU event caches used solely for scoring must be in a separate evaluator object that the baseline cannot access.

Use at least two documented selection policies on development: recency over complete events and a dependency-aware event policy. For the latter, retain control records and currently needed dependencies first, then allocate remaining capacity using an explicit public prior over future operation types. Resolve ties with a deterministic content-based rule; randomize identifiers in the metamorphic audit to detect unintended lexical priorities. No heuristic may inspect realized hidden future labels.

For natural-language inputs, add an extraction stage whose model, tokens, and errors are recorded. The machine-readable baseline and the extraction-based natural-language baseline are separate systems. Do not attribute their score difference purely to retention.

## 4. Data construction and isolation

### 4.1 Unit of independence

One latent generated world is the source cluster. Histories with changed identifiers, prose renderings, lengths, order-preserving distractors, or alternative future branches from that world remain in the same cluster. Matched variants of a history are not independent observations.

Split latent worlds first; then generate their renderings, suffixes, and questions. Every derivative inherits `source_cluster_id`. Hash manifests and assert that no source cluster appears in more than one split.

Initial data allocation:

| Split | Initial allocation | Allowed use |
|---|---:|---|
| Training | 4,096 independent source worlds | Format-compatible on-policy training and declared curriculum |
| Development | 256 independent source worlds | Baseline tuning, capacity calibration, checkpoint selection, variance and power planning |
| Locked synthetic test | Start planning at 128 independent worlds per declared cell; set final number using development power analysis | One frozen evaluation program after all choices are locked |
| Metamorphic audit | Transformations of a fixed subset of development worlds, and a separately frozen test audit subset | Correctness and invariance diagnostics; never counted as additional independent worlds |
| Natural chronological development | Initial target 50 independent source histories, subject to source availability and audit | Check whether the proposed natural protocol is feasible and unambiguous |
| Natural chronological test | Determine from development variance and resource budget before access | Independent mechanism confirmation; do not assume an arbitrary count guarantees adequate power |

These are counts of latent sources, not generated questions or API calls. Increasing K or Q improves measurement within a world but does not increase the number of independent worlds.

### 4.2 Future generation

Generate future continuations from the raw training world and a dedicated environment random stream. The distribution must be independent of the writer's sampled actions and success. Conditioning on public prior history is allowed; conditioning on a memory dump is not part of the primary protocol.

For each source and optimization iteration, create one immutable `FutureSet` with K suffixes and their hidden question pools. All G candidate prefixes use this same set. Keep the files outside policy-visible directories. A job descriptor sent to the writer contains opaque operational IDs, not semantically revealing labels; operational IDs are not rendered in a prompt.

Future families should contain both revelation and replacement. The same public input history can permit several futures. A generator that makes one visible identifier or record position perfectly predict the future is invalid.

`K=2` is a compute-conscious starting point, not an assertion that two samples characterize all future uncertainty. Any later K sweep must separate additional supervision from scheduling or estimator effects.

### 4.3 Mechanism families and generalization

Start with three exact, auditable event families: transaction cancellation, correction of a prior event, and a late request for a component of an aggregate. Add dependency chains and multi-step reversal only after the core interpreter and baseline are trustworthy.

For each family specify in the dataset card:

- Legal event schema and exact transition semantics.
- Whether repeated event IDs are idempotent, errors, or separate events.
- Behavior for missing references, cancellation of an already cancelled event, and invalid updates.
- The observable current state and which hidden past distinctions can later matter.
- Domain of randomly sampled values and identifier entropy.
- Exact question grammar and answer normalization.
- Train/development/test restrictions on templates, chain depth, dependency span, and operation composition.
- Which gold functions were independently checked by a second implementation.

Do not introduce many subtly different semantics at once. Every new operation enlarges the validation burden and can produce accidental baseline disadvantages.

An initial shift suite can hold out combinations of two known operation families, double the maximum training dependency span, and use disjoint natural-language templates. Call this compositional or span generalization as appropriate. A novel language template alone is not an unseen underlying task.

### 4.4 Calibrating capacity pressure

Vary useful independent facts separately from raw token length. Use the original candidate-count range as a starting grid, then measure the actual serialized cost of a correct event representation. Report that distribution, not just raw-history tokens divided by memory tokens.

The development cell must have genuine uncertainty and headroom: an empty or latest-only memory should not solve it, a support-complete reader should usually solve the calibrated answer task, and a strong bounded event representation must not be trivially perfect across all cases. These are diagnostic conditions, not an instruction to manufacture a cell where the learned arm wins.

Keep at least one low-pressure cell in the final study. If all methods work when required information fits, but differ when it does not, that helps interpret what changed. Keep at least one high-pressure cell that can expose failure; do not restrict presentation to the chosen primary cell.

## 5. Exact outcomes and denominators

Let a source world have declared history variants and K evaluation continuations. For each branch, compute the mean exact question score. Average branch means within a history variant, variants within the source, and sources with equal weight. Equal family weighting, if desired, must be declared separately and fixed before testing.

| Metric | Definition and interpretation |
|---|---|
| `future_accuracy` | Mean exact future-question score, aggregated to independent source worlds; primary task metric |
| `current_accuracy` | Questions about the prefix before the future arrives, computed on all sources; secondary diagnostic |
| `reveal_accuracy` | Future accuracy on declared reveal-type suffixes; report full denominator |
| `override_accuracy` | Future accuracy on suffixes that legitimately replace an older answer |
| `joint_branch_success` | Fraction of source blocks satisfying all predeclared binary branch requirements; secondary, stricter endpoint |
| `prefix_retained_tokens` | Exact charged token count at fork time |
| `final_retained_tokens` | Exact charged count after each suffix |
| `memory_overflow_rate` | Rejected attempted writes divided by attempted writes, plus per-history occurrence rate |
| `invalid_action_rate` | Invalid model actions divided by model action attempts, with cause categories |
| `model_failure_rate` | Parse failures, incomplete generations, or invalid user-level outputs that remain in the task denominator |
| `infrastructure_failure_rate` | Execution/transport failures requiring deterministic rerun, reported separately |
| `gpu_hours_allocated` | Sum of elapsed allocated time across reserved GPUs, including idle time while reserved |
| `writer_tokens_in/out` | Actual tokens processed/generated; count physical work and logical branch exposure separately |
| `reader_requests` | Logical requests, physically executed requests, and cache hits separately |
| `time_to_quality` | First scheduled development checkpoint reaching the frozen threshold; censored if never reached |

Never restrict `future_accuracy` to histories where `current_accuracy` was correct. That would change the evaluated population after the method has acted. Conditional plots can be exploratory and clearly labeled, with counts; they do not replace the primary comparison.

Joint success is sensitive to the number of questions and branches. Freeze its exact block definition and do not compare joint rates from different K or Q without qualification.

For free-form natural answers, use the original external-benchmark scorer when reproducing that benchmark. For a new natural chronological task, write a separate scoring rubric, blind method identities, audit judge errors, and retain ambiguous cases in a declared category. Do not silently switch to a more favorable scorer after seeing failures.

## 6. Resource matching: three different comparisons

The study must report these as distinct views:

1. **Supervision-matched:** same unique source worlds, same continuation manifest, same hidden questions, and same number of suffix trajectories. Prefix sampling and physical compute differ between F02 and F03. This isolates data access more cleanly than compute.
2. **Budget-matched:** same allocated training GPU-hours, hardware, model, and selected checkpoint rule. More efficient methods may complete more updates. Log partial final iterations rather than counting uncompleted work as a full batch.
3. **Curve-based:** accuracy versus allocated GPU-hours, generated writer tokens, and unique source worlds exposed. These plots disclose the trade-off instead of suggesting all quantities can be matched simultaneously.

Use training wall time from the same machine with the same resource allocation for paired runtime comparisons. Include warm-up and checkpoint costs according to a declared convention; provide a separate steady-state breakdown. Dataset generation, baseline tuning, and hyperparameter search are one-time costs reported in the resource ledger.

Equal token counts are an imperfect compute proxy because prompt length, batching, attention, and cache reuse matter. Do not claim FLOP equivalence from token counts alone.

For method comparisons, both arms use the same engine optimizations whenever semantically applicable. For a runtime optimization comparison, hold trajectories fixed and compare implementations. Do not conflate a changed sampling distribution with a faster implementation of the same computation.

## 7. Staged experiment schedule

| Study | Question | Minimum work | Required artifact and gate |
|---|---|---|---|
| `F0` | Are the experiment and estimator correct? | CPU fixtures, independent event interpreter, immutable fork tests, exact tiny-policy gradient check, actual-prompt replay, frozen reader hash | `reports/F0_contracts.json`; all semantic gates pass before paid training |
| `F1` | Is there a nontrivial bounded-memory problem? | Strong baseline screen over capacity and useful-information counts; reader calibration; identifier audit; 7B throughput profiling | `reports/F1_calibration.json`; choose primary cell and strongest non-RL comparator on development |
| `F2` | Is learning feasible and is the candidate increment plausible? | One seed each for F01/F02/F03, 20 smoke iterations using the supplied pilot suite, then 30–50 development iterations after smoke checks | Learning curves, format-versus-task analysis, cost projection, written continue/stop decision |
| `F3` | Does the result replicate across training randomness? | Full locked schedule for F01/F02/F03, three seeds; F00 as prespecified attribution control | All seeds and complete checkpoint metadata; no final external evaluation yet |
| `F4` | Does the main gain survive resource matching? | Supervision-matched comparison and equal GPU-hour comparison against F02 and locked baselines | Primary paired estimates, compute curves, all resource components |
| `F5` | Is the gain robust to unfamiliar futures and representations? | Held-out composition/span/template suite; identifier renaming; irrelevant-padding and order-preserving transformations | Per-shift outcomes with source-cluster bootstrap; transformations never inflate sample count |
| `F6` | What part of the mechanism matters? | F05, current-only objective control, fixed-reader attribution, targeted memory intervention subset | Prespecified ablation contrasts, visibility audit, failures retained |
| `F7` | Does the result extend outside the generator? | Independent natural chronological evaluation plus untouched LongMemEval-S and LoCoMo under original protocols | Separate mechanism and generic-transfer tables; no pooled benchmark average |
| `F8` | Is there a substantial systems contribution? | Naive versus optimized semantic runtime, strong caching/batching baselines, representative workload sweeps | Numerical equivalence report, throughput and time-to-quality; required only for MLSys route |
| `F9` | Does the claim depend on one backbone? | Second backbone with retrained adapters, common storage accounting, scaled replication consistent with claim | Per-backbone effects and costs; one exploratory seed is not a robust replication claim |

F0–F2 are a bounded falsification program. F3–F7 form the core method evidence if the pilot warrants further work. F8 is conditional on a measured systems bottleneck. F9 is valuable but must not consume the budget needed for properly replicated primary comparisons.

### 7.1 Avoiding an unbounded matrix

At F1, screen non-RL baselines on all 256 development sources only where cost permits; smaller screening subsets must be fixed and shared. Promote the strongest few to full development evaluation. Do not run all baseline variants, all capacities, all backbones, all K values, and all lengths with full training seeds.

The first full study should use one backbone, one training capacity, one training schedule, and the locked F02/F03 contrast. Add F01, F00, and F05 only according to their declared attribution role and measured cost. Matched-capacity retraining for the entire four-value grid is optional; testing one trained policy at other capacities is a different, cheaper capacity-transfer study.

Document every unexecuted planned arm with its reason. A resource-limited omission is acceptable when the scientific claim is narrowed accordingly; silently replacing a missing strong comparator with a weaker one is not.

## 8. Pilot decisions and stop rules

### 8.1 Engineering gates

- Zero unresolved policy-observation leakage in fixtures and a manually inspected pilot trace sample.
- Zero unexplained differences between naive and forked environment outcomes under fixed actions.
- Numerical agreement between analytical and enumerated tiny-policy gradients within the declared tolerance.
- Writer rollout and teacher-forced likelihood agreement on actual sampled tokens within a tolerance established by the backend parity probe.
- No trainable reader parameters; every reader request resolves to the expected frozen model identity.
- No persistence outside charged memory on the strict track, including hidden counters or baseline dictionaries that contain task state.
- Complete accounting for aborted or rejected model actions and infrastructure reruns.

### 8.2 Scientific continuation criteria

Use an initial smallest practically interesting accuracy gain of 0.03 as a planning convention, not a publication threshold. Consider proceeding if the pilot indicates a plausible gain of that scale over F02, or a comparably meaningful decrease in GPU-hours to the same quality, with no clear override-performance collapse. Development uncertainty can be wide; the decision must include costs and effect distributions, not just a point estimate.

Stop or revise the proposed method claim if any of the following occurs:

- F03 only improves over F00, while F02 explains the gain.
- A simple public-semantics event ledger dominates across the meaningful cost/storage range.
- Improvements disappear when the fixed reader or exact storage accounting is imposed.
- Identifier randomization or unseen rendering templates remove the advantage.
- Gains require access to future branch labels or a raw-prefix archive at deployment.
- Nearly every reward is identical and the model receives no useful task signal after format learning.
- The projected core study exceeds the actual resource allocation, and completing it would require dropping required seeds or strong baselines.

One permitted response is a narrower empirical or negative-result paper, with honest positioning and stronger independent validation. Do not add unmotivated mechanisms solely to manufacture a positive result.

## 9. Natural chronological validation

General-memory QA benchmarks do not, by themselves, identify the effect of retaining a state before a later observation arrives. Treat that mechanism claim separately.

Create a small independent chronological dataset from a permitted source with observable real revisions, cancellations, or corrections. Suitable candidates include public issue discussions with corrected configurations or versioned procedural records, provided licensing and privacy permit use. Select a domain based on accessible data and verifiable labels, not the learned model's success. If suitable data cannot be obtained, record that the natural mechanism claim is unvalidated.

Required construction protocol:

1. Group all records from one real source entity into a single source cluster before splitting.
2. Choose a chronological cut using source timestamps and an objective rule. The writer ingests only the prefix before freezing a bounded store.
3. Reveal the actual later records as the suffix. Do not insert a synthetic correction and describe the resulting sequence as naturally observed.
4. Write questions whose answers are uniquely supported by the source chronology and the declared event semantics. Record evidence spans and any ambiguity privately.
5. Have independent reviewers check whether the earlier state, later update, and final answer are correctly understood. Keep disagreements and adjudication records. A second model using the same prompt is not automatically independent human validation.
6. Reserve complete source entities for final testing. Neither model checkpoint selection nor prompt selection uses their labels.
7. Report extraction errors, missing source context, and unresolved temporal ambiguity. Apply exclusions only from a prespecified annotation-quality rule before model comparison, with a full count.
8. Use the actually observed continuation as the primary natural track. Additional counterfactual continuations can be a separate semisynthetic stress test with that label.

A coding agent can build collectors, temporal manifests, annotation forms, evidence viewers, and scorers. It cannot honestly manufacture completed human annotation or independent evidence. This dependency must remain visible in the readiness report.

Untouched LongMemEval-S and LoCoMo remain valuable generic transfer checks. Use the full eligible official sets described in MEMRL_SPEC.md. Show LongMemEval by ability and LoCoMo by conversation. Consider LongMemEval-V2 only after a cost and information-access audit; its task definition and resource scale differ and must not be silently treated as a drop-in replacement.

## 10. Statistical analysis and reporting

### 10.1 Primary contrast family

Freeze three comparisons before opening final results:

| Contrast | Estimand |
|---|---|
| `FC1` | F03 minus F02 in source-averaged future accuracy at primary capacity under the supervision-matched schedule |
| `FC2` | F03 minus F02 in future accuracy at the frozen allocated GPU-hour budget |
| `FC3` | F03 minus the development-locked strongest non-RL baseline at the same information and memory budget |

The first two answer different questions and may favor different methods. Do not present whichever is larger as the only primary result. A resource frontier can be the main scientific object, but its construction and selection rules must be frozen.

Use paired source-cluster resampling for each contrast. Preserve all questions, future branches, variants, and transformed instances of a sampled source together. When training seed pairs are deliberately aligned across arms, use the original specification's crossed seed-by-source bootstrap; otherwise resample each arm's training seeds independently while retaining paired sources. A deterministic baseline does not become three independent systems because it is compared with three learned seeds.

Report per-seed differences and all seed curves. Three seeds provide limited information about training variability. If development reveals large variation, add two seeds before the final freeze or narrow the reproducibility claim. Do not wait for an unfavorable final p-value and then add seeds until significance appears.

Use 10,000 bootstrap replicates for the final planned intervals, with a pinned analysis seed. If confirmatory p-values are reported, control the prespecified FC1–FC3 family, for example with Holm correction after verifying the chosen test's assumptions. Bootstrap intervals with three seeds are approximate; do not describe them as exact population guarantees.

### 10.2 Power planning

Estimate the source-level paired standard deviation from development, choose the meaningful effect before final testing, and calculate an initial required independent-source count using the original specification's normal-approximation helper. Then simulate clustered outcomes with the observed branch and seed structure as a sensitivity analysis.

If the required count is unaffordable, report the minimum detectable effect for the affordable sample and freeze that design. Adding more questions to the same source does not necessarily compensate for too few independent sources.

Natural sources can be much more heterogeneous than synthetic worlds. Plan their sample size separately. Ten LoCoMo conversations support limited conversation-level population inference; show their individual effects rather than implying hundreds of independent histories from their question count.

### 10.3 Missingness and errors

Model failures remain failures in the task denominator. Infrastructure failures are rerun deterministically under the same logical job ID, and all attempts remain logged. If an infrastructure failure remains unresolved at analysis time, report the missing count and sensitivity bounds; do not silently count only completed examples.

The exact same eligibility and scoring rules apply to every arm. No post-hoc removal of long histories, unsuccessful writes, or hard update families is allowed.

## 11. Ablation registry

| Ablation | Changes | Holds fixed | Interpretation boundary |
|---|---|---|---|
| Static versus future reward | F00 versus F01/F02 | Initialization, representation, reader, storage | Effect of training target; not evidence for branching itself |
| Flat versus fork | F02 versus F03 | Future distribution and logical suffix supervision | Central method contrast; show compute differences |
| Diverse versus duplicated future | F03 versus F05 | K, suffix rollout count, independent suffix-policy draws | Value of different future observations, not just more samples |
| Current-only event representation versus full bounded event ledger | Remove historical event detail under explicit policy | Parser, reader, memory cap | Representation diagnostic; dropped controls can mechanically break updates |
| Common frozen reader | Re-answer saved stores with the same reader | Memory construction | Detect whether gains come from a different reader; no retraining on test memories |
| Suffix writer frozen at initialization | Updated prefix writer with W0 suffix updater, as an optional crossed component study | Prefix snapshots and suffix observations | Separates pre-fork retention from learned update execution; this is a different deployed policy |
| Targeted information restoration | Insert certified missing evidence with matched sham/capacity control | Reader and scoring | A local intervention, not a universal decomposition of error causes |
| Identifier permutation | Apply a semantics-preserving bijection | Latent world and updates | Detect lexical shortcuts; transformed samples remain clustered |
| Archive access reference | Allow raw history retrieval explicitly | Questions and model where feasible | Privileged storage comparison, never a strict-memory baseline |

Choose the minimal ablation set that resolves concrete alternative explanations. The optional crossed suffix-writer study is useful only if the main effect is positive and it is unclear whether preservation or update execution changed. The full original writer/reader factorial is not automatically needed for every capacity and branch count.

## 12. Required paper-ready artifacts

Produce these from validated result tables rather than by manually copying numbers:

1. A primary table with every arm's future, current, reveal, and override accuracy; all seeds; confidence intervals; and charged storage.
2. Accuracy versus allocated GPU-hours for F01/F02/F03, with clear supervision-matched and budget-matched endpoints.
3. A capacity curve separating matched-capacity training from capacity transfer.
4. Per-family and out-of-distribution results, including negative cases.
5. A baseline comparison that includes the bounded event ledger and the closest learned-memory comparator.
6. A compact mechanism table with retained/visible/usable evidence and explicit uncertainty.
7. Separate natural chronological and generic external transfer tables.
8. A resource table containing training, tuning, data construction, evaluation, peak GPU memory, CPU time, and persistent bytes.
9. For the systems route, runtime breakdown, numerical-equivalence results, scaling curves, and time-to-quality.
10. A machine-readable readiness report listing passed, failed, pending, and intentionally omitted requirements, with links to exact artifacts.

Do not write the abstract as if the proposed method wins before these tables exist. The final contribution statement must match the strongest contrast actually supported by results.
