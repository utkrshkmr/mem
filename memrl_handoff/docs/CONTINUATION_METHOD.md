# Continuation-training method contract

Protocol: `continuation_blind_v3`. Status: **candidate method, unverified novelty and unmeasured benefit**. This document specifies the experiment precisely enough to implement and falsify it. It does not promise that the method will beat ordinary data augmentation. Read the direct overlap analysis in `NOVELTY_AND_VENUE.md` before writing a paper introduction.

## 1. The question worth testing

Can training across several possible future updates improve the information a fixed-capacity writer preserves, relative to training on the same updates as independent complete histories, at a declared resource budget?

The setting differs from asking many questions about a final frozen memory: **the memory must first process previously unseen future events**. A correction can depend on an old value that was unnecessary for a current answer. All methods know the public event semantics and the training distribution. No method sees its realized future before creating the prefix memory.

An illustrative toy example has two independent histories. One contains transactions `p=2,q=8`; another `p=7,q=3`. Both currently total 10. A subsequent `cancel(p)` makes their correct totals 8 and 3. Saving the present total alone loses something required for the update. This is a test fixture and prior-motivated example, not a new theorem. A strong baseline that stores transaction identities and amounts may solve it; the real experiment must impose calibrated scarcity and uncertainty over which identities will be referenced.

Do not label future usefulness unknowable in all senses: its **distribution** is declared. The realized suffix and questions are hidden. No bounded store can preserve all independently random values for all possible future questions. Scope every claim to the tested distributions and capacities.

## 2. Exact random variables and information boundary

- `h`: a raw prefix world and its rendered chronological chunks. It is sampled independently of the writer's actions.
- `u[k]`: a future event sequence sampled from a declared training kernel `P_train(u | h)`. The simulator may inspect the complete world to generate valid events and gold answers. The writer receives only the eventual event text, in order, after the fork.
- `q[k,r]`: hidden reader questions for suffix k, drawn before writer collection or by an independent named RNG stream. Gold answers are private.
- `g`: one of G independent writer trajectories through the same prefix. Each produces a possibly different bounded memory `m[g]`.
- `s[g,k]`: a suffix writer trajectory starting from a deep clone of `m[g]` and processing `u[k]`.
- `R[g,k]`: mean frozen-reader score over the Q questions for the final memory of `(g,k)`. Rewards lie in [0,1]. No auxiliary shaping is used in the primary study.

Primary defaults are H=4 prefix worlds per iteration, G=4 prefix writer samples, K=2 futures, Q=8 questions per future. This is 16 prefix trajectories, 32 suffix trajectories and 256 reader requests before exact-request caching. A cache hit does not create another statistically independent observation.

All G writers for the same world use the same sampled suffixes and questions. Writer-action RNG is independent across g and across suffix trajectories. All questions remain hidden until the relevant writer has finished. The writer sees the same task-prior instruction in every matched arm; it never sees an operator label, hidden dependency annotation or suffix count unless explicitly present in the common public task description.

The simulator's raw prefix is **private audit state after ingestion**. It is not a writer-accessible archive, tool database, language-model KV cache, filesystem path, exception string or prompt attachment. Branches clone only the permitted charged memory and legitimate cursor/budget state for writing. Auditor state may be cloned in a separately inaccessible process to compute scores.

## 3. Objective and estimator

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

## 4. Collection pseudocode

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

## 5. Required controls

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

## 6. Data construction in detail

Keep four separate layers: abstract event generator, deterministic world interpreter, public renderer, and hidden query/continuation generator. A separately implemented interpreter validates all gold labels before model collection.

**Core public operations.** Start with `create(id, amount)`, `correction(id, new_amount)` and `cancel(id)`. Core values are nonnegative integers or exact nonnegative integer minor units; never binary floats. Signed amounts require a separately versioned extension. IDs are unique within a world and independent of amount, salience and future relevance. A correction replaces the active value of an existing uncancelled ID. Cancellation deactivates that ID. Repeated cancellation, correction of an inactive/unknown ID and reused creation IDs are deterministic invalid events with no state change; render the public invalid-event semantics consistently. The supplied toy interpreter is a correctness fixture for this subset. Do not silently extend it into version rollback.

**Additional production families.** Version replacement/revocation; temporal expiry; reference/alias dependency changes; and mixed two-step updates are separate interpreters with explicit semantics and tests. Implement only after the core accounting gate passes. A revocation family needs tombstones, replay policy and version ordering; an alias family needs cycle and missing-target behavior. Register every operation by version, including invalid-event rules. Never reuse a dataset name after changing semantics.

**Prefix generation.** Sample world seeds from distinct train/dev/test namespaces. Generate 16–512 independently useful records, with candidate counts and raw length varied separately. Values should resist reconstruction from entity names and priors. Render 8k/16k/32k reference-token training histories and controlled 32k main evaluations using the frozen chunker. Use background that is genuinely disjoint by source and renderer-family split. Record rejected infeasible count/length pairs; do not truncate useful events to make a grid fit.

**Fork boundary.** Select a boundary from `{0.50, 0.75}` of event progress using an exogenous seed before policy sampling. Snap to a legal frozen chunk boundary and record the actual token/event position. Both prefix and suffix must contain at least one chunk. Each future has at least one meaningful update and matched background; keep suffix-length strata comparable. Do not branch where the policy happens to make a high-loss action in the primary method.

**Future sampling.** For in-distribution training, draw target IDs from a declared mix of recently used, old and uniformly sampled valid IDs, with probabilities `(0.4,0.2,0.4)` initially. Resolve overlapping candidate sets through a documented mixture sampler, not by assuming disjoint sets. The mixture component is not public; only actual event text becomes public at its turn. Draw K futures independently conditional on the same complete prefix world. They may coincide naturally; log this. F05 explicitly duplicates one future. Lock the kernel and selection rule on development data before final use.

**Queries.** Sample questions from totals, current individual values, active-set membership and historical values where the family permits them. A family requiring historical answers must make the semantics and representation cost explicit. Use Q=8 shared questions across g per k during training, Q=16 per evaluation cohort. `unknown` is a typed answer, not an empty response. Ensure unknown-answer cases cannot be solved by a renderer marker or exceptional question length. Evaluate only declared questions; do not quietly make every fact relevant at every moment.

**Held-out shifts.** Independently vary target-age distribution, update count, operation composition, candidate count, capacity and renderer family. Report the single-axis shifts before intersections. Hold out at least one multi-operation composition while keeping its individual operators observed in training. Rename entity and event IDs, permute legally commuting independent events, and vary distractor placement. These are metamorphic checks within a source world, not new independent test worlds.

**Natural validation.** Synthetic continuation edits to conversation text are synthetic interventions. Real chronological update examples require source-licensed histories, a defensible timestamp cut, evidence that the later correction actually occurred, two independent human label checks with adjudication, and separation by person/document. A model paraphrase is not a new natural task. LongMemEval and LoCoMo remain untouched external transfer evaluations and cannot alone prove this particular mechanism.

## 7. Failure accounting and mechanism measurements

Evaluate final accuracy on every planned world/branch/question. Also report unconditional current accuracy, invalid action rate, insufficient memory errors, charged occupancy, whole-record retention in the symbolic track, and cost. A method that only improves future accuracy by breaking ordinary current tasks has a trade-off, not dominance.

Use `certified_present`, `certified_absent` and `uncertain` for sufficient information. Free-form summaries may encode derived sufficient state. Do not declare absence because the original amount string disappeared. A typed symbolic-memory track can support exact interpretation; a general text-memory track requires independent semantic audits and reports their uncertainty.

On preselected evaluation worlds, intervene with equal-token replacement of genuinely necessary old information versus matched irrelevant information, under the same frozen reader. Existing restoration methods are diagnostics and must be cited. Do not count any gold-informed restoration as ordinary test accuracy or train-time capability.

For branch statistics, aggregate questions within branch and branches within source world before the primary paired analysis. Variant pairs, paraphrases and ID renamings stay inside their source-world cluster. Report all training seeds, not only the best one.

## 8. Optional second method: allocate continuation work with explicit inclusion weights

This is a separately gated research idea, not part of the primary F03 configuration and not an asserted first use of importance sampling. Investigate it only if profiling shows continuation evaluation dominates and the core F03 result survives F02.

Generate a fixed exogenous pool of `K_pool=8` suffixes per world. A lightweight allocator trained only on completed **earlier development/training batches** predicts which suffix groups have high learning signal per unit cost. Before sampling current policy actions, form a mixture proposal `p[k] = epsilon/K_pool + (1-epsilon)*softmax(z[k])`, with epsilon=0.25, using public prefix features and prior statistics. No current rewards or final-test labels. Sample m=2 suffix indices with replacement; evaluate every g for each sampled index to preserve the leave-one-out comparison. For duplicate draws, use fresh suffix policy RNG and log the draws separately.

Replace every branch contribution `X[k]/K_pool` in the full-pool estimator by:

```text
(1/m) * sum_draws X[k_draw] / (K_pool * p[k_draw])
```

Apply this correction to both the prefix reward contribution and suffix loss contribution. Freeze and log p before collection; detach it in the writer loss. This estimator targets the same finite-pool average in expectation under positive support. The allocator has its own separately specified supervised objective; it is not implicitly trained through detached sampling probabilities. Check exact expectation in a small enumerated pool before any GPU use. Uniform allocation, unweighted biased allocation and the full-pool method are mandatory diagnostic controls; only the first and full-pool methods are fair objective-preserving comparators.

Candidate value: learning when future diversity is worth spending computation on while preserving the target objective. Required evidence: lower end-to-end time to fixed development quality and unchanged target estimator expectation, with overhead charged. This can still be incremental; do not add it merely to increase the number of components in the paper.

## 9. Stop criteria

Stop the method-novelty narrative if gains vanish against F02 with equal compute, against the strongest budgeted event/summary baseline, or after removing renderer and ID shortcuts. If improvement is entirely explained by more reader queries, report that rather than relabeling it as a retention algorithm. If evidence stays synthetic, state that limit. Publishability depends on a meaningful supported contribution, not the number of experiments or files produced.
