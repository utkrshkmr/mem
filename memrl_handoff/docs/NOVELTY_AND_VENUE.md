# Research decision: a credible project is not yet a conference-ready paper

Date of literature check: 2026-09-23. This is a research decision document, not a claim that any experiment succeeded. Read the experiment registry in [EXPERIMENTS.md](EXPERIMENTS.md) and the implementation contract in [MEMRL_SPEC.md](../MEMRL_SPEC.md).

## 1. Direct answer about publication

The revised specification fixes serious experimental weaknesses. It does **not** make the project ready for ICLR or MLSys by itself. Acceptance cannot be guaranteed by an idea, by four H100s, or by a large experiment matrix. The current state is a well-specified research hypothesis with an unusually crowded neighborhood of prior work.

The most defensible next investment is a small falsification pilot. The pilot must determine whether learning from several possible future updates gives a useful accuracy–cost improvement over learning from the same continuations as ordinary independent training examples. If the answer is no, a new acronym, additional datasets, or more elaborate agent orchestration will not create an algorithmic contribution.

This handoff deliberately separates an executable engineering foundation from unproved scientific claims. It does not include trained checkpoints, measured H100 throughput, benchmark improvements, or a verified novelty guarantee.

## 2. The important novelty collision

**The proposed failure mode already has direct prior work.** Zhang's September 17 preprint, [Correct Now, Insufficient Later](https://arxiv.org/abs/2609.20045), studies histories with the same present answer that require different answers after a shared update. It includes a bounded-memory audit and an elementary collision argument. Its reported evidence is a small synthetic pilot, with no independent natural-task validation. Accordingly, neither this distinction nor that argument is a new contribution of our project. The candidate contribution must be a demonstrably useful learned solution and its resource trade-off.

Do not bury this paper in a long bibliography. Discuss it in the introduction and state precisely what is added. A recent preprint counts as relevant prior work regardless of its venue status. Do not characterize the project as the first study of update-sufficient memory.

## 3. Prior-work map and resulting requirements

The entries below summarize primary sources checked for this handoff. They are a scoped audit, not an exhaustive novelty certificate. Recheck new versions before freezing a submission. Do not copy a paper's numerical results into our comparison table unless its protocol is reproduced or its original setting is clearly separated.

| Prior work and primary source | Already present in that work | Consequence for this project |
|---|---|---|
| [Memory-R1, ACL 2026](https://aclanthology.org/2026.acl-long.583/) | Learning memory-management operations from downstream outcomes | CRUD tools plus policy gradients are background, not the novelty claim. Include a close learned-memory comparator. |
| [MemAgent](https://arxiv.org/abs/2507.02259) | RL for bounded streaming memory over long inputs | Long histories, recursive rewriting, and improved QA alone do not establish a new contribution. |
| [UMA, version 2](https://arxiv.org/abs/2602.18493v2) | Reusable question-blind memory, multiple QA branches supervising that memory, and Ledger-QA for evolving state | Averaging several question rewards is not new. Distinguish **new observation continuations before QA** from branching only into questions, and compare against an adaptation that shares our information budget. |
| [ChronoMem, version 2](https://arxiv.org/abs/2607.27773v2) | Memory snapshots, version history, and semantic rollback | Rollback and snapshotting are not new. Its retained historical versions must be charged or declared an archive capability in a comparison. |
| [MEMAUDIT](https://arxiv.org/abs/2605.02199) | Budgeted memory-writing evaluation with explicit evidence units, representation choices, and certified optima within a fixed candidate package | A bounded evidence audit, tombstone check, or tiny exact optimizer is a correctness instrument. Do not advertise it as the first exact memory evaluation or a universal compression optimum. |
| [What Eviction Destroys](https://arxiv.org/abs/2609.08279) | Restoration-based analysis separating eviction losses from failures to use retained information | Evidence restoration is a diagnostic control, not our headline novelty. |
| [TRACE](https://arxiv.org/abs/2608.06503) | Paired closed-loop continuations for evaluating compression events and optimizing a compression prompt | Future-execution evaluation and continuation-guided compression already have precedent. Include a continuation-informed prompt baseline when feasible. |
| [Branching Policy Optimization](https://arxiv.org/abs/2607.14171) | Sandbox snapshots, branched rollouts, and branch-based credit assignment | A rollout tree or branch baseline is not by itself a new RL algorithm. Our fixed exogenous-future sampling is a different experimental construction, not a claim to invent branching. |
| [RTMC](https://arxiv.org/abs/2604.11037) | Step-level return estimates from rollout trees and matching shared intermediate states | Avoid claiming to invent prefix-based credit assignment or tree Monte Carlo. |
| [Schedule-Level Shared-Prefix Reuse](https://arxiv.org/abs/2606.01143v3) | Reordering prefix/suffix forward and backward work with gradient-equivalence checks | Prefix reuse alone is an insufficient MLSys contribution. Compare with existing optimized inference and training, and identify a remaining measured bottleneck. |
| [MemoryWalker](https://arxiv.org/abs/2609.00865) | Training/inference conditioning errors created by compressed histories, with exact and approximate corrections | Preserve every actual rollout prompt. Reconstructing a call from the final memory can create invalid likelihoods. Correct replay is required engineering, not our novel method. |
| [OASES, version 3](https://arxiv.org/abs/2604.03675v3) | Outcome-aligned intermediate evaluation and co-training for agentic search | Learned state evaluation and process credit have related precedents. The live title at this identifier is OASES; do not cite a stale search-result title such as PRAISE. |
| [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | Environment-experience memory evaluation, including dynamic-state tracking | Existing evaluation is not uniformly static. Its large histories make it an optional, separately costed evaluation; it does not automatically substitute for a causal continuation experiment. |

The original specification's remaining bibliography still applies. This table adds close neighbors that materially change the research claim. Implementing every cited system is unnecessary. Choosing only weak convenient baselines is unacceptable.

## 4. The candidate research contribution

**Working question:** Under a hard one-pass storage cap, can a memory writer trained against several possible future observation sequences preserve update-relevant distinctions more effectively per unit of training compute than strong structured memory and ordinary continuation-augmented RL?

The internal method identifier is `F03_fork`; “forked-future training” is descriptive shorthand, not a uniqueness claim for an acronym.

For each public training prefix, sample four independent writer trajectories. For each resulting bounded memory, run two hidden exogenous continuations generated from the underlying training world, then evaluate eight hidden questions with the same frozen reader. The two continuations and questions are shared across the four writer candidates. A continuation can reveal a previously irrelevant dependency, reverse an update, cancel one transaction, or legitimately replace a previous answer. The writer must handle the observed suffix using only its bounded memory and the new chunks. It never sees the alternative future or a hidden answer.

The reward is the mean future-question score. The implementation uses an explicitly derived on-policy score-function estimator; it is not a claim to have invented a new class of RL. Shared-prefix work is counted once, and branch losses carry their specified averaging weights. See the method section of MEMRL_SPEC.md for the estimator and independence requirements.

The primary environment supplies observation sequences independently of the writer's actions. This isolates retention and updating. It does not establish performance in an interactive world whose future changes in response to the agent's choices; that would require a different environment and sampling analysis.

The central distinction is empirical and operational: **Does sharing one retained state across alternative possible continuations make training more useful under the same resource envelope?** The answer could be no. A flat training set containing the same suffixes may work equally well, or an event-aware bounded ledger may already be better and cheaper.

### 4.1 What would count as a meaningful advance

All of these are evidence requirements, not predictions:

1. `F03_fork` improves on `F02_flat`, which sees the same continuation distribution with independently sampled memory prefixes. It must not merely beat a current-only writer that never trained on updates.
2. The improvement survives a fixed reader, exact charged persistence, identical observation boundaries, and substantial memory pressure.
3. There is a useful accuracy–GPU-hour or accuracy–storage trade-off, with the complete curve and all seeds shown. Equal optimizer steps alone do not show efficiency.
4. Effects survive randomized identifiers, unseen rendering templates, longer dependency spans, and held-out combinations of update operations. A naming convention must not reveal future relevance.
5. The method accepts legitimate overrides as well as preserving old dependencies. Retaining obsolete values at the cost of failing to update is not success.
6. At least one independent natural chronological evaluation supports the intended practical claim. Untouched general-memory benchmarks provide additional transfer evidence with narrower interpretation.
7. An error analysis distinguishes retention, update execution, and reader failures without excluding unsuccessful memories from the primary denominator.

The strongest possible paper would explain **when** alternative-future supervision helps, **why** it helps, and **when a simple event representation wins**. One aggregate leaderboard improvement is much weaker.

### 4.2 What is explicitly not novel here

- Using LLMs as memory writers or readers.
- Training memory operations with RL or averaging several downstream question rewards.
- The fact that a correct current summary can omit facts required later.
- The elementary indistinguishability argument for identical compressed states.
- Storing transaction identifiers, rollback information, dependencies, or tombstones.
- Snapshotting an environment or branching model rollouts.
- Running on four GPUs, using LoRA, caching common prefixes, or providing an agent harness.
- Testing on synthetic state tracking, LoCoMo, or LongMemEval.

These can be necessary ingredients in a strong contribution. Combining them does not establish originality automatically.

## 5. Alternative route for MLSys

MLSys is a plausible route only if the project uncovers and solves a substantial systems problem. A fast implementation of an ordinary experiment does not automatically meet that bar.

A possible systems thesis is that memory-learning workloads contain a distinctive combination of long shared ingestion, bounded state snapshots, many frozen-reader jobs, and short independently updated suffixes. A scheduler might exploit those phases while maintaining exactly the intended policy likelihoods and statistical weights. This is a hypothesis to profile, not a promised new system.

Required evidence includes:

| Question | Measurement and comparison |
|---|---|
| Where is time spent? | Separate writer prefill, writer decode, reader prefill/decode, adapter publication, CPU event processing, and queue waits. |
| Is the optimized computation correct? | Compare output states, sampled-token replay log probabilities, scalar losses, and gradients against a naive semantic implementation. Use declared numerical tolerances. |
| Is the comparison strong? | Enable ordinary batching and available prefix caching in the baseline. Compare against a suitable established prefix-reuse approach where implementation access permits. |
| Does optimization improve useful work? | Measure independent worlds processed and validated branch outcomes per allocated GPU-hour, along with memory pressure and deadline misses. |
| Does the system help learning? | Time to a fixed development-quality threshold, with censoring when a run never reaches the threshold. Final test scores are not used to choose the threshold. |
| Does it generalize beyond one trace? | Sweep prefix/suffix ratios, branch count, memory capacity, writer/reader latency ratios, and at least two model sizes or architectures justified by resources. |
| Is four-GPU behavior understood? | Report one-, two-, and four-GPU measurements when meaningful, both fixed-workload speedup and per-GPU efficiency. A change in workload is not strong scaling. |

Copy-on-write CPU memory snapshots and correct weight-keyed caches are useful engineering. A submission needs a defensible contribution beyond their existence. Reusing stale KV states after an adapter update is invalid, even if it makes the benchmark faster.

If all observed gains come from enabling an existing engine's standard caching option, report that engineering result honestly and reconsider the MLSys claim.

## 6. Venue decision after development

| Evidence actually obtained | Honest framing | Decision |
|---|---|---|
| Better experimental controls, no meaningful new result | Reproducible implementation and research infrastructure | Useful project; not yet an A* submission. |
| RL beats weak summaries but loses to a strong bounded event ledger | Representation/benchmark finding | Investigate breadth; do not claim a superior learned memory method. |
| Future training beats static training; fork and flat are indistinguishable | Benefit of continuation data | No evidence for a new branching algorithm. A careful empirical paper may still need substantial new insight. |
| Fork improves the cost/quality frontier beyond flat and close learned baselines, with robust mechanism and transfer evidence | Learned continuation-aware compression | Potential ICLR-style method contribution, subject to actual novelty, significance, and review. |
| Correct runtime optimization produces substantial gains over strong systems baselines across workloads and improves time to quality | Systems contribution for memory-learning workloads | Potential MLSys-style contribution, subject to actual novelty and review. |
| Only a synthetic advantage on generator-specific cues | Benchmark overfitting | Stop the broad claim; repair the generator and rerun development. |

These are judgments about fit, not official venue acceptance rules. Check the current call for papers, artifact policy, anonymity rules, page limits, and AI-assistance disclosure requirements when choosing an actual submission cycle. No particular deadline or policy is assumed here.

## 7. How to avoid an unconvincing AI-generated research paper

Do not try to hide AI assistance through prose. Establish scientific ownership and evidence:

- Keep a claim registry that links every claim to a specific result, baseline contrast, and limitation.
- Make the strongest competing explanation explicit before running the expensive study.
- Report the simple baseline that works, the update family that fails, and all trained seeds.
- Have the researcher inspect representative raw trajectories, memory edits, scoring failures, and a sample of labels. An agent-written explanation is not an independent audit.
- Distinguish a new contribution from an implementation of a known idea in a new setting.
- Use source-versioned citations and complete benchmark provenance. Do not cite generated summaries as technical evidence.
- Keep human-authored scientific decisions in dated decision records. These should explain choices, not merely record a sequence of tool calls.
- Follow the selected venue's actual disclosure policy; do not invent one or imply that an unreviewed draft has complied.

The paper should be built around one falsifiable claim. A collection of unrelated novelty-sounding mechanisms makes attribution harder and often weakens the work.

## 8. Mandatory novelty gate before full training

Create `reports/novelty_gate.json` with:

```json
{
  "status": "pending",
  "checked_at": null,
  "main_claim": "F03_fork improves the useful accuracy-cost frontier beyond F02_flat under bounded one-pass memory",
  "closest_work": [
    "https://arxiv.org/abs/2609.20045",
    "https://arxiv.org/abs/2602.18493v2",
    "https://arxiv.org/abs/2608.06503",
    "https://arxiv.org/abs/2607.14171"
  ],
  "increment_beyond_each": [],
  "pilot_artifacts": [],
  "unresolved_overlap": [],
  "decision": "Do not assert novelty before the pilot and close-work comparison"
}
```

`pending` is a genuine research status. The coding agent may complete correctness infrastructure and pilots while this is pending. It must not change the value to `passed` because code executes, or launch the full optional sweep on that basis. The experiment gate is satisfied by recorded scientific evidence and an explicit decision, not by optimistic text generated by another agent.
