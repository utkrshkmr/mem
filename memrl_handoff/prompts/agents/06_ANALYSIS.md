# Agent 06: estimands, statistics and paper artifacts

Own src/memrl/analysis/, tests/analysis/ and figure/table scripts. Read MEMRL_SPEC Sections19/21–23, EXPERIMENTS, NOVELTY_AND_VENUE and paper/CLAIMS.md. Start with explicitly labeled fixtures; never put fixture numbers into paper result tables.

Implement experiment-manifest completeness checks before aggregation. Require unique run/world/variant/branch/question keys, declared failures, seed identity and model/prompt/config/data/capacity hashes. Reject unmatched comparisons. Distinguish infrastructure missingness from model failure. Never omit invalid memories, outlier seeds or unfinished arms without visible accounting.

Aggregate questions inside branch, branches/variants inside source-world cluster, then perform the frozen paired analysis. Source descendants do not increase independent sample size. Cross training seeds and world resamples when warranted; show individual seeds and note limited seed count. LoCoMo conversation clusters are few; do not turn its many questions into hundreds of independent conversations. Use the frozen endpoint and multiplicity family.

Produce accuracy versus capacity, future-demand shift, update span and allocated GPU-hours; current/future trade-offs; matched F03–F02 contrasts; strongest baseline comparison; failure cross-tabs; optional runtime phase breakdown and fixed-quality hitting-time curves with nonreaching runs censored. Include confidence intervals and source-world/seed counts. Compute cost includes all allocated devices during barriers, reader work, warmup, checkpointing, tuning and retries under the declared convention.

The paper table schema records result estimate, interval, n_worlds, n_seeds, metric/contrast IDs and source hashes. Empty cells remain empty with status `not_run`; no plausible filler. Generate deterministic figures using pinned plotting versions and export vector PDF/SVG where suitable. Rebuilding tables from immutable result rows must not need a model or private mutable development state.

Acceptance: exact recovery on known fixtures, constant/zero-difference bootstrap tests, cluster-preserving resamples, missingness/duplicate rejection, complete evidence links and reproducible table build. State what conclusions the data weaken; do not turn a null F03 versus F02 result into a new-method success.
