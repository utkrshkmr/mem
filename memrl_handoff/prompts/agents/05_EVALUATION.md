# Agent 05: baselines, frozen readers and evaluation

Own src/memrl/eval/, src/memrl/baselines/ and tests/evaluation/. Read MEMRL_SPEC Sections8/11/12/16/17/20, EXPERIMENTS and INTERFACES. Implement core baselines before using new-method results to choose convenient competitors.

Implement empty memory, recent raw window, FIFO, random, rolling summary, summary bank, structured facts, prompted writer, shared format control, rejection fine tuning, a close learned-memory reproduction/adaptation, and event-aware bounded ledger as staged in the spec. Charge all persistent state. FIFO and LRU are not separate arms without intervening reads. Full-context/full-archive/gold-informed references are explicitly privileged. Match common initialization, observation chunks and resource constraints; report unavoidable deviations.

Build frozen-reader all-retained evaluation that creates each memory once and reuses it for all hidden questions/cohorts. Reader uses no writer adapter. Typed synthetic scoring distinguishes exact integers, sets, dates and unknown; no substring gold checks. Current-state scoring and future-state scoring use different frozen snapshots without changing writer behavior. Build complete receipt and failure tables with actual prompt/snapshot hashes.

Adapt official LongMemEval and LoCoMo data with source revision, licenses, untouched full evaluation sets, original category mapping and official scoring where feasible. Alternative local judges are labeled and audited against human labels; no unsupported SOTA claim. These external tests never select prompts, checkpoints or hyperparameters. Preserve predeclared overlong-input and missing-evidence policies.

Implement support-present/absent/uncertain audit, equal-budget restoration/sham interventions and required attribution arms. Diagnostic improvements never enter ordinary accuracy. A parser for free-text memory cannot assume the absence of an original string proves information loss. Provide explicit uncertainty and alternative-support handling.

Acceptance: reader isolation; exact cache key; no question-dependent memory construction; all planned rows or accounted failures; strict final-test gate; baseline budget audit; official adapter fixtures; close-prior reproduction statement. Natural chronological annotations need real source and human evidence, not a generated proxy relabeled as natural.
