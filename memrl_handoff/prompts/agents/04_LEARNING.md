# Agent 04: on-policy training and branch gradients

Own src/memrl/rl/ and tests/learning/. Read MEMRL_SPEC Section14, CONTINUATION_METHOD Sections2–5, INTERFACES and RUNTIME_WIRING. Port the mathematical behavior of memrl_contract.estimator into tensor code; preserve the reference tests.

Implement shared format initialization when the development format gate requires it, ordinary rloo_pg, flat continuation augmentation and forked-future training. Prefix sampling is independent across g; hidden futures/questions are shared exogenously. Compute per-k leave-one-out across g, average those advantages for prefix calls, and use A[g,k]/K for suffix calls. Prefix rows occur once. Fork loss has H*G*Z denominator; flat ordinary full trajectories use H*K*G*Z. Never substitute group standard deviation normalization or PPO clipping.

Use exact generation-time prompt/completion IDs and behavior adapter hash. Teacher-force each call on its actual compressed context, not a concatenation of the history. Compute first completion token and EOS log probabilities correctly. Exclude prompt/padding/reader tokens. Sum completion log probabilities; use fixed Z=256. Select calls uniformly without replacement with inverse inclusion probabilities; zero-advantage groups stay in the original normalization.

Implement LoRA16 alpha32 dropout0, bf16 base, gradient checkpointing after compatibility check, one on-policy update per iteration, Adam settings from config, one global clip after accumulation and correct DDP mean compensation. All ranks participate even when a rank has no sampled real calls. Persist selected IDs, full coefficient factors, RNG, optimizer/scheduler and adapter hashes for resume.

Tests must cover exhaustive finite-policy gradients, K=1, K permutation, factor-of-K mistakes, flat versus fork normalization, unequal trajectory lengths, full versus repeated subsample expectation, padding masks, exact logprob parity, all-zero rewards, rank imbalance, single/two-rank gradients and optimizer update parity. Fault injection must reproduce the next complete update from the last committed checkpoint.

Acceptance: G05, real one-step update, DDP/resume evidence, learning health on development only. Do not implement the optional allocator until F03 survives the core controls and an explicit protocol exists for its importance weights. No trainer is done merely because loss decreases on a fake backend.
