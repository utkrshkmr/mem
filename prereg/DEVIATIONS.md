# Deviations from PLAN.md

Each entry: date, what changed, why. Nothing in `prereg/PREREG.md` changes after Gate G4 except through an entry here.

## 2026-09-25: T2.8 and T2.9 checked in fp32; bf16 shape noise reported (Phase 1)

**Finding.** In bf16, the Qwen3-4B forward pass depends on the padded sequence length T of the microbatch. Padding a call by even one token changes its final hidden states by 2–3% relative (the divergence appears around layer 17–27); changing only the batch size gives bit-identical results. The same model in fp32 agrees to ~5e-6 across every padding and batch variant, so masking, positions and packing in the Appendix A code are correct. `allow_bf16_reduced_precision_reduction=False` does not change this: it is bf16 rounding order under different kernel tilings, compounded through 36 layers.

**Effect on tests.** T2.8 (gradient at microbatch budgets 32768 vs 4096) measured cosine 0.931 and T2.9 (variance-tool mean Z vs training gradient) 0.920 in bf16, against thresholds of 0.9999. Those thresholds assume shape-invariant numerics, which no correct bf16 implementation provides.

**Change.** T2.8 and T2.9 now run on the real model in fp32 (separate process, after the vLLM engine is shut down) with the plan's thresholds unchanged (cosine >= 0.9999, relative L2 <= 1e-2). The bf16 values are still measured and recorded as the size of bf16 compute noise, with a loose sanity bound (cosine >= 0.8) that catches gross errors. Training itself stays in bf16 as the plan specifies.

**Open question for the owner.** bf16 compute noise is independent per backward pass, so it adds to the measured V_prefix and V_future (Phase 3/6). Its size should be measured before the variance results are used (see STATUS.md).

## 2026-09-25: T2.4 adapter-agreement clause measured on on-policy samples (Phase 1)

T2.4's first clause is unchanged (the v1 step moved vLLM's log-probs of the same 16 sequences by 0.53 nats/token, far above 0.02). The second clause (HF(v1) vs vLLM(v1) within T2.3 thresholds) failed only on the 99th percentile when scoring the *old v0 sequences* under v1: p99 0.50 (mean |Δ| 0.016, mean −0.001, both passing). With lr = 1e-3, Adam's first step moves every LoRA weight by about 1e-3, which pushes those tokens far off-policy (log-probs near −20). There, vLLM's bf16 LoRA weights (vLLM 0.30 accepts only fp16/bf16 for `lora_dtype`) and HF's fp32 LoRA diverge more. On fresh samples drawn from v1 itself, HF vs vLLM meets all T2.3 thresholds (diagnostic: mean |Δ| 0.012, p99 0.19, mean −0.001). The second clause now uses fresh v1 samples (the regime training uses, and a direct check that vLLM serves the new adapter). The old-sequence numbers are still recorded.

## 2026-09-25: T0.3 whoami runs with offline mode unset (criterion unchanged)

The GPU-test conftest added in Phase 1 sets `HF_HUB_OFFLINE=1` for every GPU test, so T0.3's `HfApi().whoami()` network call was refused in the full gate run (`OfflineModeIsEnabled`). The token grep itself found nothing. T0.3 now makes the whoami call in a subprocess with `HF_HUB_OFFLINE` unset.

## 2026-09-25: T2.11 cycles reuse one set of data keys; T2.4 restores v0 afterwards (criteria unchanged)

In the gate run, T2.11 measured cycle times 32.6 / 19.9 / 28.7 s (cycles 2 vs 3: 44%). Each cycle used different data (a different iteration index) on the far-moved v1 policy left over from T2.4, and at N = 16 one long chain sets the rollout's critical path. A diagnostic on GPU 1 at v0 gave 42.3 / 41.4 / 41.9 / 42.9 s with the same data keys and 25.4 / 26.2 / 39.3 / 33.0 s with different keys. The test is about the loop (no hang, no progressive slowdown), not data variance, so its three cycles now reuse one set of data keys (still full rollout → train → sync cycles at the normal learning rate). The T2.4 fixture also restores the v0 policy and a fresh optimizer once its measurements are done, so later tests do not inherit v1. Thresholds unchanged.

## 2026-09-25: T0.4 test implementation fix (criterion unchanged)

The first T0.4 run failed on the `LoRARequest(lora_name, lora_int_id, lora_path)` check because the test took the first three parameter names from a `set`, which has no order. The installed vLLM signature is correct. The check now reads the ordered `inspect.signature(...).parameters`. The pass criterion in PLAN.md §9 is unchanged.
