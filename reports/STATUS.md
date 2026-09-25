# STATUS

**Phase:** 1 complete (Gate G1 passed); Phase 2 (controls and level calibration) next
**Updated:** 2026-09-25
**GPU-hours used:** ~5.2 total (Phase 0: ~0.2; Phase 1: ~5.0 across four test runs, diagnostics and T2.13's 4-GPU runs; the plan estimated ~3 for Phases 0–1)
**Running jobs:** none

## Gate results

| Gate | Tests | Result | Notes |
|---|---|---|---|
| G0 | T1.1–T1.14 (CPU) | PASS, 78 tests | `pytest tests/unit -q` (includes the 30 Appendix A tests) |
| G0 | T0.1–T0.6 (GPU) | PASS, 6 tests | T0.4 initially failed on a test bug; T0.3 later needed its network call outside offline mode (see DEVIATIONS.md / below) |
| G1 | T2.1–T2.16 (GPU) | PASS | Full run `reports/gate_G1.log` (25/27; T0.3 and T2.11 failed on test issues), then `reports/gate_G1_rerun.log` (T0 + in-process T2: 20/20) and T2.16 again at the end |

Key G1 measurements (`reports/tests/<id>.json`):
- T2.3 HF vs vLLM at v0: mean |Δ| 0.0016, p99 0.0036, mean −0.0008 nats/token (limits 0.03 / 0.30 / 0.02).
- T2.4: v1 step changed vLLM log-probs by 0.53 nats/token; HF vs vLLM on fresh v1 samples within T2.3 limits.
- T2.5 reader frozen: max |Δ| 0.0.
- T2.7 **variant (a)**: with `VLLM_BATCH_INVARIANT=1` + per-request seeds, the two executors gave token-identical completions for all 128 calls (R10 not needed).
- T2.8 / T2.9 in fp32: cosine 0.9999998 / 1.0 (bf16 compute-noise report: 0.93 / 0.92, see open problem 1).
- T2.10 peak GPU memory: 62.5 GiB (N=128,K=4), 62.5 GiB (N=32,K=16,B=512), 63.8 GiB (N=32,K=16,B=1024); limit 78.
- T2.12 resume from checkpoint 10 with opt_step 10, finishing at 14 with 14 metrics lines; T2.13 four concurrent jobs, no port/IPC errors; T2.14 NVML UUID mapping correct.

Per-test outcomes: `reports/tests/results.jsonl`.

Git: Phase 0 committed and pushed to `github.com/utkrshkmr/mem` (`master`, tag `gate-G0`) at the owner's request; Phase 1 committed locally and tagged `gate-G1`.

## Environment record (PLAN.md §5.2)

- Node: `iad-cmp2.cse.buffalo.edu`, 4× H100 80GB HBM3 (NV6 all-to-all), 128 cores, 1 TB RAM.
- Driver 570.211.01 (< 580) → **install path B**: `vllm-0.30.0+cu129` wheel from the GitHub v0.30.0 release, torch from the PyTorch cu129 index (`--index-strategy unsafe-best-match` so non-torch packages come from PyPI). CUDA 12.9 runtime on a 12.8 driver works through CUDA minor-version compatibility.
- Installed: torch 2.13.0+cu129, vLLM 0.30.0+cu129, transformers 5.17.0, PEFT 0.21.0, huggingface_hub 1.33.0, triton 3.7.1, numpy 2.3.5, scipy 1.17.1. Full list: `env/requirements.lock`.
- Model: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` in `models/` (lock: `configs/model_lock.yaml`).
- Storage: the repo (including `models/`, `runs/`, `profiles/`) is on NFS (`iad-fs:/home`, 36 TB free). Local NVMe exists at `/scratch` (357 GB free) and `/data_local` (2.5 TB free) if NFS latency ever shows up in `t_sync` or model load times.

## Changes outside the verbatim Appendix A files

- `pyproject.toml`: added `namespaces = false` under `[tool.setuptools.packages.find]` (A.9 allows extension) so `kmatters_reference/` is not installed as a namespace package.
- `.gitignore`: added `.venv/`, `models/`, `runs/`, `profiles/`, `variance/`, `jobs/`, `*.safetensors`, `*.pt`, `.pytest_cache/`, `*.egg-info/`.

## Implementation notes

- `run_queue.py` resume mode runs the job's `resume_cmd` when present, otherwise `cmd` again; a job that exits 0 without writing `DONE` counts as failed.
- `q3.py`: in a bootstrap draw where the additive model finds no cell within budget while a measured-feasible cell exists, the draw counts as "K differs" with regret +inf (the count is reported as `frac_model_no_choice`).
- `q1.py` Q1b bootstrap recomputes the target accuracy a* from each resampled set of runs.

## Phase 1 implementation notes (2026-09-25)

- vLLM 0.30 API matches PLAN.md §5.2 with no drift. `top_k=-1` is still accepted as "disabled" (0 is the new default). Requests with `prompt_logprobs` skip the prefix cache automatically, so `score_tokens` is exact.
- Measured on GPU 0 (tiny config): engine start ~40 s with a warm compile cache (82 s cold); vLLM holds 40.4 GiB (28 GiB KV, 7.8 GiB weights, CUDA graphs).
- **Trainer allocator:** the PyTorch caching allocator reserved 37.6 GiB for a 17.4 GiB peak (fragmentation from variable microbatches), putting the GPU at 78.4 GiB. The trainer process now sets `expandable_segments:True` at runtime (vLLM's process is unaffected); total is 61.2 GiB, matching the §2 estimate.
- `Trainer.step(lr=...)`: an lr override applies to that step only (restored afterwards), so profiling's lr=0 can never leak into training.
- The loss uses the verbatim `pack_microbatches` / `completion_logprob_sums` with a loop identical to `accumulate_policy_gradient`, also returning per-token HF log-probs for the diagnostics from the same forward pass. A non-finite loss or gradient norm raises before `opt.step()`.
- Evaluations are attached to versions in `runs/<id>/evals.jsonl` (version 0 included); the metrics line of iteration `it` also embeds the eval of version `it+1` when one ran (`eval_version`).
- Memory numbers (`gpu_mem_peak_gb`, T2.10's 78 limit) are NVML GiB, the same unit as the GPU's "80 GB".
- Profiling shards write `profiles/<tag>/shard{i}of{n}/cells.jsonl` (no concurrent appends to one file on NFS); a cell's rows are written together after all its reps. Request intervals are saved per rep in `intervals/*.npz` so occupancy can be recomputed with the measured `n_sat`.
- `make_jobs.py` writes absolute interpreter paths and a per-job `cwd`; `run_queue.py` runs each job in its `cwd`.

## Open problems

1. **bf16 compute noise in gradients (found in Phase 1; owner decision needed before Phase 3 variance results are used).** In bf16 the HF forward/backward depends on the microbatch's padded length: a call's final hidden states move 2–3% when its batch is padded differently (fp32: ~5e-6, so the code is correct). Two packings of the same 48 calls give gradients with cosine 0.93 (T2.8 in bf16), and the variance tool's mean Z vs the training step's gradient has cosine 0.92 (T2.9 in bf16). In fp32 both are ≥ 0.9999998. This noise is part of every training gradient (so it is part of the system being studied), but in the variance tool it is independent per backward pass and so **adds to the measured V_future and V_prefix**. Proposed: in Phase 3, measure its share directly by recomputing a subset of Z_ik under a second packing (e.g., each future's calls alone vs. batched with another prefix's calls), and report V components with that noise share. Nothing in §11 changes. See `prereg/DEVIATIONS.md` for the test changes this caused.
2. **Per-iteration cost is training-dominated.** T2.10's base-policy iteration at N=128, K=4, L=16 took 447 s (338 s training, 108 s rollout); N=32, K=16 took 281 s (199 s training). The G4 projection will use measured pilot times, as planned.
