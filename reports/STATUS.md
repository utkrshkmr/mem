# STATUS

**Phase:** 0 complete (Gate G0 passed); Phase 1 (GPU integration) next
**Updated:** 2026-09-25
**GPU-hours used:** ~0.2 (Phase 0: T0.5 model loads, run twice)
**Running jobs:** none

## Gate results

| Gate | Tests | Result | Notes |
|---|---|---|---|
| G0 | T1.1–T1.14 (CPU) | PASS, 78 tests | `pytest tests/unit -q` (includes the 30 Appendix A tests) |
| G0 | T0.1–T0.6 (GPU) | PASS, 6 tests | `pytest tests/gpu -m gpu -k T0 -q`; T0.4 initially failed on a test bug, see DEVIATIONS.md |

Per-test outcomes: `reports/tests/results.jsonl`; measured values: `reports/tests/<id>.json`; final run: `reports/gate_G0.log`.

Git commits and gate tags are **deferred at the owner's request** (2026-09-25): `gate-G0` is not tagged and nothing is committed until the owner says so.

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

## Open problems

- None yet.
