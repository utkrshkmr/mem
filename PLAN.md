# PLAN.md: Does K matter? A fixed-K memory-writer RL study on one 4×H100 node

**Audience:** Claude Code, implementing and running this project end to end on a single node with 4× H100 80GB.
**Owner:** UK (utkarshkumar0105@gmail.com). Ask the owner before doing anything listed under "Ask first" in §4.
**Version:** 1.0, written 2026-09-25. The library versions in §5 were checked on that date (vLLM 0.30.0, transformers 5.17.0, PEFT 0.21.0, torch 2.13.0 via vLLM).

---

## 0. How to use this document

1. Read §1–§4 fully before writing any code.
2. Build the project in the order of §10 (Phases 0–7). Each phase ends in a **gate**, a list of tests from §9 that must pass before the next phase starts.
3. The reference code in **Appendix A** has been tested (CPU: torch 2.14, transformers 5.17, PEFT 0.21) and must be used verbatim for the files it covers. It is the source of truth for the environment semantics, the estimator weights, the loss, the variance decomposition, and the K* comparison. If `kmatters_reference.zip` sits next to this file, unzip it and copy the contents of its `kmatters_reference/` folder into the repo root. Otherwise, create the files from Appendix A.
4. The decision thresholds in §11 are **predeclared**. Do not change them after Phase 4 ends. Record any unavoidable deviation in `prereg/DEVIATIONS.md` with the reason and the date.
5. Keep `reports/STATUS.md` current: phase, gate results, running jobs, GPU-hours used, open problems.

---

## 1. Goal: three questions and what their answers mean

The system under study is an RL-trained **memory writer** with a frozen **reader**:

- A **prefix** (history) is `L` chunks of ledger events. The writer (base LLM + LoRA, the only trained part) reads one chunk at a time and rewrites a bounded text memory (at most `B` tokens) after each chunk. That is `L` sequential writer calls per prefix.
- A **future** starts from the memory the prefix produced. It has `U` more chunks (more writer calls), then one question. The frozen **reader** (the same base model with no LoRA) answers from the final memory only. The reward is 1 if the integer answer is exact, else 0.
- Each training iteration samples `N` independent prefixes and `K` independent futures per prefix. `F = N·K` futures per iteration is held fixed in the learning sweep.

Building a prefix is expensive (`L` calls); a future is cheaper (`U` writer calls plus 1 reader call). Futures from the same memory are correlated samples. The overall research question is whether choosing `K` well matters, and whether it needs cost-aware or adaptive treatment. This plan answers three narrower questions, each with a predeclared decision rule (§11):

| # | Question | Main measurement | What each answer implies |
|---|---|---|---|
| **Q1** | **Does K matter for learning?** (a) per sampled future at fixed `F`; (b) after pricing each arm with measured GPU time | Fixed-K sweep, K ∈ {1,2,4,8,16}, F = 64, 3 seeds; eval accuracy curves; gradient-variance components V_prefix and V_future at checkpoints | **Flat after pricing:** allocation is irrelevant, stop. **Best K always at an extreme:** no interior optimum, so no allocation problem. **Interior optimum:** the premise survives |
| **Q2** | **Does exploitable capacity survive competent scheduling?** | Cost surface T(N,K) from a competent asynchronous executor (all histories concurrent, futures start as soon as their memory exists, tuned vLLM scheduler), compared with a barrier executor; marginal completion time of extra futures vs their saturated cost; occupancy | **NO:** extra futures cost what they cost, so there is no "free future" story. **YES but immaterial:** capacity exists, but rollout is too small a share of iteration time to matter |
| **Q3** | **Does marginal-cost pricing move K\*?** | Measured V_prefix and V_future plus the measured T(N,K) surface, compared with the additive average-cost model C = N(C_p + K·C_f) | **NO:** a one-time profile plus the square-root rule suffices, and there is no systems-specific allocation contribution. **YES:** the measured surface changes the chosen K by at least a factor of 2 with at least 10% regret |

**Out of scope** (do not build): asynchronous or off-policy RL, multi-node, multi-GPU data-parallel single runs, online adaptive allocators, other allocation baselines (EPIG, TRACE, and so on), PPO clipping or multiple epochs, KL penalties, reasoning or thinking models.

---

## 2. Is this feasible on one 4×H100 node?

**Yes.** All three questions can be answered on one node if every training run uses a single GPU. The four GPUs then run four independent jobs at a time.

- **Model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, non-thinking, 36 layers, GQA 32/8, supported by transformers ≥ 4.51 and vLLM). The writer is this model plus a LoRA (r = 16, all linear layers, about 33M trainable parameters). The reader is the same base weights without LoRA, so **one vLLM engine per GPU serves both roles**.
- **Colocation per GPU:** vLLM (bf16 weights about 8 GB, KV cache 28 GiB, CUDA graphs) plus the HF training copy (bf16 weights about 8 GB, fp32 LoRA plus Adam state under 1 GB, activations with gradient checkpointing about 10–15 GB). The total is about 60 GB of 80 GB. Rollout and training alternate, so they never compete for compute.
- **Why not 7B?** A 7B model roughly doubles every phase. It is feasible for 3 seeds but leaves no slack for reruns. `configs/base.yaml` makes the model switchable. Keep 4B unless the owner asks otherwise.

**Estimated budget.** These are planning estimates only. Replace them with measured numbers at Gate G4.

| Phase | Work | GPU-hours | Wall-clock on 4 GPUs |
|---|---|---:|---:|
| 0–1 | Setup, integration tests | ~3 | 0.5–1 day, mostly development |
| 2 | Controls and difficulty calibration | ~2 | ~1 h |
| 3 | Microbenchmarks, cost grids, iteration-0 variance | ~16 | ~5 h |
| 4 | Learning-rate pilot (6 runs × 60 iterations) | ~12 | ~5 h |
| 5 | Main sweep: 5 arms × 3 seeds × 250 iterations | ~120 | ~30–36 h |
| 6 | Variance at trained checkpoints, trained-policy cost grid | ~8 | ~3 h |
| 7 | Analysis and report | CPU | ~1 h |
| | **Total** | **~160** | **about 4–5 days end to end** |

Per-iteration estimates at L = 16, B = 512, F = 64: K = 1 (N = 64) about 3 min; K = 16 (N = 4) about 1.2 min. Training forward/backward dominates at small K, and latency-bound prefix construction dominates at large K.

**Limits of a single node.** There are no inter-node placement effects, and the "GPU count" dimension of a conditional policy K(L,B,G) is not studied. Neither is needed for Q1–Q3.

---

## 3. Design at a glance

```
Iteration (policy version v, one GPU):
  sample N prefixes (fresh histories), K futures each      F = N*K = 64 in the sweep
  ROLLOUT (vLLM, one engine, LoRA v for writer, base for reader)
     prefix i: L sequential writer calls  -> memory M_i
     future (i,k): U writer calls from M_i -> M_ik ; reader(M_ik, question) -> r_ik in {0,1}
  ADVANTAGES (leave-one-prefix-out baseline b_i)
     A_pre_i = mean_k r_ik - b_i        A_fut_ik = r_ik - b_i
  LOSS  = -(1/T_norm) * sum_calls w_c * sum_tokens log pi(token)
     w_c = A_pre_i / N        for each of the L prefix calls of prefix i
     w_c = A_fut_ik / (N*K)   for each of the U future calls of future (i,k)
  one AdamW step on LoRA; save adapter v+1; hot-load into vLLM
```

| Experiment | Purpose | Phase |
|---|---|---|
| Controls (oracle, empty, shuffled memory) and level calibration | Task validity; base accuracy of 0.20–0.60 | 2 |
| Microbenchmarks (decode and prefill throughput vs batch size; training tokens/s) | Saturation point n_sat; saturated per-token costs | 3 |
| Cost grid: T(N,K) for executors {async_ready, barrier}, L ∈ {4,16}, B ∈ {256,512,1024} | Q2, Q3 | 3, 6 |
| Variance components V_prefix and V_future at checkpoints, L and B conditions | Q1 mechanism, Q3 | 3, 6 |
| Learning-rate pilot on K ∈ {1,16} | Choose one shared lr; check that learning happens | 4 |
| Fixed-K sweep, K ∈ {1,2,4,8,16}, seeds {0,1,2} | Q1 | 5 |

---

## 4. Ground rules for the implementing agent

**Engineering**
- Python 3.11, one virtualenv in `.venv` (made with `uv`), one git repository. Commit after every green test suite. Tag gates as `gate-G0` … `gate-G6`.
- There must be **one code path** for training, evaluation and profiling. `profile_costs.py` must call the same `run_iteration()` that training uses, with `lr=0`.
- Train on **exact vLLM token ids** (`prompt_token_ids` + `outputs[0].token_ids`). Never re-tokenize generated text for training.
- Never drop, time out, or resample a rollout. A failed request fails the iteration loudly, and the iteration retries from its start with the same seeds, at most twice. Selection by completion time biases the estimator.
- The writer samples at `temperature=1.0`, `top_p=1.0`, `top_k=-1` (exact on-policy sampling; no truncation). Evaluation writers and the reader use `temperature=0`.
- `lora_dropout = 0.0`. This is mandatory, because dropout would make the trained policy differ from the sampling policy.

**Long jobs**
- Claude Code's shell calls time out, so every job longer than a few minutes runs detached: `nohup … &` inside `tmux`, launched through `scripts/run_queue.py`. Poll the logs; never block on long commands.
- Stagger vLLM engine starts on the four GPUs by 45 s.
- Each job gets `CUDA_VISIBLE_DEVICES=<one gpu>`, `OMP_NUM_THREADS=<cores/4>`, `TOKENIZERS_PARALLELISM=false`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

**Secrets**
- The Hugging Face token is read from the file `hf-tok` (§8.1). It is never printed, logged, committed, or written anywhere else. `hf-tok` is in `.gitignore`.

**Ask first** (stop and ask the owner):
- Changing the model, task family, arms, seeds, F, L, U, B, or any §11 threshold.
- Spending more than 20% beyond the Gate-G4 projection.
- Deleting run outputs.
- Any change to the frozen `prereg/PREREG.md` after Phase 4.

**Do without asking**
- Fixing bugs.
- Adapting to vLLM or transformers API drift. Record the change in `reports/STATUS.md`.
- Applying the predeclared fallbacks in §13.

---

## 5. Node check, environment, Hugging Face token, model download

### 5.1 Node check (`scripts/00_check_node.sh`)

The script prints the following and writes them to `reports/node.json`. It exits non-zero if any requirement fails.

| Check | Command | Requirement |
|---|---|---|
| GPUs | `nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv` | 4 GPUs, name contains "H100", ≥ 79 GB each |
| Driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | Record it; it decides install path A or B (§5.2) |
| Topology | `nvidia-smi topo -m` | Record only (NVLink expected) |
| Idle | `nvidia-smi --query-compute-apps=pid --format=csv,noheader` | No other processes on the GPUs |
| CPU and RAM | `nproc`, `free -g` | ≥ 32 cores, ≥ 256 GB RAM (warn below this; do not fail) |
| Disk | `df -h .` | ≥ 400 GB free where `runs/` lives |
| Python | `python3.11 --version` or `uv python install 3.11` | 3.11 available |
| Token file | `test -s hf-tok` | Exists and is non-empty; warn if its mode is not 600 |

### 5.2 Python environment (`scripts/01_setup_env.sh`)

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
```

The default vLLM 0.30.0 wheel pins `torch==2.13.0`, which is built against **CUDA 13.0**. CUDA 13 needs **NVIDIA driver ≥ 580**.

**Path A (driver ≥ 580):**
```bash
uv pip install "vllm==0.30.0"
```

**Path B (driver < 580):** Install a vLLM build for CUDA 12.x. Check the assets of the GitHub release `v0.30.0` for a `+cu12x` wheel and install it together with the matching torch index, for example:
```bash
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```
If no such wheel exists for 0.30.0, use the newest vLLM release that ships one. As a last resort, use the NVIDIA NGC vLLM container, which carries CUDA forward-compatibility libraries for datacenter GPUs. Record the choice in `reports/STATUS.md`.

**Both paths**, after vLLM:
```bash
uv pip install "peft==0.21.0" accelerate "nvidia-ml-py>=13" pyyaml pandas pyarrow scipy statsmodels matplotlib orjson pytest pytest-asyncio pytest-timeout
# transformers and huggingface_hub come from vLLM's pins (transformers>=5.10.4, huggingface_hub>=1.31)
uv pip freeze > env/requirements.lock
```

Do **not** install flash-attn. HF training uses `attn_implementation="sdpa"`, and vLLM uses its own kernels.

**API notes for these versions.** Test T0.4 re-checks all of them.

| Topic | Correct usage in transformers 5 / vLLM 0.30 |
|---|---|
| Model dtype | `AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")`. The `torch_dtype=` argument is deprecated |
| Chat template to ids | `tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)`. In transformers 5 the default is `return_dict=True`, which returns a BatchEncoding, not a list |
| Engine | `from vllm.engine.arg_utils import AsyncEngineArgs`; `from vllm.v1.engine.async_llm import AsyncLLM`; `AsyncLLM.from_engine_args(args)` |
| Generate | `async for out in llm.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=rid, lora_request=lr)`. Passing raw prompt strings is deprecated; always pass token ids |
| Final-only outputs | `SamplingParams(..., output_kind=RequestOutputKind.FINAL_ONLY)` |
| Sampled-token logprob | `SamplingParams(logprobs=0)` returns the chosen token's logprob. The default `logprobs_mode="raw_logprobs"` equals processed logprobs here because temperature is 1 and there is no top-p/k |
| Scoring given tokens | `SamplingParams(max_tokens=1, prompt_logprobs=0)` on the full sequence returns per-token logprobs of the prompt (used in tests) |
| LoRA hot-swap | `await llm.add_lora(LoRARequest(lora_name=f"v{v}", lora_int_id=v+1, lora_path=dir))`, then `await llm.remove_lora(prev_id)`. `lora_int_id` must be ≥ 1 and unique per version. `LoRARequest(load_inplace=True)` also exists, but do not use it |
| KV size | `kv_cache_memory_bytes` (overrides `gpu_memory_utilization`; better for colocation) |
| Scheduler default | `max_num_seqs` defaults to 128. Set it explicitly (§7) |
| Batch-invariant mode | Environment variable `VLLM_BATCH_INVARIANT=1`; used only by test T2.7 |

### 5.3 Hugging Face token wiring (`kmatters/hf_auth.py`)

**Resolution order:** environment variable `HF_TOKEN_FILE` if set; otherwise `./hf-tok` in the repo root; otherwise `~/hf-tok`. If none exists, raise `FileNotFoundError` with this exact message:
`Hugging Face token file not found. Put your token in ./hf-tok (one line, chmod 600) or set HF_TOKEN_FILE.`

```python
# kmatters/hf_auth.py
import logging, os, stat
from pathlib import Path

_TOKEN = None

def _candidates():
    env = os.environ.get("HF_TOKEN_FILE")
    if env:
        yield Path(env).expanduser()
    yield Path.cwd() / "hf-tok"
    yield Path.home() / "hf-tok"

def load_hf_token(set_env: bool = True) -> str:
    """Read the token from hf-tok, export HF_TOKEN for huggingface_hub and vLLM, and return it.
    Call once, at the top of every entry-point script, before importing vllm/transformers."""
    global _TOKEN
    for p in _candidates():
        if p.is_file():
            tok = p.read_text().strip()
            if not tok:
                raise ValueError(f"{p} is empty")
            if not tok.startswith("hf_"):
                logging.warning("hf-tok content does not start with 'hf_'; using it anyway")
            mode = p.stat().st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH):
                logging.warning("%s is readable by group/others; run: chmod 600 %s", p, p)
            _TOKEN = tok
            if set_env:
                os.environ["HF_TOKEN"] = tok          # read by huggingface_hub and vLLM subprocesses
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            return tok
    raise FileNotFoundError("Hugging Face token file not found. Put your token in ./hf-tok "
                            "(one line, chmod 600) or set HF_TOKEN_FILE.")

class RedactFilter(logging.Filter):
    def filter(self, record):
        if _TOKEN and _TOKEN in str(record.getMessage()):
            record.msg = str(record.getMessage()).replace(_TOKEN, "hf_***")
            record.args = ()
        return True
```

`kmatters/logging_utils.setup_logging()` must attach a `RedactFilter()` to **every handler** it creates, including console and file handlers. Filters attached to a logger do not apply to records from child loggers, so handler-level filters are required.

**Rules**
- Every entry script calls `load_hf_token()` as its first statement, before importing vLLM or transformers. The token reaches vLLM's EngineCore subprocess through the inherited `HF_TOKEN` environment variable.
- Do **not** call `huggingface_hub.login()`, because it writes the token to `~/.cache/huggingface/token`.
- `.gitignore` must contain `hf-tok`, `.venv/`, `models/`, `runs/`, `profiles/`, `*.safetensors`, `*.pt`.

### 5.4 Model download (`scripts/02_download_model.py`)

```python
from kmatters.hf_auth import load_hf_token
tok = load_hf_token()
import datetime, hashlib, pathlib, yaml
from huggingface_hub import HfApi, snapshot_download
REPO = "Qwen/Qwen3-4B-Instruct-2507"
sha = HfApi(token=tok).model_info(REPO).sha
path = snapshot_download(REPO, revision=sha, local_dir="models/Qwen3-4B-Instruct-2507", token=tok)
config_sha256 = hashlib.sha256(pathlib.Path(path, "config.json").read_bytes()).hexdigest()
yaml.safe_dump({"repo": REPO, "revision": sha, "local_dir": str(path),
                "config_sha256": config_sha256,
                "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
               open("configs/model_lock.yaml", "w"))
```

- Every later job loads **only** from `models/Qwen3-4B-Instruct-2507`, so no network is needed at run time. Also set `HF_HUB_OFFLINE=1` in job environments.
- Every job asserts that `configs/model_lock.yaml` exists and that the sha256 of the local `config.json` equals its `config_sha256` field.

---

## 6. Repository layout

```
kmatters/                          # repo root (git)
├── PLAN.md                        # this file
├── README.md                      # quickstart: setup → gates → sweep → report
├── pyproject.toml                 # package "kmatters"; pytest markers: gpu, slow, stats
├── .gitignore
├── hf-tok                         # owner-provided, NOT committed
├── configs/
│   ├── base.yaml                  # all defaults (§7)
│   ├── model_lock.yaml            # generated by 02_download_model.py
│   ├── locked.yaml                # generated at Gate G2 (level) and G3 (engine tuning)
│   ├── pilot.yaml                 # Phase 4 overrides
│   ├── sweep_main.yaml            # Phase 5 arms (frozen at G4)
│   ├── grid_main.yaml             # Phase 3/6 cost-grid cells
│   ├── grid_budget.yaml           # Phase 3 B-grid cells
│   └── variance_conditions.yaml   # Phase 3/6 variance measurements
├── kmatters/
│   ├── __init__.py
│   ├── hf_auth.py                 # §5.3
│   ├── config.py                  # pydantic models, YAML load + overrides, config hash
│   ├── seeding.py                 # stable 63-bit seeds from keys (blake2b)
│   ├── logging_utils.py           # setup_logging (with RedactFilter), JSONL writer
│   ├── env/ledger.py              # Appendix A.1 (verbatim)
│   ├── env/oracle.py              # perfect-memory renderer for controls (§8.11)
│   ├── prompts.py                 # §8.4 exact text + id builders
│   ├── reward.py                  # §8.5
│   ├── engine.py                  # §8.6 vLLM wrapper
│   ├── executor.py                # §8.7 BarrierExecutor, AsyncReadyExecutor
│   ├── records.py                 # dataclasses: CallRecord, FutureRecord, PrefixRecord, RolloutBatch
│   ├── rl/estimator.py            # Appendix A.2 (verbatim)
│   ├── rl/logprobs.py             # Appendix A.3 (verbatim)
│   ├── trainer.py                 # §8.9 HF model + PEFT + AdamW + adapter I/O
│   ├── loop.py                    # §8.10 run_iteration(), train()
│   ├── evaluate.py                # §8.11
│   ├── variance.py                # §8.12 (uses analysis/variance_math.py)
│   ├── profiling.py               # §8.13 cell runner, NVML sampler, occupancy
│   ├── bench.py                   # §8.13 microbenchmarks
│   └── analysis/
│       ├── variance_math.py       # Appendix A.4 (verbatim)
│       ├── kstar.py               # Appendix A.5 (verbatim)
│       ├── curves.py              # Appendix A.6 (verbatim)
│       ├── q1.py  q2.py  q3.py    # §11 decision computations
│       ├── plots.py
│       └── report.py              # writes reports/DECISION.md
├── scripts/
│   ├── 00_check_node.sh  01_setup_env.sh  02_download_model.py
│   ├── calibrate.py               # controls + level calibration (Phase 2)
│   ├── bench.py                   # microbenchmarks (Phase 3)
│   ├── profile_costs.py           # cost grids (Phases 3, 6)
│   ├── measure_variance.py        # variance components (Phases 3, 6)
│   ├── train.py                   # one training run (Phases 4, 5)
│   ├── make_jobs.py               # writes jobs/*.jsonl for run_queue
│   ├── run_queue.py               # 1 job per GPU, LPT order, resumable
│   ├── freeze_prereg.py           # writes prereg/PREREG.md + hash (end of Phase 4)
│   └── analyze.py                 # Phase 7: q1, q2, q3, plots, report
├── tests/
│   ├── unit/                      # CPU only (T1.*); includes Appendix A.7 and A.8
│   ├── gpu/                       # T0.4–T0.6, T2.*, T3.*  (marker: gpu)
│   └── stats/                     # T4.*, T5.* (markers: gpu, stats)
├── prereg/  PREREG.md  DEVIATIONS.md
├── jobs/  runs/  profiles/  variance/  reports/   # outputs (not committed except reports/*.md)
└── env/requirements.lock
```

---

## 7. Configuration (`configs/base.yaml`)

All code reads configuration through `kmatters.config.load(paths, overrides)`. This merges YAML files left to right, then applies `--set a.b=c` overrides. The result is validated with pydantic: unknown keys are errors, and `N = F / K` must be an integer. The canonical JSON dump of the resolved config is stored in every output directory, and its sha256 is stored as `config_hash`.

```yaml
model:
  repo: Qwen/Qwen3-4B-Instruct-2507
  local_dir: models/Qwen3-4B-Instruct-2507
  dtype: bfloat16
  attn_implementation: sdpa

lora:
  r: 16
  alpha: 32
  dropout: 0.0            # MUST be 0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  init_seed_offset: 1000  # torch.manual_seed(seed + offset) before get_peft_model

vllm:
  kv_cache_memory_bytes: 30064771072    # 28 GiB; reduce if T2.10 fails
  max_model_len: 4096
  max_num_seqs: 512                     # tuned in Phase 3 (§10), then locked
  max_num_batched_tokens: 16384         # tuned in Phase 3, then locked
  enable_prefix_caching: true
  enforce_eager: false
  max_loras: 2
  max_lora_rank: 16
  max_cpu_loras: 4
  per_request_seeds: false              # true only in determinism tests (slow sampler path)

env:
  level: 3          # set by calibration (Gate G2) in configs/locked.yaml
  L: 16             # prefix chunks
  U: 2              # future chunks
  B: 512            # memory budget = writer max_tokens

sampling:
  writer_temperature: 1.0
  reader_max_tokens: 12

estimator:
  baseline: loo_prefix   # E1; "tree" = E2 ablation (not run in the main sweep)
  T_norm: 512            # constant loss normalizer; do NOT tie to B

train:
  F: 64
  K: 4                   # N = F/K
  lr: 3.0e-5             # set by the pilot (Gate G4)
  betas: [0.9, 0.99]
  eps: 1.0e-8
  weight_decay: 0.0
  grad_clip: null        # set at G4 to 5x the pilot median grad norm (catastrophe guard only)
  max_tokens_per_microbatch: 16384
  iterations: 250        # I_max; confirmed at G4
  seed: 0
  executor: async_ready  # "barrier" is allowed only in profiling
  eval_every: 10
  ckpt_every: 10
  keep_optimizer_ckpts: 2

eval:
  small: {n_prefixes: 128, futures_per_prefix: 4, eval_seed: 12345}   # every eval_every iterations
  large: {n_prefixes: 512, futures_per_prefix: 4, eval_seed: 54321}   # at it=0, I_max/2, I_max

profile:
  reps: 3
  warmup: 1
  lr: 0.0              # policy must not move during profiling (T5.3)

variance:
  N0: 64
  K0: 8

paths:
  runs: runs
  profiles: profiles
  variance: variance
  reports: reports
```

**Derived identifiers**
- `run_id = f"{tag}_K{K}_N{N}_s{seed}_{config_hash[:8]}"`
- **Data keys** (all non-negative integers, as numpy SeedSequence requires):
  - train prefix `(seed, it, i)`; train future `(seed, it, i, k)`
  - eval prefix `(eval_seed, j)`; eval future `(eval_seed, j, k)`
  - variance prefix `(vseed, ckpt_it, i)`, with `ckpt_it = 0` for the base policy
  - profile prefix `(cell_id, rep, i)`. Here `cell_id = seed_from(N, K, L, B) % 2**31` is derived from (N, K, L, B) only, **not** the executor, so executors are compared on identical data. `rep = 0` is the warmup and reps 1–3 are measured.
- **Common random numbers:** at every iteration, arms with the same seed see identical histories for the prefix indices they share (i < min N). Iteration-0 differences between arms therefore come from N and K alone.

---

## 8. Component specifications

### 8.1 `hf_auth.py`
See §5.3.

### 8.2 `seeding.py`

```python
import hashlib
def seed_from(*keys) -> int:
    """Stable across processes and machines (unlike hash()). 63-bit non-negative."""
    h = hashlib.blake2b(repr(tuple(keys)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)
```

These seeds are used for torch/numpy seeding per run, for the LoRA initialization (`seed + init_seed_offset`), and, in determinism tests only, for per-request vLLM seeds.

### 8.3 `env/ledger.py`: the task

**Use Appendix A.1 verbatim.** Its semantics:

- **Entities.** Each history draws `n_people` names and `n_cats` categories.
- **Event types**, rendered one per line:
  - `T07: Alice paid Bob $23 for groceries.`
  - `T03 was cancelled.`
  - `Correction: the amount of T05 should be $41.` The old amount is deliberately not shown, so the writer must have kept it (or the aggregates it affects).
  - Noise lines: irrelevant sentences, some containing distractor numbers.
- **Prefix.** `L` chunks of `events_per_chunk` events drawn with the level's mix.
- **Future.** `U` chunks drawn with `FUTURE_MIX` (25% cancels, 25% corrections). Cancels and corrections target a prefix transaction with probability 0.8, then one question.
- **Question types.**
  - `net` (received minus paid): 30%
  - `paid` (total paid by a person): 20%
  - `cat` (category total): 25%
  - `amount` (current amount of a transaction id; targets a prefix transaction with probability 0.7): 25%
- **Answers.** Integers computed from the exact state after the future. An independent regex re-parse (`brute_force_answer`) is used in tests.
- **Independence.** The prefix RNG and each future's RNG are separate `numpy` SeedSequence streams keyed by `(split, role, *key)`. A prefix never depends on its futures, and futures are conditionally independent given the prefix state.
- **Difficulty ladder.** `LEVELS[1..5]`. The level is chosen in Phase 2 (§10) by rule. Measured on this implementation: an oracle-perfect memory at L = 16 needs about 290–400 tokens (p95 about 340–490) depending on level. `B = 512` therefore allows a near-perfect memory, while `B = 256` forces selection. Verify with the real tokenizer in T3.1.

### 8.4 `prompts.py`: exact text

```python
WRITER_SYSTEM = """You maintain the memory of a bookkeeping assistant. You read a ledger log one chunk at a time and never see earlier chunks again: after each chunk, only your memory is kept.

Later, someone who sees ONLY your memory must answer questions such as:
- a person's net balance (total received minus total paid),
- the total amount a person has paid,
- the total amount in a spending category,
- the current amount of a specific transaction (for example T07).
More log chunks may arrive before a question, including cancellations and amount corrections of earlier transactions, so keep whatever details are needed to apply them correctly.

Rules: output only the updated memory, nothing else. Text beyond {B} tokens is cut off, so stay well under that. Ignore log lines that are not transactions, cancellations or corrections."""

WRITER_USER = """CURRENT MEMORY:
{memory}

NEW LOG CHUNK:
{chunk}

Write the updated memory."""

READER_SYSTEM = """You answer questions about a ledger using only the notes provided. Reply with a single integer (use a minus sign for negative numbers) and nothing else. If the notes do not contain the information, reply with your best guess as a single integer."""

READER_USER = """NOTES:
{memory}

QUESTION: {question}"""

EMPTY_MEMORY = "(empty)"
```

**Functions**
- `writer_ids(tok, memory, chunk, B) -> list[int]`: messages `[system=WRITER_SYSTEM.format(B=B), user=WRITER_USER.format(...)]`, converted with `apply_chat_template(..., add_generation_prompt=True, tokenize=True, return_dict=False)`.
- `reader_ids(tok, memory, question) -> list[int]`: the same, with the reader prompts.
- `memory_from_output(text) -> str`: `text.strip()`, or `EMPTY_MEMORY` if the result is empty. A truncated output (`finish_reason == "length"`) is kept as is.

**Leakage rules**, checked by T1.4:
- Writer prompts contain only the current memory and one chunk: never a question, never a future chunk during a prefix call.
- Reader prompts contain only memory and question: never a log line.

### 8.5 `reward.py`

```python
import re
_INT = re.compile(r"-?\d+")
def parse_int(text: str):
    s = text.strip().replace("−", "-").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)      # 1,234 -> 1234
    m = _INT.search(s)
    return int(m.group()) if m else None
def reward(text: str, gold: int) -> float:
    v = parse_int(text)
    return 1.0 if v is not None and v == gold else 0.0
```

### 8.6 `engine.py`: vLLM wrapper

```python
class Engine:
    """One AsyncLLM per process/GPU. Serves the writer (LoRA) and the reader (base)."""
    def __init__(self, cfg, loop: asyncio.AbstractEventLoop): ...
    async def start(self): ...                                   # builds AsyncLLM
    async def generate(self, ids, sp, rid, lora) -> GenResult: ...
    async def set_writer_adapter(self, path: str, version: int): ...
    async def score_tokens(self, prompt_ids, completion_ids, lora) -> list[float]: ...  # tests
    async def shutdown(self): ...
```

**Construction**, done in `start()`, which runs inside the persistent loop:
```python
args = AsyncEngineArgs(
    model=cfg.model.local_dir, tokenizer=cfg.model.local_dir, dtype="bfloat16",
    seed=cfg.train.seed, max_model_len=cfg.vllm.max_model_len,
    enable_lora=True, max_loras=cfg.vllm.max_loras, max_lora_rank=cfg.vllm.max_lora_rank,
    max_cpu_loras=cfg.vllm.max_cpu_loras, enable_prefix_caching=cfg.vllm.enable_prefix_caching,
    max_num_seqs=cfg.vllm.max_num_seqs, max_num_batched_tokens=cfg.vllm.max_num_batched_tokens,
    kv_cache_memory_bytes=cfg.vllm.kv_cache_memory_bytes, enforce_eager=cfg.vllm.enforce_eager,
    disable_log_stats=False)
self.llm = AsyncLLM.from_engine_args(args)
```

**Process and event-loop pattern (mandatory)**
1. The entry script calls `load_hf_token()`, sets environment variables, and creates **one** persistent loop with `loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)`.
2. Build the engine **before** loading the HF training model (vLLM's memory setup must see a clean GPU): `loop.run_until_complete(engine.start())`.
3. Load the HF model (§8.9).
4. Each phase (rollout, eval, adapter sync) runs as `loop.run_until_complete(coro)`. Never call `asyncio.run()` more than once, because AsyncLLM's output handler is bound to the loop.
5. Guard entry points with `if __name__ == "__main__":`, which the `spawn` start method requires.

**`generate()`**
- Wraps `llm.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=rid, lora_request=lora)`, with `sp.output_kind = FINAL_ONLY`.
- Records `t_submit` (time.perf_counter just before the call) and `t_end` (when the final output arrives).
- Returns a `GenResult` containing:
  - `prompt_ids`, `completion_ids = list(out.outputs[0].token_ids)` (includes EOS when the call stopped on EOS)
  - `text = out.outputs[0].text`, `finish_reason`
  - `vllm_token_logprobs`: the chosen token's logprob per completion token, when `logprobs=0`
  - `num_prompt_tokens`, `num_cached_tokens = out.num_cached_tokens or 0`
  - `t_submit`, `t_end`
- Request ids are `f"{run_id}|{phase}|{it}|{role}|{i}|{k}|{step}|{uuid4().hex[:8]}"`.

**Sampling params**

| Role | Params |
|---|---|
| Train writer | `temperature=1.0, top_p=1.0, top_k=-1, max_tokens=B, logprobs=0` (plus `seed=` only if `per_request_seeds`) |
| Eval writer | `temperature=0.0, max_tokens=B` |
| Reader | `temperature=0.0, max_tokens=reader_max_tokens`, `lora_request=None` |

**`set_writer_adapter(path, version)`**
1. `await llm.add_lora(LoRARequest(f"v{version}", version + 1, path))`
2. If a previous adapter id exists: `await llm.remove_lora(prev_id)`
3. Store the current `LoRARequest` for writer calls.

The initial adapter v0 (LoRA B = 0, so the writer equals the base model) is saved and loaded like every other version. Writer calls therefore always go through the LoRA path.

**`score_tokens()`**, used only by tests: send `prompt_ids + completion_ids` with `SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)` and return the prompt logprobs at the completion positions.

**Vendor-API drift.** If any name above differs in the installed vLLM, T0.4 fails. Adapt `engine.py` only, keep the `Engine` interface unchanged, and record the change in STATUS.md.

### 8.7 `records.py` and `executor.py`

```python
@dataclass
class CallRecord:
    role: str               # "prefix" | "future" | "reader"
    i: int; k: int; step: int      # k = -1 for prefix calls
    prompt_ids: list[int]; completion_ids: list[int]
    text: str; finish_reason: str
    vllm_lp: list[float] | None
    n_prompt: int; n_cached: int; n_completion: int
    t_submit: float; t_end: float

@dataclass
class FutureRecord:
    i: int; k: int; key: tuple
    calls: list[CallRecord]          # U writer calls
    reader: CallRecord
    question: str; q: tuple; gold: int; answer_text: str; reward: float
    final_memory: str

@dataclass
class PrefixRecord:
    i: int; key: tuple
    calls: list[CallRecord]          # L writer calls
    memory: str
    futures: list[FutureRecord]      # exactly K

@dataclass
class RolloutBatch:
    prefixes: list[PrefixRecord]
    t_start: float; t_end: float
    def rewards(self) -> np.ndarray: ...   # [N, K]
```

Both executors expose `async def collect(prefix_specs, future_specs, lora, writer_sp, reader_sp) -> RolloutBatch`. Specs come from `env.ledger.make_prefix` and `make_future` and are generated **before** rollout starts.

**AsyncReadyExecutor** (the "competent" executor, used everywhere by default):
```python
async def collect(...):
    t0 = perf_counter()
    async def run_future(i, k, mem):
        calls = []
        for u, chunk in enumerate(fspec[i][k].chunks):
            r = await eng.generate(writer_ids(tok, mem, chunk, B), writer_sp, rid(...), lora)
            calls.append(rec("future", i, k, u, r)); mem = memory_from_output(r.text)
        rr = await eng.generate(reader_ids(tok, mem, fspec[i][k].question), reader_sp, rid(...), None)
        return FutureRecord(..., calls, rec("reader", i, k, U, rr), reward=reward(rr.text, gold), final_memory=mem)
    async def run_prefix(i):
        mem, calls = EMPTY_MEMORY, []
        for t, chunk in enumerate(pspec[i].chunks):
            r = await eng.generate(writer_ids(tok, mem, chunk, B), writer_sp, rid(...), lora)
            calls.append(rec("prefix", i, -1, t, r)); mem = memory_from_output(r.text)
        futs = await asyncio.gather(*[run_future(i, k, mem) for k in range(K)])   # start immediately
        return PrefixRecord(i, pspec[i].key, calls, mem, list(futs))
    prefixes = await asyncio.gather(*[run_prefix(i) for i in range(N)])            # all chains concurrent
    return RolloutBatch(list(prefixes), t0, perf_counter())
```

**BarrierExecutor** (a diagnostic ablation that is never used for training): step-synchronous. For each `t` in `range(L)`, gather all N prefix calls of step t. Then for each `u` in `range(U)`, gather all N·K future calls of step u. Then gather all N·K reader calls. Records must have the same structure.

**Invariants** (checked in T2.6):
- Exactly N prefixes, L calls each, K futures each, U calls plus 1 reader call each.
- Every reward is in {0, 1}.
- Any exception propagates, with no partial batch.

### 8.8 `rl/estimator.py`: advantages and weights

**Use Appendix A.2 verbatim.** The estimator is

ĝ = (1/N) Σᵢ [ A_preᵢ · s_preᵢ + (1/K) Σₖ A_futᵢₖ · s_futᵢₖ ]

- `s_preᵢ` is the score (sum of ∇ log π over writer tokens) of the L prefix calls of prefix i.
- `s_futᵢₖ` is the score of the U future calls of future (i,k).
- E1 (primary) uses the baseline bᵢ = mean over j≠i of r̄ⱼ, so A_preᵢ = r̄ᵢ − bᵢ and A_futᵢₖ = rᵢₖ − bᵢ.

Properties, all verified by tests in Appendix A.7:
- **Unbiased** for E1 and E2, for any N ≥ 2 and K ≥ 1, on a toy two-stage bandit.
- **Equal weight per prefix:** the futures of a prefix together carry weight 1/N, regardless of K.
- **Variance formula:** with a constant baseline, tr Cov(ĝ) = V_prefix/N + V_future/(NK) exactly.

**No advantage normalization, no std scaling, no clipping of advantages.** Do not skip zero-advantage calls: the saving would make training cost depend on rewards.

### 8.9 `trainer.py` and `rl/logprobs.py`

**Use Appendix A.3 (`rl/logprobs.py`) verbatim.** It:
- right-pads,
- runs the decoder once,
- gathers the hidden states that predict completion tokens,
- applies `lm_head` in chunks,
- computes `log_softmax` in fp32,
- returns per-call sums plus per-token log-probabilities.

The loss is `-(1/T_norm) Σ_c w_c Σ_t log π`, microbatched by a padded-token budget. It is verified against a naive implementation and for microbatch-partition invariance (Appendix A.8).

`trainer.py`:
```python
class Trainer:
    def __init__(self, cfg, device="cuda"):
        torch.manual_seed(seed_from(cfg.train.seed, "torch"))
        base = AutoModelForCausalLM.from_pretrained(cfg.model.local_dir, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to(device)
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(cfg.train.seed + cfg.lora.init_seed_offset)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                                     target_modules=[...], task_type="CAUSAL_LM"))
        # PEFT keeps adapter weights in fp32 for a bf16 base (autocast_adapter_dtype=True); assert it.
        self.params = trainable_params(self.model)
        self.opt = torch.optim.AdamW(self.params, lr=cfg.train.lr, betas=tuple(cfg.train.betas),
                                     eps=cfg.train.eps, weight_decay=0.0, fused=True)
    def step(self, calls, T_norm, max_tokens, lr=None) -> dict:
        if lr is not None:                      # profiling passes lr=0.0; training passes None
            for g in self.opt.param_groups: g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        # microbatches: prefix calls and future calls packed SEPARATELY (timed separately)
        ...accumulate_policy_gradient(...) for each group, timing each with cuda.synchronize...
        gn = total grad norm (fp32); clipped = cfg.train.grad_clip and gn > clip -> clip_grad_norm_
        self.opt.step()
        return {"loss", "grad_norm", "clipped", "t_train_prefix", "t_train_future", tok_lp per call}
    def save_adapter(self, path): self.model.save_pretrained(path, safe_serialization=True)
    def save_state(self, path): torch.save({"opt": self.opt.state_dict(), "rng": ...}, path)
    def load_state(self, adapter_path, state_path): ...   # set_peft_model_state_dict + opt.load_state_dict
```

**Diagnostics.** These are computed from the forward pass the step already does, with no extra work.
- `lp_absdiff_mean`: mean of |lp_hf − lp_vllm| over all writer tokens.
- `lp_diff_mean`: mean of lp_hf − lp_vllm.
- `kl_k3`: mean of exp(Δ) − 1 − Δ, where Δ = lp_hf − lp_vllm.
- `neg_lp_mean`: mean of −lp_vllm (an entropy proxy).

`grad_clip` stays `null` in the pilot. At G4 it is set to 5× the pilot's median gradient norm, as a catastrophe guard only. If it activates in more than 2% of steps in any run, flag the run in STATUS.md.

### 8.10 `loop.py`: one iteration and the training run

```python
def run_iteration(ctx, it, N, K, lr_override=None) -> dict:
    """The single code path for training AND profiling.
    ctx.split/ctx.key0 = ("train", seed) in training, ("profile", cell_id) in profiling (it = rep)."""
    t0 = perf_counter()
    pspec = [make_prefix(ctx.split, (ctx.key0, it, i), ctx.L, ctx.level) for i in range(N)]
    fspec = [[make_future(ctx.split, pspec[i], k, ctx.U, ctx.level) for k in range(K)] for i in range(N)]
    tr0 = perf_counter()
    batch = ctx.loop.run_until_complete(ctx.executor.collect(pspec, fspec, lora=ctx.engine.current, ...))
    t1 = perf_counter()
    r = batch.rewards()                                          # [N, K]
    w_pre, w_fut = call_weights(r, ctx.cfg.estimator.baseline)
    calls = [Call(c.prompt_ids, c.completion_ids, w_pre[p.i]) for p in batch.prefixes for c in p.calls] \
          + [Call(c.prompt_ids, c.completion_ids, w_fut[f.i, f.k]) for p in batch.prefixes
             for f in p.futures for c in f.calls]              # reader calls are never trained
    tw = perf_counter()
    stats = ctx.trainer.step(calls, T_norm, max_tokens, lr=lr_override)   # lr_override=0.0 when profiling
    t2 = perf_counter()
    ctx.trainer.save_adapter(adir(it + 1))
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(adir(it + 1), it + 1))
    t3 = perf_counter()
    t_rollout, t_train, t_sync, t_iter = t1 - tr0, t2 - tw, t3 - t2, t3 - t0
    return metrics(batch, r, stats, t_rollout=t_rollout, t_train=t_train, t_sync=t_sync, t_iter=t_iter,
                   t_other=t_iter - t_rollout - t_train - t_sync)
```

In profiling, `adir()` points to a scratch directory and the adapter version counter keeps increasing. With lr = 0 the weights are unchanged (T5.3), but the save and load are still timed, as they are in training.

**Counting convention.** Iteration index `it` runs from 0 to I_max − 1. It samples with policy version `it` and produces version `it + 1`. "Version n" or "checkpoint n" means the state after n completed iterations. Evaluations and checkpoints are attached to versions.

**`train(cfg)`**
1. Build the engine, then the trainer.
2. Save and load adapter v0.
3. Evaluate version 0 on both the small and large sets.
4. For `it` in `[start, I_max)`:
   - run `run_iteration`, which produces version n = it + 1
   - if n % eval_every == 0: small eval of version n (plus the large eval when n ∈ {I_max/2, I_max})
   - if n % ckpt_every == 0 or n == I_max: checkpoint n
   - append to `metrics.jsonl`
5. Write `DONE` at the end.

**Checkpoint layout** `runs/<run_id>/ckpt/it{n:04d}/`, the state after n completed iterations:
- `adapter/`, which is **kept for every checkpoint** (needed for variance measurement)
- `optim.pt`, kept only for the last `keep_optimizer_ckpts`
- `state.json`: iteration, policy version, RNG states, config hash, metrics line count
- `COMPLETE`, a marker written last.

**Resume.** Find the newest checkpoint n with a `COMPLETE` marker, restore the adapter and optimizer state, load the adapter into vLLM as version n, truncate `metrics.jsonl` to the recorded line count, and continue with iteration index `it = n`. Data keys depend only on `(seed, it, i, k)`, so the data stream continues identically.

**Adapter hygiene.** Keep the `runs/<run_id>/adapters/v{n}` directories for `n % ckpt_every == 0` and the latest two. Delete the others after the next adapter has loaded.

### 8.11 `evaluate.py`, `env/oracle.py`, controls

**`evaluate(ctx, which="small"|"large")`**
- Deterministic writer (temperature 0) with the current adapter, deterministic reader.
- Fixed eval specs: `make_prefix("eval", (eval_seed, j), L, level)` for j < n_prefixes, and `make_future("eval", p, k, U, level)` for k < futures_per_prefix.
- Uses the AsyncReadyExecutor.
- Returns:
  - `acc`
  - `acc_by_type` for net, paid, cat, amount
  - `mem_len_mean`, `mem_len_p90`, `trunc_rate`
  - `n_questions`
  - per-prefix correct counts, for the clustered bootstrap.

Eval time is logged as `t_eval` and **excluded** from all cost pricing.

**`oracle.py`** renders a perfect memory from the exact state after the prefix plus the future: per-person paid, received and net; per-category totals; every live transaction `T07 Alice->Bob 23 groceries`.

**Controls**, run by `scripts/calibrate.py --controls`, each on the small eval set at the calibrated level:

| Control | Memory given to the reader | Purpose |
|---|---|---|
| Oracle | `oracle.py` rendering | Reader ceiling (T3.1) |
| Empty | `(empty)` | Guessability (T3.2) |
| Shuffled | Oracle memory of a **different** eval prefix | Reader relies on memory (T3.3) |
| Base writer | Writer v0 greedy | Starting accuracy (T3.4) |

### 8.12 `variance.py`: V_prefix and V_future at a checkpoint

`scripts/measure_variance.py --ckpt <adapter_dir|base> --L 16 --B 512 --vseed 0 --out variance/<tag>`

1. Build the engine and trainer; load the checkpoint adapter into both.
2. Collect a pool of N0 = 64 prefixes × K0 = 8 futures (split "variance", keys `(vseed, ckpt_it, i)`), with the training writer temperature, using AsyncReadyExecutor.
3. Rewards r [64, 8]. Baseline bᵢ is the leave-one-prefix-out mean, as in E1. **Shut the engine down now** (`engine.shutdown()`): no more generation is needed, and this frees about 36 GB before the backward passes.
4. For each prefix i:
   - `g_pre` = ∇ of Σ_{prefix calls} Σ_t log π / T_norm: one gradient over its L calls (microbatched with the usual token budget), followed by `flat_grad`. Use fp32 flattening and keep it on the GPU.
   - For each future k: `g_fut` = the same over its U calls (one backward per future).
   - Z_ik = (r_ik − b_i) · (g_pre + g_fut_k)
   - `acc.add_prefix([Z_i0 … Z_i7])`, using `TorchVarianceAccumulator` from Appendix A.4 (the same math as the numpy class, with float64 accumulation and a chunked Gram matrix).
5. Outputs to `variance/<tag>/result.json`:
   - V_prefix and V_future point estimates, plus 2000-draw bootstrap percentiles (2.5, 50, 97.5)
   - reward-level decomposition (the same formulas applied to scalar rewards)
   - mean reward, N0, K0, checkpoint, L, B, config hash
   - per-prefix within-variances and the Gram matrix, saved as `.npy` for re-analysis.

**Memory.** The 64 mean vectors take 64 × 33M parameters × 4 bytes ≈ 8.5 GB on the GPU. That fits easily, because the engine was shut down in step 3. If memory is still short, keep the vectors on the CPU (`store_device="cpu"`).

**Torch port.** `TorchVarianceAccumulator` in Appendix A.4 agrees with the numpy class to a relative 1e-6. That is test T1.7b, `test_torch_accumulator_matches_numpy`.

### 8.13 `profiling.py` and `bench.py`

**NVML sampler.** A background thread that samples every 50 ms: SM utilization, memory utilization, power (W), and memory used. Map the device by UUID (`torch.cuda.get_device_properties(0).uuid` → `nvmlDeviceGetHandleByUUID`), because NVML ignores `CUDA_VISIBLE_DEVICES`. Summaries are produced per phase window.

**Occupancy**, from `CallRecord` intervals:
- n(t) = the number of requests with t_submit ≤ t < t_end, sampled on a 10 ms grid over the rollout window.
- `mean_inflight`
- `idle_slot_frac` = mean(max(0, n_sat − n(t))) / n_sat
- the same split by role.

**`bench.py`**, run once per engine configuration:
- **Decode:** for n ∈ {1,2,4,8,16,32,64,128,256,512}, submit n distinct writer-shaped requests (prompt ≈ 900 tokens of real writer prompts, `max_tokens=256`, `ignore_eos=True`, prefix caching disabled by unique prompts) and time them. Report throughput(n) in tokens/s.
- `n_sat` = the smallest n with throughput(n) ≥ 0.8 × the maximum measured throughput.
- `decode_rate_sat` = the maximum throughput.
- **Prefill:** time 64 unique 1500-token prompts with `max_tokens=1`, giving `prefill_rate` in tokens/s.
- **Train:** tokens/s of `accumulate_policy_gradient` on 64 writer-shaped calls at the configured microbatch budget.
- Output `profiles/bench_<engcfg>.json`.

**Cost-grid cell runner** (`scripts/profile_costs.py --grid configs/grid_main.yaml --ckpt <...> --shard i/4`). For each cell (N, K, L, B, executor), in a seeded random order:
1. Load the checkpoint adapter.
2. Run `profile.warmup` + `profile.reps` iterations of `run_iteration(..., lr_override=0.0)` with split "profile", keys `(cell_id, rep, i)`. The same keys are used across executors, so comparisons are paired.
3. Record per rep:
   - T_iter, t_rollout, t_train_prefix, t_train_future, t_sync, t_other
   - token counts by role (prompt, cached, completion)
   - occupancy stats and NVML phase means
   - peak GPU memory
   - the adapter sha256, computed over the LoRA tensors' bytes in sorted key order rather than over the file, so file metadata cannot change it (it must stay unchanged, T5.3)
   - barrier executor only: `t_phase_prefix`, `t_phase_future` and `t_phase_reader`, the wall time of each barrier phase (§11.3 model a).
4. Output `profiles/<tag>/cells.jsonl`, one line per rep.

The first warmup rep of each cell is discarded; summaries use the median of the reps.

**Grids**

| File | Cells |
|---|---|
| `grid_main.yaml`, condition A | executor=async_ready, L=16, B=512, N ∈ {4,8,16,32,64,128}, K ∈ {1,2,4,8,16}, N·K ≤ 512 (27 cells) |
| `grid_main.yaml`, condition B | executor=barrier, the same 27 cells |
| `grid_main.yaml`, condition C | executor=async_ready, L=4, B=512, the same 27 cells |
| `grid_budget.yaml` | executor=async_ready, L=16, B ∈ {256, 1024}, (N,K) ∈ {(8,1),(8,4),(8,16),(32,1),(32,4),(32,16),(128,1),(128,4)} (16 cells) |
| trained grid (Phase 6) | condition A at the trained checkpoint, plus the barrier cell (32,4) |

**Engine tuning (Phase 3, before the grids).** On cells (8,8), (32,4) and (128,4) at L = 16 with async_ready, try `max_num_seqs` ∈ {256, 512} × `max_num_batched_tokens` ∈ {8192, 16384}. Lock the configuration with the lowest summed T_iter in `configs/locked.yaml`. Every later job uses it, including training. This is part of what makes the executor "competent".

### 8.14 `run_queue.py` and `make_jobs.py`

- `make_jobs.py --phase {pilot,sweep,grid_main,grid_budget,variance0,variance_trained,trained_grid}` writes `jobs/<phase>.jsonl`. Each line is `{job_id, cmd, out_dir, est_hours, gpu_need: 1}`. Every job writes `<out_dir>/DONE` when it finishes successfully.
- `run_queue.py jobs/<phase>.jsonl --gpus 0,1,2,3`:
  - Longest-processing-time-first ordering.
  - One job per GPU.
  - Starts are staggered by 45 s.
  - Environment per job as in §4.
  - stdout and stderr go to `jobs/logs/<job_id>.log`.
  - A job is skipped if `DONE` exists and relaunched in resume mode if a partial checkpoint exists.
  - On a non-zero exit, retry once in resume mode, then mark it FAILED and continue with the other jobs.
  - Writes `jobs/<phase>.status.json` every 60 s.
- Launch inside tmux: `tmux new -d -s kq "python scripts/run_queue.py jobs/sweep.jsonl --gpus 0,1,2,3"`.

### 8.15 Analysis (`kmatters/analysis/*`, `scripts/analyze.py`)

- `curves.py`, `kstar.py` and `variance_math.py` come from Appendix A.
- `q1.py`, `q2.py` and `q3.py` implement the computations in §11 exactly. Each writes `reports/q{1,2,3}.json` with every number the decision rule uses.
- `report.py` writes `reports/DECISION.md`: the three answers (YES, NO, or INCONCLUSIVE, with the rule evaluation shown), key numbers, figures, and deviations.
- Figures go in `reports/fig/`:

| File | Content |
|---|---|
| `q1_curves.png` | Eval accuracy vs iteration, mean ± 95% CI per K |
| `q1_priced.png` | Accuracy vs priced GPU-hours |
| `q1_auc_logk.png` | Per-run AUC vs log₂K with OLS line |
| `q1_train_reward.png` | Training reward vs iteration |
| `var_components.png` | V_prefix and V_future with CIs across checkpoints and conditions |
| `q2_occupancy_<cell>.png` | n(t) timelines for 4 representative cells, both executors |
| `q2_marginal.png` | ρ_future and ρ_prefix vs N·K |
| `q2_phase_split.png` | Stacked phase times per cell |
| `q3_mse_time.png` | Model MSE × T vs K for each N, with the linear and empirical choices marked |
| `q3_kstar.png` | K* (continuous linear, discrete linear, empirical) per condition and budget |

---

## 9. Test suite and pass criteria

Every test below must pass at its gate. Run CPU tests with `pytest tests/unit -q`, GPU tests with `pytest tests/gpu -m gpu -q` (one GPU, nothing else running), and statistical tests with `pytest tests/stats -m stats -q`. Tests save their measured values to `reports/tests/<id>.json`. A test that "reports" a number also has a pass criterion.

**Naming.** Each test function is named `test_<ID>_<slug>` with the dots replaced by underscores (e.g. `test_T2_3_logprob_agreement`), so `-k T2` selects a group. The Appendix A tests keep their names; the T1 table maps them to IDs.

**If a test fails:** fix the code. Change a test only if the test itself is wrong. In that case, write the reason in `prereg/DEVIATIONS.md` before rerunning.

### T0: Environment (Gate G0)

| ID | Test | Pass criterion |
|---|---|---|
| T0.1 | `scripts/00_check_node.sh` | Exit 0; `reports/node.json` written |
| T0.2 | Import versions | `import vllm, torch, transformers, peft` works; versions recorded; `torch.cuda.device_count()==4` in a job without `CUDA_VISIBLE_DEVICES` |
| T0.3 | HF auth | `load_hf_token()` works; `HfApi().whoami()` returns a user; the token string appears in **no** file under the repo except `hf-tok` (`grep -rF "$(cat hf-tok)" --exclude=hf-tok .` finds nothing) |
| T0.4 | vLLM API surface (gpu) | These exist with the expected parameters, checked via `inspect.signature`: `AsyncLLM.from_engine_args`, `generate` (params `lora_request`, `request_id`), `add_lora`, `remove_lora`, `LoRARequest(lora_name, lora_int_id, lora_path)`, `SamplingParams(logprobs, prompt_logprobs, output_kind, seed)`, `AsyncEngineArgs(kv_cache_memory_bytes, enable_lora, max_loras, max_lora_rank, max_cpu_loras, max_num_seqs, max_num_batched_tokens, enable_prefix_caching)`, `TokensPrompt` |
| T0.5 | Model lock (gpu) | `configs/model_lock.yaml` exists and has the fields `repo`, `revision`, `local_dir`, `config_sha256`, `downloaded_at`; the sha256 of the local `config.json` equals `config_sha256`; the model loads offline (`HF_HUB_OFFLINE=1`) in both HF and vLLM |
| T0.6 | Chat template | `apply_chat_template(..., return_dict=False)` returns `list[int]`; decoding it gives a string containing `<|im_start|>system`, the system text, and ending with `<|im_start|>assistant\n` |

### T1: Unit tests, CPU only (Gate G0)

| ID | Test | Pass criterion |
|---|---|---|
| T1.1 | `hf_auth` | Resolution order (env > ./hf-tok > ~/hf-tok) with temp files; whitespace stripped; clear error when missing; warning when the mode allows group/other read; `RedactFilter` on handlers replaces the token in log output (capture with `caplog`) |
| T1.2 | `config` | Base loads; overrides apply; unknown key raises; `F % K != 0` raises; `lora.dropout != 0` raises; config hash is stable across processes |
| T1.3 | Ledger (Appendix A.7 part 1) | All pass: answers equal the text re-parse (levels 1, 3, 5; 900 futures each); determinism; independent streams; prefix unaffected by future generation; state round trip; most common answer < 10% of answers; > 80% of futures touch prefix transactions |
| T1.4 | Prompts and leakage | For 50 sampled prefixes with 3 futures each: no prefix-call writer prompt contains any future-chunk line or any question text; no reader prompt contains any log line (regex `^T\d+:` or `was cancelled` or `Correction:`) except inside the memory string; `writer_ids` and `reader_ids` return `list[int]` |
| T1.5 | Reward parsing | `"42"`→42, `" -17\n"`→−17, `"−17"`→−17, `"$1,234"`→1234, `"The answer is 8."`→8, `"8 dollars"`→8, `"eight"`→None (reward 0), `""`→None, `"3.0"`→3, `"-0"`→0 |
| T1.6 | Estimator (Appendix A.7 part 2) | Unbiasedness for E1 and E2 at (N,K) ∈ {(2,1),(4,2),(3,5),(8,8)}, all coordinates within 4 standard errors; weight-sum rules; constant-baseline variance formula within 3% |
| T1.7 | Variance accumulator | (a) Appendix A.7 synthetic recovery: mean V_prefix within 5%, V_future within 2%. (b) The torch port matches numpy to a relative 1e-6 |
| T1.8 | K* utilities | Appendix A.7: linear surface gives identical choices and zero regret; saturating surface gives non-negative regret |
| T1.9 | Curves | Appendix A.7 curve test |
| T1.10 | Loss code (Appendix A.8) | Per-call sums match the naive computation to atol 1e-4; microbatch invariance at relative 1e-5; gradient equals the naive objective's gradient at relative 1e-4; only LoRA parameters are trainable and dropout is Identity; packing respects the budget. **Plus a sign test:** one SGD step (lr = 1e-2) on a single call with weight +1 increases that call's log-probability sum, and weight −1 decreases it |
| T1.11 | Seeding | `seed_from` is stable across two subprocesses; no collisions across 10⁶ distinct keys |
| T1.12 | Occupancy | Synthetic intervals with a known n(t) integral: `mean_inflight` and `idle_slot_frac` correct to 1e-3 |
| T1.13 | `run_queue` dry run | With 7 fake jobs (`sleep` commands, fake GPUs 0–3): never more than 1 job per GPU; LPT order; a job with `DONE` is skipped; a failing job is retried once, then marked FAILED |
| T1.14 | Decision code | `q1`, `q2` and `q3` on synthetic inputs return the expected YES, NO or INCONCLUSIVE: a planted slope gives YES, flat curves give NO, a boundary case gives INCONCLUSIVE; planted costs and variances give the expected K choices |

### T2: GPU integration (Gate G1). One GPU, 4B model.

| ID | Test | Pass criterion |
|---|---|---|
| T2.1 | Engine smoke | 8 writer calls with adapter v0 and 8 reader calls complete; completions are non-empty; `vllm_lp` present for writer calls; `finish_reason` ∈ {stop, length} |
| T2.2 | Token-id round trip | vLLM `prompt_token_ids` equals our ids exactly; `tok.decode(completion_ids, skip_special_tokens=True).strip()` equals `text.strip()` |
| T2.3 | HF vs vLLM log-probs at v0 | Over 64 sampled writer calls: mean \|lp_hf − lp_vllm\| ≤ **0.03** nats/token; 99th percentile ≤ **0.30**; \|mean(lp_hf − lp_vllm)\| ≤ **0.02** |
| T2.4 | Adapter hot-swap | Take one AdamW step with lr = 1e-3 on 16 calls with random ±1 weights; save as v1; load it. Scoring the same 16 sequences with `score_tokens`: mean \|lp_vllm(v1) − lp_vllm(v0)\| ≥ 0.02 (the adapter changed the policy), and HF(v1) vs vLLM(v1) meets the T2.3 thresholds (vLLM loaded the new adapter correctly) |
| T2.5 | Reader is frozen | `score_tokens` of 16 fixed reader sequences with `lora=None` before and after loading v1: max per-token \|Δ\| ≤ 0.01 |
| T2.6 | Executor completeness | N=8, K=4, L=4, U=2 with each executor: record counts exact (32 prefix calls, 64 future calls, 32 reader calls); every future's `i` links to its prefix; rewards ∈ {0,1}; memory passed to future k equals the prefix's final memory |
| T2.7 | Executor equivalence | (a) If `VLLM_BATCH_INVARIANT=1` with `per_request_seeds=true` starts with LoRA: the two executors produce token-identical completions for all calls (N=8, K=4, L=4). (b) Otherwise, record why and use the statistical version: 3 reps of N=32, K=4, L=4; the mean completion length differs by < 3% and the reward mean difference is < 2 pooled SE |
| T2.8 | Microbatch invariance on the real model | Gradients with `max_tokens_per_microbatch` 32768 vs 4096 on 48 real calls: cosine ≥ 0.9999 and relative L2 difference ≤ 1e-2 |
| T2.9 | Variance tool matches the training gradient | On a small pool (N0=4, K0=4, L=4): (1/N0) Σᵢ (1/K0) Σₖ Z_ik from `variance.py` equals −(the gradient from `Trainer.step` with the same calls and weights, lr=0); cosine ≥ 0.9999, relative ≤ 1e-2 |
| T2.10 | Memory headroom | One iteration each of (N=128, K=4, L=16, B=512), (N=32, K=16, L=16, B=512) and (N=32, K=16, L=16, B=1024): peak NVML memory ≤ 78 GB; no CUDA OOM |
| T2.11 | Loop robustness | 3 consecutive rollout → train → sync cycles in one process (N=16, K=4, L=4); no hang (per-cycle timeout 15 min); T_iter of cycles 2 and 3 within 20% of each other |
| T2.12 | Resume | Tiny config (N=8, K=2, L=4, ckpt_every=5, I_max=14): `kill -9` during iteration index 11, then resume. It must restart at iteration index 10 from checkpoint `it0010`; the data keys of iterations 10 and 11 must be identical to the first attempt; the optimizer `step` counter must be 10 at restart and 14 at the end; `metrics.jsonl` must have exactly one line per iteration index 0–13 |
| T2.13 | 4-GPU concurrency | `run_queue` launches 4 tiny training jobs (10 iterations) on GPUs 0–3 at the same time; all exit 0; no port or IPC errors |
| T2.14 | NVML mapping | The sampler in a job with `CUDA_VISIBLE_DEVICES=2` reports the UUID of physical GPU 2; mean power during rollout > idle power + 50 W |
| T2.15 | Timing accounting | For 5 iterations: the median `t_other` (spec generation, rewards, weights, bookkeeping) is ≤ 5% of T_iter; `t_train_prefix + t_train_future` is within 3% of `t_train` (the remainder is the optimizer step) |
| T2.16 | No token leakage | After all T2 tests, grep of `runs/`, `profiles/`, `reports/` and the logs for the token finds nothing |

### T3: Task validity (Gate G2). Small eval set, 512 questions, at the candidate level.

| ID | Test | Pass criterion |
|---|---|---|
| T3.1 | Oracle-memory reader ceiling | Accuracy ≥ **0.90**; oracle memory ≤ B tokens (real tokenizer) for ≥ 95% of eval prefix+future states at B = 512 |
| T3.2 | Empty memory | Accuracy ≤ **0.08** |
| T3.3 | Shuffled oracle memory | Accuracy ≤ **0.12** |
| T3.4 | Base-writer accuracy (level choice) | The chosen level has base accuracy in **[0.20, 0.60]** (§10 Phase 2 rule) |
| T3.5 | Learning sanity | Level 1, L=4, U=1, N=8, K=4, lr=3e-5, 40 iterations, eval on 64 level-1 prefixes × 4: accuracy gain ≥ **+0.05** absolute at iteration 40 vs 0 (mean of the last two evals). On failure, see §13 R2 before anything else |

### T4: Statistical validity of the variance measurements (Phases 3 and 6)

| ID | Test | Pass criterion |
|---|---|---|
| T4.1 | Reproducibility | Two independent VM-0 measurements (vseed 0 and 1) at L=16, B=512: the ratios of V_prefix and of V_future are in [0.67, 1.5], and their 95% bootstrap CIs overlap |
| T4.2 | Subsample consistency | Recompute one pool using only futures 0–3 (K0' = 4): V_future within 20% of the K0 = 8 value; V_prefix CIs overlap |
| T4.3 | Prediction check | Split the pool: prefixes 32–63 give the reference mean ĝ_ref. From prefixes 0–31 form 8 disjoint estimators for each (N,K) ∈ {(4,1),(2,2),(1,4)}, using futures 0..K−1. Empirical MSE minus the reference's own model error (V_p/32 + V_f/256) should equal V_p/N + V_f/(NK); the aggregate ratio over the three settings must be in **[0.7, 1.4]** |

### T5: Profiling validity (Phases 3 and 6)

| ID | Test | Pass criterion |
|---|---|---|
| T5.1 | Repeatability | Per-cell coefficient of variation of T_iter across reps: median ≤ 5%, max ≤ 10%. Cells over 10% are rerun with reps = 5 |
| T5.2 | Parallel interference | 6 predeclared cells, (4,16), (8,8), (16,4), (32,2), (64,1), (128,4) at condition A, re-measured alone on GPU 0 with GPUs 1–3 idle: \|T_parallel − T_isolated\| / T_isolated ≤ 5% for every cell. If this fails, rerun the whole grid one GPU at a time |
| T5.3 | Policy frozen during profiling | Adapter sha256 identical before and after every cell |
| T5.4 | Benchmark sanity | Decode throughput is non-decreasing in n up to n_sat (tolerance 5%); n_sat ≥ 16 |

---

## 10. Execution phases, commands and gates

In the commands below, `$Q` means `python scripts/run_queue.py`, run inside tmux.

### Phase 0: Environment (Gate G0)

1. Run `bash scripts/00_check_node.sh`.
2. Run `bash scripts/01_setup_env.sh` and choose path A or B from the driver version.
3. Create the package skeleton (§6). Copy the Appendix A files verbatim.
4. Run `python scripts/02_download_model.py`.
5. Run `pytest tests/unit -q` and `pytest tests/gpu -m gpu -k "T0" -q`.

**Gate G0:** T0.1–T0.6 and T1.1–T1.14 pass. Commit, tag `gate-G0`.

### Phase 1: GPU integration (Gate G1)

Implement §8.6–§8.14. Run `pytest tests/gpu -m gpu -k "T2" -q` on GPU 0. T2.13 uses all four GPUs.

**Gate G1:** T2.1–T2.16 pass. Commit, tag `gate-G1`.

### Phase 2: Controls and level calibration (Gate G2)

1. `python scripts/calibrate.py --levels 1 2 3 4 5 --L 16 --U 2 --B 512`. For every level, on that level's small eval set, it measures:
   - oracle-memory accuracy (T3.1) and oracle token lengths with the real tokenizer
   - empty-memory accuracy (T3.2)
   - shuffled-oracle accuracy (T3.3)
   - base-writer accuracy (T3.4).
   Run one level per GPU in parallel. Results go to `reports/calibration.json`.
2. **Level rule (predeclared):**
   - Among levels whose oracle accuracy is ≥ 0.90 and whose base accuracy is in [0.20, 0.60], choose the one with base accuracy closest to **0.35**.
   - If none qualifies, apply fallback §13 R1.
   - Write `env.level` to `configs/locked.yaml`.
3. Evaluate T3.1–T3.4 at the chosen level from `reports/calibration.json`.
4. Run T3.5, the learning sanity check.

**Gate G2:** T3.1–T3.5 pass at the locked level. Commit, tag `gate-G2`.

### Phase 3: Microbenchmarks, engine tuning, iteration-0 cost grids and variance (Gate G3)

All of this uses the base policy (adapter v0).

1. **Engine tuning** (§8.13): `python scripts/profile_costs.py --tune-engine`. Write the winner to `configs/locked.yaml`.
2. **Benchmarks:** `python scripts/bench.py` gives `profiles/bench_<engcfg>.json` (n_sat, decode_rate_sat, prefill_rate, train tokens/s). Run T5.4.
3. **Cost grids:** run `python scripts/make_jobs.py --phase grid_main --ckpt base`, then `$Q jobs/grid_main.jsonl --gpus 0,1,2,3`. There are 81 cells, sharded 4 ways by estimated cost. Then do the same with `--phase grid_budget` (16 cells).
4. **Interference check** T5.2, then T5.1 and T5.3.
5. **Variance at iteration 0 (VM-0):** `make_jobs.py --phase variance0` covers the conditions {(L=16, B=512), (L=4, B=512), (L=16, B=256), (L=16, B=1024)} × vseed {0, 1}, 8 jobs in total. Run them with `$Q`. Then run T4.1, T4.2 and T4.3 (the pool from L=16, B=512, vseed 0).
6. **Early readout:** `python scripts/analyze.py --q2 --q3 --ckpt base`. This gives a preliminary Q2 and Q3 at the initial policy, which is **reported but not final**. Append it to STATUS.md.

**Gate G3:** T4.1–T4.3 and T5.1–T5.4 pass, and all 97 cells have ≥ 3 valid reps. Commit, tag `gate-G3`.

### Phase 4: Learning-rate pilot and timing projection (Gate G4)

1. **Pilot runs:** K ∈ {1, 16} × lr ∈ {1e-5, 3e-5, 1e-4}, seed 100 (not a sweep seed), 60 iterations, eval every 10 iterations on the small set. `make_jobs.py --phase pilot`, then `$Q jobs/pilot.jsonl`. That is 6 jobs, about 2 waves.
2. **lr rule (predeclared):** for each lr, compute normalized AUC over [0, 60] for both arms. Choose the lr that **maximizes the minimum** of the two arms' AUCs. Break ties (difference < 0.005) toward the smaller lr.
3. **Learning check:** at the chosen lr, at least one arm must show an eval accuracy gain ≥ +0.03 (mean of evals at 50 and 60 minus the value at 0). Otherwise apply fallback §13 R2.
4. **grad_clip:** set it to 5× the median `grad_norm` over all pilot iterations at the chosen lr.
5. **Timing projection:** from the pilot's measured seconds per iteration (K=1 and K=16, with interpolation for K = 2, 4, 8 by N), project the sweep's wall-clock under LPT on 4 GPUs.
   - If the projection is ≤ 60 h, keep I_max = 250.
   - Otherwise apply **in order** until it is ≤ 60 h: (i) I_max = 200; (ii) small eval set 128 → 96 prefixes; (iii) drop the K = 2 arm. Record what was applied.
6. **Freeze the preregistration:** `python scripts/freeze_prereg.py` writes `prereg/PREREG.md` containing:
   - the resolved sweep config and the `configs/locked.yaml` contents
   - arms, seeds, I_max, lr, grad_clip, level, eval-set definitions
   - the full text of §11 with thresholds
   - the git commit hash of `kmatters/analysis/`
   - its own sha256, stored in `prereg/PREREG.sha256`.
   Commit it.

**Gate G4:** learning check passed, projection ≤ 60 h, PREREG committed. Tag `gate-G4`. **Nothing in PREREG changes after this point** except through a dated entry in DEVIATIONS.md.

### Phase 5: Main fixed-K sweep

1. Run `make_jobs.py --phase sweep`: 15 jobs, K ∈ {1,2,4,8,16}, N = 64/K, seeds {0,1,2}, every other setting identical.
2. Launch `$Q jobs/sweep.jsonl --gpus 0,1,2,3`. LPT ordering puts K=1 jobs first.
3. **Monitoring**, every 30–60 min: tail `jobs/sweep.status.json` and each `metrics.jsonl`. Flag a run in STATUS.md if any of these occur:
   - `lp_absdiff_mean` > 0.05 for 3 consecutive iterations. That points to an adapter-sync or numerics bug: stop that run and investigate.
   - Non-finite loss or gradient norm: stop, fix, and restart from the last checkpoint. Record a deviation.
   - `clipped` in more than 2% of iterations.
   - Truncation rate > 0.5 sustained: record only; it is part of the learned behavior.
4. Progress updates go to STATUS.md. Do not start the analysis until all 15 runs have `DONE`.

### Phase 6: Trained-checkpoint measurements (Gate G6)

1. **Reference checkpoint:** `runs/sweep_K4_N16_s0_*/ckpt/it{I_max}/adapter`, i.e. K = 4, seed 0, final. This is predeclared.
2. **Trained cost grid:** `make_jobs.py --phase trained_grid --ckpt <reference>` runs condition A (27 cells) plus the barrier cell (32,4), then T5.1–T5.3 on it.
3. **Variance at trained checkpoints:** `make_jobs.py --phase variance_trained` with:
   - (a) L=16, B=512, vseed 0 at `it ∈ {100, I_max}` for K ∈ {1, 4, 16}, seed 0 (6 jobs)
   - (b) L=4, B=512 and L=16, B ∈ {256, 1024} at the reference checkpoint (3 jobs)
   - (c) a second vseed at the reference checkpoint for L=16, B=512 (T4.1 repeated).

**Gate G6:** T4.1 (trained) and T5.1–T5.3 (trained grid) pass.

### Phase 7: Analysis, decisions, conditional extensions

1. Run `python scripts/analyze.py --all`. This produces `reports/q1.json`, `q2.json`, `q3.json`, the figures, and `reports/DECISION.md`.
2. **Conditional extensions.** These are predeclared; apply each at most once, then rerun step 1.
   - **Seed extension:** if Q1a is INCONCLUSIVE, add seeds {3, 4} for K ∈ {1, 4, 16} (6 runs).
   - **Equal-cost extension:** if an arm is censored in Q1b (it did not reach the target within I_max) but its cumulative priced GPU time at I_max is below the K=1 arm's median priced time-to-target, resume that arm's runs up to that cost horizon (the maximum of the arm's seeds' needs).
3. Send the owner `reports/DECISION.md` and a short summary: the three answers, the key numbers, and whether any extension ran.

---

## 11. Predeclared analysis and decision rules

Everything here is computed by `kmatters/analysis/q{1,2,3}.py`. Numbers are copied into `prereg/PREREG.md` at Gate G4 and are not changed afterwards.

### 11.1 Q1: Does K matter for learning?

**Data.** For each sweep run, the small-eval accuracy acc(it) at it = 0, 10, …, I_max. Let a0_run = acc(0).

**Q1a: per sampled future, at fixed F = 64**
- Per run: gain-AUC = `normalized_auc(it, acc, I_max) − a0_run`.
- Fit ordinary least squares across all runs: gain-AUC ~ α + β·log₂K. Let D = 4β, the predicted gain-AUC difference between K = 16 and K = 1.
- Report the 95% t-interval for D (df = n_runs − 2), a permutation p-value (10,000 label permutations of K across runs), per-arm means ± 95% CIs, and a one-way ANOVA across arms.
- **YES** if the 95% CI of D excludes 0 **and** |D̂| ≥ 0.02.
- **NO** if the 95% CI of D lies inside [−0.02, +0.02].
- **INCONCLUSIVE** otherwise. This triggers the seed extension (§10 Phase 7), after which the rule is applied once more with all runs.

**Q1b: after pricing**
- **Priced time.** Iteration j of an arm (N, K) costs T_price(j) = T_base(N,K) + (j/I_max)·(T_trained(N,K) − T_base(N,K)). The T values are median T_iter of condition A (async_ready, L=16, B=512) from the base and trained grids. Cumulative priced GPU-hours are G(it) = Σ_{j<it} T_price(j) / 3600. Eval time is excluded.
- **Target.** a* = a0 + 0.5·(ā_K1 − a0), where ā_K1 is the mean over K=1 seeds of the mean accuracy at the last two evals, and a0 is the mean of a0_run. If ā_K1 − a0 < 0.03, use the arm with the largest mean final gain in place of K=1.
- **Per run.** Time-to-target TTT_run = `time_to_target(G(it), smooth(acc, 3), a*)`, which may be censored.
- **Per arm.** The median TTT across seeds. A censored run counts as +∞, so an arm with 2 of 3 runs censored is censored.
- **Ratio.** R = (slowest arm median) / (fastest arm median), with a bootstrap 95% CI: 5,000 draws, resampling seeds within each arm and each cell's T from its reps.
- **YES** if R ≥ 1.25 and the CI lower bound > 1.0.
- **NO** if the CI upper bound < 1.25.
- **INCONCLUSIVE** otherwise.
- **Also report:**
  - The fastest arm.
  - Whether the optimum is **interior**: the fastest arm has K ∈ {2, 4, 8}, and its TTT is below both K=1 and K=16 in ≥ 80% of bootstrap draws.
  - The priced accuracy at a fixed budget equal to the fastest arm's median TTT.

**Mechanism check (report only).** Using VM-0 at L=16, B=512, the predicted per-iteration MSE of each arm is MSE_K = (K·V_prefix + V_future)/64. Report the Spearman correlation between −MSE_K and the arm-mean gain-AUC, together with the V_future/V_prefix ratio at iteration 0, 100 and I_max.

**Interpretation table for DECISION.md**

| Q1a | Q1b | Interpretation |
|---|---|---|
| NO | NO | K does not matter, even after pricing. Allocation is irrelevant: stop the allocation line of work |
| NO | YES, fastest = largest K | Correlated futures are nearly as informative as independent ones, so branch as much as possible. There is no interior allocation problem |
| YES | YES, interior optimum | The allocation premise survives. Continue to the conditional-static and adaptive questions |
| YES | NO | Per-sample differences cancel against cost differences. Allocation is not worth engineering at this scale |

### 11.2 Q2: Does exploitable capacity survive competent scheduling?

**Conditions**
- Primary: the trained grid, condition A (async_ready, L=16, B=512, reference checkpoint).
- Secondary: base-grid conditions A and C (L=4); the barrier executor (condition B) for contrast.

**Per cell** (medians over reps): `rollout_frac = t_rollout / T_iter`.

**Practical pairs** are (N,K) → (N,2K) with N·K ≥ 32 and N·2K ≤ 256. For each pair:
- ΔT_f^roll = (t_rollout(N,2K) − t_rollout(N,K)) / (N·K): the marginal rollout seconds per added future.
- c_f^sat: the saturated rollout cost of one future. It is computed from cell (N,2K)'s token counts per future: future-writer and reader uncached prompt tokens ÷ `prefill_rate`, plus completion tokens ÷ `decode_rate_sat` (from `bench.py`).
- ρ_f = ΔT_f^roll / c_f^sat.

**Also reported**
- ρ_p for pairs (N,K) → (2N,K): the added prefix together with its K futures, divided by the saturated cost of the same.
- The total marginal cost per added future, including training and sync, divided by the average total cost per future in that cell.
- `idle_slot_frac`, `mean_inflight` / n_sat.
- For each barrier cell, `t_rollout(barrier) / t_rollout(async_ready)`.

**Decision**
- **YES-material:** median ρ_f ≤ 0.5 over the practical pairs **and** median rollout_frac ≥ 0.25 over the cells in those pairs.
- **YES-immaterial:** median ρ_f ≤ 0.5 **and** median rollout_frac < 0.25.
- **NO:** median ρ_f ≥ 0.8.
- **PARTIAL:** otherwise.
- **Also state whether the capacity is specific to futures:** if median ρ_p ≤ 0.5 as well, spare capacity can be used just as cheaply by more independent histories. That is the competing use Q3 must price.

### 11.3 Q3: Does marginal-cost pricing move K\*?

**Inputs**
- V_prefix and V_future bootstrap draws (2000) at the reference checkpoint, L=16, B=512, vseed 0.
- The 27 cells of the trained grid, condition A, each with its median T and its reps.

**Cost models**
- **(a) Profile-once additive model** (the model in the original proposal). From the barrier cell (32,4) at the reference checkpoint:
  - C_p = (t_phase_prefix + t_train_prefix) / N
  - C_f = (t_phase_future + t_phase_reader + t_train_future) / (N·K)
  - T_a(N,K) = N·C_p + N·K·C_f, with no intercept.
  - The barrier executor must record `t_phase_prefix`, `t_phase_future` and `t_phase_reader`.
- **(b) Best linear model:** `fit_linear_costs` (a non-negative least-squares fit with an intercept) over the 27 cells.
- **(e) Measured surface:** the median T of each cell.

**Choice under budget.** Budgets are Tb = β·T_meas(32,1), with β ∈ {1, 2, 4}. For model m ∈ {a, b}:
- choice_m = the cell minimizing MSE = V_p/N + V_f/(NK) subject to T_m ≤ Tb.
- choice_e = the same subject to T_meas ≤ Tb.
- regret_m = MSE(choice_m) / MSE(choice_e) − 1. If T_meas(choice_m) > Tb, the choice is infeasible and its regret counts as +∞.

This is `compare_pricing` in Appendix A.5 applied per draw. For model (a), replace `T_lin` with T_a.

**Bootstrap.** For each of the 2000 V draws, also resample each cell's T from its reps. V_prefix draws ≤ 0 are floored at 1e-6·V_future, and the fraction floored is recorded. If more than 20% of draws are floored, Q3 = INCONCLUSIVE ("V_prefix not resolved"). The predeclared remedy is a single re-measurement with N0 = 128, followed by a rerun of Q3.

**Decision (primary = model a)**
- **YES** if for at least one β: P(K_a ≠ K_e) ≥ 0.8 **and** the median regret_a ≥ 0.10.
- **NO** if for every β: P(K_a ≠ K_e) < 0.5 **or** the median regret_a < 0.03.
- **INCONCLUSIVE** otherwise.

**Qualifier (report).** Apply the same rule to model (b).
- If (a) is YES and (b) is NO, the movement comes from a fixed per-iteration overhead or intercept that a linear fit captures. The conclusion is then "marginal pricing moves K\*, but a profiled linear model with an intercept is enough; no state-dependent scheduler is needed".
- If (b) is also YES, the cost surface is genuinely non-linear, which is the case that motivates a systems contribution.

**Also report**
- The continuous K\*_a = √(V_f·C_p / (V_p·C_f)) and K\*_b.
- The empirical choices K_e for every secondary condition: the base grid with VM-0 (L = 4 and 16; B = 256, 512, 1024, using model (b) where no barrier cell exists), and base vs trained. This shows whether K\* moves across conditions and training stages. That is a secondary result, with no decision attached.

---

## 12. Output schemas

**`runs/<run_id>/metrics.jsonl`**, one line per iteration. All times are in seconds.
```json
{"run_id":"...","it":17,"policy_version":17,"K":4,"N":16,"F":64,"L":16,"U":2,"B":512,"level":3,"seed":0,
 "lr":3e-5,"config_hash":"...",
 "reward_mean":0.41,"reward_std":0.49,"rbar_std":0.21,"frac_prefix_all0":0.12,"frac_prefix_all1":0.06,
 "adv_pre_abs_mean":0.18,"adv_fut_abs_mean":0.45,
 "n_calls":{"prefix":256,"future":128,"reader":64},
 "tokens":{"prefix":{"prompt":...,"cached":...,"completion":...},"future":{...},"reader":{...}},
 "mem_len_mean":301.2,"mem_len_p90":402,"trunc_rate":0.03,
 "loss":0.0123,"grad_norm":0.84,"clipped":false,
 "lp_absdiff_mean":0.011,"lp_diff_mean":-0.001,"kl_k3":0.0002,"neg_lp_mean":0.31,
 "t_rollout":61.2,"t_train":44.9,"t_train_prefix":38.0,"t_train_future":6.9,"t_sync":2.1,"t_other":0.8,"t_iter":109.0,
 "t_eval":null,"gpu_mem_peak_gb":61.3,"wall_time":"2026-10-02T13:05:11Z",
 "eval_small":null,"eval_large":null,
 "timing_contaminated":true}
```
- On eval iterations, `eval_small` is `{"acc":…,"acc_by_type":{"net":…,"paid":…,"cat":…,"amount":…},"mem_len_mean":…,"trunc_rate":…,"n_questions":512,"per_prefix_correct":[…]}`.
- `timing_contaminated` is always `true` in sweep runs, because four jobs share the node. **Pricing uses only the profiling grids.**

**`runs/<run_id>/rollouts/it{it:04d}.jsonl.gz`**: one line per prefix, containing:
- key, L call records (text, token counts, finish reason, times), the final memory
- K futures, each with its U call records, reader record, question, gold, answer text, and reward.

Token ids are omitted to save space.

**`profiles/<tag>/cells.jsonl`**: one line per (cell, rep), containing:
- `cell_id, executor, N, K, L, B, ckpt, rep, warmup`
- `T_iter, t_rollout, t_train_prefix, t_train_future, t_sync, t_other`
- `t_phase_prefix, t_phase_future, t_phase_reader` (barrier only)
- `tokens` by role
- `occupancy {mean_inflight, idle_slot_frac, by_role}`
- `nvml {rollout:{sm,mem,power}, train:{...}}`
- `gpu_mem_peak_gb, adapter_sha256, engine_cfg`.

**`variance/<tag>/result.json`**:
- `ckpt, it, L, B, N0, K0, vseed, reward_mean`
- `V_prefix, V_future`, each as `{est, p2_5, p50, p97_5}`
- `reward_level {V_prefix, V_future}`, `frac_Vp_nonpositive_boot`
- `config_hash`
- plus `gram.npy` and `within.npy`.

---

## 13. Risks and predeclared fallbacks

| ID | Risk | Detection | Fallback, in order; record each step in DEVIATIONS.md |
|---|---|---|---|
| R1 | No level gives base accuracy in [0.20, 0.60] with oracle ≥ 0.90 | Phase 2 | (i) If every level is < 0.20: set U = 1, recalibrate. (ii) Still < 0.20: B = 768. (iii) Every level > 0.60: L = 24. (iv) Oracle < 0.90 at every level: reword READER_SYSTEM only (this is the single allowed prompt change), then rerun controls. (v) Otherwise, stop and ask the owner |
| R2 | No learning (T3.5 or the G4 learning check fails) | Phases 2, 4 | (i) Recheck T2.3, T2.4 and T1.10's sign test. Most "no learning" bugs are adapter sync or sign errors. (ii) Inspect 20 rollouts by hand for degenerate memories. (iii) Extend the pilot lr grid with 3e-4. (iv) Extend the pilot to 120 iterations. (v) Otherwise stop and ask the owner |
| R3 | HF vs vLLM log-prob mismatch above threshold | T2.3, sweep monitor | Check in this order: chat-template ids (T2.2), LoRA scaling (`lora_alpha/r` read by vLLM from adapter_config.json), adapter dtype, `logprobs_mode`, `max_lora_rank` ≥ r. Never "fix" it by importance weighting |
| R4 | OOM | T2.10 or runtime | (i) `max_tokens_per_microbatch` 16384 → 8192. (ii) KV cache 28 → 22 GiB. (iii) `enable_sleep_mode=True`, with `sleep(level=1)` before training and `wake_up()` after. The sleep/wake time counts in t_sync, and the setting must then apply to all runs and grids |
| R5 | CUDA driver < 580 | Phase 0 | Path B (§5.2) |
| R6 | Sweep slower than projected | Phase 5 monitor | Reductions (i)–(iii) of §10 Phase 4 step 5 were fixed at G4. If Phase 5 runs more than 30% over the projection anyway, ask the owner |
| R7 | Four vLLM engines conflict (ports or IPC) | T2.13 | Stagger starts by 90 s; set a distinct `VLLM_PORT`/`MASTER_PORT` per GPU (`29500 + 10*gpu`); set `VLLM_RPC_BASE_PATH=/tmp/vllm_rpc_gpu{g}` |
| R8 | Memory collapse (constant or empty memories, reward → 0) | Monitor: `mem_len_mean` < 20 for 5 iterations | Report as an outcome of that run. Do not tune per arm |
| R9 | Eval noise hides effects | Phase 7 | Paired eval set plus AUC endpoint (already in the design); the seed extension exists for this reason |
| R10 | Batch-invariant mode unavailable with LoRA | T2.7(a) | Use T2.7(b) and record the reason |
| R11 | V_prefix not resolved (many non-positive bootstrap draws) | §11.3 | Re-measure once with N0 = 128 |
| R12 | Adam compresses variance differences between arms | Phase 7 interpretation | Report it as a limitation. An SGD arm is out of scope |

---

## 14. Deliverables checklist (end of Phase 7)

- [ ] `reports/DECISION.md`: three answers with rule evaluations, key numbers, figures, deviations, limitations
- [ ] `reports/q1.json`, `q2.json`, `q3.json`; `reports/fig/*.png`
- [ ] `prereg/PREREG.md` (frozen at G4, hash matches), `prereg/DEVIATIONS.md`
- [ ] `reports/STATUS.md` with the GPU-hours used per phase
- [ ] All test results in `reports/tests/*.json`; gates tagged in git
- [ ] `runs/`, `profiles/` and `variance/` intact (never deleted without asking)
- [ ] `env/requirements.lock`, `configs/model_lock.yaml`, `configs/locked.yaml`

---

## Appendix A: Reference code (tested; use verbatim)

These files were run in a clean CPU environment on 2026-09-25 with Python 3.11, numpy 2.4, scipy 1.17, torch 2.14 (CPU), transformers 5.17.0 and PEFT 0.21.0. Result: **30 passed**.

Place them at the paths shown, create empty `__init__.py` files in `kmatters/`, `kmatters/env/`, `kmatters/rl/` and `kmatters/analysis/`, and run:
```bash
uv pip install -e . && pytest tests/unit -q
```
All tests must pass before any GPU work. `kmatters_reference.zip`, if provided, contains exactly these files.

**Notes**
- `curves.py` uses `np.trapezoid`, which needs numpy ≥ 2.0.
- `logprobs.py` assumes a PEFT-wrapped HF causal LM whose `get_base_model()` exposes `.model` (the decoder) and `get_output_embeddings()`. This holds for Qwen3 in transformers 5.

### A.1 `kmatters/env/ledger.py`

Task environment (ledger histories, futures, questions, answers).

```python
"""Ledger environment: histories of transactions with cancels/corrections.

A history ("prefix") is L chunks of E events. A future is U more chunks
followed by one question. The writer sees chunks; the reader sees only the
final memory and the question. Ground truth comes from the exact state.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

import numpy as np

NAMES = [
    "Alice", "Bob", "Chen", "Dana", "Emeka", "Farah", "Goran", "Hana", "Ivan",
    "Jia", "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa",
    "Sven", "Tara", "Umar", "Vera", "Wen", "Ximena", "Yusuf", "Zoe", "Arjun",
    "Bea", "Cyrus", "Dmitri", "Elif", "Femi", "Gita", "Hugo", "Ines", "Jonas",
    "Kira", "Luis", "Mira", "Nils",
]
CATEGORIES = [
    "groceries", "rent", "travel", "dining", "utilities", "books", "fuel",
    "gifts", "music", "hardware",
]
NOISE_TEMPLATES = [
    "Note: {p} changed their phone number.",
    "{p} said the weather was {adj} today.",
    "Reminder: the team meeting moved to {day}.",
    "{p} ran {n} km this morning.",
    "{p} is reading a book with {n} chapters.",
    "{p} adopted a cat named {q}.",
    "The office printer on floor {n} is broken again.",
    "{p} and {q} watched a movie together.",
    "{p} bought {n} stamps at the post office (paid in cash, not part of this ledger).",
    "Room {n} is booked for {day}.",
    "{p} planted {n} tomato seedlings.",
    "{p} says hello to {q}.",
]
ADJ = ["sunny", "rainy", "windy", "cold", "humid", "pleasant"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Difficulty ladder. Level 3 is the default; calibration picks the level.
LEVELS = {
    1: dict(n_people=3, n_cats=2, events_per_chunk=3, amount_max=20,
            p_txn=0.70, p_cancel=0.05, p_amend=0.05, p_noise=0.20),
    2: dict(n_people=4, n_cats=3, events_per_chunk=4, amount_max=50,
            p_txn=0.64, p_cancel=0.08, p_amend=0.08, p_noise=0.20),
    3: dict(n_people=5, n_cats=4, events_per_chunk=4, amount_max=99,
            p_txn=0.60, p_cancel=0.10, p_amend=0.10, p_noise=0.20),
    4: dict(n_people=6, n_cats=5, events_per_chunk=5, amount_max=99,
            p_txn=0.56, p_cancel=0.12, p_amend=0.12, p_noise=0.20),
    5: dict(n_people=8, n_cats=6, events_per_chunk=6, amount_max=99,
            p_txn=0.50, p_cancel=0.15, p_amend=0.15, p_noise=0.20),
}
FUTURE_MIX = dict(p_txn=0.40, p_cancel=0.25, p_amend=0.25, p_noise=0.10)
QUESTION_MIX = {"net": 0.30, "paid": 0.20, "cat": 0.25, "amount": 0.25}
P_TARGET_PREFIX = 0.8   # future cancels/amends target prefix txns
P_AMOUNT_PREFIX = 0.7   # 'amount' questions target prefix txns

SPLIT_IDS = {"train": 0, "eval": 1, "variance": 2, "profile": 3, "calib": 4}


@dataclass
class Txn:
    tid: int
    payer: str
    payee: str
    amount: int
    cat: str


@dataclass
class LedgerState:
    people: list
    cats: list
    live: dict = field(default_factory=dict)   # tid -> Txn
    next_id: int = 1
    prefix_ids: list = field(default_factory=list)  # tids created in prefix

    def to_json(self) -> str:
        d = asdict(self)
        d["live"] = {str(k): asdict(v) for k, v in self.live.items()}
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "LedgerState":
        d = json.loads(s)
        live = {int(k): Txn(**v) for k, v in d["live"].items()}
        return LedgerState(d["people"], d["cats"], live, d["next_id"],
                           d["prefix_ids"])


def rng_for(split: str, *keys: int) -> np.random.Generator:
    """Deterministic, independent stream per (split, keys)."""
    return np.random.default_rng([SPLIT_IDS[split], *[int(k) for k in keys]])


def _pick_event_type(rng, mix) -> str:
    kinds = ["txn", "cancel", "amend", "noise"]
    p = np.array([mix["p_txn"], mix["p_cancel"], mix["p_amend"], mix["p_noise"]])
    return kinds[int(rng.choice(4, p=p / p.sum()))]


def _target(rng, st: LedgerState, prefer_prefix: bool) -> int | None:
    if not st.live:
        return None
    live = sorted(st.live)
    if prefer_prefix:
        pref = [t for t in live if t in set(st.prefix_ids)]
        if pref and rng.random() < P_TARGET_PREFIX:
            return int(rng.choice(pref))
    return int(rng.choice(live))


def _noise(rng, st: LedgerState) -> str:
    t = NOISE_TEMPLATES[int(rng.integers(len(NOISE_TEMPLATES)))]
    p, q = rng.choice(st.people, 2, replace=False)
    return t.format(p=p, q=q, adj=ADJ[int(rng.integers(len(ADJ)))],
                    day=DAYS[int(rng.integers(len(DAYS)))],
                    n=int(rng.integers(2, 60)))


def gen_event(rng, st: LedgerState, mix: dict, amount_max: int,
              in_future: bool) -> tuple:
    kind = _pick_event_type(rng, mix)
    if kind in ("cancel", "amend") and not st.live:
        kind = "txn"
    if kind == "txn":
        payer, payee = rng.choice(st.people, 2, replace=False)
        tx = Txn(st.next_id, str(payer), str(payee),
                 int(rng.integers(1, amount_max + 1)),
                 str(st.cats[int(rng.integers(len(st.cats)))]))
        return ("txn", tx)
    if kind == "cancel":
        return ("cancel", _target(rng, st, in_future))
    if kind == "amend":
        tid = _target(rng, st, in_future)
        old = st.live[tid].amount
        new = old
        while new == old:
            new = int(rng.integers(1, amount_max + 1))
        return ("amend", tid, new)
    return ("noise", _noise(rng, st))


def apply_event(st: LedgerState, ev: tuple, in_prefix: bool) -> None:
    if ev[0] == "txn":
        tx = ev[1]
        st.live[tx.tid] = copy.copy(tx)
        st.next_id = tx.tid + 1
        if in_prefix:
            st.prefix_ids.append(tx.tid)
    elif ev[0] == "cancel":
        del st.live[ev[1]]
    elif ev[0] == "amend":
        st.live[ev[1]].amount = ev[2]


def render_event(ev: tuple) -> str:
    if ev[0] == "txn":
        t = ev[1]
        return f"T{t.tid:02d}: {t.payer} paid {t.payee} ${t.amount} for {t.cat}."
    if ev[0] == "cancel":
        return f"T{ev[1]:02d} was cancelled."
    if ev[0] == "amend":
        return f"Correction: the amount of T{ev[1]:02d} should be ${ev[2]}."
    return ev[1]


def answer(st: LedgerState, q: tuple) -> int:
    kind, arg = q
    tx = list(st.live.values())
    if kind == "net":
        return sum(t.amount for t in tx if t.payee == arg) - \
            sum(t.amount for t in tx if t.payer == arg)
    if kind == "paid":
        return sum(t.amount for t in tx if t.payer == arg)
    if kind == "cat":
        return sum(t.amount for t in tx if t.cat == arg)
    if kind == "amount":
        return st.live[arg].amount
    raise ValueError(kind)


def render_question(q: tuple) -> str:
    kind, arg = q
    if kind == "net":
        return (f"What is {arg}'s net balance (total received minus total paid) "
                f"over all transactions that are not cancelled?")
    if kind == "paid":
        return f"How much has {arg} paid in total over all transactions that are not cancelled?"
    if kind == "cat":
        return f"What is the total amount of all non-cancelled transactions in the category '{arg}'?"
    if kind == "amount":
        return f"What is the current amount of transaction T{arg:02d}?"
    raise ValueError(kind)


def gen_question(rng, st: LedgerState) -> tuple:
    kinds = list(QUESTION_MIX)
    p = np.array([QUESTION_MIX[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
    if kind == "amount" and not st.live:
        kind = "net"
    if kind in ("net", "paid"):
        return (kind, str(st.people[int(rng.integers(len(st.people)))]))
    if kind == "cat":
        return (kind, str(st.cats[int(rng.integers(len(st.cats)))]))
    live = sorted(st.live)
    pref = [t for t in live if t in set(st.prefix_ids)]
    if pref and rng.random() < P_AMOUNT_PREFIX:
        return ("amount", int(rng.choice(pref)))
    return ("amount", int(rng.choice(live)))


@dataclass
class Prefix:
    key: tuple
    chunks: list          # list[str], length L
    state_json: str       # LedgerState after the prefix


@dataclass
class Future:
    key: tuple
    chunks: list          # list[str], length U
    question: str
    q: tuple
    gold: int


def make_prefix(split: str, key: tuple, L: int, level: int) -> Prefix:
    cfg = LEVELS[level]
    rng = rng_for(split, 0, *key)
    people = [str(x) for x in rng.choice(NAMES, cfg["n_people"], replace=False)]
    cats = [str(x) for x in rng.choice(CATEGORIES, cfg["n_cats"], replace=False)]
    st = LedgerState(people, cats)
    chunks = []
    for _ in range(L):
        lines = []
        for _ in range(cfg["events_per_chunk"]):
            ev = gen_event(rng, st, cfg, cfg["amount_max"], in_future=False)
            apply_event(st, ev, in_prefix=True)
            lines.append(render_event(ev))
        chunks.append("\n".join(lines))
    return Prefix(key, chunks, st.to_json())


def make_future(split: str, prefix: Prefix, k: int, U: int, level: int) -> Future:
    cfg = LEVELS[level]
    rng = rng_for(split, 1, *prefix.key, k)
    st = LedgerState.from_json(prefix.state_json)
    chunks = []
    for _ in range(U):
        lines = []
        for _ in range(cfg["events_per_chunk"]):
            ev = gen_event(rng, st, FUTURE_MIX, cfg["amount_max"], in_future=True)
            apply_event(st, ev, in_prefix=False)
            lines.append(render_event(ev))
        chunks.append("\n".join(lines))
    q = gen_question(rng, st)
    return Future((*prefix.key, k), chunks, render_question(q), q, answer(st, q))


# ---------------- reference re-implementation used only by tests -----------
def brute_force_answer(prefix: Prefix, fut: Future, level: int) -> int:
    """Recompute the answer by re-parsing the rendered text only."""
    import re
    live = {}
    for line in "\n".join(prefix.chunks + fut.chunks).splitlines():
        m = re.match(r"T(\d+): (\w+) paid (\w+) \$(\d+) for (\w+)\.$", line)
        if m:
            live[int(m[1])] = [m[2], m[3], int(m[4]), m[5]]
            continue
        m = re.match(r"T(\d+) was cancelled\.$", line)
        if m:
            del live[int(m[1])]
            continue
        m = re.match(r"Correction: the amount of T(\d+) should be \$(\d+)\.$", line)
        if m:
            live[int(m[1])][2] = int(m[2])
    kind, arg = fut.q
    if kind == "net":
        return sum(a for p, r, a, c in live.values() if r == arg) - \
            sum(a for p, r, a, c in live.values() if p == arg)
    if kind == "paid":
        return sum(a for p, r, a, c in live.values() if p == arg)
    if kind == "cat":
        return sum(a for p, r, a, c in live.values() if c == arg)
    return live[arg][2]
```

### A.2 `kmatters/rl/estimator.py`

Advantages and per-call weights (E1 primary, E2 ablation).

```python
"""Advantages and per-call loss weights for the nested (N prefixes x K futures) estimator.

g_hat = (1/N) sum_i [ A_pre_i * s_pre_i + (1/K) sum_k A_fut_ik * s_fut_ik ]
where s_* are score vectors (sum of grad log-probs of the writer tokens in those calls).

Baselines:
  "loo_prefix" (E1, primary): b_i = mean_{j != i} rbar_j, used for both prefix and future terms.
  "tree"       (E2, ablation): prefix term as E1; future term uses b_ik = mean_{l != k} r_il
                               (falls back to E1 when K == 1).
Both are unbiased: each baseline is independent of the actions it multiplies.
"""
from __future__ import annotations

import numpy as np


def advantages(r: np.ndarray, baseline: str = "loo_prefix"):
    """r: array [N, K] of rewards. Returns (A_pre [N], A_fut [N, K])."""
    r = np.asarray(r, dtype=np.float64)
    N, K = r.shape
    if N < 2:
        raise ValueError("need N >= 2 for a leave-one-out baseline")
    rbar = r.mean(axis=1)
    b = (rbar.sum() - rbar) / (N - 1)
    A_pre = rbar - b
    if baseline == "loo_prefix" or K == 1:
        A_fut = r - b[:, None]
    elif baseline == "tree":
        b_in = (r.sum(axis=1, keepdims=True) - r) / (K - 1)
        A_fut = r - b_in
    else:
        raise ValueError(baseline)
    return A_pre, A_fut


def call_weights(r: np.ndarray, baseline: str = "loo_prefix"):
    """Coefficient multiplying each call's score vector in g_hat.

    Returns (w_pre [N], w_fut [N, K]). Every writer call inside prefix i gets
    w_pre[i]; every writer call inside future (i, k) gets w_fut[i, k].
    Training loss = -(1 / T_norm) * sum_calls w_call * sum_tokens log pi.
    """
    N, K = np.asarray(r).shape
    A_pre, A_fut = advantages(r, baseline)
    return A_pre / N, A_fut / (N * K)
```

### A.3 `kmatters/rl/logprobs.py`

Completion log-probabilities, weighted loss, microbatching, flat LoRA gradient.

```python
"""Token log-probabilities of completions and the weighted policy-gradient loss.

Design rules (see PLAN.md section 8.9):
  * Train on the exact token ids vLLM returned (prompt_token_ids + token_ids).
  * Right-pad, run the decoder once, gather hidden states only at positions
    that predict completion tokens, apply lm_head in chunks, log_softmax in fp32.
  * One optimizer step per iteration; grads accumulate across microbatches.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Call:
    prompt_ids: list
    completion_ids: list
    weight: float = 0.0          # coefficient w_c from rl/estimator.call_weights


def pack_microbatches(lengths, max_tokens):
    """Greedy packing of calls (sorted by length) so padded_len * n <= max_tokens.
    Returns list of index lists. A single call longer than max_tokens gets its own
    microbatch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and new_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _decoder_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def completion_logprob_sums(model, calls, pad_id, device, head_chunk=4096):
    """Returns a 1-D tensor: for each call, sum over completion tokens of log pi.
    Differentiable w.r.t. trainable params."""
    decoder, head = _decoder_and_head(model)
    seqs = [c.prompt_ids + c.completion_ids for c in calls]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = torch.arange(T, device=device).unsqueeze(0).expand(len(seqs), T)
    hidden = decoder(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    # positions p predict token p+1; completion tokens sit at [P, P+C)
    rows, cols, targets, owner = [], [], [], []
    for b, c in enumerate(calls):
        P, C = len(c.prompt_ids), len(c.completion_ids)
        rows += [b] * C
        cols += list(range(P - 1, P + C - 1))
        targets += c.completion_ids
        owner += [b] * C
    rows = torch.tensor(rows, device=device)
    cols = torch.tensor(cols, device=device)
    targets = torch.tensor(targets, device=device)
    owner = torch.tensor(owner, device=device)
    h = hidden[rows, cols]                                # [n_tok, H]
    tok_lp = []
    for s in range(0, h.shape[0], head_chunk):
        logits = head(h[s:s + head_chunk]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp.append(lp.gather(1, targets[s:s + head_chunk, None]).squeeze(1))
    tok_lp = torch.cat(tok_lp)
    out = torch.zeros(len(calls), device=device, dtype=tok_lp.dtype)
    return out.index_add(0, owner, tok_lp), tok_lp, owner


def accumulate_policy_gradient(model, calls, T_norm, pad_id, device, max_tokens):
    """Backward of L = -(1/T_norm) * sum_c w_c * sum_t log pi, microbatched.
    Grads accumulate in .grad; caller zeroes grads before and steps after.
    Returns the scalar loss value (float) and per-call logprob sums (list)."""
    lengths = [len(c.prompt_ids) + len(c.completion_ids) for c in calls]
    total, lp_sums = 0.0, [None] * len(calls)
    for mb in pack_microbatches(lengths, max_tokens):
        sub = [calls[i] for i in mb]
        s, _, _ = completion_logprob_sums(model, sub, pad_id, device)
        w = torch.tensor([c.weight for c in sub], device=device, dtype=s.dtype)
        loss = -(w * s).sum() / T_norm
        loss.backward()
        total += float(loss.detach())
        for j, i in enumerate(mb):
            lp_sums[i] = float(s[j].detach())
    return total, lp_sums


def trainable_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad]


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten().float() if p.grad is not None
                      else torch.zeros(p.numel(), device=p.device) for p in trainable_params(model)])
```

### A.4 `kmatters/analysis/variance_math.py`

Two-level variance decomposition: numpy reference and torch port.

```python
"""Streaming two-level variance decomposition of per-future gradient vectors.

Z_ik = gradient contribution of future k of prefix i (a vector).
V_future = E[ tr Cov(Z | X) ]          (within-prefix)
V_prefix = tr Cov( E[Z | X] )          (between-prefix)
MSE(N, K) = V_prefix / N + V_future / (N K)
"""
from __future__ import annotations

import numpy as np


class VarianceAccumulator:
    """Feed one prefix at a time: add_prefix(list_of_K0_vectors).

    Keeps per-prefix mean vectors (rows of M) to form a Gram matrix for exact
    point estimates and a prefix-level bootstrap. Works with numpy or torch
    arrays via the `xp` shim (only dot/sum are needed).
    """

    def __init__(self):
        self.means = []      # m_i vectors
        self.within = []     # per-prefix unbiased within variance (trace)
        self.K0 = None

    def add_prefix(self, Z_list):
        K0 = len(Z_list)
        if K0 < 2:
            raise ValueError("need K0 >= 2 futures per prefix")
        if self.K0 is None:
            self.K0 = K0
        elif K0 != self.K0:
            raise ValueError("all prefixes must have the same K0")
        S = None
        q = 0.0
        for z in Z_list:
            z = np.asarray(z, dtype=np.float64)
            S = z.copy() if S is None else S + z
            q += float(z @ z)
        self.within.append((q - float(S @ S) / K0) / (K0 - 1))
        self.means.append(S / K0)

    def gram(self):
        M = np.stack(self.means)
        return M @ M.T

    @staticmethod
    def _estimates(G, within, K0, idx=None):
        n = G.shape[0]
        if idx is None:
            idx = np.arange(n)
        Gs = G[np.ix_(idx, idx)]
        W = (np.trace(Gs) - Gs.sum() / len(idx)) / (len(idx) - 1)
        Vf = float(np.mean(np.asarray(within)[idx]))
        Vp = float(W - Vf / K0)
        return Vp, Vf

    def estimates(self):
        return self._estimates(self.gram(), self.within, self.K0)

    def bootstrap(self, B=2000, seed=0):
        G = self.gram()
        n = G.shape[0]
        rng = np.random.default_rng(seed)
        out = np.empty((B, 2))
        for b in range(B):
            out[b] = self._estimates(G, self.within, self.K0, rng.integers(0, n, n))
        return out   # columns: Vp, Vf


def mse_model(Vp, Vf, N, K):
    return Vp / N + Vf / (N * K)


class TorchVarianceAccumulator:
    """Same estimator as VarianceAccumulator for large torch vectors (e.g. 33M LoRA grads).

    Per-prefix sums are accumulated in float64; per-prefix mean vectors are stored
    in float32 on `store_device`; the Gram matrix is built in float64 in column
    chunks so no N0 x P float64 matrix is ever materialized.
    """

    def __init__(self, store_device=None, chunk=1 << 22):
        self.means = []
        self.within = []
        self.K0 = None
        self.store_device = store_device
        self.chunk = chunk

    def add_prefix(self, Z_list):
        import torch
        K0 = len(Z_list)
        if K0 < 2:
            raise ValueError("need K0 >= 2 futures per prefix")
        if self.K0 is None:
            self.K0 = K0
        elif K0 != self.K0:
            raise ValueError("all prefixes must have the same K0")
        S = torch.zeros_like(Z_list[0], dtype=torch.float64)
        q = 0.0
        for z in Z_list:
            z64 = z.to(torch.float64)
            S += z64
            q += float(torch.dot(z64, z64))
        self.within.append((q - float(torch.dot(S, S)) / K0) / (K0 - 1))
        m = (S / K0).to(torch.float32)
        self.means.append(m if self.store_device is None else m.to(self.store_device))

    def gram(self):
        import torch
        n = len(self.means)
        G = torch.zeros((n, n), dtype=torch.float64, device=self.means[0].device)
        P = self.means[0].numel()
        for s in range(0, P, self.chunk):
            blk = torch.stack([m[s:s + self.chunk] for m in self.means]).to(torch.float64)
            G += blk @ blk.T
        return G.cpu().numpy()

    def estimates(self):
        return VarianceAccumulator._estimates(self.gram(), self.within, self.K0)

    def bootstrap(self, B=2000, seed=0):
        G = self.gram()
        n = G.shape[0]
        rng = np.random.default_rng(seed)
        out = np.empty((B, 2))
        for b in range(B):
            out[b] = VarianceAccumulator._estimates(G, self.within, self.K0, rng.integers(0, n, n))
        return out
```

### A.5 `kmatters/analysis/kstar.py`

Linear cost fit, continuous K*, budget-constrained choices, pricing comparison.

```python
"""K* under an additive (average-cost) model vs the measured cost surface."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .variance_math import mse_model


def fit_linear_costs(cells):
    """cells: list of dicts with N, K, T (median seconds). Fits
    T ~= c0 + N*Cp + N*K*Cf with non-negative coefficients."""
    X = np.array([[1.0, c["N"], c["N"] * c["K"]] for c in cells])
    y = np.array([c["T"] for c in cells])
    coef, _ = nnls(X, y)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"c0": coef[0], "Cp": coef[1], "Cf": coef[2],
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "max_rel_resid": float(np.max(np.abs(pred - y) / y))}


def kstar_continuous(Vp, Vf, Cp, Cf):
    if Vp <= 0 or Cf <= 0:
        return float("inf")
    return float(np.sqrt(Vf * Cp / (Vp * Cf)))


def choose_under_budget(cells, Vp, Vf, budget, cost_key="T"):
    """Among cells with cost <= budget, pick min model-MSE. Returns the cell or None."""
    feas = [c for c in cells if c[cost_key] <= budget]
    if not feas:
        return None
    return min(feas, key=lambda c: (mse_model(Vp, Vf, c["N"], c["K"]), c[cost_key]))


def compare_pricing(cells, Vp, Vf, budget, model=None):
    """Model-based choice vs measured-surface choice, both judged on measured T.

    cells: dicts with N, K, T (measured median seconds). Not mutated.
    model: None -> best linear fit with intercept (model b);
           dict {"Cp":..., "Cf":..., "c0": 0.0} -> profile-once additive model (model a).
    Returns the two choices, whether K differs, and regret =
    MSE(model choice) / MSE(measured choice) - 1. If the model's choice is
    infeasible under measured T, regret = +inf and lin_infeasible = True."""
    cells = [dict(c) for c in cells]
    lin = fit_linear_costs(cells) if model is None else {
        "c0": model.get("c0", 0.0), "Cp": model["Cp"], "Cf": model["Cf"],
        "r2": None, "max_rel_resid": None}
    for c in cells:
        c["T_lin"] = lin["c0"] + lin["Cp"] * c["N"] + lin["Cf"] * c["N"] * c["K"]
    pick_lin = choose_under_budget(cells, Vp, Vf, budget, "T_lin")
    pick_emp = choose_under_budget(cells, Vp, Vf, budget, "T")
    out = {"lin": pick_lin, "emp": pick_emp, "fit": lin,
           "kstar_lin_cont": kstar_continuous(Vp, Vf, lin["Cp"], lin["Cf"])}
    if pick_lin is None or pick_emp is None:
        out.update(k_differs=None, regret=None, lin_infeasible=None)
        return out
    infeasible = pick_lin["T"] > budget
    mse_l = mse_model(Vp, Vf, pick_lin["N"], pick_lin["K"])
    mse_e = mse_model(Vp, Vf, pick_emp["N"], pick_emp["K"])
    out.update(k_differs=pick_lin["K"] != pick_emp["K"],
               regret=float("inf") if infeasible else mse_l / mse_e - 1.0,
               lin_infeasible=infeasible)
    return out
```

### A.6 `kmatters/analysis/curves.py`

Normalized AUC, time-to-target with censoring, smoothing.

```python
"""Learning-curve summaries: normalized AUC and time-to-target with censoring."""
from __future__ import annotations

import numpy as np


def normalized_auc(x, y, x_max):
    """Trapezoid AUC of y over [0, x_max] divided by x_max. Requires x[0] == 0
    and x[-1] >= x_max (curve truncated/interpolated at x_max)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x[0] != 0 or x[-1] < x_max:
        raise ValueError("curve must span [0, x_max]")
    yi = np.interp(x_max, x, y)
    keep = x < x_max
    xs = np.append(x[keep], x_max)
    ys = np.append(y[keep], yi)
    return float(np.trapezoid(ys, xs) / x_max)


def time_to_target(x, y, target):
    """First x where the linearly interpolated curve reaches target.
    Returns (value, censored). Censored runs return (x[-1], True)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y[0] >= target:
        return float(x[0]), False
    for i in range(1, len(x)):
        if y[i] >= target:
            f = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1])), False
    return float(x[-1]), True


def smooth(y, window=3):
    """Centered moving average used only for time-to-target (eval noise)."""
    y = np.asarray(y, float)
    if window <= 1:
        return y
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(window) / window, mode="valid")
```

### A.7 `tests/unit/test_reference_math.py`

Unit tests for A.1, A.2, A.4, A.5, A.6 (T1.3, T1.6, T1.7, T1.8, T1.9).

```python
from collections import Counter

import numpy as np
import pytest

from kmatters.env.ledger import LedgerState, brute_force_answer, make_future, make_prefix
from kmatters.rl.estimator import advantages, call_weights
from kmatters.analysis.variance_math import VarianceAccumulator, mse_model
from kmatters.analysis.kstar import compare_pricing, fit_linear_costs
from kmatters.analysis.curves import normalized_auc, smooth, time_to_target


# ---------------------------------------------------------------- ledger ----
@pytest.mark.parametrize("level", [1, 3, 5])
def test_answers_match_text_reparse(level):
    for j in range(300):
        p = make_prefix("train", (7, 0, j), L=8, level=level)
        for k in range(3):
            f = make_future("train", p, k, U=2, level=level)
            assert brute_force_answer(p, f, level) == f.gold


def test_determinism_and_independence():
    p1 = make_prefix("train", (1, 2, 3), L=6, level=3)
    p2 = make_prefix("train", (1, 2, 3), L=6, level=3)
    assert p1.chunks == p2.chunks and p1.state_json == p2.state_json
    fa = make_future("train", p1, 0, U=2, level=3)
    fb = make_future("train", p1, 0, U=2, level=3)
    fc = make_future("train", p1, 1, U=2, level=3)
    assert fa.chunks == fb.chunks and fa.q == fb.q
    assert (fa.chunks, fa.q) != (fc.chunks, fc.q)
    # different split -> different stream
    pe = make_prefix("eval", (1, 2, 3), L=6, level=3)
    assert pe.chunks != p1.chunks


def test_prefix_does_not_depend_on_future_generation():
    a = make_prefix("train", (9, 9, 9), L=5, level=3)
    for k in range(5):
        make_future("train", a, k, U=2, level=3)
    b = make_prefix("train", (9, 9, 9), L=5, level=3)
    assert a.chunks == b.chunks and a.state_json == b.state_json


def test_state_roundtrip():
    p = make_prefix("train", (3, 1, 4), L=10, level=4)
    st = LedgerState.from_json(p.state_json)
    assert st.to_json() == p.state_json


def test_answers_not_guessable():
    golds = []
    for j in range(2000):
        p = make_prefix("calib", (0, j), L=16, level=3)
        golds.append(make_future("calib", p, 0, U=2, level=3).gold)
    top = Counter(golds).most_common(1)[0][1] / len(golds)
    assert top < 0.10, top


def test_futures_touch_prefix_transactions():
    hits = 0
    for j in range(500):
        p = make_prefix("calib", (1, j), L=16, level=3)
        f = make_future("calib", p, 0, U=2, level=3)
        pre = set(LedgerState.from_json(p.state_json).prefix_ids)
        text = "\n".join(f.chunks)
        hits += any(f"T{t:02d} was cancelled" in text or f"amount of T{t:02d}" in text
                    for t in pre)
    assert hits / 500 > 0.8


# ------------------------------------------------------------- estimator ----
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _toy_truth(th, ph, P):
    pa, pb = _softmax(th), _softmax(ph)
    J = lambda pa, pb: sum(pa[a] * 0.5 * sum(pb[b] * P[a, b, y] for b in range(2))
                           for a in range(2) for y in range(2))
    eps = 1e-6
    gth = np.array([(J(_softmax(th + eps * np.eye(2)[i]), pb) -
                     J(_softmax(th - eps * np.eye(2)[i]), pb)) / (2 * eps) for i in range(2)])
    gph = np.array([(J(pa, _softmax(ph + eps * np.eye(2)[i])) -
                     J(pa, _softmax(ph - eps * np.eye(2)[i]))) / (2 * eps) for i in range(2)])
    return np.concatenate([gth, gph])


def _toy_estimate(rng, th, ph, P, N, K, baseline, reps):
    pa, pb = _softmax(th), _softmax(ph)
    a = rng.choice(2, size=(reps, N), p=pa)
    y = rng.integers(0, 2, size=(reps, N, K))
    b = rng.choice(2, size=(reps, N, K), p=pb)
    r = (rng.random((reps, N, K)) < P[a[..., None], b, y]).astype(float)
    out = np.zeros((reps, 4))
    for t in range(reps):
        wp, wf = call_weights(r[t], baseline)
        sa = np.eye(2)[a[t]] - pa            # grad log softmax
        sb = np.eye(2)[b[t]] - pb
        out[t, :2] = (wp[:, None] * sa).sum(0)
        out[t, 2:] = (wf[..., None] * sb).sum((0, 1))
    return out


@pytest.mark.parametrize("N,K", [(2, 1), (4, 2), (3, 5), (8, 8)])
@pytest.mark.parametrize("baseline", ["loo_prefix", "tree"])
def test_estimator_unbiased(N, K, baseline):
    rng = np.random.default_rng(0)
    th, ph = np.array([0.3, -0.2]), np.array([-0.4, 0.5])
    P = rng.uniform(0.05, 0.95, size=(2, 2, 2))
    truth = _toy_truth(th, ph, P)
    est = _toy_estimate(rng, th, ph, P, N, K, baseline, reps=40000)
    se = est.std(0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(0) - truth) < 4 * se + 1e-4), (est.mean(0), truth, se)


def test_weights_sum_rules():
    rng = np.random.default_rng(1)
    for N, K in [(4, 16), (64, 1), (16, 4)]:
        r = rng.integers(0, 2, size=(N, K)).astype(float)
        wp, wf = call_weights(r)
        A_pre, A_fut = advantages(r)
        # futures of a prefix carry total weight A-average / N, like one prefix
        assert np.allclose(wf.sum(1), A_fut.mean(1) / N)
        assert np.allclose(wp, A_pre / N)


def test_fixed_baseline_variance_formula():
    """With a constant baseline, tr Cov(g_hat) == Vp/N + Vf/(NK) exactly."""
    rng = np.random.default_rng(2)
    th, ph = np.array([0.1, 0.0]), np.array([0.0, 0.2])
    pa, pb = _softmax(th), _softmax(ph)
    P = rng.uniform(0.05, 0.95, size=(2, 2, 2))
    bconst = 0.4
    # population Z = (r - b) (s_a, s_b); compute Vp, Vf by enumeration
    Zs, probs, groups = [], [], []
    for a in range(2):
        for y in range(2):
            for b in range(2):
                for rr in (0, 1):
                    pr = pa[a] * 0.5 * pb[b] * (P[a, b, y] if rr else 1 - P[a, b, y])
                    Zs.append((rr - bconst) * np.concatenate([np.eye(2)[a] - pa, np.eye(2)[b] - pb]))
                    probs.append(pr)
                    groups.append(a)
    Zs, probs, groups = np.array(Zs), np.array(probs), np.array(groups)
    mu = probs @ Zs
    cond_means = {a: (probs[groups == a] @ Zs[groups == a]) / probs[groups == a].sum() for a in range(2)}
    Vp = sum(pa[a] * np.sum((cond_means[a] - mu) ** 2) for a in range(2))
    Vf = sum(probs[i] * np.sum((Zs[i] - cond_means[groups[i]]) ** 2) for i in range(len(Zs)))
    for N, K in [(2, 1), (4, 4), (8, 2)]:
        reps = 60000
        a = rng.choice(2, size=(reps, N), p=pa)
        y = rng.integers(0, 2, size=(reps, N, K))
        b = rng.choice(2, size=(reps, N, K), p=pb)
        r = (rng.random((reps, N, K)) < P[a[..., None], b, y]).astype(float)
        sa = np.eye(2)[a] - pa
        sb = np.eye(2)[b] - pb
        g = np.concatenate([((r.mean(2) - bconst)[..., None] * sa).mean(1),
                            (((r - bconst)[..., None] * sb).mean(2)).mean(1)], axis=1)
        emp = np.sum(g.var(0))
        pred = mse_model(Vp, Vf, N, K)
        assert abs(emp / pred - 1) < 0.03, (N, K, emp, pred)


# -------------------------------------------------------------- variance ----
def test_variance_accumulator_recovers_components():
    rng = np.random.default_rng(3)
    d, N0, K0 = 50, 64, 8
    Au = rng.normal(size=(d, d)) * 0.1
    Ae = rng.normal(size=(d, d)) * 0.2
    Vp_true = np.trace(Au @ Au.T)
    Vf_true = np.trace(Ae @ Ae.T)
    est = []
    for trial in range(200):
        acc = VarianceAccumulator()
        mu = np.ones(d)
        for i in range(N0):
            u = Au @ rng.normal(size=d)
            acc.add_prefix([mu + u + Ae @ rng.normal(size=d) for _ in range(K0)])
        est.append(acc.estimates())
    est = np.array(est)
    assert abs(est[:, 0].mean() / Vp_true - 1) < 0.05
    assert abs(est[:, 1].mean() / Vf_true - 1) < 0.02
    bs = acc.bootstrap(B=500)
    assert bs.shape == (500, 2) and np.all(np.isfinite(bs))


# ----------------------------------------------------------------- kstar ----
def _grid():
    return [(N, K) for N in (4, 8, 16, 32, 64, 128) for K in (1, 2, 4, 8, 16) if N * K <= 512]


def test_linear_surface_choices_agree():
    cells = [dict(N=N, K=K, T=5 + 2.0 * N + 0.25 * N * K) for N, K in _grid()]
    fit = fit_linear_costs(cells)
    assert fit["r2"] > 0.999
    res = compare_pricing(cells, Vp=1.0, Vf=4.0, budget=100)
    assert res["k_differs"] is False and abs(res["regret"]) < 1e-9


def test_saturating_surface_moves_choice():
    # rollout futures nearly free until N*K reaches 256 (batch saturation)
    def T(N, K):
        return 5 + 2.0 * N + 0.02 * N * K + 0.5 * max(0, N * K - 256)
    cells = [dict(N=N, K=K, T=T(N, K)) for N, K in _grid()]
    res = compare_pricing(cells, Vp=1.0, Vf=8.0, budget=70)
    assert res["emp"] is not None and res["lin"] is not None
    assert res["regret"] >= 0
    assert res["k_differs"] is True        # (16, 8) vs (16, 16) for this surface
    assert "T_lin" not in cells[0]          # inputs not mutated


def test_profile_once_model_a():
    # a profile-once additive model with no intercept misprices the fixed 30 s overhead
    def T(N, K):
        return 30 + 1.0 * N + 0.25 * N * K
    cells = [dict(N=N, K=K, T=T(N, K)) for N, K in _grid()]
    ref = dict(N=32, K=4)
    Cp = (1.0 * ref["N"] + 30 * 0.5) / ref["N"]          # half of the overhead lands on prefixes
    Cf = (0.25 * ref["N"] * ref["K"] + 30 * 0.5) / (ref["N"] * ref["K"])
    res = compare_pricing(cells, Vp=1.0, Vf=6.0, budget=T(32, 1), model={"Cp": Cp, "Cf": Cf})
    assert res["regret"] >= 0
    res_b = compare_pricing(cells, Vp=1.0, Vf=6.0, budget=T(32, 1))
    assert res_b["fit"]["r2"] > 0.999 and res_b["regret"] == 0


# ---------------------------------------------------------------- curves ----
from kmatters.analysis.curves import normalized_auc, smooth, time_to_target


def test_curves():
    x = [0, 10, 20, 30]
    y = [0.2, 0.3, 0.5, 0.5]
    assert abs(normalized_auc(x, y, 30) - (0.25 * 10 + 0.4 * 10 + 0.5 * 10) / 30) < 1e-12
    assert abs(normalized_auc(x, y, 25) - (0.25 * 10 + 0.4 * 10 + 0.5 * 5) / 25) < 1e-12
    assert time_to_target(x, y, 0.4) == (15.0, False)
    assert time_to_target(x, y, 0.6) == (30.0, True)
    assert np.allclose(smooth([1, 2, 3], 3), [4 / 3, 2, 8 / 3])


def test_torch_accumulator_matches_numpy():
    torch = pytest.importorskip("torch")
    from kmatters.analysis.variance_math import TorchVarianceAccumulator
    rng = np.random.default_rng(4)
    a_np, a_t = VarianceAccumulator(), TorchVarianceAccumulator(chunk=37)
    for i in range(12):
        Z = [rng.normal(size=301) + (i % 3) for _ in range(5)]
        a_np.add_prefix(Z)
        a_t.add_prefix([torch.tensor(z, dtype=torch.float32) for z in Z])
    e_np, e_t = np.array(a_np.estimates()), np.array(a_t.estimates())
    assert np.allclose(e_np, e_t, rtol=1e-6, atol=1e-9), (e_np, e_t)
    assert np.allclose(a_np.bootstrap(B=50, seed=1), a_t.bootstrap(B=50, seed=1), rtol=1e-6, atol=1e-9)
```

### A.8 `tests/unit/test_logprobs_cpu.py`

Unit tests for A.3 on a tiny random Qwen3 + LoRA (T1.10).

```python
"""CPU tests of the loss code on a tiny randomly initialized Qwen3 + LoRA."""
import pytest
import torch

from kmatters.rl.logprobs import (Call, accumulate_policy_gradient, completion_logprob_sums,
                                  flat_grad, pack_microbatches)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=500, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    base = Qwen3ForCausalLM(cfg).float()
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, lcfg)
    # make LoRA B non-zero so the adapter matters
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    return m


def _calls(seed, n=13):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        P = int(torch.randint(3, 25, (1,), generator=g))
        C = int(torch.randint(1, 12, (1,), generator=g))
        out.append(Call(torch.randint(1, 500, (P,), generator=g).tolist(),
                        torch.randint(1, 500, (C,), generator=g).tolist(),
                        float(torch.randn(1, generator=g))))
    return out


def _naive_sums(model, calls):
    res = []
    for c in calls:
        ids = torch.tensor([c.prompt_ids + c.completion_ids])
        logits = model(input_ids=ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        P = len(c.prompt_ids)
        res.append(sum(lp[P - 1 + t, tok] for t, tok in enumerate(c.completion_ids)))
    return torch.stack(res)


def test_logprob_sums_match_naive(tiny):
    calls = _calls(1)
    s, _, _ = completion_logprob_sums(tiny, calls, pad_id=0, device="cpu", head_chunk=7)
    ref = _naive_sums(tiny, calls)
    assert torch.allclose(s, ref, atol=1e-4), (s - ref).abs().max()


def test_microbatch_partition_invariance(tiny):
    calls = _calls(2, n=17)
    grads = []
    for max_tokens in (10_000, 60, 25):
        tiny.zero_grad(set_to_none=True)
        accumulate_policy_gradient(tiny, calls, T_norm=64.0, pad_id=0,
                                   device="cpu", max_tokens=max_tokens)
        grads.append(flat_grad(tiny).clone())
    for g in grads[1:]:
        rel = (g - grads[0]).norm() / grads[0].norm()
        assert rel < 1e-5, rel


def test_gradient_matches_naive_objective(tiny):
    calls = _calls(3, n=9)
    tiny.zero_grad(set_to_none=True)
    accumulate_policy_gradient(tiny, calls, T_norm=10.0, pad_id=0, device="cpu", max_tokens=40)
    g1 = flat_grad(tiny).clone()
    tiny.zero_grad(set_to_none=True)
    w = torch.tensor([c.weight for c in calls])
    (-(w * _naive_sums(tiny, calls)).sum() / 10.0).backward()
    g2 = flat_grad(tiny).clone()
    assert (g1 - g2).norm() / g2.norm() < 1e-4


def test_only_lora_trainable_and_dropout_zero(tiny):
    names = [n for n, p in tiny.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    for mod in tiny.modules():
        if hasattr(mod, "lora_dropout"):
            for d in mod.lora_dropout.values():
                assert isinstance(d, torch.nn.Identity)


def test_packing_respects_budget():
    lens = [5, 50, 7, 30, 12, 12, 3, 80]
    mbs = pack_microbatches(lens, 60)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(lens)))
    for mb in mbs:
        assert len(mb) == 1 or max(lens[i] for i in mb) * len(mb) <= 60


def test_sign_of_update(tiny):
    """A positive weight must increase the call's log-prob after one SGD step; negative must decrease."""
    import copy
    for sign in (+1.0, -1.0):
        m = copy.deepcopy(tiny)
        c = _calls(5, n=1)[0]
        c.weight = sign
        before = float(completion_logprob_sums(m, [c], 0, "cpu")[0][0].detach())
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        opt.zero_grad()
        accumulate_policy_gradient(m, [c], T_norm=1.0, pad_id=0, device="cpu", max_tokens=10_000)
        opt.step()
        after = float(completion_logprob_sums(m, [c], 0, "cpu")[0][0].detach())
        assert (after - before) * sign > 0, (sign, before, after)
```

### A.9 `pyproject.toml`

Minimal pyproject (extend with dependencies as needed; keep the markers).

```toml
[project]
name = "kmatters"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["kmatters*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: needs one H100 and the local model",
  "slow: long-running",
  "stats: statistical validity tests (GPU, minutes)",
]
```
