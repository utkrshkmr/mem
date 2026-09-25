# kmatters: does K matter?

A fixed-K memory-writer RL study on one 4×H100 node. `PLAN.md` is the full specification; `reports/STATUS.md` tracks progress.

## Quickstart

```bash
bash scripts/00_check_node.sh          # T0.1, writes reports/node.json
bash scripts/01_setup_env.sh           # .venv; picks install path A/B from the driver
source .venv/bin/activate
python scripts/02_download_model.py    # needs ./hf-tok (chmod 600); writes configs/model_lock.yaml
pytest tests/unit -q                   # Gate G0, CPU
pytest tests/gpu -m gpu -k T0 -q       # Gate G0, GPU
```

Later phases (PLAN.md §10): GPU integration (G1) → calibration (G2) → profiling and variance (G3) → pilot and preregistration (G4) → sweep → trained-checkpoint measurements (G6) → `python scripts/analyze.py --all`, which writes `reports/DECISION.md`.

Long jobs run through the queue inside tmux:

```bash
tmux new -d -s kq "python scripts/run_queue.py jobs/<phase>.jsonl --gpus 0,1,2,3"
```

## Rules that matter

- The Hugging Face token lives only in `hf-tok`; it is never printed, logged or committed.
- `kmatters/env/ledger.py`, `kmatters/rl/{estimator,logprobs}.py`, `kmatters/analysis/{variance_math,kstar,curves}.py` and their tests are Appendix A verbatim; do not edit them.
- One code path for training, evaluation and profiling (`loop.run_iteration`).
