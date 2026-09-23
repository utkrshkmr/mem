# Decision log

| Date | Decision | Reason | Evidence / limitation |
|---|---|---|---|
| 2026-09-23 | Primary capacity 1,024 for both training and primary test | Remove mismatched training-capacity ambiguity | Planning decision; calibrate on development before freeze |
| 2026-09-23 | Candidate method is F03 versus ordinary F02 continuation augmentation | Close prior work covers generic memory RL and update sufficiency | docs/NOVELTY_AND_VENUE.md; no novelty guarantee |
| 2026-09-23 | Future-only mean reward; current accuracy secondary | Avoid adding an untested weighted objective | docs/CONTINUATION_METHOD.md |
| 2026-09-23 | Use production contracts plus separate CPU reference | Make correct semantics testable without pretending a trainer exists | CPU checks; real GPU validation pending |
| 2026-09-23 | Initial 1,500 allocated GPU-hour planning envelope | Give scheduler a finite default; pilot must measure feasibility | Planning envelope, not a measured runtime |
| 2026-09-23 | Keep decision records in records/ | The handoff already uses this directory; duplicating docs/ would split the log | This file |
| 2026-09-23 | CPU fixtures use an injected codepoint counter | Paper budgets require the pinned model tokenizer, which is a different unit | docs/DATASET.md |
| 2026-09-23 | Record vLLM parity from conda env memrl | Default Python 3.13 vLLM 0.25.1 could not start an engine on driver 570.211.01 | reports/api_probe/vllm_025_failure.json and vllm_hf.json |

Append decisions; do not overwrite history. Protocol-changing decisions name old/new protocol IDs and affected run IDs.
