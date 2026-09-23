# Parallel implementation plan

The user intends to replace the current implementation. Build in a **fresh repository or isolated branch** initialized from this kit; preserve the kit and any original data/checkpoints until the new implementation passes its gates. No deletion is necessary to begin. The kit does not depend on any present implementation.

## Worktree setup

If this directory is not already under git, initialize and commit the handoff first. Run these commands only from the new repository, not from an unrelated existing working tree:

```bash
git init
git add .
git commit -m "Add research specification and rebuild contracts"
git worktree add ../memrl-data -b build/data
git worktree add ../memrl-env -b build/environment
git worktree add ../memrl-runtime -b build/runtime
git worktree add ../memrl-learning -b build/learning
git worktree add ../memrl-eval -b build/evaluation
git worktree add ../memrl-analysis -b build/analysis
```

If commit identity is not configured, use the researcher's actual configured identity; do not invent an author. Worktrees isolate file changes, not GPUs or caches. All GPU jobs still use the single shared lease registry. Read-only model weights can be shared; mutable datasets, job queues and checkpoint outputs must use run-specific paths.

## Ownership

| Agent | Owned production paths | Task packet | Prerequisites |
|---|---|---|---|
| 00 Coordinator | `src/memrl/contracts/`, `src/memrl/config.py`, root metadata, schemas, locks, integration tests, CLI router | `prompts/agents/00_COORDINATOR.md` | CPU kit validation |
| 01 Data | `src/memrl/data/`, `tests/data/`, dataset documentation | `prompts/agents/01_DATA.md` | Frozen initial contracts |
| 02 Environment | `src/memrl/memory/`, `src/memrl/env/`, `tests/environment/` | `prompts/agents/02_ENVIRONMENT.md` | Frozen initial contracts |
| 03 Runtime | `src/memrl/runtime/`, `src/memrl/models/`, `tests/runtime/` | `prompts/agents/03_RUNTIME.md` | Contracts; real-model tests await environment |
| 04 Learning | `src/memrl/rl/`, `tests/learning/` | `prompts/agents/04_LEARNING.md` | Contracts; GPU training awaits runtime/environment |
| 05 Evaluation | `src/memrl/eval/`, `src/memrl/baselines/`, `tests/evaluation/` | `prompts/agents/05_EVALUATION.md` | Contracts; real runs await data/runtime |
| 06 Analysis | `src/memrl/analysis/`, `tests/analysis/`, figure/table scripts | `prompts/agents/06_ANALYSIS.md` | Frozen event/score schemas |

The CPU package and existing CPU tests are shared reference artifacts owned by the coordinator. Other agents can add regression proposals but must not change its semantics to fit their implementation. No agent writes another owner's tests or shared dependency files independently.

## Dependency waves

**Wave 0, coordinator only:** validate this kit; implement production types, config validator and JSON fixtures; write one fake-inference fixture explicitly labeled a test fixture; freeze schema3 and assign worktrees. A fake backend is allowed in a test process, never as an automatic fallback during a research run.

**Wave 1, parallel CPU work:** Data implements generator/interpreters/splitter; Environment implements store/tools/chunker integration; Runtime implements queue/lease/adapter scaffolding with mock-independent tests; Learning implements exact loss math and batch compiler; Evaluation implements baseline/scorer contracts; Analysis implements completeness and statistical estimands on labeled fixtures. All use the coordinator's frozen fixture schemas. No production task is “complete” after writing a placeholder.

**Wave 2, integrate CPU pipeline:** coordinator merges Data and Environment first, then Evaluation baseline mechanics, then Runtime scheduling, then Learning batch interfaces and Analysis. Run the scripted complete trajectory and branch-isolation test. Fix integration problems at their owning module. All agents sync to the updated integration commit before real model work.

**Wave 3, one GPU owner:** Runtime runs installed-API and real-model parity probes on a leased device; Environment/Evaluation can debug from captured public fixtures on CPU. Coordinator selects and pins the compatible environment. Learning then exercises one update and two-rank equivalence; Evaluation checks raw-base reader isolation after writer adapter swaps. Do not run independent agents' 7B engines on the same GPU.

**Wave 4, four-GPU pilot:** complete the runtime gate, loss gate and exact resume test; profile the 2+2 topology and optional four-device phase schedule; execute F0–F2 on development worlds. Analysis assesses baseline strength, capacity pressure, sampling variance and costs. Research design adjustments are recorded before test freeze.

**Wave 5, scientific freeze and execution:** coordinator locks hypotheses, matched compute rules, seeds, world counts, checkpoints and test manifests. Scheduler runs the primary replicated studies. Secondary methods, large backbones and optional branch allocation begin only if their dependency and compute gates pass. Evaluation and Analysis may work concurrently with training only when resources are disjoint and the reader jobs cannot contaminate selection.

## Integration requests and completion reports

Every change request supplies old and proposed field/signature, rationale, caller list, migration plan and a fixture demonstrating the issue. Coordinator either approves a compatible extension or increments schema version. Do not resolve a schema mismatch with `dict.get(..., default)` when the field is required.

Each agent returns:

```text
Task and integration commit:
Files changed (owned paths only):
Implemented behavior:
Commands executed and exact outcomes:
Evidence artifacts and hashes:
Boundary/negative cases exercised:
Unimplemented or unverified behavior:
Cross-module requests:
Ready gate / blocked gate:
```

Coordinator reviews concrete code and evidence, then commits or merges. Passing local unit tests is necessary but does not replace the cross-owner fixtures. A final human research review remains necessary before making scientific claims; agent agreement is not empirical evidence.

## Cursor use

Open each worktree in a separate agent session and supply that agent's task packet. Include the same specification commit in every initial prompt. The always-applied project rule points to the shared constraints; it does not contain the whole specification. Use the coordinating prompt to sequence the waves. If your Cursor setup cannot run parallel sessions, perform the same task packets sequentially; scientific semantics do not change.

The kit uses `.cursor/rules/*.mdc` and `AGENTS.md`, as documented in the official [Cursor rules documentation](https://cursor.com/docs/rules). It does not require an installed custom skill, a particular subscription feature, or any undocumented background-agent command.
