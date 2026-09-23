# Acceptance gates

Every gate has a status `not_started`, `in_progress`, `passed`, `failed` or `blocked`, plus code/config/data hashes, command transcript, timestamp and evidence paths. A checked Markdown box without evidence does not pass a gate. Gates are sequential dependencies, not repeated permission requests to the user.

| Gate | Required behavior and evidence | Unlocks |
|---|---|---|
| G00 kit | Supplied CPU tests and validator pass; no fake results or unresolved field accepted in production mode | Contract implementation |
| G01 contracts | Strict public/private schemas; config rejects unknown keys; byte-identical canary prompts; immutable IDs and hashes | Parallel module implementation |
| G02 data | Two independent interpreters agree; prefix/suffix ancestor splits disjoint; hidden continuations never enter prefix prompts; renderer metamorphic checks | Synthetic model pilot |
| G03 memory | Exact tokenizer charging; canonical serialization; atomic rejection; branch isolation; original history inaccessible; all-retained reader reconstruction | Model environment |
| G04 model | Pinned versions and templates; vLLM/HF sampled token/logprob parity at agreed numeric tolerance; reader adapter-free across writer swaps; exact generation budget tests | One-GPU update |
| G05 gradient | K=1 reduction; no prefix duplication; explicit flat versus fork denominators; exhaustive finite-policy check; selected-call expectation; one-rank/two-rank gradients and optimizer update within measured tolerance | Distributed training |
| G06 runtime | Atomic adapter publication and worker acknowledgements; GPU lease exclusivity; missing/duplicate job rejection; fault-injected crash/restart reproduces next declared update; allbranch barrier | Four-GPU pilot |
| G07 pilot | Frozen reader competent with feasible gold support; task not all-zero/all-one; strong summaries and ledger calibrated; capacity binds; GPU-hour projections measured | Core development study |
| G08 research | Novelty matrix refreshed; F02 control implemented; primary endpoint and meaningful-effect threshold chosen on dev; world/seed count informed by variance; resource matching declared | Protocol freeze |
| G09 freeze | Immutable manifests, seeds, prompts, checkpoints/selection rules, config and analysis plan hashes; no test selection lineage; independent review recorded | Locked final tests |
| G10 results | Complete planned rows or accounted failures; all seeds; world-cluster intervals; compute costs; no suppressed failures; natural/transfer scope correctly labeled | Paper drafting |
| G11 artifact | New checkout reproduces tables from released rows; provenance/license audit; claims supported by listed comparisons; limitations and AI assistance disclosed as venue requires | Submission consideration |

## Mandatory negative tests

Reject an unpinned model; changed chat template; writer adapter on reader; corrupt behavior-logprob length; sampled-call probability zero; G=1; K=0; duplicated prefix row; missing suffix; capacity overflow; an uncharged key; a secret answer in public metadata; a changed observation manifest in a matched comparison; a final-benchmark path in the checkpoint selector; and a reader cache hit for the wrong snapshot hash.

Fault injection must cover process death before and after adapter manifest publication, retry after partially written result files, stale worker generation, and one DDP rank receiving zero selected calls. No injected test may alter final scientific metrics or be stored under a real experiment ID.

## Completion distinctions

`implemented` means the code path exists; `CPU_validated` means CPU evidence covers it; `GPU_validated` means a real GPU test passed; `scientifically_validated` means the planned controlled experiment has results. Record these independently. The starter package is CPU reference work. Gates G01–G11 remain open until a coding agent executes their production evidence requirements.
