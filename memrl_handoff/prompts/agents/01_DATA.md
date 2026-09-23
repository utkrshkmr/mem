# Agent 01: data, worlds and independent gold

Own src/memrl/data/, tests/data/ and dataset documentation. Read MEMRL_SPEC Sections6–9 and30 onward, CONTINUATION_METHOD Sections2/6, and INTERFACES. Use coordinator contracts; request changes rather than editing shared schemas.

Implement four layers: abstract event generation, deterministic state interpretation, public rendering/chunk manifests, and private suffix/question/scoring metadata. Begin with create/correct/cancel semantics and an independently implemented gold checker. Add explicitly versioned rollback/revocation/expiry/alias families only after basic tests pass. Track unknown/inactive targets, duplicate events, integer amounts, equal corrections, cycles, expiry boundaries and source ranges. Semantic changes create new dataset versions.

Generate separate seeded train/dev/test source worlds. All descendants—suffixes, matched variants, paraphrases, ID renamings—inherit the source split. Background source books/documents and renderer families also remain disjoint where specified. Build chunks once using the pinned reference and policy tokenizer constraints. No capacity or arm may rechunk its inputs. Reject infeasible useful-record-count/length pairs with counts, rather than dropping evidence.

For continuations, sample future kernels and fork boundaries before policy collection, independent of current writer actions. Produce K futures shared across g, with hidden Q queries; duplicate-future control repeats event text but does not reuse suffix action RNG. Supply identical manifests to fork and flat arms. The public writer input contains no IDs that expose a split, family, future target, gold answer or file path. Opaque routing IDs stay outside model text.

Implement dataset CLI interfaces for generate, validate, freeze and inspect. Emit public JSONL/chunk files separately from private task records, with immutable hashes, schema version, source lineage, renderer version, operator kernel and split counts. Create a 2-world scripted fixture plus a 32-world CPU pilot; do not label these paper samples. Add property tests for state semantics, alternative supports, label equivariance, permutation of commuting events, no information leak and independent interpreter agreement.

Acceptance: G02; all labels independently checked; source ancestors split-disjoint; exact chunk reconstruction; hidden fields cannot enter WriterObservation; deterministic regeneration. Natural chronological data remain a separately documented source/annotation task, not something to synthesize and call real.
