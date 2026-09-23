# Agent 00: contracts and integration

Read AGENTS.md, README.md, MEMRL_SPEC.md revision3 and docs/{INTERFACES,PARALLEL_BUILD,ACCEPTANCE_GATES}.md. You own shared contracts/configuration, schema fixtures, dependency locks, CLI integration, production integration tests and evidence records. Do not independently rewrite an agent-owned subsystem.

Deliver a typed production package skeleton under src/memrl with **working contracts and validators**, not successful-looking placeholder services. Implement strict public/private dataclasses or equivalent validated models, canonical hashes, all required enums and state transitions. Define model/tokenizer/template identity, caller/branch identity, immutable snapshots, token/logprob records, job/result receipts, score rows and lineage. Provide valid/invalid wire fixtures. Reject unexpected fields and wrong types rather than coercing booleans to integers.

Extend the draft configuration resolver into the runtime, with explicit environment-lock merging, real tokenizer checks, test-access gate, budget gate and module registry. Resolve scientific defaults only from the specification and decision records. Add a CLI router that errors if a required module is missing; tests may explicitly select a fake backend, but a production run cannot fall back to one.

Freeze initial schema3, then assign task packets 01–06. Integrate in the documented waves. Verify each owner runs against the same schema commit. Centralize changes to pyproject/locks and export a pinned working environment after real API probes. Keep original reference tests passing; differences between toy reference and production are declared, not hidden.

Run an integration fixture through generator -> renderer -> environment -> scripted writer -> frozen test reader -> score -> branch coefficients -> result rows -> completeness audit. The fixture is labeled synthetic test data and has no paper run ID. Then schedule actual API, token-budget, one-step, DDP and restart tests before a four-GPU pilot. GPU leases and total allocated-GPU-hour budget are your responsibility.

Acceptance: G01 contracts; cross-owner CPU pipeline; production failure gates; honest IMPLEMENTATION_STATUS; verified integration commands. Return the exact commit and hashes used by every agent. No assertion of scientific novelty or completed experiments is allowed without their separate evidence.
