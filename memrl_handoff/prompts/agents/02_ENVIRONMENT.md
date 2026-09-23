# Agent 02: bounded store and writer environment

Own src/memrl/memory/, src/memrl/env/ and tests/environment/. Read MEMRL_SPEC Sections9–11, CONTINUATION_METHOD and INTERFACES. The provided memrl_contract store is a small semantic reference; production must use the pinned tokenizer and full tool protocol.

Implement strict allowed JSON actions, entry IDs, add/update/delete/read/finish behavior, canonical whole-store serialization, total/key/value/entry/index bounds, transactional preview/commit and bounded public errors. Charge every persistent model-visible key, value, ID and metadata field. Count the exact complete serialization, not the sum of field token counts. Reject over-limit proposed mutations without changing prior state or recycling committed identifiers. Never keep an uncharged shadow fact table accessible to a policy.

Assemble actual chat-template prompts with exact IDs and budgets. No evidence, question or value truncation. Clear loaded text and last tool result when moving to a new chunk. Enforce three calls/chunk and completion caps. Distinguish invalid model actions from implementation/infrastructure failures. A model error consumes its action and remains in evaluation; an invariant bug stops the run for repair.

Snapshot/clone must preserve charged state byte-for-byte and separate mutable state across branches. Do not copy private gold into a writer state. Raw historical chunks cannot be re-searched. Make all-retained reader context reconstruction a pure function of snapshot bytes and a declared ordering. Its hash/length/visible-channel ledger must match actual inference input.

Tests: exact-capacity boundary and one-token-over, Unicode, malformed/extra JSON fields, oversized keys, long IDs, deletion of absent entry, failed update atomicity, clone isolation, shared prefix immutability after suffix mutation, exhausted calls, empty memory, index overflow and prompt framing. Add a leak canary that changes private metadata while preserving model prompt bytes. Check no free memory survives chunk boundaries.

Acceptance: G03 and cross-owner scripted pipeline. Deliver documented public semantics plus events sufficient to reconstruct every evaluated reader input. Do not implement a retrieval backend until all-retained primary works; any archive reference is separately privileged and labeled.
