"""Dataset versions implemented in the production package.

`ledger_core_v1` is the primary family: create, correction, and cancel of nonnegative integers. Gold is checked by `memrl.data.private.interpreter` and, separately, by `memrl.data.private.checker`. The two implementations do not call each other.

`ledger_expiry_v1` and `ledger_alias_v1` are additional versioned interpreters. They are not a silent extension of the core family. A tick deactivates entries whose expiry is at or before the tick time. Alias creation rejects a missing target and a cycle, and it does not mutate state in those cases.

Public chunk files do not contain split names, gold answers, or future event lines. Futures are sampled from the prefix state with the mixture (recent 0.4, old 0.2, uniform 0.4) and stored only in the private record. CPU fixtures use the codepoint counter and are not reference-tokenizer budgets.

Natural chronological annotation is not included. LongMemEval and LoCoMo stay locked until a protocol freeze.
