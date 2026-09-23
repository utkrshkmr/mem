# Protocol deviations

No experimental accuracy runs have occurred. Engineering probes are listed in the resource ledger and are not protocol outcomes.

2026-09-23: vLLM 0.25.1 on the default Python failed during engine startup (`CUDA driver version is insufficient for CUDA runtime version`). The greedy token comparison was repeated in conda env `memrl` with vLLM 0.10.2. Protocol id `continuation_blind_v3` is unchanged. No final scores were visible because no evaluation run exists.


Append: date, old/new protocol, affected runs, discovered issue, whether final outcomes were already visible, corrective action, rerun scope and paper disclosure. Never erase failed or invalidated results.
