# Deviations from PLAN.md

Each entry: date, what changed, why. Nothing in `prereg/PREREG.md` changes after Gate G4 except through an entry here.

## 2026-09-25: T0.4 test implementation fix (criterion unchanged)

The first T0.4 run failed on the `LoRARequest(lora_name, lora_int_id, lora_path)` check because the test took the first three parameter names from a `set`, which has no order. The installed vLLM signature is correct. The check now reads the ordered `inspect.signature(...).parameters`. The pass criterion in PLAN.md §9 is unchanged.
