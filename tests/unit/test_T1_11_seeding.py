"""T1.11: seed_from is stable across processes and collision-free on 10^6 keys."""
import subprocess
import sys

from kmatters.seeding import seed_from

KEYS = [(0,), (1, 2, 3), ("torch", 7), (12345, 0, 63, 7), ("profile", 32, 4, 16, 512)]


def test_T1_11_stable_across_processes():
    code = f"from kmatters.seeding import seed_from; print([seed_from(*k) for k in {KEYS!r}])"
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": str(s), "PATH": ""}).stdout.strip() for s in (1, 2)]
    assert outs[0] == outs[1] == str([seed_from(*k) for k in KEYS])


def test_T1_11_no_collisions(record):
    seeds = {seed_from(a, b, c) for a in range(100) for b in range(100) for c in range(100)}
    assert len(seeds) == 10**6
    assert all(0 <= s < 2**63 for s in list(seeds)[:1000])
    record("T1.11", n_keys=10**6, n_unique=len(seeds))
