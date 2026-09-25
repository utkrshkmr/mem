"""Perfect-memory renderer for the controls (PLAN.md §8.11).

The exact state after prefix + future is rebuilt from the prefix state and the rendered future
lines (the same line formats `brute_force_answer` re-parses in the tests).
"""
from __future__ import annotations

import re

from .ledger import Future, LedgerState, Prefix, Txn

_TXN = re.compile(r"T(\d+): (\w+) paid (\w+) \$(\d+) for (\w+)\.$")
_CANCEL = re.compile(r"T(\d+) was cancelled\.$")
_AMEND = re.compile(r"Correction: the amount of T(\d+) should be \$(\d+)\.$")


def final_state(prefix: Prefix, fut: Future | None = None) -> LedgerState:
    st = LedgerState.from_json(prefix.state_json)
    for line in ("\n".join(fut.chunks).splitlines() if fut is not None else []):
        if m := _TXN.match(line):
            st.live[int(m[1])] = Txn(int(m[1]), m[2], m[3], int(m[4]), m[5])
        elif m := _CANCEL.match(line):
            del st.live[int(m[1])]
        elif m := _AMEND.match(line):
            st.live[int(m[1])].amount = int(m[2])
    return st


def render_oracle(st: LedgerState) -> str:
    lines = ["Live transactions:"]
    for tid in sorted(st.live):
        t = st.live[tid]
        lines.append(f"T{tid:02d} {t.payer}->{t.payee} {t.amount} {t.cat}")
    lines.append("Per person (paid / received / net):")
    for p in st.people:
        paid = sum(t.amount for t in st.live.values() if t.payer == p)
        recv = sum(t.amount for t in st.live.values() if t.payee == p)
        lines.append(f"{p}: paid {paid}, received {recv}, net {recv - paid}")
    lines.append("Per category totals:")
    for c in st.cats:
        lines.append(f"{c}: {sum(t.amount for t in st.live.values() if t.cat == c)}")
    return "\n".join(lines)


def oracle_memory(prefix: Prefix, fut: Future) -> str:
    return render_oracle(final_state(prefix, fut))
