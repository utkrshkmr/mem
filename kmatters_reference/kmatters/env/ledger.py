"""Ledger environment: histories of transactions with cancels/corrections.

A history ("prefix") is L chunks of E events. A future is U more chunks
followed by one question. The writer sees chunks; the reader sees only the
final memory and the question. Ground truth comes from the exact state.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

import numpy as np

NAMES = [
    "Alice", "Bob", "Chen", "Dana", "Emeka", "Farah", "Goran", "Hana", "Ivan",
    "Jia", "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa",
    "Sven", "Tara", "Umar", "Vera", "Wen", "Ximena", "Yusuf", "Zoe", "Arjun",
    "Bea", "Cyrus", "Dmitri", "Elif", "Femi", "Gita", "Hugo", "Ines", "Jonas",
    "Kira", "Luis", "Mira", "Nils",
]
CATEGORIES = [
    "groceries", "rent", "travel", "dining", "utilities", "books", "fuel",
    "gifts", "music", "hardware",
]
NOISE_TEMPLATES = [
    "Note: {p} changed their phone number.",
    "{p} said the weather was {adj} today.",
    "Reminder: the team meeting moved to {day}.",
    "{p} ran {n} km this morning.",
    "{p} is reading a book with {n} chapters.",
    "{p} adopted a cat named {q}.",
    "The office printer on floor {n} is broken again.",
    "{p} and {q} watched a movie together.",
    "{p} bought {n} stamps at the post office (paid in cash, not part of this ledger).",
    "Room {n} is booked for {day}.",
    "{p} planted {n} tomato seedlings.",
    "{p} says hello to {q}.",
]
ADJ = ["sunny", "rainy", "windy", "cold", "humid", "pleasant"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Difficulty ladder. Level 3 is the default; calibration picks the level.
LEVELS = {
    1: dict(n_people=3, n_cats=2, events_per_chunk=3, amount_max=20,
            p_txn=0.70, p_cancel=0.05, p_amend=0.05, p_noise=0.20),
    2: dict(n_people=4, n_cats=3, events_per_chunk=4, amount_max=50,
            p_txn=0.64, p_cancel=0.08, p_amend=0.08, p_noise=0.20),
    3: dict(n_people=5, n_cats=4, events_per_chunk=4, amount_max=99,
            p_txn=0.60, p_cancel=0.10, p_amend=0.10, p_noise=0.20),
    4: dict(n_people=6, n_cats=5, events_per_chunk=5, amount_max=99,
            p_txn=0.56, p_cancel=0.12, p_amend=0.12, p_noise=0.20),
    5: dict(n_people=8, n_cats=6, events_per_chunk=6, amount_max=99,
            p_txn=0.50, p_cancel=0.15, p_amend=0.15, p_noise=0.20),
}
FUTURE_MIX = dict(p_txn=0.40, p_cancel=0.25, p_amend=0.25, p_noise=0.10)
QUESTION_MIX = {"net": 0.30, "paid": 0.20, "cat": 0.25, "amount": 0.25}
P_TARGET_PREFIX = 0.8   # future cancels/amends target prefix txns
P_AMOUNT_PREFIX = 0.7   # 'amount' questions target prefix txns

SPLIT_IDS = {"train": 0, "eval": 1, "variance": 2, "profile": 3, "calib": 4}


@dataclass
class Txn:
    tid: int
    payer: str
    payee: str
    amount: int
    cat: str


@dataclass
class LedgerState:
    people: list
    cats: list
    live: dict = field(default_factory=dict)   # tid -> Txn
    next_id: int = 1
    prefix_ids: list = field(default_factory=list)  # tids created in prefix

    def to_json(self) -> str:
        d = asdict(self)
        d["live"] = {str(k): asdict(v) for k, v in self.live.items()}
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "LedgerState":
        d = json.loads(s)
        live = {int(k): Txn(**v) for k, v in d["live"].items()}
        return LedgerState(d["people"], d["cats"], live, d["next_id"],
                           d["prefix_ids"])


def rng_for(split: str, *keys: int) -> np.random.Generator:
    """Deterministic, independent stream per (split, keys)."""
    return np.random.default_rng([SPLIT_IDS[split], *[int(k) for k in keys]])


def _pick_event_type(rng, mix) -> str:
    kinds = ["txn", "cancel", "amend", "noise"]
    p = np.array([mix["p_txn"], mix["p_cancel"], mix["p_amend"], mix["p_noise"]])
    return kinds[int(rng.choice(4, p=p / p.sum()))]


def _target(rng, st: LedgerState, prefer_prefix: bool) -> int | None:
    if not st.live:
        return None
    live = sorted(st.live)
    if prefer_prefix:
        pref = [t for t in live if t in set(st.prefix_ids)]
        if pref and rng.random() < P_TARGET_PREFIX:
            return int(rng.choice(pref))
    return int(rng.choice(live))


def _noise(rng, st: LedgerState) -> str:
    t = NOISE_TEMPLATES[int(rng.integers(len(NOISE_TEMPLATES)))]
    p, q = rng.choice(st.people, 2, replace=False)
    return t.format(p=p, q=q, adj=ADJ[int(rng.integers(len(ADJ)))],
                    day=DAYS[int(rng.integers(len(DAYS)))],
                    n=int(rng.integers(2, 60)))


def gen_event(rng, st: LedgerState, mix: dict, amount_max: int,
              in_future: bool) -> tuple:
    kind = _pick_event_type(rng, mix)
    if kind in ("cancel", "amend") and not st.live:
        kind = "txn"
    if kind == "txn":
        payer, payee = rng.choice(st.people, 2, replace=False)
        tx = Txn(st.next_id, str(payer), str(payee),
                 int(rng.integers(1, amount_max + 1)),
                 str(st.cats[int(rng.integers(len(st.cats)))]))
        return ("txn", tx)
    if kind == "cancel":
        return ("cancel", _target(rng, st, in_future))
    if kind == "amend":
        tid = _target(rng, st, in_future)
        old = st.live[tid].amount
        new = old
        while new == old:
            new = int(rng.integers(1, amount_max + 1))
        return ("amend", tid, new)
    return ("noise", _noise(rng, st))


def apply_event(st: LedgerState, ev: tuple, in_prefix: bool) -> None:
    if ev[0] == "txn":
        tx = ev[1]
        st.live[tx.tid] = copy.copy(tx)
        st.next_id = tx.tid + 1
        if in_prefix:
            st.prefix_ids.append(tx.tid)
    elif ev[0] == "cancel":
        del st.live[ev[1]]
    elif ev[0] == "amend":
        st.live[ev[1]].amount = ev[2]


def render_event(ev: tuple) -> str:
    if ev[0] == "txn":
        t = ev[1]
        return f"T{t.tid:02d}: {t.payer} paid {t.payee} ${t.amount} for {t.cat}."
    if ev[0] == "cancel":
        return f"T{ev[1]:02d} was cancelled."
    if ev[0] == "amend":
        return f"Correction: the amount of T{ev[1]:02d} should be ${ev[2]}."
    return ev[1]


def answer(st: LedgerState, q: tuple) -> int:
    kind, arg = q
    tx = list(st.live.values())
    if kind == "net":
        return sum(t.amount for t in tx if t.payee == arg) - \
            sum(t.amount for t in tx if t.payer == arg)
    if kind == "paid":
        return sum(t.amount for t in tx if t.payer == arg)
    if kind == "cat":
        return sum(t.amount for t in tx if t.cat == arg)
    if kind == "amount":
        return st.live[arg].amount
    raise ValueError(kind)


def render_question(q: tuple) -> str:
    kind, arg = q
    if kind == "net":
        return (f"What is {arg}'s net balance (total received minus total paid) "
                f"over all transactions that are not cancelled?")
    if kind == "paid":
        return f"How much has {arg} paid in total over all transactions that are not cancelled?"
    if kind == "cat":
        return f"What is the total amount of all non-cancelled transactions in the category '{arg}'?"
    if kind == "amount":
        return f"What is the current amount of transaction T{arg:02d}?"
    raise ValueError(kind)


def gen_question(rng, st: LedgerState) -> tuple:
    kinds = list(QUESTION_MIX)
    p = np.array([QUESTION_MIX[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
    if kind == "amount" and not st.live:
        kind = "net"
    if kind in ("net", "paid"):
        return (kind, str(st.people[int(rng.integers(len(st.people)))]))
    if kind == "cat":
        return (kind, str(st.cats[int(rng.integers(len(st.cats)))]))
    live = sorted(st.live)
    pref = [t for t in live if t in set(st.prefix_ids)]
    if pref and rng.random() < P_AMOUNT_PREFIX:
        return ("amount", int(rng.choice(pref)))
    return ("amount", int(rng.choice(live)))


@dataclass
class Prefix:
    key: tuple
    chunks: list          # list[str], length L
    state_json: str       # LedgerState after the prefix


@dataclass
class Future:
    key: tuple
    chunks: list          # list[str], length U
    question: str
    q: tuple
    gold: int


def make_prefix(split: str, key: tuple, L: int, level: int) -> Prefix:
    cfg = LEVELS[level]
    rng = rng_for(split, 0, *key)
    people = [str(x) for x in rng.choice(NAMES, cfg["n_people"], replace=False)]
    cats = [str(x) for x in rng.choice(CATEGORIES, cfg["n_cats"], replace=False)]
    st = LedgerState(people, cats)
    chunks = []
    for _ in range(L):
        lines = []
        for _ in range(cfg["events_per_chunk"]):
            ev = gen_event(rng, st, cfg, cfg["amount_max"], in_future=False)
            apply_event(st, ev, in_prefix=True)
            lines.append(render_event(ev))
        chunks.append("\n".join(lines))
    return Prefix(key, chunks, st.to_json())


def make_future(split: str, prefix: Prefix, k: int, U: int, level: int) -> Future:
    cfg = LEVELS[level]
    rng = rng_for(split, 1, *prefix.key, k)
    st = LedgerState.from_json(prefix.state_json)
    chunks = []
    for _ in range(U):
        lines = []
        for _ in range(cfg["events_per_chunk"]):
            ev = gen_event(rng, st, FUTURE_MIX, cfg["amount_max"], in_future=True)
            apply_event(st, ev, in_prefix=False)
            lines.append(render_event(ev))
        chunks.append("\n".join(lines))
    q = gen_question(rng, st)
    return Future((*prefix.key, k), chunks, render_question(q), q, answer(st, q))


# ---------------- reference re-implementation used only by tests -----------
def brute_force_answer(prefix: Prefix, fut: Future, level: int) -> int:
    """Recompute the answer by re-parsing the rendered text only."""
    import re
    live = {}
    for line in "\n".join(prefix.chunks + fut.chunks).splitlines():
        m = re.match(r"T(\d+): (\w+) paid (\w+) \$(\d+) for (\w+)\.$", line)
        if m:
            live[int(m[1])] = [m[2], m[3], int(m[4]), m[5]]
            continue
        m = re.match(r"T(\d+) was cancelled\.$", line)
        if m:
            del live[int(m[1])]
            continue
        m = re.match(r"Correction: the amount of T(\d+) should be \$(\d+)\.$", line)
        if m:
            live[int(m[1])][2] = int(m[2])
    kind, arg = fut.q
    if kind == "net":
        return sum(a for p, r, a, c in live.values() if r == arg) - \
            sum(a for p, r, a, c in live.values() if p == arg)
    if kind == "paid":
        return sum(a for p, r, a, c in live.values() if p == arg)
    if kind == "cat":
        return sum(a for p, r, a, c in live.values() if c == arg)
    return live[arg][2]
