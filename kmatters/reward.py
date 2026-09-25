import re

_INT = re.compile(r"-?\d+")


def parse_int(text: str):
    s = text.strip().replace("−", "-").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)      # 1,234 -> 1234
    m = _INT.search(s)
    return int(m.group()) if m else None


def reward(text: str, gold: int) -> float:
    v = parse_int(text)
    return 1.0 if v is not None and v == gold else 0.0
