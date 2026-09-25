import hashlib


def seed_from(*keys) -> int:
    """Stable across processes and machines (unlike hash()). 63-bit non-negative."""
    h = hashlib.blake2b(repr(tuple(keys)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)
