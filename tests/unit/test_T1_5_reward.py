"""T1.5: integer parsing and reward."""
import pytest

from kmatters.reward import parse_int, reward

CASES = [("42", 42), (" -17\n", -17), ("−17", -17), ("$1,234", 1234), ("The answer is 8.", 8),
         ("8 dollars", 8), ("eight", None), ("", None), ("3.0", 3), ("-0", 0)]


@pytest.mark.parametrize("text,expected", CASES)
def test_T1_5_parse(text, expected):
    assert parse_int(text) == expected


def test_T1_5_reward_values():
    assert reward("42", 42) == 1.0
    assert reward("41", 42) == 0.0
    assert reward("eight", 8) == 0.0
    assert reward("", 0) == 0.0
    assert reward("-0", 0) == 1.0
