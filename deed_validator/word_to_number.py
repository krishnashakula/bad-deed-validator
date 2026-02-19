"""
Convert written-out English number words to their numeric value.

THIS IS A CRITICAL FINANCIAL COMPONENT.

We implement it ourselves rather than trusting a third-party library because:
  1. An error here could record a fraudulent amount on the blockchain.
  2. We need full auditability — every line of the converter is under our control.
  3. It's ~60 lines of well-tested code; no magic, no dependencies.

Supported patterns:
    "One Million Two Hundred Fifty Thousand" → 1,250,000
    "Three Hundred Forty Five"               → 345
    "One Hundred"                            → 100
    "Twelve"                                 → 12
    "One Million Two Hundred Thousand Dollars" → 1,200,000  (ignores "Dollars")
"""

from __future__ import annotations

from decimal import Decimal

# ─── Word Lookup Tables ──────────────────────────────────────────────

_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Words to strip from the input (not part of the number itself)
_IGNORE: set[str] = {
    "and",
    "dollars",
    "dollar",
    "cents",
    "cent",
    "only",
    "the",
    "of",
}


# ─── Word Classifier ─────────────────────────────────────────────────


def _classify_and_apply(
    word: str, current: Decimal, result: Decimal, source: str
) -> tuple[Decimal, Decimal]:
    """Classify a single number word and update the running accumulators.

    Returns:
        (new_current, new_result) after processing the word.

    Raises:
        ValueError: If the word is not a recognised number token.
    """
    if word in _ONES:
        return current + _ONES[word], result
    if word in _TENS:
        return current + _TENS[word], result
    if word == "hundred":
        effective = current if current else Decimal(1)
        return effective * 100, result
    if word in _SCALES:
        effective = current if current else Decimal(1)
        return Decimal(0), result + effective * _SCALES[word]
    raise ValueError(f"Unrecognized number word: {word!r} in {source!r}")


# ─── Main Converter ─────────────────────────────────────────────────


def words_to_number(text: str) -> Decimal:
    """Convert English number words to a Decimal value.

    Args:
        text: e.g. "One Million Two Hundred Thousand Dollars"

    Returns:
        Decimal(1200000)

    Raises:
        ValueError: If the text is empty or contains unrecognized words.

    Algorithm:
        We maintain two accumulators:
        - `result`: completed scale groups (e.g., after processing "million")
        - `current`: the number being built in the current scale group

        For each word:
        - ones/teens/tens → add to `current`
        - "hundred"       → multiply `current` by 100
        - scale word      → flush `current * scale` into `result`, reset `current`

        At the end, `result + current` is the final value.
    """
    if not text or not text.strip():
        raise ValueError("Empty text cannot be converted to a number")

    # Normalize: lowercase, strip parentheses/brackets, remove hyphens/commas
    normalized = text.strip().strip("()[]")
    words = normalized.lower().replace("-", " ").replace(",", " ").split()
    words = [w for w in words if w not in _IGNORE]

    if not words:
        raise ValueError(f"No number words found in: {text!r}")

    result = Decimal(0)  # Accumulator for completed scale groups
    current = Decimal(0)  # Number being built in current group

    for word in words:
        current, result = _classify_and_apply(word, current, result, normalized)

    result += current

    # Safety: if we parsed tokens but got 0 and "zero" wasn't in input, something's wrong
    if result == 0 and "zero" not in normalized.lower():
        raise ValueError(f"Could not parse a valid number from: {text!r}")

    return result
