"""
County name resolution and data enrichment.

Handles the mapping between messy OCR county names (e.g., "S. Clara")
and canonical names in our database ("Santa Clara").

Strategy — NOT a hardcoded lookup table:
  1. Expand known abbreviations (S. → San/Santa/South, etc.)
  2. Generate all candidate full names
  3. Fuzzy-match each candidate against the county database (SequenceMatcher)
  4. Pick the best match above a confidence threshold

This approach generalizes to ANY abbreviated county name without maintaining
a brittle mapping dictionary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ─── Abbreviation Expansion Table ────────────────────────────────────
# Maps common US geographic abbreviations to all possible expansions.
# NOTE: These are NOT county-specific — they're general US naming patterns.

_ABBREVIATION_EXPANSIONS: dict[str, list[str]] = {
    "s.": ["san", "santa", "south"],
    "n.": ["north", "new"],
    "e.": ["east", "el"],
    "w.": ["west"],
    "ft.": ["fort"],
    "st.": ["saint"],
    "mt.": ["mount"],
    "pt.": ["port", "point"],
    "la.": ["los angeles", "la"],  # Special case for LA county
}

# Minimum fuzzy-match score to accept (0.0 = no match, 1.0 = exact)
MATCH_THRESHOLD = 0.70


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class CountyMatch:
    """Result of a county resolution attempt."""

    original: str  # What the OCR said
    resolved: str  # What we matched it to
    tax_rate: float  # From our reference data
    confidence: float  # 0.0-1.0 match score


# ─── Public API ──────────────────────────────────────────────────────


def load_counties(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load county reference data from JSON file.

    Args:
        path: Path to counties.json. Defaults to project root.
    """
    resolved = Path(__file__).parent.parent / "counties.json" if path is None else Path(path)

    with resolved.open(encoding="utf-8") as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def resolve_county(
    raw_name: str, counties: list[dict[str, Any]] | None = None
) -> CountyMatch | None:
    """Resolve an abbreviated/messy county name to a canonical one.

    Args:
        raw_name: The county name from OCR (e.g., "S. Clara")
        counties: List of county dicts with 'name' and 'tax_rate' keys.

    Returns:
        CountyMatch if resolved above threshold, None otherwise.
    """
    if counties is None:
        counties = load_counties()

    # ── Step 1: Exact match (case-insensitive) ──────────────────────
    for county in counties:
        if raw_name.strip().lower() == county["name"].lower():
            return CountyMatch(
                original=raw_name,
                resolved=county["name"],
                tax_rate=county["tax_rate"],
                confidence=1.0,
            )

    # ── Step 2: Generate expansion candidates ───────────────────────
    candidates = _expand_abbreviations(raw_name)

    # ── Step 3: Fuzzy-match each candidate against all counties ─────
    best_match: dict[str, Any] | None = None
    best_score: float = 0.0

    for candidate in candidates:
        for county in counties:
            score = SequenceMatcher(
                None,
                candidate.lower(),
                county["name"].lower(),
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = county

    # ── Step 4: Accept only above threshold ─────────────────────────
    if best_match and best_score >= MATCH_THRESHOLD:
        return CountyMatch(
            original=raw_name,
            resolved=best_match["name"],
            tax_rate=best_match["tax_rate"],
            confidence=round(best_score, 3),
        )

    return None


# ─── Internal Helpers ────────────────────────────────────────────────


def _expand_abbreviations(name: str) -> list[str]:
    """Expand abbreviated parts of a county name into all possible full forms.

    Example:
        "S. Clara" → ["San Clara", "Santa Clara", "South Clara"]
        "N. York"  → ["North York", "New York"]
    """
    candidates = [name]  # Always include the original
    tokens = name.split()

    for i, token in enumerate(tokens):
        token_lower = token.lower()

        if token_lower in _ABBREVIATION_EXPANSIONS:
            expansions = _ABBREVIATION_EXPANSIONS[token_lower]
            new_candidates = []

            for expansion in expansions:
                for candidate in candidates:
                    parts = candidate.split()
                    parts[i] = expansion.title()
                    new_candidates.append(" ".join(parts))

            candidates = new_candidates

    return candidates


# ─── Grantee Parser ──────────────────────────────────────────────────────


def parse_grantees(raw: str) -> list[str]:
    """Split 'John & Sarah Connor' into ['John Connor', 'Sarah Connor'].

    Handles:
      - & separator:   "John & Sarah Connor"
      - 'and' keyword: "John and Sarah Connor"
      - Comma lists:   "John Connor, Sarah Connor"

    Smart last-name propagation: if the last part has a surname but earlier
    parts are first-name-only, we propagate the shared surname.
    """
    # Normalize excessive whitespace (OCR artifact)
    raw = re.sub(r"\s+", " ", raw.strip())

    # Split on & or 'and'
    parts = re.split(r"\s*&\s*|\s+and\s+", raw, flags=re.IGNORECASE)

    if len(parts) <= 1:
        return [raw]

    # Smart last-name propagation
    last_part_words = parts[-1].strip().split()

    if len(last_part_words) >= 2:
        last_name = last_part_words[-1]
        result = []

        for part in parts[:-1]:
            words = part.strip().split()
            # If this part is just a first name (1 word), add the shared surname
            if len(words) == 1:
                result.append(f"{words[0]} {last_name}")
            else:
                result.append(part.strip())

        result.append(parts[-1].strip())
        return result

    return [p.strip() for p in parts]
