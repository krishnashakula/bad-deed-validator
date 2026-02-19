"""
Deterministic regex-based extraction from OCR text.

This module extracts deed fields using PURE REGEX — no AI, no guessing.
It serves as the GROUND TRUTH baseline that we compare the LLM output against.

Philosophy: It's better to extract nothing than to extract wrong data.
              Every pattern is conservative; we'd rather return None than a bad value.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from .models import RawDeedExtraction


def extract_with_regex(raw_text: str) -> RawDeedExtraction:
    """Extract deed fields from raw OCR text using regex patterns.

    Args:
        raw_text: The raw OCR-scanned deed text.

    Returns:
        RawDeedExtraction with all fields that could be deterministically extracted.
    """
    return RawDeedExtraction(
        document_id=_extract_doc_id(raw_text),
        county=_extract_county(raw_text),
        state=_extract_state(raw_text),
        date_signed=_extract_date(raw_text, "Signed"),
        date_recorded=_extract_date(raw_text, "Recorded"),
        grantor=_extract_labeled_field(raw_text, "Grantor"),
        grantee=_extract_labeled_field(raw_text, "Grantee"),
        amount_numeric=_extract_amount_numeric(raw_text),
        amount_words=_extract_amount_words(raw_text),
        apn=_extract_labeled_field(raw_text, "APN"),
        status=_extract_labeled_field(raw_text, "Status"),
    )


# ─── Individual Field Extractors ─────────────────────────────────────


def _extract_doc_id(text: str) -> str | None:
    """Match 'Doc: DEED-TRUST-0042' or similar patterns."""
    match = re.search(r"Doc:\s*(\S+)", text)
    return match.group(1).strip() if match else None


def _extract_county(text: str) -> str | None:
    """Match 'County: S. Clara  |  State: CA' — grab text before the pipe."""
    match = re.search(r"County:\s*([^|\n]+)\s*(?:\||$)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_state(text: str) -> str | None:
    """Match 'State: CA' — strictly 2 uppercase letters."""
    match = re.search(r"State:\s*([A-Z]{2})", text)
    return match.group(1).strip() if match else None


def _extract_date(text: str, label: str) -> date | None:
    """Match 'Date Signed: 2024-01-15' or 'Date Recorded: 2024-01-10'.

    Uses strict ISO format parsing — if the date is malformed, we return None
    rather than guessing.
    """
    pattern = rf"Date\s+{label}:\s*(\d{{4}}-\d{{2}}-\d{{2}})"
    match = re.search(pattern, text)
    if match:
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None
    return None


def _extract_labeled_field(text: str, label: str) -> str | None:
    """Generic extractor for 'Label:  value' lines.

    Captures everything after the colon until end-of-line, then normalizes
    excessive whitespace (common in OCR output).
    """
    pattern = rf"{label}:\s*(.+?)$"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        value = match.group(1).strip()
        # Normalize multiple spaces to single (OCR artifact cleanup)
        value = re.sub(r"\s+", " ", value)
        return value
    return None


def _extract_amount_numeric(text: str) -> Decimal | None:
    """Extract the dollar amount in numeric form: $1,250,000.00

    Handles commas, optional decimal places, and the $ prefix.
    Converts to Decimal for exact financial arithmetic (never float!).
    """
    match = re.search(r"\$([0-9,]+(?:\.\d{1,2})?)", text)
    if match:
        try:
            return Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            return None
    return None


def _extract_amount_words(text: str) -> str | None:
    """Extract the written-out amount from parentheses: (One Million ... Dollars)

    Legal documents typically enclose the word-form amount in parentheses
    as a cross-check against the numeric form.  We anchor the pattern to
    parentheses that appear right after a dollar amount so we don't
    accidentally match earlier parenthesized text like '(LLC)'.
    """
    match = re.search(r"\$[0-9,]+(?:\.\d{1,2})?\s*\(([^)]+)\)", text)
    if match:
        return match.group(1).strip()
    return None
