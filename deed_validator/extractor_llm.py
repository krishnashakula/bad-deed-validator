"""
LLM-based extraction from OCR text using OpenAI structured output.

The LLM is used as a "smart OCR post-processor" — it understands context
and handles messy formatting better than regex. BUT we never trust it blindly.
Every field it returns gets cross-checked by the deterministic validators.

Design:
  - JSON mode enforced (structured output, not free text)
  - GPT-5 uses default temperature; determinism comes from structured output
  - Graceful fallback: no API key → returns None → system uses regex only
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from decimal import Decimal

from .models import RawDeedExtraction

logger = logging.getLogger(__name__)


# ─── System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a legal document data extractor specializing in property deeds.
Parse the given OCR-scanned deed text into a structured JSON object.

CRITICAL RULES:
1. Extract EXACTLY what is written — do NOT correct errors or inconsistencies.
2. If a field looks wrong (e.g., impossible dates), extract it AS-IS.
3. Do not infer or hallucinate values for missing fields.
4. Preserve abbreviations as they appear in the text.

Return a JSON object with these exact keys:
{
    "document_id": "string or null",
    "county": "string exactly as written, or null",
    "state": "2-letter state code or null",
    "date_signed": "YYYY-MM-DD or null",
    "date_recorded": "YYYY-MM-DD or null",
    "grantor": "string or null",
    "grantee": "string or null",
    "amount_numeric": number (no $ sign, no commas) or null,
    "amount_words": "the written-out amount inside parentheses, or null",
    "apn": "string or null",
    "status": "string or null"
}

IMPORTANT: Extract raw data only. Validation happens in a separate step.
"""


def extract_with_llm(raw_text: str) -> RawDeedExtraction | None:
    """Extract deed fields using an LLM (OpenAI GPT-5).

    Returns:
        RawDeedExtraction if LLM succeeds, None if unavailable or fails.
        Failure is NOT an error — the system gracefully falls back to regex-only.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("No OPENAI_API_KEY set — skipping LLM extraction (regex-only mode)")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extract structured data from this OCR-scanned deed:\n\n"
                        f"{raw_text}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            # GPT-5 only supports default temperature (1.0)
            # Determinism is ensured by structured output + explicit system prompt
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content")
            return None
        data = json.loads(content)

        logger.info("LLM extraction succeeded")
        return RawDeedExtraction(
            document_id=data.get("document_id"),
            county=data.get("county"),
            state=data.get("state"),
            date_signed=_safe_date(data.get("date_signed")),
            date_recorded=_safe_date(data.get("date_recorded")),
            grantor=data.get("grantor"),
            grantee=data.get("grantee"),
            amount_numeric=_safe_decimal(data.get("amount_numeric")),
            amount_words=data.get("amount_words"),
            apn=data.get("apn"),
            status=data.get("status"),
        )

    except ImportError:
        logger.warning("openai package not installed — pip install openai")
        return None
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None


# ─── Safe Type Converters ────────────────────────────────────────────


def _safe_date(value: object) -> date | None:
    """Safely convert an LLM output to a date. Returns None on failure."""
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _safe_decimal(value: object) -> Decimal | None:
    """Safely convert an LLM output to Decimal. Returns None on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
