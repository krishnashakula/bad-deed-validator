"""
Pydantic models for deed data — strict typing as our first line of defense.

Every field is explicitly typed. No `Any`, no `dict`. If data doesn't fit
the model, it fails loudly at the boundary — not silently downstream.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Severity Levels ────────────────────────────────────────────────


class Severity(str, Enum):
    """Severity of a validation finding."""

    ERROR = "ERROR"  # Fatal — deed MUST be rejected
    WARNING = "WARNING"  # Suspicious — needs human review
    INFO = "INFO"  # Informational observation


# ─── Validation Finding ─────────────────────────────────────────────


class ValidationFinding(BaseModel):
    """A single validation finding with severity, machine-readable code, and details."""

    severity: Severity
    code: str  # Machine-readable, e.g. "DATE_LOGIC_VIOLATION"
    field: str  # Which deed field this relates to
    message: str  # Human-readable explanation
    details: dict = Field(default_factory=dict)


# ─── Extraction Models ──────────────────────────────────────────────


class RawDeedExtraction(BaseModel):
    """What the LLM (or regex) extracts from raw OCR text.

    Fields are Optional because extraction may fail for individual fields.
    We validate completeness later in the pipeline.
    """

    document_id: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    date_signed: Optional[date] = None
    date_recorded: Optional[date] = None
    grantor: Optional[str] = None
    grantee: Optional[str] = None
    amount_numeric: Optional[Decimal] = None
    amount_words: Optional[str] = None
    apn: Optional[str] = None
    status: Optional[str] = None


# ─── Enriched Deed ──────────────────────────────────────────────────


class EnrichedDeed(BaseModel):
    """Deed data after enrichment: county resolved, tax rate attached, grantees parsed."""

    document_id: str
    county_raw: str
    county_resolved: str
    state: str
    date_signed: date
    date_recorded: date
    grantor: str
    grantee: list[str]
    amount_numeric: Decimal
    amount_words: str
    amount_from_words: Optional[Decimal] = None
    apn: str
    status: str
    tax_rate: Optional[float] = None
    estimated_transfer_tax: Optional[Decimal] = None
    estimated_closing_costs: Optional[Decimal] = None  # Transfer tax + est. fees


# ─── Validation Report ──────────────────────────────────────────────


class ValidationReport(BaseModel):
    """The final output of the validation pipeline."""

    document_id: str
    is_valid: bool
    findings: list[ValidationFinding] = Field(default_factory=list)
    deed: Optional[EnrichedDeed] = None
    extraction_method: str = "unknown"
    original_hash: str = ""  # SHA-256 of original OCR text for audit trail
