"""
Deterministic validation engine — the "paranoid" layer.

These validators run PURE CODE checks on the extracted deed data.
They NEVER call an LLM.  They NEVER guess.  They catch what the AI missed.

Each validator function:
  - Takes an EnrichedDeed
  - Returns a list of ValidationFinding objects (empty = all clear)
  - Is independently testable

The validate_all() function runs every check and aggregates findings.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from .models import EnrichedDeed, Severity, ValidationFinding


# ─── Constants ───────────────────────────────────────────────────────

VALID_RECORDING_STATUSES: frozenset[str] = frozenset({
    "RECORDED", "FINAL", "APPROVED", "EXECUTED",
})

VALID_US_STATES: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI", "AS", "MP",
})


# ─── Orchestrator ────────────────────────────────────────────────────


def validate_all(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Run ALL validators and collect findings."""
    findings: list[ValidationFinding] = []
    findings.extend(validate_date_logic(deed))
    findings.extend(validate_amount_consistency(deed))
    findings.extend(validate_apn_format(deed))
    findings.extend(validate_status(deed))
    findings.extend(validate_grantee_parties(deed))
    findings.extend(validate_state_code(deed))
    findings.extend(validate_future_dates(deed))
    findings.extend(validate_grantor_name(deed))
    return findings


# ─── Individual Validators ───────────────────────────────────────────


def validate_date_logic(deed: EnrichedDeed) -> list[ValidationFinding]:
    """A deed CANNOT be recorded before it is signed.

    This is a fundamental legal constraint — recording requires a signed document.
    We check this with a simple date comparison, NOT by asking an LLM.
    """
    findings: list[ValidationFinding] = []

    if deed.date_recorded < deed.date_signed:
        gap = (deed.date_signed - deed.date_recorded).days
        findings.append(
            ValidationFinding(
                severity=Severity.ERROR,
                code="DATE_LOGIC_VIOLATION",
                field="date_recorded",
                message=(
                    f"Document was recorded ({deed.date_recorded}) BEFORE it was "
                    f"signed ({deed.date_signed}). A deed cannot be recorded before "
                    f"signing. Gap: {gap} day(s)."
                ),
                details={
                    "date_signed": str(deed.date_signed),
                    "date_recorded": str(deed.date_recorded),
                    "gap_days": gap,
                },
            )
        )

    return findings


def validate_amount_consistency(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Cross-check the numeric amount against the written-out amount.

    Legal documents include both forms ($1,250,000 and "One Million...")
    as a built-in integrity check. If they disagree, we MUST flag it.
    We do NOT silently pick one — that would be the opposite of paranoid.

    The word→number conversion is performed during enrichment so this
    validator stays pure (no side effects, no mutation).
    """
    findings: list[ValidationFinding] = []

    amount_from_words = deed.amount_from_words

    # If enrichment couldn't parse the words, report it
    if amount_from_words is None:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                code="AMOUNT_WORDS_UNPARSEABLE",
                field="amount_words",
                message=f"Could not parse written amount: '{deed.amount_words}'",
                details={"raw_words": deed.amount_words},
            )
        )
        return findings

    # Compare
    if amount_from_words != deed.amount_numeric:
        discrepancy = abs(deed.amount_numeric - amount_from_words)
        findings.append(
            ValidationFinding(
                severity=Severity.ERROR,
                code="AMOUNT_MISMATCH",
                field="amount_numeric",
                message=(
                    f"DISCREPANCY: Numeric amount (${deed.amount_numeric:,.2f}) does "
                    f"not match written amount \"{deed.amount_words}\" "
                    f"(=${amount_from_words:,.2f}). "
                    f"Difference: ${discrepancy:,.2f}. "
                    f"Both values must agree before recording."
                ),
                details={
                    "amount_numeric": str(deed.amount_numeric),
                    "amount_words": deed.amount_words,
                    "amount_from_words": str(amount_from_words),
                    "discrepancy": str(discrepancy),
                },
            )
        )

    return findings


def validate_apn_format(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Validate Assessor's Parcel Number format.

    Standard CA APNs are formatted as NNN-NNN-NNN (all numeric with dashes).
    Alpha characters are unusual and may indicate OCR errors or data corruption.
    """
    findings: list[ValidationFinding] = []

    # Check for non-numeric, non-dash characters
    alpha_chars = re.sub(r"[\d\-]", "", deed.apn)
    if alpha_chars:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                code="APN_CONTAINS_ALPHA",
                field="apn",
                message=(
                    f"APN '{deed.apn}' contains non-numeric characters: "
                    f"'{alpha_chars}'. Standard APNs are numeric-only. "
                    f"This may indicate an OCR scanning error."
                ),
                details={"apn": deed.apn, "invalid_chars": alpha_chars},
            )
        )

    return findings


def validate_status(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Check document status for recording readiness.

    Only certain statuses indicate a deed is ready for blockchain recording.
    A PRELIMINARY deed should never be committed.
    """
    findings: list[ValidationFinding] = []

    if deed.status.upper() not in VALID_RECORDING_STATUSES:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                code="STATUS_NOT_RECORDABLE",
                field="status",
                message=(
                    f"Document status is '{deed.status}', which is not a valid "
                    f"recording status. Expected one of: "
                    f"{', '.join(sorted(VALID_RECORDING_STATUSES))}. "
                    f"This deed should not be committed to the blockchain."
                ),
                details={
                    "current_status": deed.status,
                    "valid_statuses": sorted(VALID_RECORDING_STATUSES),  # noqa: C414
                },
            )
        )

    return findings


def validate_grantee_parties(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Flag multi-party grantees for additional review.

    Multi-party deeds (e.g., "John & Sarah Connor") require verification
    of ownership split / tenancy type.
    """
    findings: list[ValidationFinding] = []

    if len(deed.grantee) > 1:
        findings.append(
            ValidationFinding(
                severity=Severity.INFO,
                code="MULTI_PARTY_GRANTEE",
                field="grantee",
                message=(
                    f"Multiple grantees detected ({len(deed.grantee)}): "
                    f"{', '.join(deed.grantee)}. Verify ownership split / "
                    f"tenancy type (joint tenants, tenants-in-common, etc.)."
                ),
                details={"parties": deed.grantee, "count": len(deed.grantee)},
            )
        )

    return findings


def validate_state_code(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Validate state code is a recognized US state/territory."""
    findings: list[ValidationFinding] = []

    if deed.state.upper() not in VALID_US_STATES:
        findings.append(
            ValidationFinding(
                severity=Severity.ERROR,
                code="INVALID_STATE_CODE",
                field="state",
                message=f"'{deed.state}' is not a recognized US state code.",
                details={"state": deed.state},
            )
        )

    return findings


def validate_future_dates(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Flag any dates that are in the future — potential data entry error."""
    findings: list[ValidationFinding] = []
    today = date.today()

    if deed.date_signed > today:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                code="FUTURE_DATE_SIGNED",
                field="date_signed",
                message=f"Date signed ({deed.date_signed}) is in the future.",
                details={"date_signed": str(deed.date_signed), "today": str(today)},
            )
        )

    if deed.date_recorded > today:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                code="FUTURE_DATE_RECORDED",
                field="date_recorded",
                message=f"Date recorded ({deed.date_recorded}) is in the future.",
                details={
                    "date_recorded": str(deed.date_recorded),
                    "today": str(today),
                },
            )
        )

    return findings


def validate_grantor_name(deed: EnrichedDeed) -> list[ValidationFinding]:
    """Flag unusual patterns in grantor name that may indicate OCR artifacts.

    E.g., "T.E.S.L.A. Holdings LLC" has dots between each letter,
    which could be a real entity name OR an OCR scanning artifact.
    """
    findings: list[ValidationFinding] = []

    # Detect excessive dots (potential OCR artifact)
    dot_count = deed.grantor.count(".")
    word_count = len(deed.grantor.split())

    if dot_count >= 3 and dot_count > word_count:
        findings.append(
            ValidationFinding(
                severity=Severity.INFO,
                code="GRANTOR_NAME_UNUSUAL",
                field="grantor",
                message=(
                    f"Grantor name '{deed.grantor}' contains an unusual number "
                    f"of periods ({dot_count}). This may be an OCR artifact — "
                    f"verify against the original document."
                ),
                details={"grantor": deed.grantor, "dot_count": dot_count},
            )
        )

    return findings
