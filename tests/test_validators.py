"""
Comprehensive test suite for the Bad Deed Validator.

Tests the DETERMINISTIC components — no LLM, no network, no flakiness.
Every business rule is verified in isolation AND as part of the full pipeline.

Run: pytest tests/ -v
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from deed_validator.enrichment import resolve_county
from deed_validator.extractor_regex import extract_with_regex
from deed_validator.models import EnrichedDeed, Severity
from deed_validator.pipeline import DeedValidationPipeline, _parse_grantees
from deed_validator.validators import (
    validate_all,
    validate_amount_consistency,
    validate_apn_format,
    validate_date_logic,
    validate_future_dates,
    validate_grantor_name,
    validate_grantee_parties,
    validate_state_code,
    validate_status,
)
from deed_validator.word_to_number import words_to_number


# ─── Test Data ───────────────────────────────────────────────────────

RAW_OCR = """\
*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara  |  State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-10
Grantor:  T.E.S.L.A. Holdings LLC
Grantee:  John  &  Sarah  Connor
Amount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***"""

COUNTIES = [
    {"name": "Santa Clara", "tax_rate": 0.012},
    {"name": "San Mateo", "tax_rate": 0.011},
    {"name": "Santa Cruz", "tax_rate": 0.010},
]


def _make_deed(**overrides: Any) -> EnrichedDeed:
    """Factory for test deeds with sensible defaults (containing known errors)."""
    kwargs: dict[str, Any] = {
        "document_id": "DEED-TRUST-0042",
        "county_raw": "S. Clara",
        "county_resolved": "Santa Clara",
        "state": "CA",
        "date_signed": date(2024, 1, 15),
        "date_recorded": date(2024, 1, 10),  # BAD: before signing
        "grantor": "T.E.S.L.A. Holdings LLC",
        "grantee": ["John Connor", "Sarah Connor"],
        "amount_numeric": Decimal("1250000.00"),
        "amount_words": "One Million Two Hundred Thousand Dollars",
        "apn": "992-001-XA",
        "status": "PRELIMINARY",
        "tax_rate": 0.012,
        "estimated_transfer_tax": Decimal("15000.00"),
    }
    kwargs.update(overrides)
    # Pre-compute amount_from_words if not explicitly provided
    if "amount_from_words" not in kwargs:
        try:
            kwargs["amount_from_words"] = words_to_number(kwargs["amount_words"])
        except ValueError:
            kwargs["amount_from_words"] = None
    return EnrichedDeed(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# WORD TO NUMBER
# ═══════════════════════════════════════════════════════════════════════


class TestWordToNumber:
    """Verify our custom English-to-number converter."""

    def test_simple_hundred(self):
        assert words_to_number("One Hundred") == Decimal(100)

    def test_one_million(self):
        assert words_to_number("One Million") == Decimal(1_000_000)

    def test_million_with_thousands(self):
        assert words_to_number("One Million Two Hundred Thousand") == Decimal(1_200_000)

    def test_full_matching_amount(self):
        assert words_to_number("One Million Two Hundred Fifty Thousand") == Decimal(1_250_000)

    def test_ignores_dollar_word(self):
        result = words_to_number("One Million Two Hundred Thousand Dollars")
        assert result == Decimal(1_200_000)

    def test_three_hundred_forty_five(self):
        assert words_to_number("Three Hundred Forty Five") == Decimal(345)

    def test_twelve(self):
        assert words_to_number("Twelve") == Decimal(12)

    def test_ninety_nine(self):
        assert words_to_number("Ninety Nine") == Decimal(99)

    def test_hyphenated(self):
        assert words_to_number("Twenty-One") == Decimal(21)

    def test_zero(self):
        assert words_to_number("Zero") == Decimal(0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            words_to_number("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            words_to_number("   ")

    def test_nonsense_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            words_to_number("banana")

    def test_large_number(self):
        result = words_to_number("Two Billion Three Hundred Million")
        assert result == Decimal(2_300_000_000)


# ═══════════════════════════════════════════════════════════════════════
# DATE LOGIC VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestDateLogic:
    """The date check uses pure date comparison — NOT AI."""

    def test_recorded_before_signed_is_error(self):
        deed = _make_deed(
            date_signed=date(2024, 1, 15),
            date_recorded=date(2024, 1, 10),
        )
        findings = validate_date_logic(deed)
        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR
        assert findings[0].code == "DATE_LOGIC_VIOLATION"
        assert findings[0].details["gap_days"] == 5

    def test_same_day_is_ok(self):
        deed = _make_deed(
            date_signed=date(2024, 1, 15),
            date_recorded=date(2024, 1, 15),
        )
        assert validate_date_logic(deed) == []

    def test_recorded_after_signed_is_ok(self):
        deed = _make_deed(
            date_signed=date(2024, 1, 10),
            date_recorded=date(2024, 1, 15),
        )
        assert validate_date_logic(deed) == []

    def test_large_gap_detected(self):
        deed = _make_deed(
            date_signed=date(2024, 6, 1),
            date_recorded=date(2024, 1, 1),
        )
        findings = validate_date_logic(deed)
        assert findings[0].details["gap_days"] == 152


# ═══════════════════════════════════════════════════════════════════════
# AMOUNT CONSISTENCY VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestAmountConsistency:
    """$1,250,000 vs 'One Million Two Hundred Thousand' = $50k gap."""

    def test_mismatch_is_flagged_as_error(self):
        deed = _make_deed(
            amount_numeric=Decimal("1250000.00"),
            amount_words="One Million Two Hundred Thousand Dollars",
        )
        findings = validate_amount_consistency(deed)
        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR
        assert findings[0].code == "AMOUNT_MISMATCH"
        assert findings[0].details["discrepancy"] == "50000.00"

    def test_matching_amounts_pass(self):
        deed = _make_deed(
            amount_numeric=Decimal("1250000.00"),
            amount_words="One Million Two Hundred Fifty Thousand Dollars",
        )
        findings = validate_amount_consistency(deed)
        assert findings == []

    def test_unparseable_words_flagged(self):
        deed = _make_deed(
            amount_words="gibberish amount text",
            amount_from_words=None,
        )
        findings = validate_amount_consistency(deed)
        assert len(findings) == 1
        assert findings[0].code == "AMOUNT_WORDS_UNPARSEABLE"

    def test_small_discrepancy_still_flagged(self):
        """Even a $1 discrepancy is an error in financial validation."""
        deed = _make_deed(
            amount_numeric=Decimal("1001"),
            amount_words="One Thousand Dollars",
        )
        findings = validate_amount_consistency(deed)
        assert len(findings) == 1
        assert findings[0].code == "AMOUNT_MISMATCH"


# ═══════════════════════════════════════════════════════════════════════
# APN FORMAT VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestAPNFormat:
    def test_alpha_chars_flagged(self):
        deed = _make_deed(apn="992-001-XA")
        findings = validate_apn_format(deed)
        codes = [f.code for f in findings]
        assert "APN_CONTAINS_ALPHA" in codes

    def test_valid_numeric_apn_passes(self):
        deed = _make_deed(apn="992-001-123")
        findings = validate_apn_format(deed)
        assert findings == []


# ═══════════════════════════════════════════════════════════════════════
# STATUS VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestStatus:
    def test_preliminary_flagged(self):
        deed = _make_deed(status="PRELIMINARY")
        findings = validate_status(deed)
        assert len(findings) == 1
        assert findings[0].code == "STATUS_NOT_RECORDABLE"

    def test_recorded_passes(self):
        deed = _make_deed(status="RECORDED")
        assert validate_status(deed) == []

    def test_final_passes(self):
        deed = _make_deed(status="FINAL")
        assert validate_status(deed) == []

    def test_approved_passes(self):
        deed = _make_deed(status="APPROVED")
        assert validate_status(deed) == []


# ═══════════════════════════════════════════════════════════════════════
# STATE CODE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestStateCode:
    def test_ca_is_valid(self):
        deed = _make_deed(state="CA")
        assert validate_state_code(deed) == []

    def test_invalid_state(self):
        deed = _make_deed(state="ZZ")
        findings = validate_state_code(deed)
        assert findings[0].code == "INVALID_STATE_CODE"


# ═══════════════════════════════════════════════════════════════════════
# GRANTEE PARSER
# ═══════════════════════════════════════════════════════════════════════


class TestGranteeParser:
    def test_ampersand_with_shared_surname(self):
        result = _parse_grantees("John & Sarah Connor")
        assert result == ["John Connor", "Sarah Connor"]

    def test_single_grantee(self):
        result = _parse_grantees("John Connor")
        assert result == ["John Connor"]

    def test_multiple_spaces(self):
        """OCR often introduces extra whitespace."""
        result = _parse_grantees("John  &  Sarah  Connor")
        assert result == ["John Connor", "Sarah Connor"]

    def test_and_keyword(self):
        result = _parse_grantees("John and Sarah Connor")
        assert result == ["John Connor", "Sarah Connor"]


# ═══════════════════════════════════════════════════════════════════════
# GRANTOR NAME VALIDATOR
# ═══════════════════════════════════════════════════════════════════════


class TestGrantorName:
    def test_dotted_name_flagged(self):
        deed = _make_deed(grantor="T.E.S.L.A. Holdings LLC")
        findings = validate_grantor_name(deed)
        assert any(f.code == "GRANTOR_NAME_UNUSUAL" for f in findings)

    def test_normal_name_passes(self):
        deed = _make_deed(grantor="Acme Holdings LLC")
        findings = validate_grantor_name(deed)
        assert findings == []


# ═══════════════════════════════════════════════════════════════════════
# COUNTY RESOLUTION (FUZZY MATCHING)
# ═══════════════════════════════════════════════════════════════════════


class TestCountyResolution:
    def test_s_clara_resolves_to_santa_clara(self):
        match = resolve_county("S. Clara", COUNTIES)
        assert match is not None
        assert match.resolved == "Santa Clara"
        assert match.tax_rate == pytest.approx(0.012)

    def test_exact_match_has_full_confidence(self):
        match = resolve_county("Santa Clara", COUNTIES)
        assert match is not None
        assert match.confidence == pytest.approx(1.0)

    def test_san_mateo_exact(self):
        match = resolve_county("San Mateo", COUNTIES)
        assert match is not None
        assert match.tax_rate == pytest.approx(0.011)

    def test_unknown_county_returns_none(self):
        match = resolve_county("Atlantis", COUNTIES)
        assert match is None

    def test_s_cruz_resolves(self):
        match = resolve_county("S. Cruz", COUNTIES)
        assert match is not None
        assert match.resolved == "Santa Cruz"


# ═══════════════════════════════════════════════════════════════════════
# REGEX EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════


class TestRegexExtractor:
    def test_extracts_doc_id(self):
        result = extract_with_regex(RAW_OCR)
        assert result.document_id == "DEED-TRUST-0042"

    def test_extracts_county(self):
        result = extract_with_regex(RAW_OCR)
        assert result.county == "S. Clara"

    def test_extracts_state(self):
        result = extract_with_regex(RAW_OCR)
        assert result.state == "CA"

    def test_extracts_dates(self):
        result = extract_with_regex(RAW_OCR)
        assert result.date_signed == date(2024, 1, 15)
        assert result.date_recorded == date(2024, 1, 10)

    def test_extracts_amount_numeric(self):
        result = extract_with_regex(RAW_OCR)
        assert result.amount_numeric == Decimal("1250000.00")

    def test_extracts_amount_words(self):
        result = extract_with_regex(RAW_OCR)
        assert result.amount_words is not None
        assert "One Million" in result.amount_words
        assert "Two Hundred Thousand" in result.amount_words

    def test_extracts_grantor(self):
        result = extract_with_regex(RAW_OCR)
        assert result.grantor is not None
        assert "T.E.S.L.A." in result.grantor
        assert "Holdings LLC" in result.grantor

    def test_extracts_grantee(self):
        result = extract_with_regex(RAW_OCR)
        assert result.grantee is not None
        assert "John" in result.grantee
        assert "Sarah" in result.grantee
        assert "Connor" in result.grantee

    def test_extracts_apn(self):
        result = extract_with_regex(RAW_OCR)
        assert result.apn == "992-001-XA"

    def test_extracts_status(self):
        result = extract_with_regex(RAW_OCR)
        assert result.status == "PRELIMINARY"


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Run the entire pipeline end-to-end on the sample OCR text."""

    def test_pipeline_rejects_bad_deed(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert report.is_valid is False

    def test_pipeline_catches_date_error(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        codes = {f.code for f in report.findings}
        assert "DATE_LOGIC_VIOLATION" in codes

    def test_pipeline_catches_amount_mismatch(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        codes = {f.code for f in report.findings}
        assert "AMOUNT_MISMATCH" in codes

    def test_pipeline_catches_apn_issue(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        codes = {f.code for f in report.findings}
        assert "APN_CONTAINS_ALPHA" in codes

    def test_pipeline_catches_status_issue(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        codes = {f.code for f in report.findings}
        assert "STATUS_NOT_RECORDABLE" in codes

    def test_pipeline_resolves_county(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert report.deed is not None
        assert report.deed.county_resolved == "Santa Clara"
        assert report.deed.tax_rate == pytest.approx(0.012)

    def test_pipeline_computes_transfer_tax(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert report.deed is not None
        assert report.deed.estimated_transfer_tax == Decimal("0.012") * Decimal("1250000.00")

    def test_pipeline_has_audit_hash(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert len(report.original_hash) == 64  # SHA-256 hex = 64 chars

    def test_pipeline_catches_all_critical_issues(self):
        """The two MUST-CATCH issues from the assignment:
        1. Date logic violation
        2. Amount mismatch ($50K discrepancy)
        """
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)

        error_codes = {
            f.code for f in report.findings if f.severity == Severity.ERROR
        }

        # These two are the dealbreakers
        assert "DATE_LOGIC_VIOLATION" in error_codes, \
            "CRITICAL: Failed to catch date recorded before date signed!"
        assert "AMOUNT_MISMATCH" in error_codes, \
            "CRITICAL: Failed to catch $50K discrepancy between numeric and written amounts!"


# ═══════════════════════════════════════════════════════════════════════
# FULL VALIDATE_ALL ON THE TEST DEED
# ═══════════════════════════════════════════════════════════════════════


class TestValidateAll:
    """Verify validate_all() catches everything at once."""

    def test_catches_all_known_issues(self):
        deed = _make_deed()
        findings = validate_all(deed)
        codes = {f.code for f in findings}

        assert "DATE_LOGIC_VIOLATION" in codes
        assert "AMOUNT_MISMATCH" in codes
        assert "APN_CONTAINS_ALPHA" in codes
        assert "STATUS_NOT_RECORDABLE" in codes
        assert "MULTI_PARTY_GRANTEE" in codes
        assert "GRANTOR_NAME_UNUSUAL" in codes

    def test_error_count(self):
        deed = _make_deed()
        findings = validate_all(deed)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        # Should have at least: DATE_LOGIC_VIOLATION + AMOUNT_MISMATCH
        assert len(errors) >= 2


# ═══════════════════════════════════════════════════════════════════════
# WORD-TO-NUMBER EDGE CASES (LLM returns parens, etc.)
# ═══════════════════════════════════════════════════════════════════════


class TestWordToNumberEdgeCases:
    """The LLM sometimes returns amount_words with surrounding parentheses."""

    def test_parentheses_stripped(self):
        """LLM may return '(One Million Two Hundred Thousand Dollars)'."""
        result = words_to_number("(One Million Two Hundred Thousand Dollars)")
        assert result == Decimal(1_200_000)

    def test_brackets_stripped(self):
        result = words_to_number("[Five Hundred Thousand Dollars]")
        assert result == Decimal(500_000)

    def test_and_keyword_ignored(self):
        result = words_to_number("One Hundred and Fifty Thousand")
        assert result == Decimal(150_000)

    def test_compound_with_ones(self):
        result = words_to_number("Two Million Four Hundred Sixty Seven Thousand Eight Hundred Ninety One")
        assert result == Decimal(2_467_891)


# ═══════════════════════════════════════════════════════════════════════
# CLOSING COST CALCULATION
# ═══════════════════════════════════════════════════════════════════════


class TestClosingCosts:
    """The spec says: 'You'll need this to calculate closing costs later.'"""

    def test_pipeline_computes_closing_costs(self):
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert report.deed is not None
        assert report.deed.estimated_closing_costs is not None
        # Transfer tax ($15,000) + recording fee ($75)
        assert report.deed.estimated_closing_costs == Decimal("15075.00")

    def test_closing_costs_use_correct_tax_rate(self):
        """S. Clara → Santa Clara → tax_rate 0.012 → $1,250,000 * 0.012 = $15,000."""
        pipeline = DeedValidationPipeline()
        report = pipeline.run(RAW_OCR)
        assert report.deed is not None
        assert report.deed.estimated_transfer_tax == Decimal("15000.000")


# ═══════════════════════════════════════════════════════════════════════
# AMOUNT MISMATCH DETAIL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════


class TestAmountMismatchDetails:
    """Verify the $50K discrepancy is precisely reported."""

    def test_discrepancy_is_exactly_50k(self):
        deed = _make_deed(
            amount_numeric=Decimal("1250000.00"),
            amount_words="One Million Two Hundred Thousand Dollars",
        )
        findings = validate_amount_consistency(deed)
        assert len(findings) == 1
        # The discrepancy must be exactly $50,000
        assert Decimal(findings[0].details["discrepancy"]) == Decimal("50000.00")

    def test_message_mentions_both_values(self):
        deed = _make_deed(
            amount_numeric=Decimal("1250000.00"),
            amount_words="One Million Two Hundred Thousand Dollars",
        )
        findings = validate_amount_consistency(deed)
        msg = findings[0].message
        assert "1,250,000" in msg
        assert "1,200,000" in msg
        assert "50,000" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
