"""
Main validation pipeline — orchestrates the full workflow.

Flow:
  ┌─────────┐
  │ Raw OCR │
  └────┬────┘
       │
  ┌────▼────┐     ┌──────────┐
  │  Regex  │     │   LLM    │   ← Dual extraction
  │ Extract │     │ Extract  │
  └────┬────┘     └────┬─────┘
       │               │
       └───────┬───────┘
               │
        ┌──────▼──────┐
        │ Reconciler  │   ← Flag disagreements
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Enrichment  │   ← Fuzzy county match, tax calc
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Validators  │   ← Pure code checks
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   Report    │   ← Typed findings + pass/fail
        └─────────────┘

Design principles:
  - The regex extractor ALWAYS runs (deterministic baseline).
  - The LLM extractor is optional (graceful degradation).
  - Reconciliation catches cases where the LLM "helpfully" corrects errors.
  - Validators are pure functions — no side effects, no network calls.
  - The original OCR text is SHA-256 hashed for audit trail.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from decimal import Decimal

from .enrichment import load_counties, parse_grantees, resolve_county
from .extractor_llm import extract_with_llm
from .extractor_regex import extract_with_regex
from .models import (
    EnrichedDeed,
    RawDeedExtraction,
    Severity,
    ValidationFinding,
    ValidationReport,
)
from .validators import validate_all
from .word_to_number import words_to_number

logger = logging.getLogger(__name__)


class DeedValidationPipeline:
    """Orchestrates the full deed validation workflow.

    Usage:
        pipeline = DeedValidationPipeline()
        report = pipeline.run(raw_ocr_text)
        if not report.is_valid:
            # deed has errors — do not record
            for finding in report.findings:
                print(finding)
    """

    def __init__(self, counties_path: str | None = None):
        self.counties = load_counties(counties_path)

    def run(self, raw_text: str) -> ValidationReport:
        """Execute the full pipeline on raw OCR text.

        Args:
            raw_text: The raw OCR-scanned deed text.

        Returns:
            ValidationReport with findings and pass/fail verdict.
        """
        # ── Step 0: Audit hash of original input ────────────────────
        doc_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # ── Step 1: Dual extraction ─────────────────────────────────
        logger.info("Starting regex extraction...")
        regex_result = extract_with_regex(raw_text)

        logger.info("Starting LLM extraction...")
        llm_result = extract_with_llm(raw_text)

        # Determine primary extraction source
        if llm_result is not None:
            primary = llm_result
            extraction_method = "LLM (GPT-5) + Regex cross-check"
        else:
            primary = regex_result
            extraction_method = "Regex-only (no LLM API key)"

        # ── Step 2: Reconcile extractions (if both available) ───────
        reconciliation_findings: list[ValidationFinding] = []
        if llm_result is not None:
            reconciliation_findings = self._reconcile(regex_result, llm_result)

        # ── Step 3: Validate extraction completeness ────────────────
        missing = self._check_required_fields(primary)
        if missing:
            return ValidationReport(
                document_id=primary.document_id or "UNKNOWN",
                is_valid=False,
                findings=missing,
                extraction_method=extraction_method,
                original_hash=doc_hash,
            )

        # ── Step 4: Enrich (county resolution, tax calc) ───────────
        enriched, enrichment_findings = self._enrich(primary)

        # ── Step 5: Validate (all deterministic checks) ─────────────
        validation_findings = validate_all(enriched)

        # ── Step 6: Compile final report ────────────────────────────
        all_findings = (
            reconciliation_findings + enrichment_findings + validation_findings
        )
        has_errors = any(f.severity == Severity.ERROR for f in all_findings)

        return ValidationReport(
            document_id=enriched.document_id,
            is_valid=not has_errors,
            findings=all_findings,
            deed=enriched,
            extraction_method=extraction_method,
            original_hash=doc_hash,
        )

    # ─── Reconciliation ─────────────────────────────────────────────

    def _reconcile(
        self, regex: RawDeedExtraction, llm: RawDeedExtraction
    ) -> list[ValidationFinding]:
        """Compare LLM and regex extractions field-by-field.

        Any disagreement is flagged as a WARNING. This catches cases where
        the LLM "helpfully" corrects an error that should be reported,
        or when the regex misparses noisy OCR.
        """
        findings: list[ValidationFinding] = []

        fields_to_check = [
            ("document_id", "Document ID"),
            ("county", "County"),
            ("state", "State"),
            ("date_signed", "Date Signed"),
            ("date_recorded", "Date Recorded"),
            ("amount_numeric", "Amount (Numeric)"),
            ("apn", "APN"),
            ("status", "Status"),
        ]

        for field_name, display_name in fields_to_check:
            regex_val = getattr(regex, field_name)
            llm_val = getattr(llm, field_name)

            if regex_val is not None and llm_val is not None:
                # Use type-aware comparison: Decimal and date compare by value,
                # not by string representation (avoids 1250000 != 1250000.00)
                if isinstance(regex_val, Decimal) and isinstance(llm_val, Decimal):
                    values_match = regex_val == llm_val
                elif isinstance(regex_val, date) and isinstance(llm_val, date):
                    values_match = regex_val == llm_val
                else:
                    values_match = (
                        str(regex_val).strip().lower()
                        == str(llm_val).strip().lower()
                    )

                if not values_match:
                    findings.append(
                        ValidationFinding(
                            severity=Severity.WARNING,
                            code="EXTRACTION_DISAGREEMENT",
                            field=field_name,
                            message=(
                                f"LLM and regex disagree on {display_name}: "
                                f"regex='{regex_val}', LLM='{llm_val}'. "
                                f"Manual review recommended."
                            ),
                            details={
                                "regex_value": str(regex_val),
                                "llm_value": str(llm_val),
                            },
                        )
                    )

        return findings

    # ─── Required Field Check ────────────────────────────────────────

    def _check_required_fields(
        self, extraction: RawDeedExtraction
    ) -> list[ValidationFinding]:
        """Ensure all critical fields were extracted.

        If any required field is missing, we cannot proceed with validation.
        Better to reject early than validate incomplete data.
        """
        findings: list[ValidationFinding] = []

        required = [
            ("document_id", "Document ID"),
            ("county", "County"),
            ("state", "State"),
            ("date_signed", "Date Signed"),
            ("date_recorded", "Date Recorded"),
            ("grantor", "Grantor"),
            ("grantee", "Grantee"),
            ("amount_numeric", "Amount (Numeric)"),
            ("amount_words", "Amount (Words)"),
            ("apn", "APN"),
            ("status", "Status"),
        ]

        for field_name, display_name in required:
            if getattr(extraction, field_name) is None:
                findings.append(
                    ValidationFinding(
                        severity=Severity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                        field=field_name,
                        message=f"Required field '{display_name}' could not be extracted from OCR text.",
                    )
                )

        return findings

    # ─── Enrichment ──────────────────────────────────────────────────

    def _enrich(
        self, extraction: RawDeedExtraction
    ) -> tuple[EnrichedDeed, list[ValidationFinding]]:
        """Resolve county name, parse grantees, calculate transfer tax."""
        findings: list[ValidationFinding] = []

        # Type narrowing — _check_required_fields verified these are non-None
        assert extraction.document_id is not None
        assert extraction.county is not None
        assert extraction.state is not None
        assert extraction.date_signed is not None
        assert extraction.date_recorded is not None
        assert extraction.grantor is not None
        assert extraction.grantee is not None
        assert extraction.amount_numeric is not None
        assert extraction.amount_words is not None
        assert extraction.apn is not None
        assert extraction.status is not None

        # ── County resolution ───────────────────────────────────────
        county_match = resolve_county(extraction.county, self.counties)

        if county_match:
            county_resolved = county_match.resolved
            tax_rate = county_match.tax_rate

            if county_match.confidence < 1.0:
                findings.append(
                    ValidationFinding(
                        severity=Severity.INFO,
                        code="COUNTY_FUZZY_MATCHED",
                        field="county",
                        message=(
                            f"County '{extraction.county}' fuzzy-matched to "
                            f"'{county_resolved}' "
                            f"(confidence: {county_match.confidence:.1%})"
                        ),
                        details={
                            "original": extraction.county,
                            "resolved": county_resolved,
                            "confidence": county_match.confidence,
                        },
                    )
                )
        else:
            county_resolved = extraction.county
            tax_rate = None
            findings.append(
                ValidationFinding(
                    severity=Severity.ERROR,
                    code="COUNTY_RESOLUTION_FAILED",
                    field="county",
                    message=(
                        f"Could not resolve county '{extraction.county}' "
                        f"to any known county in the database."
                    ),
                    details={"raw_county": extraction.county},
                )
            )

        # ── Parse grantee(s) ────────────────────────────────────────
        grantees = parse_grantees(extraction.grantee)
        # ── Convert written amount to Decimal (before validators run) ──
        amount_from_words: Decimal | None = None
        try:
            amount_from_words = words_to_number(extraction.amount_words)
        except ValueError:
            pass  # Validator will report AMOUNT_WORDS_UNPARSEABLE
        # ── Calculate estimated transfer tax & closing costs ───
        estimated_tax = None
        estimated_closing = None
        if tax_rate and extraction.amount_numeric:
            estimated_tax = Decimal(str(tax_rate)) * extraction.amount_numeric
            # Closing costs = transfer tax + estimated recording/escrow fees
            # (typical 0.1%–0.2% of sale price for recording + flat fees)
            recording_fee = Decimal("75.00")  # Typical county recording fee
            estimated_closing = estimated_tax + recording_fee

        # ── Build enriched deed ─────────────────────────────────────
        enriched = EnrichedDeed(
            document_id=extraction.document_id,
            county_raw=extraction.county,
            county_resolved=county_resolved,
            state=extraction.state,
            date_signed=extraction.date_signed,
            date_recorded=extraction.date_recorded,
            grantor=extraction.grantor,
            grantee=grantees,
            amount_numeric=extraction.amount_numeric,
            amount_words=extraction.amount_words,
            amount_from_words=amount_from_words,
            apn=extraction.apn,
            status=extraction.status,
            tax_rate=tax_rate,
            estimated_transfer_tax=estimated_tax,
            estimated_closing_costs=estimated_closing,
        )

        return enriched, findings


# Backward-compatible alias (function moved to enrichment.py)
_parse_grantees = parse_grantees
