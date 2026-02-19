#!/usr/bin/env python3
"""
Bad Deed Validator — Entry Point
=================================

Demonstrates the full validation pipeline on a sample OCR-scanned deed.

Usage:
    python main.py                          # Regex-only mode (no API key needed)
    OPENAI_API_KEY=sk-... python main.py    # LLM + Regex dual extraction
"""

from __future__ import annotations

import sys

from deed_validator.models import Severity
from deed_validator.pipeline import DeedValidationPipeline

# ─── Load .env if available (optional dependency) ────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ─── The Exact OCR Output — Ugly on Purpose ─────────────────────────

RAW_OCR_TEXT = """\
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


# ─── ANSI Color Constants ───────────────────────────────────────────

_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_WIDTH = 72


# ─── Pretty Printer Helpers ─────────────────────────────────────────


def _print_deed_details(deed) -> None:
    """Print enriched deed fields."""
    print(f"  County:      {deed.county_raw} {_DIM}→{_RESET} {_BOLD}{deed.county_resolved}{_RESET}")
    print(f"  State:       {deed.state}")
    print(f"  Signed:      {deed.date_signed}")
    print(f"  Recorded:    {deed.date_recorded}")
    print(f"  Grantor:     {deed.grantor}")
    print(f"  Grantee(s):  {', '.join(deed.grantee)}")
    print(f"  Amount ($):  ${deed.amount_numeric:,.2f}")
    print(f"  Amount (w):  {deed.amount_words}")
    if deed.amount_from_words is not None:
        print(f"  Words -> $:  ${deed.amount_from_words:,.2f}")
    print(f"  APN:         {deed.apn}")
    print(f"  Status:      {deed.status}")
    if deed.tax_rate:
        print(f"  Tax Rate:    {deed.tax_rate:.3f}")
    if deed.estimated_transfer_tax:
        print(f"  Est. Tax:    ${deed.estimated_transfer_tax:,.2f}")
    if deed.estimated_closing_costs:
        print(f"  Close Cost:  ${deed.estimated_closing_costs:,.2f}")


def _print_findings_group(findings, color: str, label: str) -> None:
    """Print a categorized group of findings (errors or warnings)."""
    if not findings:
        return
    print(f"\n  {color}{_BOLD}{label} ({len(findings)}){_RESET}")
    for f in findings:
        print(f"    {color}[{f.code}]{_RESET}")
        print(f"    {f.message}")
        for k, v in f.details.items():
            print(f"      {_DIM}{k}: {v}{_RESET}")
        print()


def _print_info_group(findings) -> None:
    """Print informational findings (compact format)."""
    if not findings:
        return
    print(f"  {_CYAN}INFO ({len(findings)}){_RESET}")
    for f in findings:
        print(f"    [{f.code}] {f.message}")
    print()


# ─── Pretty Printer ─────────────────────────────────────────────────


def print_report(report) -> int:
    """Pretty-print the validation report with ANSI color codes.

    Returns:
        0 if deed passed, 1 if rejected.
    """
    print(f"\n{'=' * _WIDTH}")
    print(f"{_BOLD}{_CYAN}  DEED VALIDATION REPORT{_RESET}")
    print(f"{'=' * _WIDTH}")
    print(f"  Document:    {report.document_id}")
    print(f"  Audit Hash:  {_DIM}{report.original_hash[:16]}...{_RESET}")
    print(f"  Extraction:  {report.extraction_method}")
    print(f"{'─' * _WIDTH}")

    if report.deed:
        _print_deed_details(report.deed)

    print(f"{'─' * _WIDTH}")

    errors = [f for f in report.findings if f.severity == Severity.ERROR]
    warnings = [f for f in report.findings if f.severity == Severity.WARNING]
    infos = [f for f in report.findings if f.severity == Severity.INFO]

    _print_findings_group(errors, _RED, "ERRORS")
    _print_findings_group(warnings, _YELLOW, "WARNINGS")
    _print_info_group(infos)

    print(f"{'=' * _WIDTH}")
    if report.is_valid:
        print(f"  {_GREEN}{_BOLD}DEED PASSED ALL CHECKS{_RESET}")
    else:
        print(f"  {_RED}{_BOLD}DEED REJECTED  --  {len(errors)} error(s) found{_RESET}")
    print(f"{'=' * _WIDTH}\n")

    return 0 if report.is_valid else 1


# ─── Main ────────────────────────────────────────────────────────────


def main():
    """Run the full validation pipeline and print the report."""
    print("\n  Starting Bad Deed Validator...")
    print("  Analyzing OCR-scanned deed...\n")

    pipeline = DeedValidationPipeline()
    report = pipeline.run(RAW_OCR_TEXT)
    exit_code = print_report(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
