"""
Bad Deed Validator — Paranoid validation for OCR-scanned property deeds.

Architecture: Dual extraction (Regex + LLM) → Reconciliation → Enrichment → Validation
Philosophy:  Trust the AI to parse. Trust only code to validate.
"""

__version__ = "1.0.0"
