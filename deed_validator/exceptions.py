"""
Custom exception hierarchy for deed validation.

Each exception type maps to a specific category of validation failure,
enabling precise error handling and reporting in the pipeline.
"""

from __future__ import annotations


class DeedValidationError(Exception):
    """Base exception for all deed validation failures."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)


class DateLogicViolation(DeedValidationError):
    """A deed cannot be recorded before it is signed."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__("DATE_LOGIC_VIOLATION", message, details)


class AmountMismatchError(DeedValidationError):
    """The numeric amount disagrees with the written-out amount."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__("AMOUNT_MISMATCH", message, details)


class CountyResolutionError(DeedValidationError):
    """The county name cannot be matched to any known county."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__("COUNTY_RESOLUTION_FAILED", message, details)


class APNFormatError(DeedValidationError):
    """The Assessor's Parcel Number has an invalid format."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__("APN_FORMAT_INVALID", message, details)


class ExtractionError(DeedValidationError):
    """Data extraction (LLM or regex) failed completely."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__("EXTRACTION_FAILED", message, details)
