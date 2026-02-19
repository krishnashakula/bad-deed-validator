"""
Bad Deed Validator — FastAPI Server
====================================

RESTful API for validating OCR-scanned property deeds.

Endpoints:
    POST /validate          Validate raw OCR deed text
    POST /validate/file     Upload a text file for validation
    GET  /health            Health check / readiness probe

Run:
    uvicorn api:app --reload              # Dev (http://localhost:8000)
    uvicorn api:app --host 0.0.0.0        # Production

Docs:
    http://localhost:8000/docs             # Swagger UI (auto-generated)
    http://localhost:8000/redoc            # ReDoc (alternative)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import asyncio

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from deed_validator.models import EnrichedDeed, ValidationFinding, ValidationReport
from deed_validator.pipeline import DeedValidationPipeline

# ─── Load .env if available ──────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ─── Application Lifespan (pre-warm pipeline) ───────────────────────

_pipeline: DeedValidationPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the pipeline (load counties.json) on startup."""
    global _pipeline  # noqa: PLW0603
    _pipeline = DeedValidationPipeline()
    yield
    _pipeline = None


# ─── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(
    title="Bad Deed Validator API",
    description=(
        "Paranoid validation for OCR-scanned property deeds. "
        "Dual extraction (Regex + LLM), deterministic code-based validation, "
        "fuzzy county resolution, and closing-cost estimation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response Schemas ─────────────────────────────────────


class ValidateRequest(BaseModel):
    """Request body for the /validate endpoint."""

    raw_ocr_text: str = Field(
        ...,
        min_length=10,
        description="The raw OCR-scanned deed text to validate.",
        json_schema_extra={
            "example": (
                "*** RECORDING REQ ***\n"
                "Doc: DEED-TRUST-0042\n"
                "County: S. Clara  |  State: CA\n"
                "Date Signed: 2024-01-15\n"
                "Date Recorded: 2024-01-10\n"
                "Grantor:  T.E.S.L.A. Holdings LLC\n"
                "Grantee:  John  &  Sarah  Connor\n"
                "Amount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)\n"
                "APN: 992-001-XA\n"
                "Status: PRELIMINARY\n"
                "*** END ***"
            )
        },
    )


class SeverityOut(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class FindingOut(ValidationFinding):
    """API-facing finding (inherits all fields from ValidationFinding)."""

    severity: SeverityOut  # type: ignore[assignment]  # narrow to str enum for OpenAPI


class DeedOut(EnrichedDeed):
    """API-facing deed (inherits all fields from EnrichedDeed)."""


class ValidateResponse(BaseModel):
    """Structured validation report returned by the API."""

    document_id: str
    is_valid: bool
    extraction_method: str
    original_hash: str = Field(description="SHA-256 hash of the raw OCR input")
    error_count: int
    warning_count: int
    findings: list[FindingOut]
    deed: Optional[DeedOut] = None

    model_config = {"json_schema_extra": {"example": {
        "document_id": "DEED-TRUST-0042",
        "is_valid": False,
        "extraction_method": "Regex-only (no LLM API key)",
        "original_hash": "a1b2c3d4...",
        "error_count": 2,
        "warning_count": 2,
        "findings": [
            {
                "severity": "ERROR",
                "code": "DATE_LOGIC_VIOLATION",
                "field": "date_recorded",
                "message": "Recorded before signed — 5 day gap",
                "details": {"gap_days": 5},
            }
        ],
        "deed": None,
    }}}


class HealthResponse(BaseModel):
    status: str
    version: str
    counties_loaded: int


# ─── Helpers ─────────────────────────────────────────────────────────


def _get_pipeline() -> DeedValidationPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


def _build_response(report: ValidationReport) -> ValidateResponse:
    """Convert the internal ValidationReport to the API response schema."""
    deed_out = DeedOut.model_validate(report.deed, from_attributes=True) if report.deed else None

    findings_out = [
        FindingOut.model_validate(f, from_attributes=True)
        for f in report.findings
    ]

    error_count = sum(1 for f in report.findings if f.severity.value == "ERROR")
    warning_count = sum(1 for f in report.findings if f.severity.value == "WARNING")

    return ValidateResponse(
        document_id=report.document_id,
        is_valid=report.is_valid,
        extraction_method=report.extraction_method,
        original_hash=report.original_hash,
        error_count=error_count,
        warning_count=warning_count,
        findings=findings_out,
        deed=deed_out,
    )


# ─── Endpoints ───────────────────────────────────────────────────────


@app.post(
    "/validate",
    summary="Validate a deed from raw OCR text",
    tags=["Validation"],
    responses={503: {"description": "Pipeline not yet initialised"}},
)
def validate_deed(request: ValidateRequest) -> ValidateResponse:
    """Run the full validation pipeline on raw OCR-scanned deed text.

    Returns a structured report with:
    - **is_valid**: `true` if the deed passes all checks
    - **findings**: detailed list of errors, warnings, and info items
    - **deed**: the enriched deed data (county resolved, tax calculated, etc.)
    - **original_hash**: SHA-256 of the input for audit trail
    """
    pipeline = _get_pipeline()
    report = pipeline.run(request.raw_ocr_text)
    return _build_response(report)


@app.post(
    "/validate/file",
    summary="Validate a deed from an uploaded text file",
    tags=["Validation"],
    responses={
        413: {"description": "File too large (max 1 MB)"},
        400: {"description": "File is not valid UTF-8 text"},
        422: {"description": "File content too short to be a deed"},
        503: {"description": "Pipeline not yet initialised"},
    },
)
async def validate_deed_file(file: UploadFile) -> ValidateResponse:
    """Upload a `.txt` file containing raw OCR deed text for validation.

    Accepts any text file up to 1 MB.
    """
    if file.size and file.size > 1_048_576:
        raise HTTPException(status_code=413, detail="File too large (max 1 MB)")

    content = await file.read()
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    if len(raw_text.strip()) < 10:
        raise HTTPException(status_code=422, detail="File content too short to be a deed")

    pipeline = _get_pipeline()
    report = await asyncio.to_thread(pipeline.run, raw_text)
    return _build_response(report)


@app.get(
    "/health",
    summary="Health check",
    tags=["System"],
    responses={503: {"description": "Pipeline not yet initialised"}},
)
def health_check() -> HealthResponse:
    """Returns service status and configuration info."""
    pipeline = _get_pipeline()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        counties_loaded=len(pipeline.counties),
    )
