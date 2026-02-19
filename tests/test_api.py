"""
FastAPI endpoint tests for the Bad Deed Validator API.

Uses httpx + FastAPI TestClient — no real server needed, no LLM calls.
"""

from __future__ import annotations

import api
import pytest
from api import app
from fastapi.testclient import TestClient

from deed_validator.pipeline import DeedValidationPipeline

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _warm_pipeline() -> None:
    """Initialise the pipeline once for all API tests (bypasses lifespan)."""
    api._pipeline = DeedValidationPipeline()
    yield  # type: ignore[misc]
    api._pipeline = None


# ─── Sample OCR text (same as main.py) ──────────────────────────────

RAW_OCR = (
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


class TestHealthEndpoint:
    def test_health_returns_200(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["counties_loaded"] >= 1


class TestValidateEndpoint:
    def test_rejects_bad_deed(self) -> None:
        resp = client.post("/validate", json={"raw_ocr_text": RAW_OCR})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is False
        assert data["document_id"] == "DEED-TRUST-0042"

    def test_catches_date_and_amount_errors(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        codes = {f["code"] for f in data["findings"]}
        assert "DATE_LOGIC_VIOLATION" in codes
        assert "AMOUNT_MISMATCH" in codes

    def test_error_and_warning_counts(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        assert data["error_count"] >= 2
        assert data["warning_count"] >= 1

    def test_deed_is_populated(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        deed = data["deed"]
        assert deed is not None
        assert deed["county_resolved"] == "Santa Clara"
        assert deed["state"] == "CA"
        assert len(deed["grantee"]) == 2

    def test_original_hash_present(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        assert len(data["original_hash"]) == 64  # SHA-256 hex

    def test_extraction_method_populated(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        assert "Regex" in data["extraction_method"] or "LLM" in data["extraction_method"]

    def test_closing_costs_in_response(self) -> None:
        data = client.post("/validate", json={"raw_ocr_text": RAW_OCR}).json()
        deed = data["deed"]
        assert deed["estimated_closing_costs"] is not None
        assert deed["estimated_transfer_tax"] is not None


class TestRequestValidation:
    def test_empty_body_returns_422(self) -> None:
        resp = client.post("/validate", json={})
        assert resp.status_code == 422

    def test_too_short_text_returns_422(self) -> None:
        resp = client.post("/validate", json={"raw_ocr_text": "short"})
        assert resp.status_code == 422

    def test_missing_content_type_returns_422(self) -> None:
        resp = client.post("/validate")
        assert resp.status_code == 422


class TestFileUploadEndpoint:
    def test_upload_text_file(self) -> None:
        resp = client.post(
            "/validate/file",
            files={"file": ("deed.txt", RAW_OCR.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is False
        assert data["document_id"] == "DEED-TRUST-0042"

    def test_upload_too_short_file(self) -> None:
        resp = client.post(
            "/validate/file",
            files={"file": ("deed.txt", b"tiny", "text/plain")},
        )
        assert resp.status_code == 422
