"""Pytest configuration — ensures the project root is importable."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _no_llm_calls():
    """Prevent real LLM API calls during tests — keeps the suite fast and free."""
    with patch("deed_validator.pipeline.extract_with_llm", return_value=None):
        yield
