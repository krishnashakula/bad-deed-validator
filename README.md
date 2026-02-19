# Bad Deed Validator

**Paranoid validation for OCR-scanned property deeds** — built for the intersection of fuzzy AI and rigorous financial logic.

> *"Trust the AI to parse. Trust only code to validate."*

---

## Architecture: Zero-Trust AI Pipeline

```
Raw OCR Text
     │
     ▼
┌──────────────┐     ┌──────────────┐
│ Regex Extract│     │  LLM Extract │   ← Dual extraction paths
│ (deterministic)    │ (smart, but  │
│              │     │ untrustworthy)│
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
         ┌──────▼──────┐
         │ Reconciler  │   ← Flag ANY disagreements between paths
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ Enrichment  │   ← Fuzzy county matching + tax calculation
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ Validators  │   ← Pure code. No AI. No mercy.
         │  • Dates    │
         │  • Amounts  │
         │  • APN      │
         │  • Status   │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │   Report    │   ← Typed findings + pass/fail verdict
         └─────────────┘
```

The key insight: **extraction** is where AI excels (understanding messy text), but **validation** must be deterministic code (catching errors AI would gloss over).

---

## Design Philosophy

### 1. Dual Extraction with Reconciliation

We extract data with **BOTH** regex and an LLM (GPT-5). The regex path is our deterministic ground truth. The LLM handles messy formatting better but can "helpfully" correct errors that should be flagged.

When both run, a **reconciler** compares them field-by-field using **type-aware comparison** (Decimal values compared numerically, dates compared as dates — not strings). Any disagreement generates a WARNING. This catches a subtle but dangerous failure mode: the LLM silently "fixing" a date or amount that was genuinely wrong in the original document.

### 2. Custom Word-to-Number Converter

We wrote our own English number parser (~60 lines) instead of using a third-party library. Why?

- Financial amounts demand **full auditability** — we need to understand every line
- The converter handles legal document patterns (`"One Million Two Hundred Thousand Dollars"`)
- It's tested with 18 test cases covering edge cases (including LLM parentheses edge cases)
- An error here could mean recording a fraudulent amount on the blockchain

### 3. Fuzzy County Resolution (Not a Lookup Table)

"S. Clara" → "Santa Clara" is NOT solved with a hardcoded mapping. Instead:

1. **Expand abbreviations**: `"S."` → `["San", "Santa", "South"]`
2. **Generate candidates**: `["San Clara", "Santa Clara", "South Clara"]`
3. **Fuzzy-match** each against the county database using `SequenceMatcher`
4. **Accept** only matches above a confidence threshold (70%)

This generalizes to any abbreviated county name (`"S. Cruz" → "Santa Cruz"`, `"N. York" → "New York"`) without maintaining a brittle lookup table.

### 4. Date Validation: Code, Not AI

```python
if deed.date_recorded < deed.date_signed:
    # ERROR: recorded before signed — impossible
```

This is a one-line check that **can never hallucinate**. We don't ask the LLM "are these dates logical?" because LLMs are notoriously bad at temporal reasoning. Pure `datetime` comparison is infinitely more reliable.

### 5. Audit Trail

Every validation run includes a **SHA-256 hash** of the original OCR text. This ensures we can later prove exactly what input produced what validation result — essential for blockchain recording integrity.

### 6. Graceful Degradation

No API key? No problem. The system runs in **regex-only mode** with full validation. The LLM is an enhancement, not a dependency. This means:
- CI/CD tests run without API costs (autouse pytest fixture mocks the LLM)
- All 84 tests complete in <1 s with zero network calls
- Production works even if the LLM provider is down
- We can verify all business rules without any network calls

---

## What It Catches

| Issue | Severity | Caught By |
|-------|----------|-----------|
| Recorded before Signed (Jan 10 < Jan 15) | **ERROR** | Date validator (pure code) |
| $1,250,000 ≠ "One Million Two Hundred Thousand" ($50K gap) | **ERROR** | Amount validator + word parser |
| APN `992-001-XA` has alpha characters | WARNING | APN format validator |
| Status `PRELIMINARY` not recordable | WARNING | Status validator |
| Grantor name has unusual dot pattern | INFO | Grantor name validator |
| Multiple grantees detected | INFO | Grantee parser |
| `S. Clara` → `Santa Clara` (fuzzy match) | INFO | County enrichment |
| LLM vs Regex extraction disagrees | WARNING | Reconciler |

---

## Quick Start

### CLI Mode (original)

```bash
# Clone the repo
git clone <repo-url>
cd bad-deed-validator

# Install dependencies
pip install -r requirements.txt

# Run the validator (no API key needed — regex-only mode)
python main.py

# Run with LLM dual-extraction (optional)
# Windows:
set OPENAI_API_KEY=sk-your-key-here
python main.py
# Linux/Mac:
OPENAI_API_KEY=sk-your-key-here python main.py

# Run the full test suite (84 tests, ~0.6 s — no network calls)
pytest tests/ -v
```

### API Mode (FastAPI)

```bash
# Start the API server (must cd into bad-deed-validator/)
cd bad-deed-validator
uvicorn api:app --reload

# Swagger UI docs auto-generated at:
# http://localhost:8000/docs
# ReDoc alternative at:
# http://localhost:8000/redoc
```

#### Validate a deed (POST JSON)
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"raw_ocr_text": "*** RECORDING REQ ***\nDoc: DEED-TRUST-0042\nCounty: S. Clara  |  State: CA\nDate Signed: 2024-01-15\nDate Recorded: 2024-01-10\nGrantor: T.E.S.L.A. Holdings LLC\nGrantee: John & Sarah Connor\nAmount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)\nAPN: 992-001-XA\nStatus: PRELIMINARY\n*** END ***"}'
```

#### Upload a file
```bash
curl -X POST http://localhost:8000/validate/file \
  -F "file=@deed_scan.txt"
```

#### Health check
```bash
curl http://localhost:8000/health
# → {"status":"healthy","version":"1.0.0","counties_loaded":3}
```

---

## Project Structure

```
bad-deed-validator/
├── main.py                        # CLI entry point — runs pipeline on sample deed
├── api.py                         # FastAPI server — REST API with Swagger docs
├── counties.json                  # Reference tax data
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
├── .gitignore
├── conftest.py                    # Pytest config + LLM mock (autouse)
│
├── deed_validator/                # Core package
│   ├── __init__.py
│   ├── models.py                  # Pydantic models — strict typing as first defense
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── extractor_regex.py         # Deterministic regex extraction (ground truth)
│   ├── extractor_llm.py           # LLM extraction via OpenAI (optional)
│   ├── enrichment.py              # Fuzzy county matching + abbreviation expansion
│   ├── word_to_number.py          # English words → Decimal (custom, auditable)
│   ├── validators.py              # ALL business rule checks (pure code, no AI)
│   └── pipeline.py                # Orchestrator — ties everything together
│
└── tests/
    ├── __init__.py
    ├── test_validators.py         # 70 unit/integration tests (15 test classes)
    └── test_api.py                # 14 FastAPI endpoint tests (4 test classes)
```

---

## Sample Output

```
  Starting Bad Deed Validator...
  Analyzing OCR-scanned deed...

========================================================================
  DEED VALIDATION REPORT
========================================================================
  Document:    DEED-TRUST-0042
  Audit Hash:  a1b2c3d4e5f6...
  Extraction:  LLM (GPT-5) + Regex cross-check
────────────────────────────────────────────────────────────────────────
  County:      S. Clara → Santa Clara
  State:       CA
  Signed:      2024-01-15
  Recorded:    2024-01-10
  Grantor:     T.E.S.L.A. Holdings LLC
  Grantee(s):  John Connor, Sarah Connor
  Amount ($):  $1,250,000.00
  Amount (w):  One Million Two Hundred Thousand Dollars
  Words -> $:  $1,200,000.00
  APN:         992-001-XA
  Status:      PRELIMINARY
  Tax Rate:    0.012
  Est. Tax:    $15,000.00
  Close Cost:  $15,075.00
────────────────────────────────────────────────────────────────────────

  ERRORS (2)
    [DATE_LOGIC_VIOLATION]
    Document was recorded (2024-01-10) BEFORE it was signed (2024-01-15).
    A deed cannot be recorded before signing. Gap: 5 day(s).

    [AMOUNT_MISMATCH]
    DISCREPANCY: Numeric amount ($1,250,000.00) does not match written
    amount "One Million Two Hundred Thousand Dollars" (=$1,200,000.00).
    Difference: $50,000.00. Both values must agree before recording.

  WARNINGS (2)
    [APN_CONTAINS_ALPHA]
    APN '992-001-XA' contains non-numeric characters: 'XA'.

    [STATUS_NOT_RECORDABLE]
    Document status is 'PRELIMINARY', not a valid recording status.

  INFO (2)
    [MULTI_PARTY_GRANTEE] Multiple grantees detected: John Connor, Sarah Connor
    [GRANTOR_NAME_UNUSUAL] Grantor name has unusual number of periods

========================================================================
  DEED REJECTED  --  2 error(s) found
========================================================================
```

---

## Dependencies

Intentionally minimal for a financial system:

| Package | Purpose | Required? |
|---------|---------|-----------|
| `pydantic` | Strict data models | Yes |
| `fastapi` | REST API framework | Yes |
| `uvicorn` | ASGI server | Yes |
| `openai` | LLM extraction | No (falls back to regex) |
| `python-dotenv` | .env file loading | No (convenience) |
| `python-multipart` | File upload support | Yes (for /validate/file) |
| `pytest` | Test runner | Dev only |
| `httpx` | FastAPI TestClient transport | Dev only |

Zero third-party dependencies for the core validation logic. `difflib.SequenceMatcher` (stdlib) handles fuzzy matching. Our custom `word_to_number` replaces the need for external number parsing libraries.

---

## Engineering Decisions FAQ

**Q: Why not just use an LLM for everything?**
A: Because LLMs hallucinate. A date comparison (`<`) never hallucinates. A Decimal subtraction never hallucinates. For blockchain-immutable financial records, we need mathematical certainty, not probabilistic confidence.

**Q: Why dual extraction instead of just regex?**
A: Regex handles the fields we know the format of. But OCR text is unpredictable — the LLM handles edge cases regex misses (weird spacing, line breaks, format variations). By running both and comparing, we get the best of both worlds.

**Q: Why a custom word-to-number converter?**
A: For the same reason banks don't use random npm packages in their transaction processing. When a bug could mean recording $1.2M instead of $1.25M on an immutable blockchain, we want to own and audit every line of that converter.

**Q: Why Pydantic?**
A: Type errors at the model boundary are infinitely cheaper than type errors in production. Pydantic gives us schema validation, serialization, and documentation in one package.
