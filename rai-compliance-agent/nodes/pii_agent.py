"""
nodes/pii_agent.py
------------------
PII Detection Agent — Microsoft Presidio implementation.

Scans current_text for Personally Identifiable Information, builds a
structured result, and appends one entry to the audit log and step_trace.
"""

from datetime import datetime, timezone
from state import ComplianceState, PIIResult

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialise once at module level — expensive to recreate per call
_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

ENTITIES_TO_DETECT = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "LOCATION",
    "NRP",           # nationalities, religions, political groups
]

# DATE_TIME excluded: Presidio over-fires on date ranges (e.g. "2018-2023"),
# which are not GDPR-relevant personal data in this context.

# Raised from 0.6 — reduces Presidio over-firing on policy/governance text.
CONFIDENCE_THRESHOLD = 0.75

# Minimum printable alphanumeric characters required in an entity value.
# Filters out box-drawing chars (━ ─ │), lone digits, punctuation artifacts.
MIN_ALNUM_CHARS = 3

# Exact-match denylist: known false positives in governance/policy documents.
# These are Presidio mis-classifications of formatting artifacts, acronyms,
# and generic terms that carry no personal data risk.
_FP_DENYLIST: set[str] = {
    "tier 1", "tier 2", "tier 3",           # risk tiers, not locations
    "model drift detection",                  # technical term, not a location
    "aiec",                                   # committee acronym
    "indian",                                 # nationality adjective, not a person
}

import re as _re
_ALNUM_RE = _re.compile(r"[a-zA-Z0-9]")


def _is_false_positive(value: str) -> bool:
    """
    Returns True when a Presidio hit should be suppressed.

    Rules:
    1. Value is too short in real alphanumeric content (catches ━, ─, lone digits).
    2. Value (lowercased, stripped) is in the known false-positive denylist.
    3. Value consists entirely of non-alphanumeric characters (formatting lines).
    """
    stripped = value.strip()
    alnum_count = len(_ALNUM_RE.findall(stripped))
    if alnum_count < MIN_ALNUM_CHARS:
        return True
    if stripped.lower() in _FP_DENYLIST:
        return True
    # Pure symbol/punctuation strings — e.g. "━\n1" has only 1 alnum char
    return False


def pii_agent_node(state: ComplianceState) -> dict:
    """
    Scans current_text for PII using Microsoft Presidio.
    Returns pii_result, step_trace entry, and appends to audit_log.
    """
    print(f"\n[PII AGENT] Scanning for PII...")

    text = state["current_text"]

    results = _analyzer.analyze(
        text=text,
        entities=ENTITIES_TO_DETECT,
        language="en",
    )

    entities_found = [
        {
            "entity_type": r.entity_type,
            "value": text[r.start:r.end],
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
        }
        for r in results
        if r.score >= CONFIDENCE_THRESHOLD
        and not _is_false_positive(text[r.start:r.end])
    ]

    has_pii = len(entities_found) > 0

    if has_pii:
        anonymized = _anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )
        redacted_text = anonymized.text
    else:
        redacted_text = None

    pii_result: PIIResult = {
        "has_pii": has_pii,
        "entities_found": entities_found,
        "redacted_text": redacted_text,
        "passed": not has_pii,
    }

    new_violations = ["PII_DETECTED"] if has_pii else []

    status = "FAIL — PII detected" if has_pii else "PASS"
    print(f"[PII AGENT] Result: {status}")
    for e in entities_found:
        print(f"           → {e['entity_type']}: '{e['value']}' (confidence {e['score']})")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "pii_agent",
        "action": "PII_SCAN",
        "result": "fail" if has_pii else "pass",
        "detail": {
            "entities_found": len(entities_found),
            "entity_types": list({e["entity_type"] for e in entities_found}),
            "redacted": has_pii,
        },
    }

    summary = (
        f"Found {len(entities_found)} PII entities: "
        f"{', '.join(set(e['entity_type'] for e in entities_found))}."
        if entities_found
        else "No PII detected. Text is clean."
    )

    step_entry = {
        "step":  "pii_agent",
        "label": "PII Detection",
        "status": "fail" if has_pii else "pass",
        "prompt": None,
        "response": {
            "entities_scanned_for": ENTITIES_TO_DETECT,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "entities_found": entities_found,
            "has_pii": has_pii,
            "redacted_text": redacted_text,
        },
        "summary": summary,
    }

    return {
        "pii_result": pii_result,
        "violations": new_violations,
        "current_node": "pii_agent",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }
