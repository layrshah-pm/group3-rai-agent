"""
nodes/pii_agent.py
------------------
PII Detection Agent — Microsoft Presidio implementation.

Scans current_text for Personally Identifiable Information, builds a
structured result, and appends one entry to the audit log.
Redaction is performed here so correction_node can use it directly.
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

CONFIDENCE_THRESHOLD = 0.6


def pii_agent_node(state: ComplianceState) -> dict:
    """
    Scans current_text for PII using Microsoft Presidio.
    Returns pii_result and appends to audit_log.
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
    ]

    has_pii = len(entities_found) > 0

    # Redact immediately so correction_node can use it without an LLM call
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
        "correction_count": state["correction_count"],
    }

    return {
        "pii_result": pii_result,
        "violations": new_violations,
        "current_node": "pii_agent",
        "audit_log": [log_entry],
    }
