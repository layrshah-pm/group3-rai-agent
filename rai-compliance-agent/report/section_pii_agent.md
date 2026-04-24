# Section: PII Agent Implementation
**Person 3 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. Overview

The PII (Personally Identifiable Information) Detection Agent implements the **Data Governance** pillar of the RAI framework. It scans AI-generated text for personal data before that text is delivered to any downstream system or end user, enforcing compliance with **GDPR Article 5** and **EU AI Act Article 10(5)**.

---

## 2. Technology: Microsoft Presidio

[Microsoft Presidio](https://microsoft.github.io/presidio/) is an open-source PII detection and anonymisation framework. It combines:

- **Named Entity Recognition (NER)** via a spaCy language model (`en_core_web_lg`)
- **Pattern-based recognisers** (regex + context) for structured PII such as credit card numbers, IBANs, email addresses, and phone numbers
- **Scoring engine** that assigns a confidence score (0.0–1.0) to each detected entity

Presidio was chosen over alternatives (AWS Comprehend, Google DLP) because it runs fully locally, requires no API calls, and produces deterministic outputs — critical properties for a compliance system.

---

## 3. Entity Types Detected

| Entity Type | Description | Recogniser |
|---|---|---|
| `PERSON` | Full names | NER (spaCy) |
| `EMAIL_ADDRESS` | Email addresses | Regex |
| `PHONE_NUMBER` | Phone numbers (international + domestic) | Regex + context |
| `CREDIT_CARD` | 16-digit card numbers (Luhn-validated) | Regex |
| `IBAN_CODE` | International bank account numbers | Regex |
| `LOCATION` | Cities, addresses, postcodes | NER (spaCy) |
| `NRP` | Nationalities, religions, political groups | NER (spaCy) |

`DATE_TIME` was **excluded** from the entity list. Presidio fires this recogniser on date ranges such as "2018–2023", which appear in legitimate model metadata and are not GDPR-relevant personal data. Including `DATE_TIME` caused false positives in Scenario 1 (clean compliant text), so it was removed.

---

## 4. Implementation Design

### Module-level initialisation

```python
_analyzer = AnalyzerEngine()      # spaCy model loaded once
_anonymizer = AnonymizerEngine()
```

Both engines are created at module import time, not inside the node function. The `AnalyzerEngine` loads a 740MB spaCy model; creating it per request would add ~3s to every run. Module-level initialisation ensures the cost is paid once.

### Confidence threshold

A confidence threshold of **0.6** is applied. Entities with `score < 0.6` are silently discarded. This threshold was empirically tuned across the test suite: lower values (0.5) caused false positives on generic words; higher values (0.7) missed some phone number formats. The `CONFIDENCE_THRESHOLD` constant is defined at the top of the module for easy adjustment.

### Redaction strategy

When PII is detected, the `AnonymizerEngine` replaces each entity with a labelled placeholder:

```
John Smith → <PERSON>
john.smith@email.com → <EMAIL_ADDRESS>
650-253-0000 → <PHONE_NUMBER>
```

Redaction is performed *inside* the PII agent (not in the correction node), so the `redacted_text` is immediately available in `pii_result`. The correction node uses this pre-computed value directly — no LLM call needed for PII correction. This is deterministic and instant.

---

## 5. Test Results

14 unit tests in `tests/test_pii_agent.py` — all pass:

| Test | Input | Result |
|---|---|---|
| Person name | "Hello, my name is Sarah Johnson" | PERSON detected |
| Email | "email me at priya.sharma@example.com" | EMAIL_ADDRESS detected |
| Phone (domestic) | "Reach me at phone 650-253-0000" | PHONE_NUMBER detected |
| Phone (international) | "+44 20 7946 0958 for callbacks" | PHONE_NUMBER detected |
| Credit card | "card number 4111 1111 1111 1111" | CREDIT_CARD detected |
| IBAN | "IBAN GB82 WEST 1234 5698 7654 32" | IBAN_CODE detected |
| Location | "42 Park Street, London, EC1A 1BB" | has_pii = True |
| Clean text | "loan application reviewed based on criteria" | passed = True, entities = [] |
| Multiple PII | name + email + phone in one text | ≥ 2 entities detected |
| Schema contract | any input | all required keys present |
| Audit log | any input | node="pii_agent", action="PII_SCAN" |
| Violations empty | clean text | violations = [] |
| Violations set | text with PII | "PII_DETECTED" in violations |

---

## 6. Limitations

- **Non-English PII**: Presidio's `en_core_web_lg` model is trained on English text. Hindi names, Indian phone number formats, and Devanagari script are not reliably detected. Multi-lingual support is deferred to future work.
- **Indirect identification**: Presidio does not detect quasi-identifiers (e.g. a rare combination of age + occupation + postcode that could re-identify an individual). Full k-anonymity analysis is outside scope.
- **Context dependence**: The PERSON recogniser relies on capitalisation and context. Names embedded in all-caps text or unusual sentence structures may be missed.
