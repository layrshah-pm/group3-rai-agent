# Section: Regulatory Mapping and LLM Prompt Design
**Person 5 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. Overview

The Policy Compliance Agent implements the **Strategic Alignment** and **Org Oversight** pillars of the RAI framework. It evaluates AI-generated text against a set of regulatory criteria derived from the **EU AI Act** and **NIST AI Risk Management Framework**, using a locally-running Large Language Model (Ollama/Gemma4) with Pydantic-structured output.

The Correction Node is also owned by this section — it handles the LLM-based text rewriting for policy and bias violations.

---

## 2. Regulatory Criteria

Eight criteria were identified by reviewing EU AI Act Articles 8–51 and NIST AI RMF core functions. Each maps to a specific regulatory obligation for high-risk AI systems (which includes credit scoring under Annex III):

| ID | Obligation | Primary Reference |
|---|---|---|
| `TRANSPARENCY` | AI-generated content must be identifiable as such | EU AI Act Article 50 / NIST GOVERN 1.1 |
| `EXPLAINABILITY` | Decisions must be explainable to affected individuals | EU AI Act Article 13 / NIST MEASURE 2.5 |
| `HUMAN_OVERSIGHT` | Mechanism for human review must be available | EU AI Act Article 14 / NIST MANAGE 1.3 |
| `DATA_QUALITY` | Data sources and limitations must be acknowledged | EU AI Act Article 10 / NIST MAP 3.5 |
| `NON_DISCRIMINATION` | Output must not contain discriminatory content | EU AI Act Article 5(1)(b) / NIST MEASURE 2.2 |
| `PRIVACY` | Output must not expose unnecessary personal data | GDPR Article 5 / EU AI Act Article 10(5) |
| `ACCOUNTABILITY` | Responsible party for the AI decision must be identifiable | EU AI Act Article 25 / NIST GOVERN 6.1 |
| `ROBUSTNESS` | Uncertainty or confidence levels should be communicated | EU AI Act Article 15 / NIST MEASURE 2.6 |

---

## 3. LLM Setup: Ollama/Gemma4

**Model:** `gemma4:latest` running locally via Ollama.

Gemma4 was selected because:
- It runs on consumer hardware (no GPU required for inference at this text length)
- It reliably follows JSON output instructions for structured extraction
- No API costs or data privacy concerns (no data leaves the machine)
- Temperature = 0 ensures deterministic outputs for compliance auditing

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="gemma4:latest", temperature=0)
```

---

## 4. Structured Output via Pydantic

Rather than parsing raw LLM text, the agent uses LangChain's `with_structured_output()` method, which forces the model to return a JSON object matching a Pydantic schema:

```python
class PolicyViolation(BaseModel):
    id: str                    # must match one of VALID_IDS
    description: str           # one sentence
    severity: Literal["low", "medium", "high"]
    article_reference: str
    remediation: str           # one sentence fix

class PolicyCheckOutput(BaseModel):
    violations: list[PolicyViolation]
    summary: str
```

After receiving the structured response, a validation step filters violations to `VALID_IDS` — the set of 8 known criterion IDs. This prevents the LLM from hallucinating criterion names not in the defined set.

---

## 5. Prompt Engineering

The prompt design went through multiple iterations to eliminate false positives. The key insight was that Gemma4 tends to be overly cautious: it flags `PRIVACY` for aggregate statistics ("36 months"), `ACCOUNTABILITY` when no phone number is present, and `DATA_QUALITY` unless exhaustive source documentation is provided.

The final prompt includes explicit negative rules:

```
IMPORTANT RULES:
- Flag a criterion ONLY if it is CLEARLY AND EXPLICITLY violated.
- Do NOT flag PRIVACY unless the text contains real personal identifiers
  (names, emails, phone numbers, ID numbers). Aggregate statistics and
  anonymised metrics are NOT privacy violations.
- Do NOT flag ACCOUNTABILITY if a responsible organisation is named,
  even without a specific contact number.
- Do NOT flag DATA_QUALITY if the text names at least one data source
  and mentions at least one limitation.
- If in doubt, do NOT flag. Return an empty violations array for borderline cases.
```

This mirrors the legal principle of *strict construction* — a compliance auditor should not penalise for things that are not explicitly required, and should err on the side of a passing verdict when the evidence is ambiguous.

### Prompt structure

```
You are a Responsible AI compliance auditor evaluating AI-generated text.

TEXT TO EVALUATE:
"""[current_text]"""

REGULATORY CRITERIA:
[list of 8 criteria with IDs and references]

IMPORTANT RULES:
[negative rules as above]

For each clear violation return:
  id / description / severity / article_reference / remediation

Return a JSON object with:
  violations: [...] (empty array [] if none)
  summary: one paragraph plain-English summary

Return ONLY valid JSON. No markdown fences, no text outside the JSON.
```

---

## 6. Correction Node

The correction node handles five violation types:

| Violation | Strategy |
|---|---|
| `PII_DETECTED` | Returns `pii_result["redacted_text"]` directly — no LLM call |
| `BIAS_DETECTED` | LLM prompt: rewrite to be demographically neutral |
| `POLICY_TRANSPARENCY` | LLM prompt: add AI disclosure sentence |
| `POLICY_EXPLAINABILITY` | LLM prompt: add two-sentence explanation of decision factors |
| `POLICY_HUMAN_OVERSIGHT` | LLM prompt: add sentence offering human review |
| All others | Generic LLM prompt referencing the violation ID |

The PII case is handled without the LLM because Presidio's redaction is deterministic, instantaneous, and more accurate than generative rewriting. Calling an LLM to "remove personal data" would risk incomplete redaction.

---

## 7. Test Results

**Policy agent:** 8 unit tests — all pass (live LLM, 273s total)

| Test | Input | Expected | Result |
|---|---|---|---|
| Clean compliant text | Full Scenario 1 text | zero high violations | PASS |
| No AI disclosure | Bare denial notice | TRANSPARENCY flagged | PASS |
| No explanation | "AI-generated. Application rejected." | EXPLAINABILITY flagged | PASS |
| No human oversight | "This decision is automated and final." | HUMAN_OVERSIGHT flagged | PASS |
| Discriminatory language | "Female applicants present higher risk." | NON_DISCRIMINATION flagged | PASS |
| Schema contract | any | all keys present, severity in enum | PASS |
| No hallucinated IDs | any | all violation IDs in VALID_IDS | PASS |
| Audit log | any | node="policy_agent", action="POLICY_CHECK" | PASS |

**Correction node:** 6 unit tests (mocked LLM) — all pass

---

## 8. Limitations

- **LLM non-determinism at scale**: Even with `temperature=0`, Gemma4 occasionally produces slightly different responses on the same input across runs. The explicit negative rules in the prompt reduce but do not eliminate this variability. The `validate_demo.py` script quantifies this.
- **No RAG pipeline**: The policy agent evaluates text against static criteria. A production system would embed the full EU AI Act text and use retrieval-augmented generation to cite specific articles more precisely.
- **English only**: Policy prompts and criteria descriptions are in English. Non-English AI outputs are not evaluated.
