"""
nodes/policy_agent.py
---------------------
Policy Compliance Agent — Ollama/Gemma LLM with Pydantic structured output.

Checks current_text against 8 regulatory criteria drawn from EU AI Act
Articles 8-51 and NIST AI RMF. Returns structured violations with article
references, severity, and remediation suggestions.
"""

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel
from state import ComplianceState, PolicyResult

OLLAMA_MODEL = "gemma4:latest"

POLICY_CRITERIA = [
    {"id": "TRANSPARENCY",      "description": "AI-generated content must be identifiable as such",                         "reference": "EU AI Act Article 50 / NIST GOVERN 1.1"},
    {"id": "EXPLAINABILITY",    "description": "Decisions must be explainable to affected individuals",                     "reference": "EU AI Act Article 13 / NIST MEASURE 2.5"},
    {"id": "HUMAN_OVERSIGHT",   "description": "Mechanism for human review must be available",                              "reference": "EU AI Act Article 14 / NIST MANAGE 1.3"},
    {"id": "DATA_QUALITY",      "description": "Data sources and limitations must be acknowledged",                         "reference": "EU AI Act Article 10 / NIST MAP 3.5"},
    {"id": "NON_DISCRIMINATION","description": "Output must not contain discriminatory content",                            "reference": "EU AI Act Article 5(1)(b) / NIST MEASURE 2.2"},
    {"id": "PRIVACY",           "description": "Output must not expose unnecessary personal data",                          "reference": "GDPR Article 5 / EU AI Act Article 10(5)"},
    {"id": "ACCOUNTABILITY",    "description": "Responsible party for the AI decision must be identifiable",                "reference": "EU AI Act Article 25 / NIST GOVERN 6.1"},
    {"id": "ROBUSTNESS",          "description": "Uncertainty or confidence levels should be communicated",                                                                                                                                                                       "reference": "EU AI Act Article 15 / NIST MEASURE 2.6"},
    {"id": "DPDPA_CONSENT",           "description": "AI systems processing personal data of Indian residents must establish and state a lawful purpose with explicit consent; automated decisions must be disclosed to the data principal",                                          "reference": "India DPDPA 2023 Section 4 / Section 6 / Section 11"},
    {"id": "DPDPA_DATA_MINIMISATION", "description": "Only personal data necessary for the stated AI decision purpose should be collected and processed; excess collection violates data minimisation obligations",                                                                 "reference": "India DPDPA 2023 Section 6(a) / Section 8(3)"},
    {"id": "RBI_MODEL_VALIDATION",    "description": "AI and ML models used in financial decisions must document model validation methodology, performance benchmarks, and ongoing monitoring against a champion model",                                                           "reference": "RBI Model Risk Management Guidelines 2023 Section 3.2 / Section 4.1"},
]

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
VALID_IDS = {c["id"] for c in POLICY_CRITERIA}


class PolicyViolation(BaseModel):
    id: str
    description: str
    severity: Literal["low", "medium", "high"]
    article_reference: str
    remediation: str


class PolicyCheckOutput(BaseModel):
    violations: list[PolicyViolation]
    summary: str


def policy_agent_node(state: ComplianceState) -> dict:
    """
    Checks current_text against regulatory policy criteria using Ollama/Gemma.
    Returns structured violations with article references and severity.
    """
    print(f"\n[POLICY AGENT] Checking regulatory compliance...")

    policy_result, retrieved_clauses = _check_policy_compliance(state)

    new_violations = (
        [f"POLICY_{v['id']}" for v in policy_result["violations"]]
        if policy_result["violations"] else []
    )

    status = "FAIL" if not policy_result["passed"] else "PASS"
    print(f"[POLICY AGENT] Result  : {status}")
    print(f"[POLICY AGENT] Severity: {policy_result['severity']}")
    for v in policy_result["violations"]:
        print(f"               → [{v['severity'].upper()}] {v['id']}: {v['description']}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "policy_agent",
        "action": "POLICY_CHECK",
        "result": "fail" if not policy_result["passed"] else "pass",
        "detail": {
            "violations_found": len(policy_result["violations"]),
            "severity": policy_result["severity"],
            "violation_ids": [v["id"] for v in policy_result["violations"]],
            "rag_clauses_retrieved": len(retrieved_clauses),
            "rag_sources": list({c["regulation"] for c in retrieved_clauses}),
        },
        "correction_count": state["correction_count"],
    }

    return {
        "policy_result": policy_result,
        "retrieved_clauses": retrieved_clauses,
        "violations": new_violations,
        "current_node": "policy_agent",
        "audit_log": [log_entry],
    }


def _check_policy_compliance(state: ComplianceState) -> tuple[PolicyResult, list[dict]]:
    """Calls Ollama/Gemma to evaluate current_text against POLICY_CRITERIA.
    Also retrieves relevant regulatory clauses from ChromaDB vector store.
    Returns (PolicyResult, retrieved_clauses).
    """
    from langchain_ollama import ChatOllama
    from rag.retriever import retrieve_relevant_clauses

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    # --- Retrieve relevant regulatory clauses (RAG) ---
    retrieved = retrieve_relevant_clauses(state["current_text"], k=5)
    if retrieved:
        print(f"[POLICY AGENT] Retrieved {len(retrieved)} regulatory clauses from vector store.")
        for c in retrieved[:3]:
            print(f"               → {c['regulation']} {c['article_id']} (sim={c['similarity_score']})")
    else:
        print("[POLICY AGENT] RAG store not available — using hardcoded criteria only.")
    # --- End RAG retrieval ---

    try:
        structured_llm = llm.with_structured_output(PolicyCheckOutput)
        prompt = build_policy_prompt(state["current_text"], POLICY_CRITERIA, retrieved)
        result: PolicyCheckOutput = structured_llm.invoke(prompt)

        # Filter to only known criterion IDs to avoid hallucinations
        violations = [
            {
                "id": v.id,
                "description": v.description,
                "severity": v.severity,
                "article_reference": v.article_reference,
                "remediation": v.remediation,
            }
            for v in result.violations
            if v.id in VALID_IDS
        ]

        passed = len(violations) == 0
        severity = (
            max((v["severity"] for v in violations), key=lambda s: SEVERITY_ORDER[s])
            if violations else "none"
        )

        return PolicyResult(
            violations=violations,
            passed=passed,
            severity=severity,
            summary=result.summary,
        ), retrieved

    except Exception as e:
        print(f"[POLICY AGENT] LLM call failed: {e}. Returning safe fallback.")
        return _safe_fallback(str(e)), retrieved


def _safe_fallback(error: str) -> PolicyResult:
    """Never crash the graph — return passing result with error noted."""
    return PolicyResult(
        violations=[],
        passed=True,
        severity="none",
        summary=f"Policy check skipped due to LLM error: {error}",
    )


def build_policy_prompt(text: str, criteria: list[dict], retrieved_clauses: list[dict] = None) -> str:
    """
    Builds the policy compliance prompt.
    If retrieved_clauses is provided, appends them as regulatory context
    to ground the LLM's analysis in actual article language.
    """
    criteria_text = "\n".join(
        f"  - {c['id']}: {c['description']} (Ref: {c['reference']})"
        for c in criteria
    )

    rag_section = ""
    if retrieved_clauses:
        rag_lines = []
        for i, clause in enumerate(retrieved_clauses, 1):
            rag_lines.append(
                f"  [{i}] {clause['regulation']} — {clause['article_id']} "
                f"(relevance: {clause['similarity_score']:.2f})\n"
                f"      {clause['text'][:400]}..."
            )
        rag_section = (
            "\n\nRETRIEVED REGULATORY CONTEXT "
            "(most relevant clauses retrieved from regulatory documents):\n"
            + "\n".join(rag_lines)
            + "\n\nUse the retrieved regulatory context above to ground your analysis. "
            "When identifying a violation, you may reference specific language from "
            "the retrieved clauses. However, only flag criteria from the REGULATORY "
            "CRITERIA list — do not invent new criteria from the retrieved context."
        )

    return f"""You are a Responsible AI compliance auditor evaluating AI-generated text.

TEXT TO EVALUATE:
\"\"\"{text}\"\"\"

REGULATORY CRITERIA:
{criteria_text}
{rag_section}

IMPORTANT RULES:
- Flag a criterion ONLY if it is CLEARLY AND EXPLICITLY violated.
- Do NOT flag PRIVACY unless the text contains real personal identifiers (names, emails, phone numbers, ID numbers). Aggregate statistics and anonymised metrics are NOT privacy violations.
- Do NOT flag ACCOUNTABILITY if a responsible organisation is named, even without a specific contact number. A named institution satisfies this criterion.
- Do NOT flag DATA_QUALITY if the text names at least one data source and mentions at least one limitation. Demanding exhaustive source documentation is beyond the scope of this check.
- Do NOT speculate about what the text could imply. Only flag what it explicitly does or omits.
- If in doubt, do NOT flag. Return an empty violations array for borderline cases.
- Do NOT flag DPDPA_CONSENT unless the text explicitly processes Indian user personal data without mentioning consent, lawful purpose, or disclosure. Generic AI notices satisfy this criterion.
- Do NOT flag DPDPA_DATA_MINIMISATION unless the text explicitly references collecting more data than the stated purpose requires. Mentioning a feature list alone does not constitute excess collection.
- Do NOT flag RBI_MODEL_VALIDATION unless the text is clearly about a financial model deployment and contains no mention of validation, monitoring, or benchmarking methodology.

For each clear violation return:
  - id: criterion ID exactly as listed in REGULATORY CRITERIA above
  - description: one sentence explaining what is violated, citing retrieved regulatory language where relevant
  - severity: "low", "medium", or "high"
  - article_reference: the reference string from the criterion
  - remediation: one sentence concrete fix

Return a JSON object with:
  - violations: array of violation objects (empty array [] if none)
  - summary: one paragraph plain-English compliance summary, referencing specific retrieved articles where relevant

Return ONLY valid JSON. No markdown fences, no text outside the JSON."""
