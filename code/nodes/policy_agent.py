"""
nodes/policy_agent.py
---------------------
Policy Compliance Agent — Ollama/Gemma LLM with Pydantic structured output.

Two modes:
  - "text" / "model_output": checks AI-generated text against 11 regulatory criteria
  - "policy_document": checks a company policy document against 5 RAI pillar criteria
"""

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel
from state import ComplianceState, PolicyResult

OLLAMA_MODEL = "gemma4:latest"

POLICY_DOCUMENT_CRITERIA = [
    {
        "id": "GOVERNANCE_ACCOUNTABILITY",
        "pillar": "governance",
        "pillar_key": "strategic_alignment",
        "description": "Policy must name a responsible owner or AI Ethics Committee for AI systems, with a human override mechanism",
        "reference": "EU AI Act Art.17 / RBI MRM Sec.4 / DPDPA 2023 Sec.8",
        "check_questions": [
            "Is a named individual or committee accountable for AI governance?",
            "Is there a model inventory or register requirement?",
            "Is human override of AI decisions mandated?",
        ],
    },
    {
        "id": "FAIRNESS_BIAS",
        "pillar": "fairness",
        "pillar_key": "model_governance",
        "description": "Policy must require pre-deployment bias testing, training data assessment, and fairness metric selection",
        "reference": "EU AI Act Art.10 / RBI MRM Sec.7 / DPDPA 2023 Sec.8",
        "check_questions": [
            "Does the policy require bias/fairness testing before deployment?",
            "Is training data assessment for historical bias required?",
            "Are proxy variable reviews mandated?",
        ],
    },
    {
        "id": "TRANSPARENCY_EXPLAINABILITY",
        "pillar": "transparency",
        "pillar_key": "org_oversight",
        "description": "Policy must require disclosure of AI use in decisions and provide an explanation standard for affected individuals",
        "reference": "EU AI Act Art.13 / DPDPA 2023 Sec.11",
        "check_questions": [
            "Does the policy require informing people when AI influences their decisions?",
            "Is there an explanation standard (e.g. counterfactual, plain-language)?",
            "Are model cards or documentation required?",
        ],
    },
    {
        "id": "ROBUSTNESS_MONITORING",
        "pillar": "robustness",
        "pillar_key": "continuous_monitoring",
        "description": "Policy must define performance monitoring, drift detection, periodic review, and an incident response protocol",
        "reference": "EU AI Act Art.72 / RBI MRM Sec.8",
        "check_questions": [
            "Are performance thresholds and monitoring protocols defined?",
            "Is there a requirement for periodic model review?",
            "Is there an incident response or escalation procedure?",
        ],
    },
    {
        "id": "PRIVACY_DATA_STEWARDSHIP",
        "pillar": "privacy",
        "pillar_key": "data_governance",
        "description": "Policy must require DPDPA-compliant consent for AI training data, purpose limitation, and vendor AI data assessments",
        "reference": "DPDPA 2023 Sec.6 / Sec.8 / EU AI Act Art.10(5)",
        "check_questions": [
            "Does the policy address consent for using personal data in AI training?",
            "Is purpose limitation for AI data use defined?",
            "Are third-party/vendor AI tools subject to data assessment?",
        ],
    },
]

VALID_DOC_IDS = {c["id"] for c in POLICY_DOCUMENT_CRITERIA}

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


class PillarDocumentScore(BaseModel):
    id: str
    compliance_level: int          # 0=absent, 1=minimal, 2=partial, 3=adequate
    gap_description: str           # what is MISSING (empty string if score == 3)
    evidence_from_policy: str      # direct quote or section reference that was evaluated
    pass_reason: str               # one sentence explaining why it passed (if score >= 2)


class PolicyDocumentOutput(BaseModel):
    pillar_scores: list[PillarDocumentScore]
    summary: str


def policy_agent_node(state: ComplianceState) -> dict:
    """
    Checks current_text against regulatory policy criteria using Ollama/Gemma.
    Branches on input_type: "policy_document" uses 5-pillar criteria;
    all other modes use the original 11-criterion text audit.
    """
    if state.get("input_type") == "policy_document":
        return _policy_document_node(state)
    return _text_policy_node(state)


def _text_policy_node(state: ComplianceState) -> dict:
    """Original text/model_output path — 11-criterion audit."""
    print(f"\n[POLICY AGENT] Checking regulatory compliance (text mode)...")

    policy_result, retrieved_clauses, prompt_text, raw_response = \
        _check_policy_compliance(state)

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
    }

    pvs = policy_result.get("violations", [])
    summary = (
        f"Found {len(pvs)} violation(s): "
        f"{', '.join(v['id'] for v in pvs[:3])}{'...' if len(pvs) > 3 else ''}."
        if pvs
        else "All 11 policy criteria satisfied."
    )

    step_entry = {
        "step":  "policy_agent",
        "label": "Policy Compliance (LLM)",
        "status": "fail" if not policy_result["passed"] else "pass",
        "prompt": prompt_text,
        "response": raw_response,
        "summary": summary,
    }

    return {
        "policy_result": policy_result,
        "retrieved_clauses": retrieved_clauses,
        "violations": new_violations,
        "current_node": "policy_agent",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }


def _policy_document_node(state: ComplianceState) -> dict:
    """Policy document audit path — 5-pillar RAI governance criteria."""
    print(f"\n[POLICY AGENT] Auditing policy document against 5 RAI pillars...")

    policy_result, pillar_gaps, retrieved_clauses, prompt_text, raw_response = \
        _check_policy_document_compliance(state)

    new_violations = (
        [f"POLICY_{v['id']}" for v in policy_result["violations"]]
        if policy_result["violations"] else []
    )

    status = "FAIL" if not policy_result["passed"] else "PASS"
    print(f"[POLICY AGENT] Result: {status}")
    for criterion_id, gap_data in pillar_gaps.items():
        level = gap_data.get("level", 0)
        print(f"               → {criterion_id}: {level}/3 — {gap_data.get('gap', '')[:80]}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "policy_agent",
        "action": "POLICY_DOCUMENT_AUDIT",
        "result": "fail" if not policy_result["passed"] else "pass",
        "detail": {
            "mode": "policy_document",
            "pillars_evaluated": len(pillar_gaps),
            "pillars_failing": len([g for g in pillar_gaps.values() if g.get("level", 0) < 2]),
            "rag_clauses_retrieved": len(retrieved_clauses),
        },
    }

    failing_pillars = [cid for cid, g in pillar_gaps.items() if g.get("level", 0) < 2]
    summary = (
        f"Policy document fails {len(failing_pillars)} pillar(s): "
        f"{', '.join(failing_pillars[:3])}{'...' if len(failing_pillars) > 3 else ''}."
        if failing_pillars
        else "Policy document satisfies all 5 RAI pillar criteria."
    )

    step_entry = {
        "step":  "policy_agent",
        "label": "Policy Document Audit (LLM)",
        "status": "fail" if not policy_result["passed"] else "pass",
        "prompt": prompt_text,
        "response": raw_response,
        "summary": summary,
    }

    return {
        "policy_result": policy_result,
        "pillar_gaps": pillar_gaps,
        "retrieved_clauses": retrieved_clauses,
        "violations": new_violations,
        "current_node": "policy_agent",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }


def _check_policy_document_compliance(
    state: ComplianceState,
) -> tuple[PolicyResult, dict, list[dict], str, dict]:
    """Calls Ollama/Gemma to evaluate a policy document against the 5 RAI pillar criteria.
    Returns (PolicyResult, pillar_gaps, retrieved_clauses, prompt_text, raw_response_dict).
    """
    from langchain_ollama import ChatOllama
    from rag.retriever import retrieve_relevant_clauses

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    retrieved = retrieve_relevant_clauses(state["current_text"], k=5)
    if retrieved:
        print(f"[POLICY AGENT] Retrieved {len(retrieved)} regulatory clauses from vector store.")
    else:
        print("[POLICY AGENT] RAG store not available — using hardcoded criteria only.")

    try:
        structured_llm = llm.with_structured_output(PolicyDocumentOutput)
        prompt_text = _build_policy_document_prompt(state["current_text"], POLICY_DOCUMENT_CRITERIA, retrieved)
        result: PolicyDocumentOutput = structured_llm.invoke(prompt_text)

        raw_response_dict = {
            "pillar_scores": [s.model_dump() for s in result.pillar_scores],
            "summary": result.summary,
        }

        # Build pillar_gaps dict — filter to valid IDs only
        pillar_gaps: dict = {}
        for score in result.pillar_scores:
            if score.id not in VALID_DOC_IDS:
                continue
            level = max(0, min(3, score.compliance_level))
            criterion = next(c for c in POLICY_DOCUMENT_CRITERIA if c["id"] == score.id)
            pillar_gaps[score.id] = {
                "level": level,
                "gap": score.gap_description,
                "pass_reason": getattr(score, "pass_reason", ""),
                "evidence": getattr(score, "evidence_from_policy", ""),
                "pillar_key": criterion["pillar_key"],
                "reference": criterion["reference"],
                "description": criterion["description"],
            }

        # Fill in any missing criteria with level 0
        for c in POLICY_DOCUMENT_CRITERIA:
            if c["id"] not in pillar_gaps:
                pillar_gaps[c["id"]] = {
                    "level": 0,
                    "gap": "Not assessed.",
                    "pass_reason": "",
                    "evidence": "",
                    "pillar_key": c["pillar_key"],
                    "reference": c["reference"],
                    "description": c["description"],
                }

        # Build violations for pillars scoring < 2
        violations = []
        for c in POLICY_DOCUMENT_CRITERIA:
            gap_data = pillar_gaps[c["id"]]
            if gap_data["level"] < 2:
                severity = "high" if gap_data["level"] == 0 else "medium"
                violations.append({
                    "id": c["id"],
                    "description": gap_data["gap"] or c["description"],
                    "severity": severity,
                    "article_reference": c["reference"],
                    "remediation": f"Update policy to address: {c['description']}",
                })

        passed = len(violations) == 0
        severity = "none"
        if violations:
            severity = max((v["severity"] for v in violations), key=lambda s: {"none": 0, "low": 1, "medium": 2, "high": 3}[s])

        return PolicyResult(
            violations=violations,
            passed=passed,
            severity=severity,
            summary=result.summary,
        ), pillar_gaps, retrieved, prompt_text, raw_response_dict

    except Exception as e:
        print(f"[POLICY AGENT] Policy document LLM call failed: {e}. Returning safe fallback.")
        fallback_result, placeholder_prompt, placeholder_response = _safe_fallback(str(e))
        fallback_gaps = {
            c["id"]: {"level": 0, "gap": f"Assessment failed: {e}", "pillar_key": c["pillar_key"]}
            for c in POLICY_DOCUMENT_CRITERIA
        }
        return fallback_result, fallback_gaps, [], placeholder_prompt, placeholder_response


def _build_policy_document_prompt(text: str, criteria: list[dict], retrieved_clauses: list[dict] = None) -> str:
    """Builds the policy document audit prompt for the 5 RAI pillar criteria."""
    criteria_text = ""
    for c in criteria:
        questions = "\n".join(f"    • {q}" for q in c["check_questions"])
        criteria_text += (
            f"\n  {c['id']}:\n"
            f"    What we need: {c['description']}\n"
            f"    Regulatory reference: {c['reference']}\n"
            f"    Check questions:\n{questions}\n"
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
            "\n\nRETRIEVED REGULATORY CONTEXT:\n"
            + "\n".join(rag_lines)
            + "\n\nUse the retrieved context to ground your analysis. "
            "When citing a gap, reference the specific regulatory article."
        )

    return f"""You are a Responsible AI compliance auditor. Your job is to evaluate the policy document below against 5 RAI governance pillars and produce a structured, evidence-based assessment.

POLICY DOCUMENT TEXT:
\"\"\"{text[:4000]}\"\"\"

EVALUATION CRITERIA (assess each of the 5 pillars):
{criteria_text}
{rag_section}

SCORING RUBRIC:
3 = Fully addressed — policy explicitly and specifically covers ALL key aspects of this pillar with named owners, defined thresholds, or concrete procedures.
2 = Partially addressed — policy mentions the area but is missing at least one specific requirement (e.g. no defined thresholds, no named owner, no procedure).
1 = Minimally mentioned — only a brief or vague reference; no actionable commitment.
0 = Absent — policy does not address this pillar at all.

IMPORTANT SCORING RULES:
- Score 3 only if ALL three check questions are clearly answered by the policy.
- Score 2 if some but not all check questions are answered.
- Score 1 if the topic is mentioned but without substance.
- Score 0 if the topic is completely absent.
- Do NOT give a 3 unless the policy has SPECIFICS (e.g. named committee, defined metric, stated timeline).

For EACH of the 5 criteria, return an object with these EXACT fields:
  - id: the criterion ID exactly as listed (e.g. "GOVERNANCE_ACCOUNTABILITY")
  - compliance_level: integer 0, 1, 2, or 3 per the rubric above
  - evidence_from_policy: a SHORT direct quote (max 60 words) from the policy text that is most relevant to this pillar — or "Not found in policy." if absent
  - gap_description: one specific sentence describing EXACTLY what requirement is missing or inadequate (reference the regulatory article where possible). Write an empty string "" only if score is 3.
  - pass_reason: one sentence explaining WHY this pillar passes (what the policy does well). Write an empty string "" only if score is 0 or 1.

Return a JSON object with:
  - pillar_scores: array of exactly 5 objects (one per criterion, in the order listed above)
  - summary: 2-3 sentence plain-English verdict: overall RAI maturity, strongest pillar, most critical gap

Return ONLY valid JSON. No markdown fences. No text outside the JSON object."""


def _check_policy_compliance(
    state: ComplianceState,
) -> tuple[PolicyResult, list[dict], str, dict]:
    """Calls Ollama/Gemma to evaluate current_text against POLICY_CRITERIA.
    Also retrieves relevant regulatory clauses from ChromaDB vector store.
    Returns (PolicyResult, retrieved_clauses, prompt_text, raw_response_dict).
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

    try:
        structured_llm = llm.with_structured_output(PolicyCheckOutput)
        prompt_text = build_policy_prompt(state["current_text"], POLICY_CRITERIA, retrieved)
        result: PolicyCheckOutput = structured_llm.invoke(prompt_text)

        # Serialize result for step_trace
        raw_response_dict = {
            "violations": [v.model_dump() for v in result.violations],
            "summary": result.summary,
        }

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
        ), retrieved, prompt_text, raw_response_dict

    except Exception as e:
        print(f"[POLICY AGENT] LLM call failed: {e}. Returning safe fallback.")
        fallback, placeholder_prompt, placeholder_response = _safe_fallback(str(e))
        return fallback, retrieved, placeholder_prompt, placeholder_response


def _safe_fallback(error: str) -> tuple[PolicyResult, str, dict]:
    """
    Returns a FAILING result when the LLM is unavailable.

    Returning passed=True here would silently mask policy failures during
    Ollama outages. We instead surface a synthetic LLM_UNAVAILABLE violation
    at HIGH severity so the audit log captures the gap and operators know
    the check did not run. This is intentionally conservative: the governance
    framework requires every pillar to be positively verified, not assumed clear.
    """
    synthetic_violation = {
        "id": "LLM_UNAVAILABLE",
        "description": f"Policy compliance check could not run: {error[:200]}",
        "severity": "high",
        "article_reference": "EU AI Act Art.9 (risk management system)",
        "remediation": "Restore the Ollama LLM service and re-run the compliance audit.",
    }
    return (
        PolicyResult(
            violations=[synthetic_violation],
            passed=False,
            severity="high",
            summary=f"Policy check SKIPPED — LLM unavailable: {error[:200]}",
        ),
        "LLM unavailable — policy check skipped.",
        {"error": error},
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
