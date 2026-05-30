"""
nodes/suggestion_agent.py
--------------------------
Suggestion Agent — standalone Ollama/Gemma call for policy rewrite suggestions.

Not a LangGraph node — invoked from the UI when the user requests policy
improvement suggestions after a failed audit.

Public API:
    run_suggestion_agent(state)               — LLM + rule-based fallback
    run_suggestion_agent_with_audit(state, thread_id)  — same + SQLite audit entry
"""

import json
import logging
from datetime import datetime, timezone
from pydantic import BaseModel

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "gemma4:latest"

PILLAR_CRITERION_NAMES = {
    "GOVERNANCE_ACCOUNTABILITY":    "Governance & Accountability",
    "FAIRNESS_BIAS":                "Fairness & Bias Mitigation",
    "TRANSPARENCY_EXPLAINABILITY":  "Transparency & Explainability",
    "ROBUSTNESS_MONITORING":        "Robustness & Monitoring",
    "PRIVACY_DATA_STEWARDSHIP":     "Privacy & Data Stewardship",
}

PILLAR_KEY_TO_CRITERION = {
    "strategic_alignment":   "GOVERNANCE_ACCOUNTABILITY",
    "model_governance":      "FAIRNESS_BIAS",
    "org_oversight":         "TRANSPARENCY_EXPLAINABILITY",
    "continuous_monitoring": "ROBUSTNESS_MONITORING",
    "data_governance":       "PRIVACY_DATA_STEWARDSHIP",
}

CRITERION_REGULATORY_REFS = {
    "GOVERNANCE_ACCOUNTABILITY":   "EU AI Act Art.17 / RBI MRM Sec.4 / DPDPA 2023 Sec.8",
    "FAIRNESS_BIAS":               "EU AI Act Art.10 / RBI MRM Sec.7 / DPDPA 2023 Sec.8",
    "TRANSPARENCY_EXPLAINABILITY": "EU AI Act Art.13 / DPDPA 2023 Sec.11",
    "ROBUSTNESS_MONITORING":       "EU AI Act Art.72 / RBI MRM Sec.8",
    "PRIVACY_DATA_STEWARDSHIP":    "DPDPA 2023 Sec.6 / Sec.8 / EU AI Act Art.10(5)",
}

_FALLBACK_LANGUAGE = {
    "GOVERNANCE_ACCOUNTABILITY": [
        "The AI Ethics Committee, chaired by the Chief Risk Officer, shall maintain a registry of all AI systems in production use.",
        "All AI deployment decisions must be reviewable by a designated human officer within 48 hours of an affected party's request.",
        "An AI governance charter shall be reviewed and reaffirmed by the Board on an annual basis.",
    ],
    "FAIRNESS_BIAS": [
        "Prior to deployment, all AI models must undergo a bias and fairness assessment using standardised metrics including disparate impact ratio (≥ 0.8).",
        "Training datasets must be reviewed for historical bias, and any identified proxy variables must be excluded unless explicitly justified and documented.",
        "Fairness thresholds shall be defined and documented before model deployment and reassessed at each major retraining cycle.",
    ],
    "TRANSPARENCY_EXPLAINABILITY": [
        "All individuals subject to AI-assisted decisions shall be informed of this fact in plain language at the point of engagement.",
        "Upon request, the organisation shall provide a plain-language explanation of the primary factors influencing an AI-assisted decision.",
        "Model cards shall be maintained for all production AI systems, documenting model purpose, training data sources, known limitations, and performance benchmarks.",
    ],
    "ROBUSTNESS_MONITORING": [
        "Performance thresholds for all production AI models shall be defined and monitored on at least a monthly basis, with automated alerts for threshold breaches.",
        "An incident response procedure shall be established to detect, escalate, and remediate AI model failures or anomalous outputs within defined SLAs.",
        "Models shall undergo formal re-validation at least annually, or immediately following any significant change in underlying data distribution.",
    ],
    "PRIVACY_DATA_STEWARDSHIP": [
        "Personal data used in AI model training shall only be processed with a documented lawful basis and, where applicable, explicit consent per DPDPA 2023 Section 6.",
        "Data used for AI purposes shall be subject to purpose limitation — it shall not be repurposed for AI model training without a fresh consent review.",
        "Third-party AI tools and vendors shall be subject to data processing agreements that include obligations on data minimisation and AI training data use restrictions.",
    ],
}


class PillarSuggestion(BaseModel):
    pillar_id: str
    pillar_name: str
    current_score: int
    gap_description: str
    suggested_language: list[str]
    regulatory_basis: str


class SuggestionsOutput(BaseModel):
    suggestions: list[PillarSuggestion]


def run_suggestion_agent(state: dict) -> list[dict]:
    """
    Takes a ComplianceState dict (post-audit), returns a list of suggestion dicts.
    Falls back to rule-based suggestions if LLM call fails.
    """
    from langchain_ollama import ChatOllama

    current_text = state.get("current_text", state.get("raw_input", ""))
    rai_scores = state.get("rai_scores") or {}
    pillar_gaps = state.get("pillar_gaps") or {}

    failing_pillars = []
    for pillar_key, score in rai_scores.items():
        if score < 3:
            criterion_id = PILLAR_KEY_TO_CRITERION.get(pillar_key, pillar_key)
            gap_data = pillar_gaps.get(criterion_id, {})
            failing_pillars.append({
                "pillar_key":    pillar_key,
                "criterion_id":  criterion_id,
                "name":          PILLAR_CRITERION_NAMES.get(criterion_id, criterion_id),
                "score":         score,
                "gap":           gap_data.get("gap", "No specific gap recorded."),
                "evidence":      gap_data.get("evidence", ""),
                "reference":     CRITERION_REGULATORY_REFS.get(criterion_id, ""),
            })

    if not failing_pillars:
        return []

    audit_findings_json = json.dumps(
        [
            {
                "pillar": p["name"],
                "criterion_id": p["criterion_id"],
                "current_score": p["score"],
                "identified_gap": p["gap"],
                "existing_policy_text": p["evidence"] or "(not found in policy)",
                "regulatory_reference": p["reference"],
            }
            for p in failing_pillars
        ],
        indent=2,
    )

    prompt = f"""You are a Responsible AI governance expert helping a company improve their AI policy to meet regulatory standards.

EXISTING POLICY TEXT (excerpt):
\"\"\"{current_text[:3000]}\"\"\"

AUDIT FINDINGS — pillars that are missing or inadequate:
{audit_findings_json}

YOUR TASK:
For each failing pillar, produce specific, actionable improvements. Your suggestions must:
1. Be written as ACTUAL POLICY LANGUAGE the company can copy-paste into their document.
2. Directly address the identified gap — do not write generic statements.
3. Be grounded in the regulatory reference listed for that pillar.
4. If the policy already has partial text (see "existing_policy_text"), EXTEND or STRENGTHEN it rather than replacing it.

WRITING RULES:
- Write in the third person ("The organisation shall...", "All AI models must...", "The AI Ethics Committee is required to...")
- Be specific: name metrics where relevant (e.g. "disparate impact ratio ≥ 0.8"), timelines (e.g. "within 48 hours"), or roles (e.g. "Model Owner")
- Each suggested sentence should be independently meaningful — avoid vague filler like "adequate measures shall be taken"
- Provide exactly 3 draft sentences per pillar

Return a JSON object with:
- suggestions: array of objects, one per failing pillar, each containing:
  - pillar_id: the criterion ID exactly as listed (e.g. "GOVERNANCE_ACCOUNTABILITY")
  - pillar_name: the display name of the pillar
  - current_score: the current compliance score (integer 0-3)
  - gap_description: one precise sentence describing what is missing, referencing the regulatory article
  - suggested_language: array of exactly 3 draft policy sentences ready for insertion
  - regulatory_basis: the applicable regulatory reference string

Return ONLY valid JSON. No markdown fences. No text outside the JSON object."""

    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
        structured_llm = llm.with_structured_output(SuggestionsOutput)
        result: SuggestionsOutput = structured_llm.invoke(prompt)
        return [s.model_dump() for s in result.suggestions]
    except Exception as e:
        print(f"[SUGGESTION AGENT] LLM call failed: {e}. Returning rule-based fallback.")
        return _rule_based_fallback(failing_pillars)


def _rule_based_fallback(failing_pillars: list[dict]) -> list[dict]:
    results = []
    for p in failing_pillars:
        cid = p["criterion_id"]
        results.append({
            "pillar_id":          cid,
            "pillar_name":        p["name"],
            "current_score":      p["score"],
            "gap_description":    p["gap"] or f"The policy does not adequately address {p['name']}.",
            "suggested_language": _FALLBACK_LANGUAGE.get(cid, ["Add policy language addressing this pillar."]),
            "regulatory_basis":   p["reference"],
        })
    return results


def run_suggestion_agent_with_audit(state: dict, thread_id: str | None = None) -> list[dict]:
    """
    Runs the suggestion agent and writes a SUGGESTION_GENERATED entry to the
    SQLite audit DB so that suggestion requests are traceable alongside the
    main compliance audit trail.

    Args:
        state:     ComplianceState dict (post-audit), same as run_suggestion_agent.
        thread_id: The LangGraph thread_id of the parent audit run, used to link
                   the suggestion log entry to the original audit checkpoint.
                   If None, the entry is still written but without a thread link.

    Returns:
        list of suggestion dicts (same schema as run_suggestion_agent).
    """
    suggestions = run_suggestion_agent(state)

    # --- Write audit log entry to SQLite ---
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).parent.parent / "rai_audit.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS suggestion_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                thread_id   TEXT,
                pillars     TEXT,
                count       INTEGER,
                source      TEXT
            )
            """
        )
        pillar_ids = [s.get("pillar_id", "") for s in suggestions]
        conn.execute(
            "INSERT INTO suggestion_log (timestamp, thread_id, pillars, count, source) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                thread_id,
                json.dumps(pillar_ids),
                len(suggestions),
                "ui_suggestion_agent",
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            "[SUGGESTION AGENT] Audit entry written — %d suggestion(s) for thread %s",
            len(suggestions), thread_id,
        )
    except Exception as exc:
        # Audit write failure must not crash the UI — log and continue.
        logger.warning("[SUGGESTION AGENT] Could not write audit log entry: %s", exc)

    return suggestions
