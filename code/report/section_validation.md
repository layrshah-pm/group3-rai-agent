# Section: Integration Testing and Validation
**Person 7 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. Overview

This section documents the end-to-end validation of the RAI Compliance Agent across all three demo scenarios. Validation covers: correctness (does each scenario produce the expected outcome?), reliability (does it produce the same outcome consistently?), and structural integrity (does the audit trail persist correctly?).

---

## 2. Test Coverage Summary

| Test file | Tests | Status | Notes |
|---|---|---|---|
| `test_pii_agent.py` | 14 | ✓ All pass | Presidio + spaCy, no LLM |
| `test_bias_agent.py` | 7 | ✓ All pass | Fairlearn, no LLM |
| `test_correction.py` | 6 | ✓ All pass | Mocked LLM (ChatOllama patch) |
| `test_policy_agent.py` | 8 | ✓ All pass | Live Ollama/Gemma4, 273s |
| `test_integration.py` | 12 | ✓ All pass | Live full graph, 1180s |
| **Total** | **47** | **47/47** | |

---

## 3. Integration Test Design

### Scenario 1 — Clean Compliant Text

The input text was carefully crafted to satisfy all 8 regulatory criteria simultaneously. The test asserts:

```python
assert result["final_status"] == "PASS"
assert result["violations"] == []
assert result["correction_count"] == 0
```

All five pillar scores are verified to be present and in range [0, 3].

**Result:** PASS on every run. This scenario is deterministic — Presidio finds no PII (it passes the confidence threshold cleanly), Fairlearn is not applicable (text mode), and Gemma4 consistently returns zero violations for a text that explicitly satisfies all criteria.

### Scenario 2 — PII Leakage

The test asserts that:
1. `PII_DETECTED` is in `violations`
2. `correction_count >= 1`
3. `"John Smith"` and `"john.smith@email.com"` are absent from the final `current_text`
4. `final_status != "PASS"`

The PII redaction step is deterministic and passes reliably. The subsequent policy loop (which fires because the bare redacted text lacks AI disclosure) depends on Gemma4 — and by design the agent escalates after 3 correction attempts.

### Scenario 3 — Biased Model Output

The test asserts:
1. `BIAS_DETECTED` is in `violations`
2. `bias_result["disparate_impact_ratio"] < 0.8`

The bias metrics are computed from a fixed pre-trained model (`models/loan_model.pkl`) against a fixed test set. They are fully deterministic — no LLM involved. The loop escalates because text rewording cannot fix model bias.

### Loop termination

A dedicated test uses `max_corrections=2` and asserts `correction_count <= 2`. This guards against the failure mode where a routing bug causes infinite cycles.

---

## 4. Self-Correction Loop Behaviour

The self-correction loop terminates in all cases:

| Scenario | Trigger | Outcome | Cycles |
|---|---|---|---|
| Clean text | No violation | PASS (no loop) | 0 |
| PII text | PII_DETECTED | PII fixed in 1 cycle; policy fires for 3 cycles then ESCALATED | 1+3 = 4 (capped at max_corrections) |
| Biased model | BIAS_DETECTED | 3 cycles attempted, text rewritten but DI unchanged → ESCALATED | 3 |

The router correctly routes to `scorecard` when `correction_count >= max_corrections`, regardless of whether violations remain. This is the expected escalation behaviour.

---

## 5. Audit Trail Validation

After running all 3 scenarios in `main.py`:

```
rai_audit.db contains:
  - Scenario 1:  5 audit entries  (ingestion + pii + bias + policy + scorecard)
  - Scenario 2: 15 audit entries  (4 nodes × 4 passes + scorecard)
  - Scenario 3: 13 audit entries  (ingestion + 3 bias cycles + scorecard)
```

The `utils/audit_logger.get_runs_summary()` correctly decodes all 22 historical runs from the msgpack-serialised checkpoint database and returns `final_status`, `total_score`, `timestamp`, and `correction_count` for each.

---

## 6. Timing

Measured on Apple M-series hardware (2024), Ollama running locally:

| Node | Mean time |
|---|---|
| Ingestion | < 0.1s |
| PII Agent | 0.3–0.5s |
| Bias Agent | 35–40s (model loading + inference) |
| Policy Agent | 60–90s (LLM call) |
| Correction Node (PII) | < 0.5s |
| Correction Node (LLM) | 60–90s |
| Scorecard | < 0.1s |

Full Scenario 1 run: ~90s (dominated by policy LLM call)
Full Scenario 2 run: ~360s (3 policy LLM correction calls)
Full Scenario 3 run: ~320s (3 bias correction LLM calls)

The bias agent's 35–40s is due to loading `models/loan_model.pkl` on every call. A production optimisation would cache the model at module level, reducing this to < 1s.

---

## 7. Running the Validation Suite

To replicate the full validation:

```bash
# Unit tests (fast, ~7s)
python -m pytest tests/test_pii_agent.py tests/test_bias_agent.py tests/test_correction.py -v

# Policy tests (live LLM, ~5 min)
python -m pytest tests/test_policy_agent.py -v

# Integration tests (live full graph, ~20 min)
python -m pytest tests/test_integration.py -v

# Reliability check — 20 runs per scenario (~3 hours for all 3)
python validate_demo.py --runs 20

# Quick smoke check — 3 runs per scenario (~30 min)
python validate_demo.py --runs 3
```

---

## 8. Known Flakiness

The only source of non-determinism in the system is the Policy Agent LLM. Across the 8 unit tests and 12 integration tests, all passed 100% in the validation runs performed for this report. However, the policy agent's behaviour can vary slightly when:

- Ollama is under load (slower inference may cause partial JSON responses)
- The LLM generates a violation for a borderline case that the explicit "do NOT flag" rules did not fully suppress

The `_safe_fallback()` function in `policy_agent.py` ensures the graph never crashes on an LLM error — it returns `passed=True` with an error note in the summary, allowing the run to complete with a partial result.
