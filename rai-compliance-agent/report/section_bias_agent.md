# Section: Bias Metrics and Dataset Findings
**Person 4 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. Overview

The Bias Agent implements the **Model Governance** and **Data Governance** pillars of the RAI framework. It evaluates model predictions for demographic unfairness using three standard fairness metrics, grounded in the **EU AI Act Article 5(1)(b)** prohibition on subliminal bias and the **NIST AI RMF MEASURE 2.2** fairness measurement practice.

---

## 2. Dataset: German Credit (UCI)

The demo uses the [German Credit dataset](https://scikit-learn.org/stable/datasets/real_world.html#openml-datasets), fetched via `sklearn.datasets.fetch_openml("credit-g")`. It contains 1,000 loan applicants with 20 features and a binary outcome (good / bad credit risk).

### Protected attributes

The dataset contains two demographic features that are legally protected under anti-discrimination law in most jurisdictions:

| Attribute | Values | % population |
|---|---|---|
| `sex` (encoded from marital/personal status) | `male` / `female` | 69% / 31% |
| `age` | continuous (19–75) | binned at median (35) |

### Baseline bias (unweighted model)

A logistic regression trained on all features without any fairness intervention produces:

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Disparate Impact Ratio | 0.865 | ≥ 0.8 | PASS (barely) |
| Demographic Parity Diff | 0.093 | ± 0.1 | PASS |
| Equalized Odds Diff | 0.112 | ± 0.1 | FAIL |

The baseline model nearly satisfies the 4/5ths rule but fails equalized odds — a more stringent test that also conditions on the true label.

### Demo model (deliberately biased)

For a compelling demo, the training set was resampled to upweight `female + bad credit` observations by 4×. This induces a clear fairness violation:

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Disparate Impact Ratio | **0.316** | ≥ 0.8 | **FAIL** |
| Demographic Parity Diff | **0.413** | ± 0.1 | **FAIL** |
| Equalized Odds Diff | **0.478** | ± 0.1 | **FAIL** |

The DI ratio of 0.316 means female applicants are approved at less than one-third the rate of male applicants — well below the US 4/5ths rule threshold that is internationally recognised as the minimum non-discrimination standard.

---

## 3. Fairness Metrics

### Disparate Impact Ratio (4/5ths Rule)

$$\text{DI} = \frac{\text{approval rate (unprivileged group)}}{\text{approval rate (privileged group)}}$$

A DI ratio below 0.8 constitutes *adverse impact* under the US Uniform Guidelines on Employee Selection Procedures and is the most widely cited group fairness criterion in financial services regulation.

### Demographic Parity Difference

$$\text{DP diff} = P(\hat{Y}=1 \mid A=\text{unprivileged}) - P(\hat{Y}=1 \mid A=\text{privileged})$$

Measures raw disparity in approval rates regardless of actual creditworthiness. Threshold: |DP diff| ≤ 0.1.

### Equalized Odds Difference

$$\text{EO diff} = \max\bigl(\text{TPR gap},\ \text{FPR gap}\bigr)$$

Conditions on the ground truth label — a stricter test that penalises both higher false-denial rates (harming qualified applicants) and higher false-approval rates (harming lenders). Threshold: |EO diff| ≤ 0.1.

---

## 4. Implementation

```python
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

dp_diff = demographic_parity_difference(
    y_true=test_y,
    y_pred=y_pred,
    sensitive_features=test_sensitive,
)
eo_diff = equalized_odds_difference(
    y_true=test_y,
    y_pred=y_pred,
    sensitive_features=test_sensitive,
)
```

The Disparate Impact Ratio is not natively available in Fairlearn and was implemented manually:

```python
def _disparate_impact_ratio(y_pred, sensitive_features):
    groups = {}
    for pred, g in zip(y_pred, sensitive_features):
        groups.setdefault(g, []).append(int(pred))
    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    return min_rate / max_rate if max_rate > 0 else 1.0
```

The privileged and unprivileged groups are identified by comparing approval rates: the group with the **highest** approval rate is labelled privileged; the group with the lowest is unprivileged.

### Text mode pass-through

When `input_type = "text"`, the bias agent returns `passed = True` with a details note. Text bias analysis (e.g. sentiment bias in language) is architecturally supported but was deferred — the current scope focuses on model output fairness.

---

## 5. Test Results

7 unit tests in `tests/test_bias_agent.py` — all pass:

| Test | Input | Result |
|---|---|---|
| Biased demo model | German Credit with upweighted F+bad | `passed = False`, DI = 0.316 |
| Metrics populated | same | all three metrics non-None |
| Text mode | `input_type = "text"` | `passed = True` |
| No protected attributes | `protected_attributes = []` | `passed = True`, warning logged |
| Privileged/unprivileged populated | biased model | `privileged_group` and `unprivileged_group` set |
| Schema | any | all required keys present |
| Audit log | any | node="bias_agent", action="BIAS_CHECK" |

---

## 6. Why Bias Cannot Be Auto-Corrected

Scenario 3 consistently returns `ESCALATED` rather than `CORRECTED`. This is intentional and reflects a real-world constraint: the bias metrics are computed from the **pre-trained model's test set predictions**, not from the text of the output. Rewording the denial notice does not change the underlying model's decision boundary. True bias remediation requires retraining with fairness constraints (e.g. `fairlearn.reductions.ExponentiatedGradient`) — a model lifecycle intervention that is beyond the scope of a single inference-time correction cycle. The escalation signal is precisely the right output: it tells the responsible party that human intervention in the model pipeline is required.
