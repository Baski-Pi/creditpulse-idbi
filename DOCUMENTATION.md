# CreditPulse — Technical Documentation

End-to-end reference for the CreditPulse default-prediction system
(IDBI Innovate 2026, Track 4). Written for a maintainer who needs to
understand, defend, and extend every stage of the pipeline.

Companion notebook: [notebooks/creditpulse_walkthrough.ipynb](notebooks/creditpulse_walkthrough.ipynb)
executes each stage described here with live outputs.

---

## 1. Data Understanding

### 1.1 What data is used, and why

| Dataset | Role | Size | Why chosen |
|---|---|---|---|
| **Lending Club accepted loans** (Kaggle: `wordsforthewise/lending-club`) | Primary training data for both model layers | 2.26M loans, 151 columns, 2007–2018 | The only large public dataset where the **timing** of default is derivable per loan (`issue_d` → `last_pymnt_d`), which the time-to-default model requires |
| **UCI "default of credit card clients"** (Taiwan, 2005) | Behavioral-data demonstration | 30,000 customers, 25 columns | Contains 6 months of true **monthly repayment behavior** (bill, payment, delay status) — proves the pipeline's uplift when behavioral data is available, standing in for the bank's internal data |
| **AmEx Default Prediction** (parquet mirror) | Downloaded, held in reserve | 459k customers × 13 monthly statements | Monthly statement time series; not used in the current build (features are anonymized, no default date) |

### 1.2 Structure of the primary dataset

Each Lending Club row is one loan at origination plus its final outcome:

- **Loan terms**: `loan_amnt`, `term` (36/60 months), `int_rate`, `installment`, `grade`/`sub_grade` (lender's internal rating)
- **Borrower profile**: `annual_inc`, `emp_length`, `home_ownership`, `dti` (debt-to-income), `addr_state`
- **Credit history**: `fico_range_*`, `revol_util` (revolving utilization — the classic #1 credit signal), `delinq_2yrs`, `inq_last_6mths`, `earliest_cr_line`, counts of past 90/120-DPD accounts
- **Outcome**: `loan_status` (Fully Paid / Charged Off / Current / Late...), `issue_d`, `last_pymnt_d`

### 1.3 Key assumptions (and their limits)

1. **"Default" is operationalized as payment cessation**: the event is `loan_status ∈ {Charged Off, Default}` and the event *time* is the last payment date. Formal charge-off is booked ~4–6 months after payments stop; we deliberately model the earlier, more actionable event. Median time from disbursal to payment cessation: **14 months**.
2. **US personal loans proxy the bank's internal database pillar.** Feature *schema* transfers to Indian bank data (utilization, DTI, vintage, delinquency counts map to CBS + CIBIL fields); learned *coefficients* do not — the model must be retrained in the IDBI sandbox.
3. **Fully-observable cohort**: models train/validate only on loans whose full term fits inside the data window (issue + term ≤ 2018-12). Consequence (disclosed): the 2015 hold-out is **exclusively 36-month loans**. The architecture is term-agnostic.
4. **Censoring is informative-free**: loans still running at data end are treated as right-censored (not as non-defaults). Loans that never made a payment inherit `duration = data end` — a known edge case (~2.3k of 2.26M) documented rather than fixed pre-deadline, since fixing it would churn every published number.
5. **Public-domain pillar is architecture-ready but not data-backed** in the prototype (only a regional factor). First sandbox integration target: RBI sectoral GNPA and macro series.

---

## 2. Data Preprocessing

Script: [src/prep_lending_club.py](src/prep_lending_club.py) (LC) and stage 1–2 of
[src/baseline_uci.py](src/baseline_uci.py) (UCI).

| Step | What | Why | Impact |
|---|---|---|---|
| Column selection | 151 → 40 columns at load | Excludes **post-outcome columns** (`recoveries`, `settlement_*`, `hardship_*`, `last_fico_*`) | Prevents target leakage — these fields are only populated after distress; including them yields fake AUC ≈ 0.95+ |
| Date parsing | `'Dec-2015'` → datetime | Needed for durations & time-based splits | Enables the survival target and honest validation |
| Percent strings | `'13.5%'` → 13.5 (defensive) | Dataset-version differences | Robust re-runs |
| `emp_length` parsing | `'10+ years'`→10, `'< 1 year'`→0 | Ordinal signal | Small but real importance |
| Survival fields | `event` (charged off), `duration_months` (issue → last payment; censored at data end) | The Layer-2 target | The core of the WHEN model |
| Derived ratios | `loan_to_income`, `payment_to_income`, `balance_to_income`, `fico` midpoint, `credit_hist_months`, `util_x_delinq` | Ratio features dominated the Home Credit winning solutions | Among top-15 importances |
| UCI sentinel repair | `EDUCATION ∈ {0,5,6}` → "other"; `MARRIAGE 0` → "other" | Undocumented category codes | Cleanliness; avoids spurious splits |
| UCI protected attributes | `SEX`, `MARRIAGE` **excluded** from features | Fair-lending: gender/marital status must not drive credit decisions | AUC unchanged (0.786) — demonstrating the signal loss is nil |
| UCI behavioral features | utilization (bill/limit) stats, payment-coverage stats, deterioration deltas (`last − mean`), after-pay residual | The AmEx-competition-winning feature families | These provide the 0.72 → 0.79 AUC gap vs origination-only data |

**General principle:** every feature must be computable *at scoring time* from
information available *before* the outcome window. That single rule drives all
exclusions.

---

## 3. Exploratory Data Analysis & Statistical Validation

Performed checks (reproduced in the notebook):

1. **Class balance**: lifetime default rate 11.9% (full LC), 14.8% (modeling cohort), 5.2% for the 12-month task, 22.1% (UCI). Consequence: raw accuracy is a misleading headline (a do-nothing model scores 94.8% on the 12-month task) → we report AUC/KS/lift/calibration alongside.
2. **Signal sanity — utilization monotonicity**: default rate rises monotonically across utilization deciles in both datasets (the model's monotone response is *learned*, not imposed — and matches credit-officer intuition).
3. **Timing distribution**: histogram of `duration_months` for defaulters — median 14, mode ~8–16; validates that a 12-month early-warning window is actionable (most defaults have not yet happened at scoring time + 12 months... they are being predicted, not observed).
4. **Empirical survival curve**: Kaplan-Meier-style cumulative default by month-on-book, by grade — grade-separated hazard levels justify segment-aware features.
5. **Temporal stability / drift**: per-quarter AUC on the 2015 hold-out (0.695–0.708 across four quarters) — the production-readiness check the 2024 Home Credit "stability" competition institutionalized.
6. **Calibration** (see §5): predicted vs observed by decile + portfolio-level totals.

Assumption tests worth naming in Q&A: *stationarity* (mitigated by time-based
validation, not assumed), *no unobserved-censoring bias* (censored loans treated
via the discrete-time likelihood, not dropped), *feature availability at score
time* (enforced by construction, §2).

---

## 4. Modeling Approach

### 4.1 Two layers, one engine

**Layer 1 — lifetime risk score** ([src/train_lc_layer1.py](src/train_lc_layer1.py)):
LightGBM binary classifier on completed loans (Fully Paid vs Charged Off).
Purpose: a competitive ranking score and feature-importance evidence.

**Layer 2 — discrete-time hazard model** ([src/train_lc_layer2_hazard.py](src/train_lc_layer2_hazard.py)) — the differentiator:

1. Expand each loan into **loan-month rows** ("person-period" format): a loan
   observed 9 months contributes 9 rows; the target is *"did payments stop in
   THIS month?"* (6.24M rows from 250k loans; monthly hazard rate ≈ 0.56%).
2. LightGBM learns the monthly hazard `h(t | features, month_on_book)`.
3. Chain hazards into survival: `S(t) = Π(1−h(m))`, cumulative PD `= 1 − S(t)`
   → a **PD term structure** for months 1..12 per account.
4. **Running-account scoring**: for an account alive at month 6, chain months
   7..18 instead — same model, conditioned on survival, no retraining. This is
   what the dashboard's viewpoint toggle does.

Why this construction: it answers the bank's actual question ("how far in
advance?") natively; it handles censoring correctly for free (a loan observed 4
months contributes 4 honest rows — naive 12-month labeling would silently
mislabel it); and it is the standard IFRS 9 lifetime-PD construction, so it
reads as industry practice, not a hackathon trick.

### 4.2 Alternatives considered and rejected

| Alternative | Why not |
|---|---|
| Logistic-regression scorecard (WoE binning) | Interpretable and bank-classic, but 3–5 AUC points weaker on this data; we get interpretability via SHAP + monotone sanity instead |
| XGBoost / CatBoost | Interchangeable with LightGBM here; LightGBM chosen for speed on 6.2M-row panels and native categorical handling. The AmEx competition showed the three are within noise of each other |
| Cox proportional hazards (lifelines) | Elegant but assumes proportional hazards & linear covariate effects; discrete-time + GBDT drops both assumptions and stays explainable |
| XGBoost AFT / xgbse | Viable; poorer calibration out-of-the-box (needs xgbse's extra machinery) vs. the naturally-calibrated discrete-time probabilities |
| DeepSurv / transformers on sequences | Data-hungry, opaque, and the AmEx leaderboard showed GBDT-on-aggregations captures ~95% of achievable signal; not justified for a 6-day, judge-audited build |
| Separate classifiers per horizon (3/6/9/12m) | Simpler but needs monotonicity enforcement across horizons and four models; the hazard model gives all horizons coherently from one model |

### 4.3 Training & validation methodology

- **Time-based split**: train on loans issued ≤ 2014, test on 2015 — mirrors
  production ("score tomorrow's book with yesterday's model"). Random K-fold
  would leak macro conditions and overstate performance.
- **Early stopping** tunes tree count on a **2014 slice of training data**,
  never the test year (fixed after external review — see §8).
- **No hyperparameter search on test**: parameters are sensible defaults;
  the only tuned quantity (n_trees) uses the 2014 slice.
- Reproducibility: fixed seeds; deterministic retrains produce identical
  metrics.

---

## 5. Evaluation Metrics

| Metric | Value (2015 hold-out, 283,173 loans) | Why this metric |
|---|---|---|
| **Accuracy** (12-month task) | **94.8%** | The bank's stated requirement (">90%"). Reported with its caveat: base rate is 5.2%, so a do-nothing model matches it — accuracy alone cannot demonstrate skill on imbalanced data |
| **AUC** | **0.72** | Ranking power: probability a random defaulter scores above a random good account. Threshold-free, imbalance-robust |
| **KS statistic** | **0.33** | Bankers' standard scorecard metric (max separation of score distributions); 0.3+ is a usable scorecard |
| **Lift @ top-15%** | **2.47×** (37% of defaulters captured) | Operational meaning for a capacity-constrained watchlist |
| **Calibration by decile** | Riskiest decile: predicted 14.6% vs observed 14.2%; portfolio: 239 predicted vs 246 actual | A PD used for provisioning/pricing must *mean what it says*; mid-book deciles underpredict modestly (10–20% relative) — disclosed, recalibrated on bank data |
| **RAG blind validation** | RED 11.0% / AMBER 7.7% / GREEN 3.3% (at-disbursal); RED 15.5% / GREEN 5.1% (month-6 view) | End-to-end proof that the delivered artifact (buckets, not raw scores) separates real outcomes |
| **Per-quarter AUC stability** | 0.695–0.708 across 2015 Q1–Q4 | Production-readiness: performance must not decay across the deployment window |

Known trade-off, pre-framed: GREEN still contains ~48% of eventual defaulters
(at 3.3% rate). Watchlist size is a **capacity dial** — RED+AMBER at ~30% of the
book captures 52% of defaults at 2.5× lift; widening the buckets trades officer
workload for coverage.

---

## 6. AI and Advanced Techniques

### 6.1 Where ML/AI is used

1. **Layer 1 & Layer 2**: gradient-boosted decision trees (LightGBM) — the ML core.
2. **Explainability**: TreeSHAP-style per-prediction contributions
   (`booster.predict(..., pred_contrib=True)`) → top positive contributors are
   mapped through a hand-curated dictionary to **plain-language reason codes**
   ("Revolving credit utilization is high"). This mirrors US adverse-action
   practice and RBI model-governance expectations.
3. **Hybrid rules + ML**: transparent EWS triggers (utilization ≥ 90%,
   delinquency in 2 years, ≥3 enquiries in 6 months, DTI ≥ 35, ever-120-DPD)
   run alongside the score and can promote an account's bucket. Verified: rule-promoted
   REDs default at 11.8% vs 10.8% for score-only REDs — the rules add real signal.

### 6.2 "RAG" — important acronym clarification

In this project **RAG = Red-Amber-Green**, the color-coded risk bucketing the
bank explicitly requested in the AMA. It is **not** Retrieval-Augmented
Generation. **No LLM or retrieval-augmented generation is used anywhere in the
current system.** The only LLM-adjacent element is on the roadmap: an *agentic
memo-drafter* that would draft the recommended-action note for each RED account
(retrieving the account's reason codes, curve, and RBI EWS references as
context — which WOULD be a retrieval-augmented generation pattern), always
subject to officer approval.

### 6.3 Architecture overview

```
Data pillars (behavior / internal DB / public domain)
        │  monthly batch
        ▼
Feature engine (utilization, deterioration deltas, coverage, segments)
        │
        ├──► Layer 1: LightGBM risk score
        ├──► Layer 2: discrete-time hazard → 12-month PD curve (any account age)
        ▼
Layer 3: RAG bucketing = ML score thresholds + EWS rule triggers
        │
        ├──► Reason codes (SHAP → plain language)
        ▼
Dashboard / API → credit officer (human decides) → actions & case workflow
```

Production mapping (sandbox-ready): S3 + Glue (data lake, monthly batch) →
SageMaker endpoint (Layers 1–2) → Lambda rules engine → API Gateway →
dashboard/case management. All open-source, on-prem deployable, CPU-only.

---

## 7. Overall System Flow & Design Decisions

**End-to-end runtime flow** (what happens each month in production):

1. Ingest refreshed account data (repayments, balances, bureau, public-domain).
2. Recompute features per account (as-of the scoring date — never future data).
3. Score every account: Layer-1 score + Layer-2 PD curve for the *next* 12
   months conditioned on the account's current age on book.
4. Apply RAG thresholds + rule triggers → watchlist with reason codes.
5. Officers work RED (immediate) and AMBER (watch) queues; decisions and
   outcomes feed back for monitoring and eventual retraining.

**Key design decisions and their trade-offs:**

| Decision | Trade-off accepted |
|---|---|
| Predict payment cessation, not formal charge-off | Slightly unconventional label ↔ 4–6 months earlier warning |
| Discrete-time hazard over binary classifier | More engineering (panel expansion) ↔ native "months ahead" output + correct censoring |
| LightGBM over scorecard | Less native transparency ↔ +3–5 AUC points, recovered via SHAP reason codes |
| Time-based over random validation | Lower reported metrics ↔ honest production estimate |
| Hybrid rules + ML | Redundancy ↔ officer trust + RBI EWS alignment + measured extra signal |
| Precomputed demo portfolio (5,000 accounts) in the app | Not live-scoring ↔ instant free-tier dashboard; scoring service is the documented production path |
| Human-in-the-loop only | No automation win ↔ exactly what the bank asked for; AI recommends, officer decides |

**Pipeline execution order** (to rebuild everything from scratch):

```
pip install -r requirements-train.txt
# place raw data per §1.1 into data/raw/
python src/baseline_uci.py             # stage 1: behavioral baseline
python src/prep_lending_club.py        # stage 2a: clean + survival fields
python src/train_lc_layer1.py          # stage 2b: Layer-1 scorer
python src/train_lc_layer2_hazard.py   # stage 3: hazard model + 12m metrics
python src/build_dashboard_data.py     # stage 4: demo portfolio + RAG + reasons
streamlit run app/streamlit_app.py     # stage 5: dashboard
```

---

## 8. Known Limitations (external-review register)

Kept deliberately visible — knowing these is part of maintaining the system:

1. **~2.3k charged-off loans with no payments** inherit `duration = data end`
   and are mislabeled as 12-month survivors (200 in the test set). Fix requires
   full retrain; deferred post-competition.
2. **Layer-2 features are static** (origination snapshot); the only dynamic
   input is `month_on_book`. Individual behavioral *timing* arrives with the
   bank's monthly data in the sandbox.
3. **Mid-book calibration** underpredicts 10–20% relative; recalibrate
   (isotonic) on bank data.
4. **RAG thresholds** are quantiles of the demo portfolio; production thresholds
   must be fixed on a prior book.
5. **36-month cohort** for validation (see §1.3); retrain per segment on bank data.
6. **Behavioral uplift (0.72 → 0.79)** is directional evidence across different
   datasets, not a quantified forecast.
