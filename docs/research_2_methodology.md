# Methodology Research — Kaggle winning solutions + survival/EWS/explainability

Research date: 2026-07-03.

## 1. AmEx Default Prediction (2022) — what won
- Consensus: LightGBM/XGBoost on flattened per-customer aggregations ≈ 95% of achievable score. Transformers/NNs only small ensemble diversity. 1st place: 2 LGB + 2 NN blend, GroupKFold by customer. 2nd place: near-pure LightGBM + **separate model for short-history customers**.
- **Feature recipes (reusable):**
  - Aggregation flatten per customer per column: `mean, std, min, max, last`. `last` = single most important family (recency dominates).
  - Trend/deterioration: `last − first`, `last / first`, `last − mean`.
  - **"After-pay" features**: balance/spend minus payment (`B_x − P_2` etc.) = residual balance after payment — signature handcrafted win.
  - Categorical: count, last, nunique, frequency encoding. Null-count per customer = thin-file signal.
- Validation: 5-fold StratifiedGroupKFold by customer; never aggregate over label window (leakage).

## 2. Home Credit (2018 + 2024)
- 1st place 2018: feature engineering beat model tuning. groupby-agg per time window (last 6/12/24 mo separately), weighted moving averages (recent higher).
- Top ratio features: `credit/annuity`, `annuity/income`, installment paid/due, days-late per installment, `days_employed/days_birth`.
- 2024 stability lessons: evaluate per time-slice (plot AUC/Gini by month); drop drift-heavy features (adversarial validation); **time-based validation split** (train early, validate late) mirrors deployment.

## 3. Give Me Some Credit
- RevolvingUtilization = strongest single predictor. Core credit signal = utilization + past-delinquency counts + DTI.
- Hygiene: cap utilization >1, sentinel values 96/98 in past-due cols → "ever severely delinquent" flag, income NaN → impute + missing indicator.

## 4. Survival / time-to-default (the novelty anchor)
- **Discrete-time hazard model = highest value/effort.** Expand loans into loan-month rows; target = "defaulted this month"; add time-on-book splines + behavioral covariates; fit LightGBM/logistic on the stacked panel. Hazard h(t) → S(t)=∏(1−h) → cumulative PD term structure PD(≤m), m=1..12. "Months-to-default" = first month cumulative PD crosses threshold, or Σ S(t). Native 12-month advance-notice output; SHAP-explainable; this is how IFRS 9 lifetime PD is built (arXiv 2507.15441 tutorial).
- Cox PH (lifelines) = interpretable benchmark ("2× utilization → 1.8× hazard"). Dirick et al. (JORS 2017): Cox with splines best, AFT close second.
- XGBoost `survival:cox` / `survival:aft`; xgbse for calibrated survival curves; scikit-survival for C-index/Brier.
- Calibration: time-dependent AUC, integrated Brier, predicted vs empirical cumulative default curves per decile. Fallback route: separate calibrated classifiers per horizon (3/6/9/12 mo) + isotonic + monotonicity across horizons.
- **Censoring pitfall:** short-history non-defaulters are NOT clean negatives for a 12-month label; discrete-time handles free.

## 5. Early-warning systems (EWS)
- Signal families: payment behavior (missed/partial, DPD bucket migration/roll rates), utilization (spikes, sustained >90%, min-pay-only), cash flow (salary credits stop, bounce rates), bureau (inquiry bursts, score drop), relationship (dormancy).
- **RBI hook (India-differentiator):** RBI Fraud Risk Master Directions (July 2024, from 2015 circular) mandate EWS with ~42–45 published Early Warning Signals + Red Flagged Account (RFA) classification ≥ ₹50 cr, integrated with CBS. Signals include installment delays, cheque bounces, drop in account credits, frequent limit breaches.
- Architecture pattern: ingestion → indicator engine → ML score + rule triggers → green/amber/red tiers → watchlist/case workflow. **Hybrid rules+ML is the credible design.**

## 6. Explainability
- TreeSHAP on LightGBM: global beeswarm + local waterfall per alert.
- **Reason codes:** map top-k negative SHAP feature-GROUPS to a curated human-readable dictionary ("Credit utilization too high relative to limits"). US adverse-action precedent (ECOA/CFPB 2022-03); India framing = RBI Digital Lending Guidelines + model-governance/FREE-AI expectations.
- SHAP stability caveat: fix seeds, average over folds (arXiv 2508.01851).
- **Monotonic constraints** in LightGBM on utilization/DPD = cheap regulator-credibility, kills counterintuitive artifacts.

## Top 10 transferable ideas (impact per effort)
1. LightGBM on flattened per-customer aggregations (mean/std/min/max/last).
2. `last`-vs-mean/first trend "deterioration" features.
3. **Discrete-time hazard model → cumulative PD curve → predicted months-to-default.**
4. TreeSHAP → grouped reason codes + waterfall per alert.
5. After-pay/coverage features (payment−min_due, payment/balance, balance−payment).
6. Ratio features (annuity/income, credit/income, paid/due, balance/limit).
7. Time-based validation + per-month AUC/Gini stability chart.
8. Hybrid EWS: rule triggers + ML score → green/amber/red; cite RBI EWS/RFA 42-signal list.
9. Per-horizon calibrated probabilities (3/6/9/12 mo, isotonic, monotone) as fallback.
10. Monotonic constraints + separate thin-history segment model.

## Cross-cutting pitfalls
(a) Label leakage — freeze features at observation date, label from following 12 months. (b) GroupKFold by customer. (c) Censoring. (d) Out-of-fold for any target encoding. (e) Sentinel values (96/98; DAYS_EMPLOYED=365243).