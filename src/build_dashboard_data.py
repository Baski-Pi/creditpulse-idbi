"""
Stage 4a: Precompute everything the dashboard needs into ONE small parquet.

For a 5,000-account demo portfolio (drawn from the 2015 hold-out book):
- full 12-month PD curve from the Layer 2 hazard model
- RAG bucket (Red / Amber / Green) = ML score + transparent rule triggers
- top-3 plain-language reason codes from per-account SHAP contributions
- the ACTUAL 12-month outcome (these are historical loans, so the demo can
  prove the buckets were right - "validation view")

Run:  .venv\\Scripts\\python.exe src\\build_dashboard_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
MODELS = BASE / "models"
APPDATA = BASE / "app" / "data"
APPDATA.mkdir(parents=True, exist_ok=True)

N_PORTFOLIO = 5000
SEED = 42

# ----------------------------------------------------- rebuild the test frame
df = pd.read_parquet(PROC / "lc_clean.parquet")
df["issue_d"] = pd.to_datetime(df["issue_d"])
months_off = df["term_months"].astype(int).values
observable = df["issue_d"].values.astype("datetime64[M]") + months_off.astype(
    "timedelta64[M]") <= np.datetime64("2018-12")
cohort = df[observable & df["issue_year"].between(2010, 2015)].copy()
CATS = ["grade", "sub_grade", "purpose", "home_ownership",
        "verification_status", "addr_state", "application_type"]
for c in CATS:
    cohort[c] = cohort[c].astype("category")
META = ["id", "issue_d", "last_pymnt_d", "duration_months", "event",
        "completed", "issue_year"]
FEATS = [c for c in cohort.columns if c not in META]

test = cohort[cohort["issue_year"] == 2015]
# stratified-ish sample: keep it interesting - oversample risky grades a bit
sample = test.sample(n=N_PORTFOLIO, random_state=SEED).reset_index(drop=True)

booster = lgb.Booster(model_file=str(MODELS / "lc_layer2_hazard.txt"))
feat_order = FEATS + ["month_on_book"]

# ----------------------------------------------------- PD curve months 1..12
surv = np.ones(len(sample))
curve = np.zeros((len(sample), 12))
for m in range(1, 13):
    rows = sample[FEATS].copy()
    rows["month_on_book"] = m
    h = booster.predict(rows[feat_order])
    surv = surv * (1 - h)
    curve[:, m - 1] = 1 - surv

pd12 = curve[:, 11]

# expected timing: probability-weighted average default month (given default by 12m)
inc = np.diff(np.concatenate([np.zeros((len(sample), 1)), curve], axis=1), axis=1)
with np.errstate(invalid="ignore", divide="ignore"):
    exp_month = np.where(pd12 > 0, (inc * np.arange(1, 13)).sum(axis=1) / pd12, np.nan)

# ----------------------------------------------------- rule triggers (EWS layer)
rules = pd.DataFrame(index=sample.index)
rules["Utilization above 90%"] = sample["revol_util"].fillna(0) >= 90
rules["Delinquency in last 2 years"] = sample["delinq_2yrs"].fillna(0) >= 1
rules["Ever 120+ days past due"] = sample["num_accts_ever_120_pd"].fillna(0) >= 1
rules["3+ credit enquiries in 6 months"] = sample["inq_last_6mths"].fillna(0) >= 3
rules["Debt-to-income above 35%"] = sample["dti"].fillna(0) >= 35
rule_count = rules.sum(axis=1)
rule_list = rules.apply(lambda r: [c for c in rules.columns if r[c]], axis=1)

# ----------------------------------------------------- RAG assignment
red_thr = float(np.quantile(pd12, 0.93))     # ~top 7% by model
amber_thr = float(np.quantile(pd12, 0.80))   # next ~13%
rag = np.where((pd12 >= red_thr) | ((pd12 >= amber_thr) & (rule_count >= 2)), "RED",
      np.where((pd12 >= amber_thr) | (rule_count >= 2), "AMBER", "GREEN"))

# ----------------------------------------------------- reason codes via SHAP
REASON = {
    "revol_util": "Revolving credit utilization is high",
    "bc_util": "Card utilization is high",
    "percent_bc_gt_75": "Most cards are above 75% of their limit",
    "dti": "Debt-to-income ratio is elevated",
    "delinq_2yrs": "Delinquencies recorded in the last 2 years",
    "mths_since_last_delinq": "Recent delinquency on record",
    "inq_last_6mths": "Multiple recent credit enquiries",
    "acc_open_past_24mths": "Many new credit accounts opened recently",
    "num_accts_ever_120_pd": "History of severe (120+ days) past-due accounts",
    "num_tl_90g_dpd_24m": "90+ days past-due in the last 24 months",
    "pct_tl_nvr_dlq": "Low share of accounts with clean history",
    "annual_inc": "Income is low relative to peers",
    "loan_to_income": "Loan amount is high relative to income",
    "payment_to_income": "EMI burden is high relative to monthly income",
    "balance_to_income": "Outstanding balances high relative to income",
    "fico": "Credit score is low",
    "credit_hist_months": "Credit history is short",
    "emp_length_yrs": "Short employment tenure",
    "sub_grade": "High-risk internal credit rating",
    "grade": "High-risk internal credit rating",
    "int_rate": "Priced at a high risk premium",
    "term_months": "Longer loan tenor increases exposure",
    "loan_amnt": "Large loan size",
    "installment": "High installment amount",
    "month_on_book": "Loan is in its highest-risk seasoning period",
    "purpose": "Loan purpose carries elevated risk",
    "home_ownership": "Housing status adds risk",
    "util_x_delinq": "High utilization combined with past delinquency",
    "avg_cur_bal": "Account balance pattern signals stress",
    "tot_hi_cred_lim": "Limited total credit capacity",
    "total_rev_hi_lim": "Limited revolving credit capacity",
    "revol_bal": "High revolving balance",
    "open_acc": "Number of open accounts",
    "total_acc": "Total accounts profile",
    "pub_rec": "Public derogatory records",
    "pub_rec_bankruptcies": "Bankruptcy on record",
    "mort_acc": "Mortgage exposure",
    "verification_status": "Income verification status",
    "addr_state": "Regional risk factor",
    "application_type": "Application type",
    "tax_liens": "Tax liens on record",
}

rows12 = sample[FEATS].copy()
rows12["month_on_book"] = 12
contrib = booster.predict(rows12[feat_order], pred_contrib=True)[:, :-1]  # drop bias
cols = np.array(feat_order)
reasons = []
for i in range(len(sample)):
    top = np.argsort(-contrib[i])[:5]
    rs = []
    for j in top:
        if contrib[i, j] > 0 and cols[j] in REASON:
            rs.append(REASON[cols[j]])
        if len(rs) == 3:
            break
    reasons.append(rs)

# ----------------------------------------------------- assemble portfolio file
out = pd.DataFrame({
    "account_id": ["IDBI-" + str(100000 + i) for i in range(len(sample))],
    "rag": rag,
    "pd_12m": pd12.round(4),
    "pd_3m": curve[:, 2].round(4),
    "pd_6m": curve[:, 5].round(4),
    "pd_9m": curve[:, 8].round(4),
    "expected_risk_month": np.round(exp_month, 1),
    "rule_flags": [json.dumps(r) for r in rule_list],
    "n_rules": rule_count.values,
    "reasons": [json.dumps(r) for r in reasons],
    "grade": sample["grade"].astype(str),
    "purpose": sample["purpose"].astype(str),
    "state": sample["addr_state"].astype(str),
    "loan_amnt": sample["loan_amnt"].values,
    "annual_inc": sample["annual_inc"].values,
    "revol_util": sample["revol_util"].values,
    "dti": sample["dti"].values,
    "fico": sample["fico"].values,
    "term_months": sample["term_months"].values,
    # ground truth for the validation view
    "actual_default_12m": ((sample["event"] == 1)
                           & (sample["duration_months"] <= 12)).astype(int).values,
    "actual_default_ever": sample["event"].values,
})
for m in range(1, 13):
    out[f"pd_m{m}"] = curve[:, m - 1].round(4)

out.to_parquet(APPDATA / "portfolio.parquet", index=False)

summary = {
    "n_accounts": len(out),
    "rag_counts": out["rag"].value_counts().to_dict(),
    "observed_12m_default_rate_by_rag": out.groupby("rag")["actual_default_12m"]
                                            .mean().round(4).to_dict(),
    "red_threshold_pd12": round(red_thr, 4),
    "amber_threshold_pd12": round(amber_thr, 4),
}
with open(APPDATA / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
print("Saved ->", APPDATA / "portfolio.parquet")