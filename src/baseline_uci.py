"""
Stage 1 baseline: end-to-end default-prediction pipeline on the UCI Taiwan dataset.

Pipeline stages (the same shape we will reuse for Lending Club):
  1. LOAD      raw file -> DataFrame
  2. CLEAN     fix sentinel/invalid values
  3. FEATURES  behavioral signals: utilization, payment coverage, delinquency, trends
  4. MODEL     LightGBM with 5-fold stratified cross-validation
  5. EVALUATE  AUC, default-recall, KS, accuracy (bank's headline metric)
  6. EXPLAIN   global feature importance (what drives risk)

Run:  .venv\\Scripts\\python.exe src\\baseline_uci.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "uci_credit_default" / "default of credit card clients.xls"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_excel(RAW, header=1)
df = df.rename(columns={"default payment next month": "target", "PAY_0": "PAY_1"})
# PAY_1..PAY_6: repayment status per month (1 = most recent). -2/-1/0 = paid/ok,
# 1..9 = months of delay. BILL_AMT1/PAY_AMT1 = most recent bill / payment.

# ---------------------------------------------------------------- 2. CLEAN
# Undocumented category codes -> collapse into 'other' buckets.
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

pay_cols = [f"PAY_{i}" for i in range(1, 7)]
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

# ---------------------------------------------------------------- 3. FEATURES
f = pd.DataFrame(index=df.index)
limit = df["LIMIT_BAL"].clip(lower=1)

# -- Utilization: balance / credit limit (the #1 credit-risk signal family)
util = df[bill_cols].div(limit, axis=0).clip(-1, 3)
f["util_last"] = util["BILL_AMT1"]
f["util_mean"] = util.mean(axis=1)
f["util_max"] = util.max(axis=1)
f["util_trend"] = f["util_last"] - f["util_mean"]          # deterioration delta
f["util_high_months"] = (util > 0.9).sum(axis=1)           # sustained high utilization

# -- Payment coverage: how much of the bill does the customer actually pay?
bills = df[bill_cols].clip(lower=1)
coverage = (df[amt_cols].values / bills.values)
f["coverage_last"] = np.clip(coverage[:, 0], 0, 2)
f["coverage_mean"] = np.clip(coverage.mean(axis=1), 0, 2)
f["coverage_trend"] = f["coverage_last"] - f["coverage_mean"]

# -- After-pay residual (AmEx-competition signature feature): balance left after paying
f["afterpay_last"] = (df["BILL_AMT1"] - df["PAY_AMT1"]) / limit

# -- Delinquency behavior from PAY_ status codes
pay = df[pay_cols]
f["delay_last"] = pay["PAY_1"]
f["delay_max"] = pay.max(axis=1)
f["delay_months"] = (pay > 0).sum(axis=1)                  # how many months delayed
f["delay_trend"] = pay["PAY_1"] - pay.mean(axis=1)         # worsening vs own average

# -- Scale / demographics / segment
# Protected attributes (SEX, MARRIAGE) are deliberately EXCLUDED - fair-lending
# hygiene: gender/marital status must not drive credit decisions.
f["limit_bal"] = df["LIMIT_BAL"]
f["age"] = df["AGE"]
f["EDUCATION"] = df["EDUCATION"].astype("category")

y = df["target"]

# ---------------------------------------------------------------- 4. MODEL (5-fold CV)
params = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=600,
    min_child_samples=50,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
oof = np.zeros(len(f))
models, importances = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(skf.split(f, y)):
    m = lgb.LGBMClassifier(**params)
    m.fit(
        f.iloc[tr], y.iloc[tr],
        eval_set=[(f.iloc[va], y.iloc[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof[va] = m.predict_proba(f.iloc[va])[:, 1]
    models.append(m)
    importances.append(m.feature_importances_)

# ---------------------------------------------------------------- 5. EVALUATE
auc = roc_auc_score(y, oof)

# KS statistic: max separation between cumulative distributions of scores
order = np.argsort(-oof)
y_sorted = y.values[order]
cum_bad = np.cumsum(y_sorted) / y.sum()
cum_good = np.cumsum(1 - y_sorted) / (len(y) - y.sum())
ks = float(np.max(cum_bad - cum_good))

# Threshold chosen on OOF to maximize accuracy (the bank's stated metric),
# but we report recall/precision alongside because accuracy alone is gameable.
thresholds = np.linspace(0.1, 0.9, 81)
accs = [accuracy_score(y, oof >= t) for t in thresholds]
t_best = float(thresholds[int(np.argmax(accs))])
pred = oof >= t_best

metrics = {
    "dataset": "UCI Taiwan credit default (30k, 22.1% default rate)",
    "cv": "5-fold stratified, out-of-fold metrics",
    "auc": round(float(auc), 4),
    "ks": round(ks, 4),
    "accuracy_at_best_threshold": round(float(np.max(accs)), 4),
    "threshold": t_best,
    "default_recall": round(float(recall_score(y, pred)), 4),
    "default_precision": round(float(precision_score(y, pred)), 4),
    "baseline_accuracy_predict_all_good": round(float(1 - y.mean()), 4),
}
print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------- 6. EXPLAIN
imp = (
    pd.DataFrame({"feature": f.columns, "importance": np.mean(importances, axis=0)})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\nTop 10 risk drivers:")
print(imp.head(10).to_string(index=False))

imp.to_csv(OUT / "uci_feature_importance.csv", index=False)
with open(OUT / "uci_baseline_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
print(f"\nSaved: {OUT / 'uci_baseline_metrics.json'}")