"""
Stage 2a: Lending Club data preparation.

Reads the raw 2.26M-loan file, keeps the columns we need, cleans them,
derives model features AND the survival fields (duration, event) that
Layer 2 (months-to-default) will use, then saves a compact parquet.

Key decisions (documented for the PPT):
- Loans are "portfolio accounts": we model default risk of running loans.
- Survival fields: event = charged off / default;
  duration_months = issue date -> last payment date (event) or data end (censored).
- We deliberately EXCLUDE post-outcome columns (recoveries, settlement,
  last_fico_*, hardship_*) - they leak the outcome.

Run:  .venv\\Scripts\\python.exe src\\prep_lending_club.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "lending_club" / "accepted_2007_to_2018Q4.csv.gz"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

USECOLS = [
    # identity / terms
    "id", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "issue_d", "purpose", "application_type",
    # borrower profile (internal-database pillar)
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "addr_state", "dti",
    # credit history & behavior (borrower-behavior pillar)
    "earliest_cr_line", "fico_range_low", "fico_range_high",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc",
    "revol_bal", "revol_util", "bc_util", "percent_bc_gt_75",
    "acc_open_past_24mths", "avg_cur_bal", "mort_acc", "pub_rec_bankruptcies",
    "num_accts_ever_120_pd", "num_tl_90g_dpd_24m", "pct_tl_nvr_dlq",
    "mths_since_last_delinq", "tot_hi_cred_lim", "total_rev_hi_lim",
    # outcome / survival
    "loan_status", "last_pymnt_d",
]

print("Loading raw file (2.26M rows, gzip) ...")
df = pd.read_csv(RAW, usecols=USECOLS, low_memory=False)
print("Loaded:", df.shape)

# ------------------------------------------------------------- CLEAN
df = df.dropna(subset=["loan_status", "issue_d", "loan_amnt"])

# Dates come as 'Dec-2015'
for c in ["issue_d", "last_pymnt_d", "earliest_cr_line"]:
    df[c] = pd.to_datetime(df[c], format="%b-%Y", errors="coerce")

# Percent columns may arrive as '13.5%' strings or floats depending on version
for c in ["int_rate", "revol_util"]:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c].astype(str).str.rstrip("%"), errors="coerce")

df["term_months"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0])
df["emp_length_yrs"] = (
    df["emp_length"].astype(str)
    .str.replace("10+ years", "10", regex=False)
    .str.replace("< 1 year", "0", regex=False)
    .str.extract(r"(\d+)")[0].astype(float)
)

# ------------------------------------------------------------- TARGET + SURVIVAL FIELDS
BAD = {"Charged Off", "Default",
       "Does not meet the credit policy. Status:Charged Off"}
GOOD_DONE = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

df["event"] = df["loan_status"].isin(BAD).astype(int)
df["completed"] = df["loan_status"].isin(BAD | GOOD_DONE).astype(int)

DATA_END = df["last_pymnt_d"].max()  # ~end of observation window
end_date = df["last_pymnt_d"].fillna(DATA_END)
df["duration_months"] = (
    (end_date.dt.year - df["issue_d"].dt.year) * 12
    + (end_date.dt.month - df["issue_d"].dt.month)
).clip(lower=0)
# Censored-at-data-end for loans still running
running = df["completed"] == 0
df.loc[running, "duration_months"] = (
    (DATA_END.year - df.loc[running, "issue_d"].dt.year) * 12
    + (DATA_END.month - df.loc[running, "issue_d"].dt.month)
)

# ------------------------------------------------------------- FEATURES
df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df["credit_hist_months"] = (
    (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
    + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
)
inc = df["annual_inc"].clip(lower=1)
df["loan_to_income"] = df["loan_amnt"] / inc
df["payment_to_income"] = df["installment"] / (inc / 12)
df["balance_to_income"] = df["revol_bal"] / inc
df["util_x_delinq"] = df["revol_util"].fillna(0) * (df["delinq_2yrs"].fillna(0) > 0)
df["issue_year"] = df["issue_d"].dt.year

drop = ["term", "emp_length", "fico_range_low", "fico_range_high",
        "earliest_cr_line", "loan_status"]
df = df.drop(columns=drop)

df.to_parquet(OUT / "lc_clean.parquet", index=False)
print("Saved:", OUT / "lc_clean.parquet", df.shape)
print("\nStatus mix: event(default)=", df["event"].mean().round(4),
      "| completed=", df["completed"].mean().round(4))
print("Duration (defaults) median months:",
      df.loc[df.event == 1, "duration_months"].median())
print("Issue years:", df["issue_year"].min(), "-", df["issue_year"].max())