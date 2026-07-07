# Dataset Research — Track 4 (Credit Utilization & Default Prediction)

Research date: 2026-07-03. Goal: datasets supporting time-to-default / survival modeling + early-warning demo, < 5 GB, fast access, 6-day solo build.

## Ranked shortlist

### #1 — Lending Club ("All Lending Club loan data" by wordsforthewise, Kaggle dataset)
- **Why #1:** only large dataset where true time-to-default is directly computable: `issue_d` → `last_pymnt_d` for Charged Off loans = survival time in months; `loan_status` = event indicator; Current/Fully Paid = censored. Textbook survival setup.
- Has `revol_util` (credit utilization), `dti`, `delinq_2yrs`, payment amounts, FICO ranges.
- 2.26M loans, 151 cols, ~640 MB gz / ~1.6 GB CSV. Plain Kaggle account download (NO competition rules).
- URL: kaggle.com/datasets/wordsforthewise/lending-club
- Limitation: loan-level snapshot, not monthly panel — pair with AmEx for trajectories.

### #2 — AmEx Default Prediction (use raddar parquet mirror, NOT raw 48 GB CSVs)
- Up to 13 monthly statements × 458,913 customers, 190 features (D_=delinquency, S_=spend, P_=payment, B_=balance, R_=risk). Statement date `S_2`.
- **Best for the early-warning demo**: "watch risk score climb month-over-month".
- Label is binary (default within 120 days of last statement) — NO default date → dynamic early-warning framing, not survival.
- Use: `kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format` (~1.7 GB train parquet; skip test). Plain dataset, no rules acceptance.
- Features anonymized (weaker India narrative slide).

### #3 — Freddie Mac Single-Family Loan-Level SAMPLE
- 50k loans/vintage year, origination + monthly performance (delinquency status per month, zero-balance codes). Real survival panels.
- Free registration required (Clarity portal, ~10 min friction). US mortgages. Optional.

## Other candidates (evaluated, deprioritized)
- **UCI Taiwan credit default** — ALREADY DOWNLOADED (`data/raw/uci_credit_default`), 30k rows, 22.1% default rate, 6 months PAY_/BILL_AMT/PAY_AMT history. Quick baseline + demo-friendly.
- **Home Credit Default Risk** — great repayment-behavior tables (installments_payments: due vs paid), but binary target, no default date. Feature-engineering reference only. Requires competition rules acceptance.
- **Give Me Some Credit** — snapshot only; 30-min baseline at most. RevolvingUtilization = strongest predictor lesson.
- **Berka/PKDD'99 Czech bank** — real transaction ledger + 682 loans (76 bad). Direct download, 70 MB. Optional transaction-behavior garnish; too few events for survival stats.
- **Fannie Mae** — 48 GB compressed; skip (Freddie sample dominates).
- **Indian datasets:** NO public loan-level Indian dataset with a time dimension exists. L&T Vehicle Loan Default (kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction, 233k loans, CIBIL-style score) and "Leading Indian Bank & CIBIL" (51k rows) = transfer-evidence slide, not modeling base. RBI/data.gov.in = aggregates only.

## Recommended stack (~3.5–4 GB)
LC (primary survival) + AmEx raddar parquet (early-warning layer) + UCI (baseline, in hand) + Berka (optional).

## Access checklist
- kaggle.json token at `C:\Users\BaskarBarijatham\.kaggle\kaggle.json` — REQUIRED, user action.
- LC, raddar AmEx, Berka, L&T: plain datasets — token is enough.
- Only original AmEx/HomeCredit/GMSC competition files need website "Join/Accept rules" — we avoid those.