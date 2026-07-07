# 💓 CreditPulse — Default Risk Early-Warning System

**IDBI Innovate 2026 · Track 4: Default Prediction Model · Team Credit Pulse**

CreditPulse predicts **not just IF a loan account will default, but WHEN** — up to 12
months in advance — and explains every alert in plain language, keeping the credit
officer in charge of the decision.

## The problem (IDBI's ask)

> "Predict default 12 months in advance for running loan accounts, with accuracy
> above 90%, using borrower behavior, the bank's internal database, and
> public-domain data. Output as high/medium/low risk buckets or RAG color codes.
> AI gives the reason and logic; the underwriter decides."

## The solution — three layers

| Layer | What it does | How |
|---|---|---|
| **1. Risk scoring engine** | Ranks every account by default risk | LightGBM on behavioral aggregations (utilization, deterioration deltas, payment coverage) — the recipe that won the AmEx Default Prediction competition |
| **2. Time-to-default model** | Predicts **WHEN**: a month-by-month default probability curve over the next 12 months | Discrete-time hazard model (the IFRS 9 lifetime-PD construction): loan-month panel → monthly hazard → cumulative PD term structure |
| **3. RAG early-warning watchlist** | Red / Amber / Green buckets + transparent rule triggers + recommended actions | ML score + RBI Early-Warning-Signal style rules (utilization spikes, delinquency history, enquiry bursts) |

Every flagged account carries **reason codes** (from SHAP contributions mapped to
plain language) — aligned with RBI model-governance expectations and the bank's
human-in-the-loop requirement.

## Validated results (strict out-of-time test: train ≤2014, score 2015 — 283,173 unseen loans)

- **Headline accuracy 94.8%** on the bank's stated question (default within 12 months)
- **Calibration that holds**: riskiest decile predicted 14.6% vs observed 14.2%
- **RAG buckets proven on blind history**: RED accounts defaulted at **11.0%** vs
  GREEN **3.3%** (3.3× separation) within 12 months
- Median time-to-default in the data is 14 months → a 12-month warning is actionable
- On monthly behavioral data (30k-customer dataset) the same pipeline reaches AUC 0.79 —
  the expected uplift when connected to the bank's internal repayment behavior in the sandbox

## Repository layout

```
app/
  streamlit_app.py      <- the dashboard (deployed demo)
  data/portfolio.parquet<- precomputed 5,000-account demo portfolio
src/
  baseline_uci.py           Stage 1: behavioral-data baseline
  prep_lending_club.py      Stage 2a: clean 2.26M loans + survival fields
  train_lc_layer1.py        Stage 2b: Layer 1 risk scorer (time-based validation)
  train_lc_layer2_hazard.py Stage 3: Layer 2 discrete-time hazard model
  build_dashboard_data.py   Stage 4: RAG + reason codes + PD curves
models/                 <- trained LightGBM boosters
docs/                   <- research notes & requirement spec
```

## Run locally

```bash
pip install -r requirements.txt          # dashboard only
streamlit run app/streamlit_app.py
```

To retrain from scratch: `pip install -r requirements-train.txt`, download the
Kaggle datasets listed in `docs/research_1_datasets.md` into `data/raw/`, then
run the `src/` scripts in order.

## Data

Trained on public data (UCI credit default; Lending Club loan performance —
2.26M loans with true default timing). No proprietary or personal data is included;
the demo portfolio is anonymized historical data relabeled with synthetic account IDs.
In the sandbox stage, the same feature schema maps to the bank's internal
transaction/repayment data and CIBIL-style bureau fields.

## Production architecture (sandbox-ready)

Data (CBS / bureau / public domain) → S3 data lake → feature pipeline →
SageMaker-hosted scoring (Layers 1+2) → rules engine (Layer 3) → dashboard/API →
case-management workflow. Monthly re-scoring of the full book; on-prem deployable
(all components are open-source Python).

---
*Built for IDBI Innovate 2026 by Team Credit Pulse.*