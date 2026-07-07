# Track 4 — Default Prediction Model: Requirement Spec
Compiled from: Orientation session + Problem Statement Explainer/AMA session (speaker: Salil Bakshi, DGM, IDBI Bank).

## Official track lineup (per explainer session)
1. Digital Wealth Management (avatar-based, hybrid text+voice, multi-language, lead-gen to RM)
2. Prospect Assist AI (loan lead quality: interest + repayment capacity from bank statements/behavior; current lead conversion <1%)
3. Financial Health Score (new-to-bank/new-to-credit assessment via alternative data: electricity consumption, EPFO, fuel costs)
4. **Default Prediction Model (OUR TRACK)**
5. Open/Novel Innovation

## Track 4 stated requirements (verbatim intent)
- Target: **loan accounts already running in the bank** (portfolio monitoring, NOT application scoring).
- Predict default **12 months in advance**.
- Accuracy: bank insists **> 90%** (a participant challenged this re: imbalanced data; bank re-affirmed ">90%, that's fine"). → Strategy: report accuracy >90% headline AND AUC/recall/precision for statistical honesty.
- Model must encompass THREE data pillars: **(1) borrower behavior, (2) bank's internal database, (3) public-domain data**.
- Output format (AMA answer): **bucket-wise categorization — high/medium/low risk — or RAG (Red-Amber-Green) color-coded**. ← Direct validation of our watchlist design.
- Scoring framework (AMA answer): **hybrid** — segment-dependent (vintage, occupation, experience, qualification, age group) feeding a **unified score**.
- Explainability (AMA answer): "AI will give us the **reason and logic**; credit decision stays with the underwriter; human intervention will NOT be removed." ← Human-in-the-loop + reason codes required.

## Cross-cutting rules from AMA
- AI coding agents (Claude Code etc.) explicitly ALLOWED; code must be original (no copyright issues); abide by RBI AI norms.
- Sandbox (for shortlisted teams) is **AWS-based**; bank will provide internal APIs + synthetic datasets + dummy request/response sets. GCP only if callable via API; prefer AWS services.
- Bank currently has **NO LLM/AI models in production** (all UAT) — deployable practicality is a selling point.
- Solutions evaluated for on-prem or cloud deployability.
- Tech stack must stay within regulatory guidelines (RBI, SEBI, DPDP as applicable).

## Implications adopted into our plan
1. RAG buckets + risk categories = primary output UI (bank asked for exactly this).
2. Add third data pillar: public-domain signals (RBI sectoral stress/GNPA, macro indicators) as features/architecture slot.
3. Add segmentation: segment-aware modeling (e.g., thin-history vs seasoned; grade/purpose segments) + unified score.
4. Dual metric reporting: headline accuracy >90% + AUC/recall/KS to satisfy both the bank's ask and statistical rigor.
5. Frame everything as PORTFOLIO MONITORING of running accounts (monthly re-scoring), not loan-application decisioning.
6. Architecture slide must show AWS deployability (S3 + SageMaker/Lambda + API Gateway) even though prototype deploys on Streamlit Cloud.
7. Pitch line: human-in-the-loop early-warning co-pilot for underwriters, aligned with RBI EWS/RFA framework.