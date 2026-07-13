"""
CreditPulse - Default Risk Early-Warning Dashboard
IDBI Innovate 2026 | Track 4: Default Prediction Model | Team Credit Pulse

Run locally:  .venv\\Scripts\\streamlit.exe run app\\streamlit_app.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("creditpulse")

st.set_page_config(page_title="CreditPulse — Early Warning System",
                   page_icon="💓", layout="wide")

DATA = Path(__file__).resolve().parent / "data"

RAG_COLORS = {"RED": "#d62728", "AMBER": "#ff9f1c", "GREEN": "#2a9d3a"}
RAG_EMOJI = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}
ACTIONS = {
    "RED": ["Immediate relationship-manager call",
            "Review & consider limit reduction / restructuring options",
            "Move to monthly (or fortnightly) monitoring",
            "Verify income / employment status"],
    "AMBER": ["Add to quarterly watchlist review",
              "Soft outreach: payment reminder & financial-health nudge",
              "Re-score next month; escalate if PD rises further"],
    "GREEN": ["Standard annual review cycle",
              "Eligible for pre-approved offers / cross-sell"],
}


@st.cache_data
def load(data_version: float):
    # data_version (file mtime) is part of the cache key: whenever
    # portfolio.parquet changes on deploy, the cache invalidates itself.
    df = pd.read_parquet(DATA / "portfolio.parquet")
    with open(DATA / "summary.json") as fh:
        summary = json.load(fh)
    return df, summary


df, summary = load((DATA / "portfolio.parquet").stat().st_mtime)

# ------------------------------------------------------------------ header
st.title("💓 CreditPulse — Default Risk Early-Warning System")
st.caption("Predicts **WHEN** an account is likely to default — up to 12 months in advance — "
           "with plain-language reasons for every alert. Human-in-the-loop by design: "
           "CreditPulse flags and explains; the credit officer decides. "
           "(Demo portfolio: 5,000 real historical loans, scored blind on a hold-out year.)")

view = st.radio(
    "Scoring viewpoint",
    ["At disbursal (new loan)", "At month 6 on book (running account)"],
    horizontal=True, key="viewpoint")
if view.startswith("At month 6"):
    n_off = int((df["on_book_m6"] == 0).sum())
    df = df[df["on_book_m6"] == 1].copy()
    df["rag"] = df["rag_m6"]
    df["pd_12m"] = df["pd6_m12"]
    df["pd_6m"] = df["pd6_m6"]
    for _m in range(1, 13):
        df[f"pd_m{_m}"] = df[f"pd6_m{_m}"]
    df["actual_default_12m"] = df["actual_default_12m_m6"]
    st.info(f"**Running-account monitoring view** — the {n_off} accounts that stopped "
            f"paying before month 6 have already left the book. The hazard model scores "
            f"the {len(df):,} survivors over their **next 12 months** (book months 7–18), "
            f"conditioned on having survived to month 6. Same model, no retraining — "
            f"the discrete-time formulation makes mid-life scoring native.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accounts monitored", f"{len(df):,}")
k2.metric("🔴 Red (act now)", int((df.rag == "RED").sum()))
k3.metric("🟡 Amber (watch)", int((df.rag == "AMBER").sum()))
k4.metric("Expected 12-mo defaults", int(df.pd_12m.sum().round()))
k5.metric("Portfolio avg PD (12m)", f"{df.pd_12m.mean():.1%}")

# Single-section navigation (segmented control) instead of st.tabs:
# only ONE section renders per rerun, which structurally prevents the
# known st.tabs glitch where panels bleed content into each other.
SECTIONS = ["📋 Portfolio Watchlist", "👤 Account 360°", "✅ Proof: Model Validation"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0],
                               key="section", label_visibility="collapsed")
section = section or SECTIONS[0]
log.info("section=%s | view=%s", section, view)

# ------------------------------------------------------------------ SECTION 1
if section == SECTIONS[0]:
    c1, c2, c3 = st.columns([1, 1, 2])
    rag_f = c1.multiselect("Risk bucket", ["RED", "AMBER", "GREEN"], default=["RED", "AMBER"])
    grade_f = c2.multiselect("Internal grade", sorted(df.grade.unique()))
    view = df[df.rag.isin(rag_f)]
    if grade_f:
        view = view[view.grade.isin(grade_f)]
    view = view.sort_values("pd_12m", ascending=False)

    show = view[["account_id", "rag", "pd_12m", "pd_6m",
                 "n_rules", "grade", "purpose", "loan_amnt", "revol_util", "dti"]].copy()
    show["rag"] = show["rag"].map(RAG_EMOJI) + " " + view["rag"]
    st.dataframe(
        show, width="stretch", height=430, hide_index=True,
        column_config={
            "account_id": "Account",
            "rag": "Bucket",
            "pd_12m": st.column_config.ProgressColumn(
                "PD within 12 months", format="percent", min_value=0, max_value=0.5),
            "pd_6m": st.column_config.ProgressColumn(
                "PD within 6 months", format="percent", min_value=0, max_value=0.5),
            "n_rules": "EWS rules hit",
            "grade": "Grade", "purpose": "Purpose",
            "loan_amnt": st.column_config.NumberColumn("Loan amt", format="$%d"),
            "revol_util": st.column_config.NumberColumn("Utilization %", format="%.0f%%"),
            "dti": st.column_config.NumberColumn("DTI", format="%.0f"),
        })
    st.caption(f"{len(view):,} accounts in view — sorted by 12-month default probability. "
               "EWS rule triggers follow RBI Early-Warning-Signal style indicators.")

# ------------------------------------------------------------------ SECTION 2
elif section == SECTIONS[1]:
    # Progressive filtering: pick a bucket first so the account list stays
    # small and fast (officers work a queue, not the whole book).
    fc1, fc2 = st.columns([1, 3])
    bucket_pick = fc1.selectbox("Risk bucket", ["RED", "AMBER", "GREEN", "ALL"],
                                index=0, key="bucket_pick")
    pool = df if bucket_pick == "ALL" else df[df.rag == bucket_pick]
    MAX_LIST = 300
    listed = pool.sort_values("pd_12m", ascending=False)["account_id"].tolist()[:MAX_LIST]
    acc = fc2.selectbox(
        f"Select account — {len(listed)} of {len(pool):,} shown, riskiest first",
        listed, index=0, key="account_pick")
    row = df[df.account_id == acc].iloc[0]
    log.info("view=%s | account=%s | rag=%s | pd12=%.3f",
             view, acc, row.rag, row.pd_12m)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"### {RAG_EMOJI[row.rag]} {acc} — "
            f"<span style='color:{RAG_COLORS[row.rag]}'>{row.rag}</span>",
            unsafe_allow_html=True)
        st.markdown(f"**12-month default probability: {row.pd_12m:.1%}** "
                    f"({row.pd_12m / df.pd_12m.mean():.1f}× portfolio average)")
        st.markdown(f"**Risk trajectory: {row.pd_6m:.1%} by month 6 → {row.pd_12m:.1%} "
                    f"by month 12** — the gap between those numbers is the intervention window.")

        curve = pd.DataFrame({
            "Month ahead": np.arange(1, 13),
            "Cumulative default probability": [row[f"pd_m{m}"] for m in range(1, 13)],
        }).set_index("Month ahead")
        st.line_chart(curve, height=260)

        reasons = json.loads(row.reasons)
        rules = json.loads(row.rule_flags)
        if row.rag == "GREEN":
            rel = row.pd_12m / df.pd_12m.mean()
            level = ("LOW" if rel <= 1.0
                     else "within the GREEN range (upper end — periodic review is enough)")
            st.markdown("**Risk notes** *(account is healthy — for awareness, not action)*:")
            st.markdown(f"- ✅ Overall risk is {level}: {row.pd_12m:.1%} over 12 months "
                        f"({rel:.1f}× portfolio average). Items below are this account's "
                        "*relatively* weakest points, not warnings.")
        else:
            st.markdown("**Why this account is flagged (reason codes):**")
        for r in reasons:
            st.markdown(f"- 🧠 *Model:* {r}")
        for r in rules:
            st.markdown(f"- 📏 *EWS rule:* {r}")
        if not reasons and not rules:
            st.markdown("- ✅ No adverse signals at all — clean profile.")

    with right:
        st.markdown("### Account profile")
        st.table(pd.DataFrame({
            "Value": [row.grade, row.purpose, f"${row.loan_amnt:,.0f}",
                      f"${row.annual_inc:,.0f}", f"{row.revol_util:.0f}%",
                      f"{row.dti:.0f}", f"{row.fico:.0f}", f"{row.term_months:.0f} months"],
        }, index=["Internal grade", "Purpose", "Loan amount", "Annual income",
                  "Revolving utilization", "Debt-to-income", "Credit score", "Tenor"]))

        st.markdown("### Recommended actions (officer decides)")
        for a in ACTIONS[row.rag]:
            st.checkbox(a, key=f"{acc}-{a}")

# ------------------------------------------------------------------ SECTION 3
elif section == SECTIONS[2]:
    st.markdown(f"#### These {len(df):,} accounts are real historical loans (2015 vintage) "
                "scored **blind** — the model never saw their outcomes. Here is what actually happened:")
    obs = df.groupby("rag")["actual_default_12m"].mean().to_dict()
    cnt = df["rag"].value_counts().to_dict()
    import altair as alt
    BUCKETS = ["RED", "AMBER", "GREEN"]
    RANGE = [RAG_COLORS[b] for b in BUCKETS]

    colA, colB = st.columns([1, 1.9])

    with colA:
        # Question 1: how is the book bucketed? (composition -> donut)
        st.markdown(f"**How the {len(df):,} accounts are bucketed**")
        comp = pd.DataFrame({
            "label": [f"{b} — {cnt.get(b, 0):,} ({cnt.get(b, 0)/len(df):.0%})"
                      for b in BUCKETS],
            "n": [cnt.get(b, 0) for b in BUCKETS],
        })
        donut = (alt.Chart(comp).mark_arc(innerRadius=55, cornerRadius=3, padAngle=0.02)
                 .encode(theta=alt.Theta("n", sort=None),
                         color=alt.Color("label", sort=None,
                                         scale=alt.Scale(domain=comp["label"].tolist(),
                                                         range=RANGE),
                                         legend=alt.Legend(title=None, orient="bottom",
                                                           columns=1)))
                 .properties(height=260))
        st.altair_chart(donut, width="stretch")

    with colB:
        # Question 2: was the sorting right? (icon array - 100 dots per bucket)
        st.markdown("**Out of every 100 accounts in each bucket — "
                    "how many went bad within 12 months?**")
        rows, hdrs = [], []
        for b in BUCKETS:
            k = int(round(obs.get(b, 0) * 100))
            hdr = f"{b}: {k} of 100"
            hdrs.append(hdr)
            for i in range(100):
                rows.append({"bucket": hdr, "x": i % 10, "y": i // 10,
                             "fill": "BAD" if i < k else b})
        dots = pd.DataFrame(rows)
        # red always = went bad; "stayed fine" wears the bucket's own shade
        FINE = {"RED": "#f3b3b3", "AMBER": "#f5c26b", "GREEN": "#2a9d3a"}
        icon = (alt.Chart(dots).mark_square(size=160)
                .encode(
                    x=alt.X("x:O", axis=None),
                    y=alt.Y("y:O", axis=None),
                    color=alt.Color("fill:N", legend=None,
                                    scale=alt.Scale(
                                        domain=["BAD", "RED", "AMBER", "GREEN"],
                                        range=[RAG_COLORS["RED"], FINE["RED"],
                                               FINE["AMBER"], FINE["GREEN"]])),
                    column=alt.Column("bucket:N", sort=hdrs,
                                      header=alt.Header(title=None, labelFontSize=14,
                                                        labelFontWeight="bold")),
                )
                .properties(width=185, height=185))
        st.altair_chart(icon)
        st.caption("Each square = one account out of 100 in that bucket. "
                   "🟥 red = actually stopped repaying · other shades = stayed fine. "
                   "GREEN bucket: right about ~97 of 100; RED bucket: concentrates "
                   "more than double the book's trouble.")
    red_r, green_r = obs.get("RED", 0), obs.get("GREEN", 1e-9)
    c1, c2, c3 = st.columns(3)
    c1.metric("RED bucket — actually defaulted", f"{red_r:.1%}")
    c1.caption(f"≈ 1 in {round(1 / max(red_r, 1e-9))} RED accounts went bad")
    c2.metric("GREEN bucket — actually defaulted", f"{green_r:.1%}")
    c2.caption(f"≈ 1 in {round(1 / max(green_r, 1e-9))} GREEN accounts went bad")
    c3.metric("Risk separation (RED vs GREEN)", f"{red_r / max(green_r, 1e-9):.1f}×")
    c3.caption("how much more trouble RED holds than GREEN")

    st.markdown("##### Model report card *(measured on 283,173 unseen loans)*")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Accuracy", "94.8%")
    r1.caption("bank's ask: above 90% ✓ (12-month question)")
    r2.metric("AUC", "0.72")
    r2.caption("ranking power: riskier account ranked higher 72 times out of 100")
    r3.metric("KS", "0.33")
    r3.caption("scorecard separation; 0.30+ is a usable scorecard")
    r4.metric("Lift @ top 15%", "2.5×")
    r4.caption("watchlist catches 2.5× more defaulters than random review")
    r5.metric("Promise vs reality", "14.6% → 14.2%")
    r5.caption("model predicted 14.6% of the riskiest tenth would go bad — "
               "14.2% actually did (near-exact)")

    st.markdown(
        "**The takeaway:** the buckets were assigned *before* knowing any outcome — and "
        "the outcomes followed the colors. RED + AMBER (~30% of the book) captured **52% "
        "of all 12-month defaults at 2.5× lift**; bucket sizes are a capacity dial the "
        "bank sets, not a model ceiling."
    )
    with st.expander("Full validation details & fine print"):
        st.markdown(
            "- 'Default' here means the account **stops repaying** (and is subsequently "
            "charged off) — the earliest actionable signal; formal charge-off is recorded "
            "months later.\n"
            "- Headline accuracy on the bank's stated question (stops repaying within 12 "
            "months): **94.8%** (base rate 5.2%; a do-nothing model scores the same, which "
            "is why we lead with ranking power and calibration).\n"
            "- Ranking power **AUC 0.72, KS 0.33** on 283,173 unseen loans; riskiest decile "
            "predicted 14.6% vs observed 14.2% (full decile table in the deck appendix).\n"
            "- Validation cohort: 36-month personal loans (the fully-observable segment of "
            "the public data); the architecture itself is term-agnostic.\n"
            "- With the bank's internal **monthly repayment behavior** (sandbox stage), "
            "ranking power improves materially — shown directionally on a 30k-customer "
            "behavioral dataset; IDBI's own data connects via the sandbox APIs."
        )