# app.py
# Strategic Evaluation of AI-Driven Risk Management in Engineering Projects
# Non-editable, comprehensible data analytics dashboard

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategic Evaluation • AI-Driven Risk Management",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Fixed Parameters (non-editable)
# ─────────────────────────────────────────────────────────────────────────────
RISK_ID = ["RISK_ID_1","RISK_ID_2","RISK_ID_3"]
RISK_ASSESS = ["RISK_ASSESS_1","RISK_ASSESS_2","RISK_ASSESS_3"]
RISK_MIT = ["RISK_MIT_1","RISK_MIT_2","RISK_MIT_3"]
AI_READY = ["AI_READY_1","AI_READY_2","AI_READY_3","AI_READY_4","AI_READY_5","AI_READY_6"]
ALL_RISK = RISK_ID + RISK_ASSESS + RISK_MIT
REQUIRED = {"ID","AGE","GENDER","ROLE","EXP","PTYPE"} | set(ALL_RISK) | set(AI_READY)

# Fixed strategy settings
K_CLUSTERS = 3
CHAMPION_Z = 1.0
VULN_RISK_Z = 0.5
VULN_AI_Z = -0.5
ANOMALY_SHARE = 0.10

SAMPLE_CSV = """ID,AGE,GENDER,ROLE,EXP,PTYPE,RISK_ID_1,RISK_ID_2,RISK_ID_3,RISK_ASSESS_1,RISK_ASSESS_2,RISK_ASSESS_3,RISK_MIT_1,RISK_MIT_2,RISK_MIT_3,AI_READY_1,AI_READY_2,AI_READY_3,AI_READY_4,AI_READY_5,AI_READY_6
1,27,1,2,4,4,5,4,3,2,2,5,5,5,4,2,2,4,2,5,2
2,40,1,2,3,1,2,5,5,3,4,2,2,3,5,5,5,2,5,4,5
3,35,1,2,4,4,4,4,5,2,2,5,2,5,3,2,4,2,2,4,2
4,31,2,1,2,4,5,2,5,2,3,3,5,2,2,5,5,5,2,4,5
5,28,2,2,3,4,5,5,5,4,3,3,5,2,4,5,4,4,4,5,5
6,41,2,1,4,3,3,2,5,3,5,3,4,5,2,4,5,2,3,3,5
7,27,2,2,1,3,4,5,3,5,3,4,3,4,2,4,2,5,5,3,2
8,39,2,3,2,3,4,5,3,5,4,4,5,4,3,3,5,4,4,5,4
9,43,1,3,4,1,2,3,4,4,2,5,2,2,5,4,2,5,2,4,5
10,31,2,1,1,4,4,2,5,4,5,5,4,2,3,3,5,3,3,4,3
11,31,2,3,4,3,2,4,3,3,2,2,5,4,3,3,5,2,4,3,2
12,44,1,3,1,3,4,4,4,5,2,2,2,4,3,3,3,3,3,4,5
13,41,2,2,2,1,3,2,5,5,4,5,2,5,3,3,2,4,4,2,5
14,24,1,1,3,3,4,4,2,5,3,4,3,5,4,3,3,3,3,3,3
15,28,2,2,1,1,2,4,4,2,3,3,4,4,5,2,2,3,2,4,4
16,44,1,2,4,2,2,2,3,5,2,5,4,5,2,4,3,4,5,3,4
17,23,2,2,2,3,3,5,2,2,5,2,3,5,2,3,4,5,2,2,4
18,42,2,2,1,2,4,2,2,3,3,4,3,3,5,5,5,3,5,3,5
19,41,1,2,4,1,4,5,2,2,5,5,4,5,2,4,2,5,3,3,5
20,22,1,2,4,4,3,4,4,3,3,5,4,2,5,4,2,3,4,3,5
21,44,1,2,4,3,4,4,3,5,5,4,5,5,2,3,5,5,2,3,2
22,32,1,1,1,1,4,4,2,5,4,5,5,5,3,2,2,4,5,2,3
23,26,1,3,1,4,2,3,5,3,5,4,3,4,2,3,2,3,2,5,5
24,22,1,2,1,4,4,5,2,4,4,3,5,2,5,2,3,4,5,3,3
25,41,1,2,3,2,4,3,2,3,4,4,2,5,5,5,2,3,2,3,3
26,21,1,2,1,1,3,3,4,4,5,4,5,5,4,4,4,5,4,4,4
27,32,2,2,1,4,3,2,4,2,4,4,5,3,5,5,4,4,5,3,2
28,42,2,2,1,3,5,3,3,2,2,5,2,5,5,2,5,3,3,4,5
29,32,1,2,3,3,2,2,5,2,4,5,3,5,4,2,2,3,5,5,4
30,37,2,3,1,2,4,2,4,5,3,4,2,3,4,5,5,3,3,2,4
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def cronbach_alpha(df: pd.DataFrame) -> float:
    df = df.dropna()
    k = df.shape[1]
    if k < 2:
        return np.nan
    v_sum = df.var(ddof=1).sum()
    v_tot = df.sum(axis=1).var(ddof=1)
    if v_tot == 0:
        return np.nan
    return (k/(k-1)) * (1 - v_sum / v_tot)

def _safe_z(series: pd.Series) -> pd.Series:
    m = series.mean()
    s = series.std(ddof=1)
    if pd.isna(s) or s == 0:
        return pd.Series(0.0, index=series.index)
    return (series - m) / s

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RISK_SUM"] = out[ALL_RISK].sum(axis=1)
    out["RISK_SCORE"] = out["RISK_SUM"]/9.0  # 9 risk items
    out["AI_SUM"] = out[AI_READY].sum(axis=1)
    out["AI_SCORE"] = out["AI_SUM"]/6.0      # 6 AI items
    # Robust z-scores (avoid divide-by-zero)
    out["Z_RISK"] = _safe_z(out["RISK_SCORE"])
    out["Z_AI"]   = _safe_z(out["AI_SCORE"])
    return out

def pct(mask: pd.Series) -> str:
    return f"{(mask.mean()*100):.0f}%"

# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion (non-editable parameters)
# ─────────────────────────────────────────────────────────────────────────────
st.title("Strategic Evaluation of AI-Driven Risk Management in Engineering Projects")
st.caption("Data analytics dashboard")

df_raw = pd.read_csv(StringIO(SAMPLE_CSV))

missing = sorted(list(REQUIRED - set(df_raw.columns)))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Compute analytics
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.sort_values("ID").reset_index(drop=True)
df = compute_scores(df)

alpha = {
    "Risk Identification": cronbach_alpha(df[RISK_ID]),
    "Risk Assessment": cronbach_alpha(df[RISK_ASSESS]),
    "Risk Mitigation": cronbach_alpha(df[RISK_MIT]),
    "AI Readiness": cronbach_alpha(df[AI_READY]),
}

# K-means clustering on standardized Z-scores
scaler = StandardScaler()
X = scaler.fit_transform(df[["Z_RISK","Z_AI"]].values)
# Fix: n_init as int for compatibility (avoids 'auto' errors on older sklearn)
km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=42)
labels = km.fit_predict(X)
df["CLUSTER"] = labels

# Order clusters by (Z_RISK + Z_AI) and map to strategic names
centers = (
    df.groupby("CLUSTER")[["Z_RISK","Z_AI"]].mean()
    .assign(SUM=lambda d: d["Z_RISK"]+d["Z_AI"])
    .sort_values("SUM")
)
names = ["Traditional / Unprepared","Developing / Transitional","Advanced / Ready"]
cluster_name_map = {int(cid): names[i] for i, cid in enumerate(centers.index)}
df["SEGMENT"] = df["CLUSTER"].map(cluster_name_map)

# Silhouette score
try:
    silhouette = silhouette_score(X, labels)
except Exception:
    silhouette = np.nan

# Champions & Vulnerable (fixed thresholds)
champions = df[(df["Z_RISK"]>=CHAMPION_Z) & (df["Z_AI"]>=CHAMPION_Z)]
vulnerable = df[(df["Z_RISK"]>=VULN_RISK_Z) & (df["Z_AI"]<=VULN_AI_Z)]

# Isolation Forest anomalies (fixed contamination) — guard against NaNs
iso = IsolationForest(n_estimators=100, contamination=ANOMALY_SHARE, random_state=42)
df["ANOMALY"] = iso.fit_predict(df[["Z_RISK","Z_AI"]].fillna(0.0))
df["ANOMALY_FLAG"] = df["ANOMALY"].map({-1:"Anomaly", 1:"Normal"})

# Correlations (be explicit; avoids future dtype warnings)
corr_targets = df[["AGE","GENDER","ROLE","EXP","PTYPE","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI"]].corr(numeric_only=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Respondents", f"{len(df):,}")
k2.metric("Mean Risk Score", f"{df['RISK_SCORE'].mean():.2f}")
k3.metric("Mean AI Score", f"{df['AI_SCORE'].mean():.2f}")
k4.metric("Silhouette (k=3)", f"{silhouette:.3f}" if pd.notna(silhouette) else "n/a")
k5.metric("AI Readiness α", f"{alpha['AI Readiness']:.2f}" if pd.notna(alpha["AI Readiness"]) else "n/a")

with st.expander("Reliability (Cronbach’s α) by Construct"):
    st.table(pd.DataFrame(alpha, index=["Cronbach α"]).T)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_seg, tab_champ, tab_insight, tab_plan, tab_data = st.tabs(
    ["Segments & Misalignment","Champions & Vulnerable","Strategic Insights","90-Day Plan","Data & Downloads"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Segments & Misalignment
# ─────────────────────────────────────────────────────────────────────────────
with tab_seg:
    cA,cB,cC,cD = st.columns(4)
    cA.metric("Traditional / Unprepared", pct(df["SEGMENT"]=="Traditional / Unprepared"))
    cB.metric("Developing / Transitional", pct(df["SEGMENT"]=="Developing / Transitional"))
    cC.metric("Advanced / Ready", pct(df["SEGMENT"]=="Advanced / Ready"))
    cD.metric("Risk–AI Misalignment", pct((df["Z_RISK"]>=VULN_RISK_Z) & (df["Z_AI"]<=VULN_AI_Z)))

    fig = px.scatter(
        df, x="Z_RISK", y="Z_AI", color="SEGMENT",
        hover_data=["ID","AGE","ROLE","EXP","RISK_SCORE","AI_SCORE"],
        title="Z-Risk vs Z-AI by Strategic Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Segment Summary**")
    seg_summary = df.groupby("SEGMENT")[["RISK_SCORE","AI_SCORE","Z_RISK","Z_AI"]].agg(["mean","min","max"]).round(2)
    st.dataframe(seg_summary)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Champions & Vulnerable
# ─────────────────────────────────────────────────────────────────────────────
with tab_champ:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### AI Champions (High Z-Risk & High Z-AI)")
        st.caption(f"Fixed thresholds: Z-Risk ≥ {CHAMPION_Z}, Z-AI ≥ {CHAMPION_Z}")
        st.dataframe(champions[["ID","SEGMENT","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI","ROLE","EXP"]])
    with c2:
        st.markdown("### Vulnerable (High Z-Risk & Low Z-AI)")
        st.caption(f"Fixed definition: Z-Risk ≥ {VULN_RISK_Z} and Z-AI ≤ {VULN_AI_Z}")
        st.dataframe(vulnerable[["ID","SEGMENT","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI","ROLE","EXP"]])

    st.markdown("### Anomaly Map (Isolation Forest)")
    a_fig = px.scatter(
        df, x="Z_RISK", y="Z_AI", color="ANOMALY_FLAG",
        hover_data=["ID","SEGMENT","RISK_SCORE","AI_SCORE"],
        title="Isolation Forest Outliers on Z-Scores"
    )
    st.plotly_chart(a_fig, use_container_width=True)

    st.write("Flagged anomalies:")
    st.dataframe(df.loc[df["ANOMALY_FLAG"]=="Anomaly", ["ID","SEGMENT","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI"]])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Strategic Insights (auto narrative, static visuals)
# ─────────────────────────────────────────────────────────────────────────────
with tab_insight:
    st.markdown("### Correlation Heatmap (Key Fields)")
    try:
        heat = px.imshow(
            corr_targets.round(2),
            text_auto=True, aspect="auto", title="Correlation Heatmap"
        )
    except TypeError:
        # Older plotly without text_auto
        heat = px.imshow(
            corr_targets.round(2),
            aspect="auto", title="Correlation Heatmap"
        )
    st.plotly_chart(heat, use_container_width=True)

    # Lightweight “driver” view: mean AI score per category vs overall
    st.markdown("### Key Driver Lift (vs Overall AI Mean)")
    overall_ai = df["AI_SCORE"].mean()
    lifts = []
    for col in ["ROLE","EXP","SEGMENT"]:
        for val, sub in df.groupby(col):
            lifts.append({
                "Attribute": f"{col} = {val}",
                "AI_mean": sub["AI_SCORE"].mean(),
                "Lift_vs_overall": sub["AI_SCORE"].mean() - overall_ai,
                "n": len(sub)
            })
    lift_df = pd.DataFrame(lifts).sort_values("Lift_vs_overall", ascending=False)
    st.dataframe(lift_df.style.format({"AI_mean":"{:.2f}","Lift_vs_overall":"{:+.2f}"}))

    # Auto Executive Summary (static)
    st.markdown("### Executive Summary")
    n = len(df)
    p_trad = pct(df["SEGMENT"]=="Traditional / Unprepared")
    p_mid  = pct(df["SEGMENT"]=="Developing / Transitional")
    p_adv  = pct(df["SEGMENT"]=="Advanced / Ready")
    misalign = pct((df["Z_RISK"]>=VULN_RISK_Z) & (df["Z_AI"]<=VULN_AI_Z))
    n_champ = len(champions)
    n_vuln  = len(vulnerable)

    summary = f"""
**Overview.** We analyzed **{n}** responses and segmented them into **{p_trad} Traditional/Unprepared**, **{p_mid} Developing/Transitional**, and **{p_adv} Advanced/Ready** groups.  
**Risk–AI Misalignment.** About **{misalign}** show higher risk maturity/exposure but **low AI readiness**, indicating priority targets for training and tooling.  
**Personas.** Identified **{n_champ} AI Champions** to spearhead pilots and **{n_vuln} Vulnerable** cases requiring immediate support.  
**Implication.** Focus on closing misalignment, scaling proven AI use-cases via champions, and institutionalizing cadence (periodic assessment, anomaly alerts, and budget gates).
"""
    st.markdown(summary)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — 90-Day Plan (static)
# ─────────────────────────────────────────────────────────────────────────────
with tab_plan:
    st.markdown("### 90-Day Action Plan (Static)")
    plan = pd.DataFrame({
        "Workstream":["Foundational Training","Champion Pilot","Governance & Cadence"],
        "Owner":["PMO + HR","AI Champion + Eng Lead","PMO"],
        "Month 1":["Risk/AI bootcamps (Traditional)","Select pilot use-case + KPIs","Define monthly risk reviews"],
        "Month 2":["Mentorship & tools rollout","Build & deploy MVP","Integrate anomaly alerts"],
        "Month 3":["Assessment & certifications","Measure ROI; share playbook","Budget gates tied to mitigation index"]
    })
    st.dataframe(plan)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Data & Downloads (static outputs)
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Enriched Dataset")
    st.dataframe(df)

    # Assemble Markdown report
    md_parts = []
    md_parts.append("# Strategic Reporting & Recommendations\n")
    md_parts.append("## Executive Summary\n")
    md_parts.append(summary)
    md_parts.append("\n## Segments (per respondent)\n")
    md_parts.append(df[["ID","SEGMENT","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI"]].to_markdown(index=False))
    md_parts.append("\n## 90-Day Action Plan\n")
    md_parts.append(plan.to_markdown(index=False))
    report_md = "\n".join(md_parts)

    st.markdown("### Downloads")
    st.download_button(
        "Download Strategic Report (.md)",
        data=report_md.encode("utf-8"),
        file_name="strategic_report.md"
    )
    st.download_button(
        "Download Enriched Dataset (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="enriched_ai_risk_dataset.csv"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Analytics (extra)
# ─────────────────────────────────────────────────────────────────────────────
tab_analytics, = st.tabs(["Analytics"])

with tab_analytics:
    # ---- Helper prep ----
    ROLE_MAP = {1: "Student", 2: "Engineer", 3: "Project Manager", 4: "Other"}
    GENDER_MAP = {1: "Male", 2: "Female", 3: "Prefer not to say"}
    PTYPE_MAP = {1: "Construction", 2: "IT/Software", 3: "Manufacturing", 4: "Energy", 5: "Other"}

    dfa = df.copy()
    dfa["ROLE_LBL"] = dfa["ROLE"].map(ROLE_MAP).fillna(dfa["ROLE"].astype(str))
    dfa["GENDER_LBL"] = dfa["GENDER"].map(GENDER_MAP).fillna(dfa["GENDER"].astype(str))
    dfa["PTYPE_LBL"] = dfa["PTYPE"].map(PTYPE_MAP).fillna(dfa["PTYPE"].astype(str))

    # Risk subscale means
    RISK_ID = ["RISK_ID_1","RISK_ID_2","RISK_ID_3"]
    RISK_ASSESS = ["RISK_ASSESS_1","RISK_ASSESS_2","RISK_ASSESS_3"]
    RISK_MIT = ["RISK_MIT_1","RISK_MIT_2","RISK_MIT_3"]
    AI_READY = ["AI_READY_1","AI_READY_2","AI_READY_3","AI_READY_4","AI_READY_5","AI_READY_6"]

    dfa["RISK_ID_MEAN"] = dfa[RISK_ID].mean(axis=1)
    dfa["RISK_ASSESS_MEAN"] = dfa[RISK_ASSESS].mean(axis=1)
    dfa["RISK_MIT_MEAN"] = dfa[RISK_MIT].mean(axis=1)

    st.markdown("## Analytics")

    # Row 1: Age distribution + AI readiness hist + Risk vs AI scatter
    c1, c2, c3 = st.columns(3)

    # Age Distribution (bar / histogram)
    fig_age = px.histogram(dfa, x="AGE", nbins=10, title="Age Distribution")
    fig_age.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c1.plotly_chart(fig_age, use_container_width=True)

    # AI Readiness (overall) distribution
    fig_ai = px.histogram(dfa, x="AI_SCORE", nbins=10, title="AI Readiness (Overall) Distribution")
    fig_ai.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c2.plotly_chart(fig_ai, use_container_width=True)

    # Risk Identification vs AI Readiness (point graph)
    # Safe fallback if statsmodels isn't installed (trendline requires it)
    try:
        fig_scatter = px.scatter(
            dfa, x="RISK_ID_MEAN", y="AI_SCORE",
            hover_data=["ID","AGE","ROLE_LBL","EXP"],
            trendline="ols", title="Risk Identification vs. AI Readiness"
        )
    except Exception:
        fig_scatter = px.scatter(
            dfa, x="RISK_ID_MEAN", y="AI_SCORE",
            hover_data=["ID","AGE","ROLE_LBL","EXP"],
            title="Risk Identification vs. AI Readiness"
        )
    fig_scatter.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c3.plotly_chart(fig_scatter, use_container_width=True)

    # Row 2: Risk practice by role + AI readiness item distributions (grid)
    st.markdown("### Risk Practice by Role & AI Readiness Item Distributions")

    c4, c5 = st.columns([1.2, 1.8])

    # Risk Practice Scores by Role (box or violin)
    fig_role_risk = px.box(
        dfa, x="ROLE_LBL", y="RISK_SCORE",
        title="Risk Practice Scores by Role",
        points="all"
    )
    fig_role_risk.update_layout(xaxis_title="", margin=dict(l=0,r=0,t=40,b=0))
    c4.plotly_chart(fig_role_risk, use_container_width=True)

    # AI Readiness Item Distributions (small multiples)
    # Build a long-form frame for AI items
    ai_long = dfa[["ID"] + AI_READY].melt(id_vars="ID", var_name="AI_ITEM", value_name="SCORE")
    fig_ai_items = px.histogram(
        ai_long, x="SCORE", facet_col="AI_ITEM", facet_col_wrap=3,
        category_orders={"AI_ITEM": AI_READY},
        title="AI Readiness Item Distributions (Q15–Q20)"
    )
    fig_ai_items.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c5.plotly_chart(fig_ai_items, use_container_width=True)

    # Row 3: Risk boxplots (ID / ASSESS / MIT)
    st.markdown("### Risk Construct Boxplots")

    c6, c7, c8 = st.columns(3)

    fig_id_box = px.box(
        dfa[RISK_ID].melt(var_name="Item", value_name="Score"),
        x="Item", y="Score", title="Risk Identification Boxplots"
    )
    fig_id_box.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c6.plotly_chart(fig_id_box, use_container_width=True)

    fig_assess_box = px.box(
        dfa[RISK_ASSESS].melt(var_name="Item", value_name="Score"),
        x="Item", y="Score", title="Risk Assessment Boxplots"
    )
    fig_assess_box.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c7.plotly_chart(fig_assess_box, use_container_width=True)

    fig_mit_box = px.box(
        dfa[RISK_MIT].melt(var_name="Item", value_name="Score"),
        x="Item", y="Score", title="Risk Mitigation Boxplots"
    )
    fig_mit_box.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    c8.plotly_chart(fig_mit_box, use_container_width=True)

    # Row 4: Correlation heatmap
    st.markdown("### Correlation Heatmap")
    corr = dfa[["AGE","GENDER","ROLE","EXP","PTYPE","RISK_ID_MEAN","RISK_ASSESS_MEAN","RISK_MIT_MEAN","RISK_SCORE","AI_SCORE","Z_RISK","Z_AI"]].corr(numeric_only=True).round(2)
    try:
        fig_heat = px.imshow(
            corr, text_auto=True, aspect="auto", title="Correlation Heatmap (Demographics, Risk Constructs, AI)"
        )
    except TypeError:
        fig_heat = px.imshow(
            corr, aspect="auto", title="Correlation Heatmap (Demographics, Risk Constructs, AI)"
        )
    fig_heat.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Footnotes (captions)
    st.caption(
        "Notes: Risk Identification/Assessment/Mitigation are 3-item means; AI Readiness is a 6-item composite. "
        "Boxes show median and IQR; whiskers extend to 1.5×IQR. Trendline via OLS (if statsmodels available)."
    )
