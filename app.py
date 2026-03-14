"""
SIIP — Sovereign Investment Intelligence Platform
===================================================
Home Dashboard: Executive intelligence overview
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_data, load_frameworks, load_interpretations, load_classification_results, load_clustering_results, load_arm_rules
from utils.styles import inject_css, kpi_card, section_header, metric_highlight, format_currency, format_pct, COLORS, PLOTLY_COLORS
from utils.charts import donut_chart, bar_chart, apply_theme

st.set_page_config(
    page_title="SIIP — Sovereign Investment Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(inject_css(), unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🏛️ SIIP")
    st.markdown("**Sovereign Investment Intelligence Platform**")
    st.markdown("---")
    st.markdown("##### Navigation")
    st.markdown("""
    - 🏠 **Home** — Executive Overview
    - 📊 Pipeline Intelligence
    - 📈 Risk Engine
    - 🎯 Prediction Engine
    - 🧬 Segmentation Lab
    - 🔗 Pattern Discovery
    - 🛠️ Deal Evaluator
    - 💬 Sentiment Intelligence
    """)
    st.markdown("---")
    st.caption("Built for SWF Decision Intelligence")
    st.caption("Data: 6,798 ventures × 12 sectors × 8 regions")

# --- Load data ---
df = load_data()
frameworks = load_frameworks()
interp = load_interpretations()
stats = frameworks['overall_stats']

# --- Header ---
st.markdown("# 🏛️ SIIP — Sovereign Investment Intelligence Platform")
st.caption("Real-time venture intelligence for sovereign wealth fund decision-making")

# --- Row 1: KPI Cards ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi_card("Total Ventures", f"{stats['total_ventures']:,}", f"{len(stats['sectors'])} sectors"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("Capital Deployed", format_currency(stats['total_capital']), f"{len(stats['regions'])} regions"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Success Rate", format_pct(stats['overall_success_rate']), f"{stats['outcome_dist'].get('Successful Exit', 0):,} exits"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Active Pipeline", format_pct(stats['active_pct']), f"{stats['active_count']:,} ventures"), unsafe_allow_html=True)

st.markdown("")

# --- Row 2: Outcome Distribution + Sector Deal Flow ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(section_header("Portfolio Outcome Distribution"), unsafe_allow_html=True)
    outcome_data = stats['outcome_dist']
    fig = donut_chart(
        labels=list(outcome_data.keys()),
        values=list(outcome_data.values()),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(interp['home']['kpi_summary'])

with col2:
    st.markdown(section_header("Deal Flow by Sector"), unsafe_allow_html=True)
    sector_counts = df['sector'].value_counts().sort_values()
    fig = bar_chart(
        x=sector_counts.index.tolist(),
        y=sector_counts.values.tolist(),
        orientation='h', height=320
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Top sectors: {', '.join(sector_counts.index[-3:][::-1])}. Together they represent {sector_counts.values[-3:].sum()/len(df):.0%} of the pipeline.")

# --- Row 3: Success Rate by Stage + Region Distribution ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(section_header("Success Rate by Funding Stage"), unsafe_allow_html=True)
    stage_order = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'Pre-IPO']
    stage_success = df.groupby('funding_stage')['outcome_binary'].mean().reindex(stage_order)
    colors = [COLORS['danger'] if v < 0.15 else COLORS['warning'] if v < 0.20 else COLORS['success'] for v in stage_success.values]
    fig = bar_chart(
        x=stage_order, y=stage_success.values.tolist(),
        color=colors, height=320,
        text=[f"{v:.1%}" for v in stage_success.values]
    )
    fig.update_layout(yaxis_title='Success Rate', yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    best_stage = stage_success.idxmax()
    st.caption(f"{best_stage} ventures show the highest success rate at {stage_success.max():.1%}. Later stages correlate with higher exit probability.")

with col2:
    st.markdown(section_header("Ventures by Region"), unsafe_allow_html=True)
    region_data = df.groupby('region').agg(
        count=('venture_id', 'count'),
        success_rate=('outcome_binary', 'mean')
    ).sort_values('count', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=region_data.index, x=region_data['count'], orientation='h',
        marker_color=COLORS['accent2'], name='Deal Count',
        hovertemplate='%{y}: %{x} ventures<extra></extra>'
    ))
    fig.update_layout(xaxis_title='Number of Ventures')
    fig = apply_theme(fig, 320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(interp['home']['portfolio_health'])

# --- Row 4: Intelligence Module Summary ---
st.markdown("---")
st.markdown(section_header("Intelligence Modules — Key Findings"), unsafe_allow_html=True)

# Load module results for summary cards
try:
    clf_results = load_classification_results()
    best_clf = max(clf_results, key=lambda k: clf_results[k]['auc_roc'])
    best_auc = clf_results[best_clf]['auc_roc']
except:
    best_clf, best_auc = "N/A", 0

try:
    clust = load_clustering_results()
    best_k = clust['best_k']
except:
    best_k = "N/A"

try:
    arm = load_arm_rules()
    n_rules = arm['n_rules']
except:
    n_rules = 0

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.info("**📈 Risk Engine**")
    st.markdown("Linear models explain **6.2%** of outcome variance (R²=0.062). Statistically significant (p<1e-81) but structurally limited for categorical outcomes.")
    st.caption("→ Go to Risk Engine for regression analysis")

with m2:
    st.info(f"**🎯 Prediction Engine**")
    st.markdown(f"Best classifier: **{best_clf}** (AUC={best_auc:.3f}). Non-linear models capture complex feature interactions that linear regression misses.")
    st.caption("→ Go to Prediction Engine for classification")

with m3:
    st.info(f"**🧬 Segmentation Lab**")
    st.markdown(f"**{best_k} venture archetypes** identified via K-Means clustering. Each archetype has distinct risk-return profiles requiring tailored allocation strategies.")
    st.caption("→ Go to Segmentation Lab for cluster analysis")

with m4:
    st.info(f"**🔗 Pattern Discovery**")
    st.markdown(f"**{n_rules:,} association rules** discovered via Apriori. Feature combinations reveal hidden success/failure patterns for screening.")
    st.caption("→ Go to Pattern Discovery for ARM analysis")

with m5:
    st.info("**💬 Sentiment Intelligence**")
    st.markdown("**3 NLP methods** (VADER, TextBlob, Naive Bayes) analyze analyst memos. Head-to-head comparison reveals sentiment signal quality.")
    st.caption("→ Go to Sentiment Intelligence for NLP analysis")

# --- Analytics Pipeline ---
st.markdown("---")
st.markdown(section_header("Analytics Pipeline: Descriptive → Diagnostic → Predictive → Prescriptive"), unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.success("**DESCRIPTIVE**")
    st.markdown("EDA, distributions, correlations, data quality assessment")
with p2:
    st.warning("**DIAGNOSTIC**")
    st.markdown("Regression analysis, feature significance, multicollinearity")
with p3:
    st.info("**PREDICTIVE**")
    st.markdown("Classification, clustering, ARIMA forecasting, anomaly detection")
with p4:
    st.error("**PRESCRIPTIVE**")
    st.markdown("Deal scoring, strategy recommendations, screening rules")
