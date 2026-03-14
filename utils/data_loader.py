"""
SIIP Dashboard — Data Loader
==============================
Cached loading of data and pre-computed model artifacts
"""

import json
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"


@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    df = pd.read_csv(DATA_DIR / "SIIP_cleaned.csv")
    # Impute NaNs same as precompute
    df['esg_composite'] = df['esg_composite'].fillna(df['esg_composite'].median())
    df['esg_x_competition'] = df['esg_x_competition'].fillna(df['esg_x_competition'].median())
    return df


@st.cache_data
def load_regression_results():
    with open(MODELS_DIR / "regression_results.json") as f:
        return json.load(f)


@st.cache_data
def load_classification_results():
    with open(MODELS_DIR / "classification_results.json") as f:
        return json.load(f)


@st.cache_data
def load_classification_models():
    with open(MODELS_DIR / "classification_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_shap_values():
    with open(MODELS_DIR / "shap_values.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_decision_tree():
    with open(MODELS_DIR / "decision_tree.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_clustering_results():
    with open(MODELS_DIR / "clustering_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_pca_results():
    with open(MODELS_DIR / "pca_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_anomaly_results():
    with open(MODELS_DIR / "anomaly_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_arm_rules():
    with open(MODELS_DIR / "arm_rules.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_arima_results():
    with open(MODELS_DIR / "arima_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_frameworks():
    with open(MODELS_DIR / "frameworks.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_interpretations():
    with open(MODELS_DIR / "interpretations.json") as f:
        return json.load(f)


@st.cache_data
def load_prep_artifacts():
    with open(MODELS_DIR / "prep_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scaler():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_threshold_analysis():
    with open(MODELS_DIR / "threshold_analysis.json") as f:
        return json.load(f)


@st.cache_data
def load_sentiment_results():
    with open(MODELS_DIR / "sentiment_results.pkl", "rb") as f:
        return pickle.load(f)


# Feature name mappings for display
FEATURE_DISPLAY_NAMES = {
    'funding_stage_num': 'Funding Stage',
    'revenue_growth_pct': 'Revenue Growth (%)',
    'gross_margin_pct': 'Gross Margin (%)',
    'company_age': 'Company Age (yrs)',
    'country_risk_composite': 'Country Risk',
    'bilateral_composite': 'Bilateral Alignment',
    'competitive_intensity': 'Competition',
    'team_size': 'Team Size',
    'patent_count': 'Patents',
    'ip_protection_score': 'IP Protection',
    'tech_transfer_proxy': 'Tech Transfer',
    'esg_composite': 'ESG Score',
    'national_strategy_alignment': 'Strategy Alignment',
    'is_pre_revenue': 'Pre-Revenue',
    'stability_x_stage': 'Stability x Stage',
    'runway_months': 'Runway (months)',
    'regulatory_moat': 'Regulatory Moat',
    'total_capital_raised': 'Capital Raised',
    'last_valuation': 'Last Valuation',
    'burn_rate_monthly': 'Monthly Burn',
    'outcome_binary': 'Success (Binary)',
    'outcome_numeric': 'Outcome (0-3)',
}


def display_name(feat):
    """Get display name for a feature"""
    return FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())


# SWF Priority dimension groupings
SWF_DIMENSIONS = {
    'Financial Return': ['funding_stage_num', 'revenue_growth_pct', 'gross_margin_pct', 'runway_months'],
    'Strategic Value': ['patent_count', 'ip_protection_score', 'tech_transfer_proxy', 'competitive_intensity', 'regulatory_moat'],
    'Geopolitical Safety': ['country_risk_composite', 'bilateral_composite', 'stability_x_stage'],
    'Domestic Impact': ['national_strategy_alignment', 'team_size', 'is_pre_revenue'],
    'ESG': ['esg_composite', 'company_age']
}
