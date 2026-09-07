"""
Streamlit UI — AI-Powered Credit Risk Intelligence Platform
Sections: Overview | EDA | Risk Prediction | Explainability | Business Rules | Talk-to-Data

Run: streamlit run ui/app.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.loader import load_full_dataset
from src.ml.evaluate import evaluate_predictions
from src.utils.config import MODEL_CFG, MODEL_DIR
from src.utils.docker_utils import ensure_data_available, ensure_model_available

st.set_page_config(page_title="Credit Risk Intelligence", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💳 Credit Risk Platform")
data_mode = ensure_data_available()
model_ready = ensure_model_available()

st.sidebar.caption(f"Data mode: **{data_mode}**" + (" (synthetic demo data — drop the real Kaggle CSV into /data for production numbers)" if data_mode == "synthetic" else ""))
if not model_ready:
    st.sidebar.warning("No trained model found. Run `python -m src.ml.train` first.")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🎯 Risk Prediction", "🔍 Explainability", "📋 Business Rules", "💬 Talk-to-Data"],
)


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_full_dataset()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("AI-Powered Credit Risk Intelligence Platform")
    st.markdown(
        "End-to-end platform built on the "
        "Home Credit Default Risk dataset: EDA → ML risk scoring → explainability → "
        "business rules → natural-language talk-to-data chatbot."
    )
    col1, col2, col3, col4 = st.columns(4)
    df = get_data()
    col1.metric("Applications", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['TARGET'].mean()*100:.2f}%")
    if model_ready:
        metrics_path = MODEL_DIR / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            col3.metric("Model ROC-AUC (OOF)", f"{m['oof_overall']['roc_auc']:.3f}")
            col4.metric("Model PR-AUC (OOF)", f"{m['oof_overall']['pr_auc']:.3f}")
    st.divider()
    st.subheader("Architecture")
    st.markdown(
        """
        ```
        Kaggle CSV ─▶ loader.py ─▶ preprocessor.py ─▶ LightGBM (train.py)
                                         │                    │
                                         ▼                    ▼
                                  SQLite (talk-to-data)   models/*.pkl
                                         │                    │
                          ┌──────────────┴───────┐   ┌───────┴────────┐
                          ▼                       ▼   ▼                ▼
                Claude NL→SQL agent        SHAP explainer      Rule surrogate tree
                          │                       │                    │
                          └───────────────┬───────┴────────────────────┘
                                           ▼
                                   Streamlit UI (this app)
        ```
        """
    )

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif page == "📊 EDA":
    st.title("Exploratory Data Analysis")
    df = get_data()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Default Drivers", "Data Quality"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            p99_income = df["AMT_INCOME_TOTAL"].quantile(0.99)
            df_income_plot = df[df["AMT_INCOME_TOTAL"] <= p99_income]

            fig_income = px.histogram(
                df_income_plot,
                x="AMT_INCOME_TOTAL",
                nbins=40,
                title="Income distribution (<= 99th percentile)",
                )
            fig_income.update_layout(bargap=0.05)
            st.plotly_chart(fig_income, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="AMT_CREDIT", nbins=60, title="Credit amount distribution")
            st.plotly_chart(fig, use_container_width=True)
        fig = px.pie(df, names="NAME_EDUCATION_TYPE", title="Education level share")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rate_by_edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
        st.plotly_chart(px.bar(rate_by_edu, title="Default rate (%) by education", labels={"value": "Default rate %"}), use_container_width=True)

        rate_by_income = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
        st.plotly_chart(px.bar(rate_by_income, title="Default rate (%) by income type", labels={"value": "Default rate %"}), use_container_width=True)

    with tab3:
        missing = (df.isna().mean() * 100).sort_values(ascending=False)
        missing = missing[missing > 0].head(15)
        st.plotly_chart(px.bar(missing, orientation="h", title="Top missing-value columns (%)"), use_container_width=True)
        st.info("DAYS_EMPLOYED contains a known sentinel value (365243) for pensioners — "
                "flagged as DAYS_EMPLOYED_ANOMALY in the feature pipeline rather than silently mis-encoded.")

# ---------------------------------------------------------------------------
# Risk Prediction
# ---------------------------------------------------------------------------
elif page == "🎯 Risk Prediction":
    st.title("Score a Loan Applicant")
    if not model_ready:
        st.error("Train the model first: `python -m src.ml.train`")
        st.stop()

    from src.ml.predict import RiskModel
    model = RiskModel()

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Annual income", min_value=0, value=180000, step=5000)
        credit = c1.number_input("Loan amount", min_value=0, value=600000, step=5000)
        annuity = c1.number_input("Annuity", min_value=0, value=25000, step=500)
        goods_price = c1.number_input("Goods price", min_value=0, value=580000, step=5000)
        age = c2.slider("Age", 18, 75, 35)
        employed_years = c2.slider("Years employed", 0, 45, 5)
        children = c2.number_input("Number of children", 0, 10, 0)
        education = c3.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        income_type = c3.selectbox("Income type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
        gender = c3.selectbox("Gender", ["M", "F"])
        submitted = st.form_submit_button("Score Applicant", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "SK_ID_CURR": 999999999,
            "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": gender,
            "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y", "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price, "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment", "OCCUPATION_TYPE": "Laborers",
            "ORGANIZATION_TYPE": "Business Entity Type 3",
            "DAYS_BIRTH": -int(age * 365.25), "DAYS_EMPLOYED": -int(employed_years * 365.25),
            "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.5,
        }])
        scored = model.score(row)
        band = scored["RISK_BAND"].iloc[0]
        score = scored["RISK_SCORE"].iloc[0]

        color = {"Low": "green", "Medium": "orange", "High": "red"}[band]
        st.markdown(f"### Predicted default probability: **{score:.1%}**")
        st.markdown(f"### Risk band: :{color}[**{band}**]")
        st.session_state["last_scored_row"] = row
        st.session_state["last_scored_result"] = scored

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
elif page == "🔍 Explainability":
    st.title("Explain a Prediction (SHAP)")
    if not model_ready:
        st.error("Train the model first: `python -m src.ml.train`")
        st.stop()
    if "last_scored_row" not in st.session_state:
        st.info("Score an applicant on the 'Risk Prediction' page first, then come back here.")
        st.stop()

    from src.ml.predict import RiskModel
    model = RiskModel()
    row = st.session_state["last_scored_row"]
    explanation = model.explain(row)[0]

    exp_df = pd.DataFrame(explanation)
    exp_df["direction"] = exp_df["shap_value"].apply(lambda v: "Increases risk" if v > 0 else "Decreases risk")
    fig = px.bar(
        exp_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
        color="direction", color_discrete_map={"Increases risk": "#D62728", "Decreases risk": "#2CA02C"},
        title="Top feature contributions to this prediction (SHAP values)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(exp_df[["feature", "feature_value", "shap_value"]], use_container_width=True)
    st.caption("Positive SHAP value = pushes prediction toward higher default risk; negative = toward lower risk.")

# ---------------------------------------------------------------------------
# Business Rules
# ---------------------------------------------------------------------------
elif page == "📋 Business Rules":
    st.title("Business-Readable Decision Rules")
    rules_path = MODEL_DIR / "business_rules.json"
    if not rules_path.exists():
        st.error("Train the model first: `python -m src.ml.train`")
        st.stop()
    rules = json.loads(rules_path.read_text())
    st.caption("Rules distilled from the ML model via a shallow surrogate decision tree — "
               "each rule shows its predicted default rate and how much of the portfolio it covers.")
    for r in rules:
        badge = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[r["risk_band"]]
        with st.expander(f"{badge} {r['risk_band']} risk — {r['predicted_default_rate']:.1%} predicted default rate ({r['support_pct']}% of applicants)"):
            st.code(r["rule"], language="text")

# ---------------------------------------------------------------------------
# Talk-to-Data
# ---------------------------------------------------------------------------
elif page == "💬 Talk-to-Data":
    st.title("Talk to Your Data")
    st.caption("Ask questions in plain English — answers are generated from validated, "
               "read-only SQL run against the applications database.")

    examples = [
        "What is the overall default rate?",
        "Show average income by education level",
        "Which occupation types have the highest default rate?",
        "List the top 5 applicants by credit amount who are high risk",
        "How many applicants have children and own a car?",
        "What is the risk band distribution?",
    ]
    st.write("Try:", " · ".join(f"`{e}`" for e in examples))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.chat_input("Ask a question about the loan portfolio...")
    if question:
        from src.talk_to_data.nl_to_sql import answer_question
        with st.spinner("Thinking..."):
            result = answer_question(question)
        st.session_state.chat_history.append(result)

    for r in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(r.question)
        with st.chat_message("assistant"):
            st.write(r.answer)
            if r.sql:
                with st.expander("Show generated SQL"):
                    st.code(r.sql, language="sql")
            st.caption(f"mode: {r.mode} · prompt version: {r.prompt_version}")
