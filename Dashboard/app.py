"""
Financial Fraud Detection System — Professional Streamlit Dashboard
================================================================
Features:
  • KPI overview cards
  • Live Transaction Detector (probability + risk level + reason)
  • Interactive Plotly analytics
  • Model performance (Confusion Matrix, ROC, Feature Importance)
  • High-risk transactions table
  • CSV / Excel upload support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2a2f45;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; }
    
    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #EE322F;
    }
    
    /* Result cards */
    .fraud-card {
        background: linear-gradient(135deg, #3b0a0a 0%, #5c0e0e 100%);
        border: 2px solid #EE322F;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #fecaca;
    }
    .legit-card {
        background: linear-gradient(135deg, #0a2e1a 0%, #0e3d24 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #bbf7d0;
    }
    .reason-box {
        background: #1e293b;
        border-left: 4px solid #EE322F;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 12px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #EE322F, #c41e1a);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #c41e1a, #a01815);
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_default_data():
    path = "/home/workdir/attachments/financial_fraud_detection_dataset.csv"
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def read_uploaded_file(uploaded):
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        elif name.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(uploaded, engine="openpyxl")
            except Exception:
                return pd.read_excel(uploaded, engine="xlrd")
        else:
            # try csv first
            return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def clean_and_engineer(df_raw: pd.DataFrame):
    """Reproduce the notebook pipeline with proper encoders & scaler."""
    df = df_raw.copy()

    # Required columns check
    required = [
        "Transaction_Amount", "Merchant_Category", "Payment_Method",
        "Device_Type", "Location", "Is_International", "Previous_Transactions",
        "Average_Spend", "Account_Age_Days", "Suspicious_Keyword", "Fraudulent"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop IDs if present
    for col in ["Transaction_ID", "Customer_ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Parse date
    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], dayfirst=True, errors="coerce")
        df["Hour"] = df["Transaction_Date"].dt.hour.fillna(12).astype(int)
        df["Day"] = df["Transaction_Date"].dt.day.fillna(15).astype(int)
        df["Month"] = df["Transaction_Date"].dt.month.fillna(6).astype(int)
        df["Weekday"] = df["Transaction_Date"].dt.weekday.fillna(0).astype(int)
        df["Is_Weekend"] = (df["Weekday"] >= 5).astype(int)
        df = df.drop(columns=["Transaction_Date"])
    else:
        df["Hour"] = 12
        df["Day"] = 15
        df["Month"] = 6
        df["Weekday"] = 0
        df["Is_Weekend"] = 0

    # Ratio features
    df["Amount_to_AvgSpend_Ratio"] = df["Transaction_Amount"] / (df["Average_Spend"] + 1)
    df["Amount_to_PrevTxn_Ratio"] = df["Transaction_Amount"] / (df["Previous_Transactions"] + 1)

    # Encode categoricals (separate encoder per column)
    cat_cols = ["Merchant_Category", "Payment_Method", "Device_Type", "Location", "Suspicious_Keyword"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Scale numerical
    num_cols = [
        "Transaction_Amount", "Previous_Transactions", "Average_Spend",
        "Account_Age_Days", "Amount_to_AvgSpend_Ratio", "Amount_to_PrevTxn_Ratio",
        "Hour", "Day", "Month"
    ]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Keep only modeling columns + target
    feature_cols = [
        "Transaction_Amount", "Merchant_Category", "Payment_Method", "Device_Type",
        "Location", "Is_International", "Previous_Transactions", "Average_Spend",
        "Account_Age_Days", "Suspicious_Keyword",
        "Hour", "Day", "Month", "Weekday", "Is_Weekend",
        "Amount_to_AvgSpend_Ratio", "Amount_to_PrevTxn_Ratio"
    ]
    # Ensure all exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols]
    y = df["Fraudulent"].astype(int)

    return X, y, encoders, scaler, feature_cols, df_raw


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
    return results, X_test, y_test


def build_reason(raw_inputs: dict, prob: float) -> str:
    reasons = []
    amount = raw_inputs.get("Transaction_Amount", 0)
    avg = raw_inputs.get("Average_Spend", 1)
    prev = raw_inputs.get("Previous_Transactions", 1)
    ratio = amount / (avg + 1)

    if ratio > 2.5:
        reasons.append(f"Transaction amount ({amount:.0f}) is unusually high vs average spend ({avg:.0f})")
    if amount > 300:
        reasons.append("Very high absolute transaction amount")
    if raw_inputs.get("Is_International", 0) == 1:
        reasons.append("International transaction")
    if raw_inputs.get("Suspicious_Keyword", "No") == "Yes":
        reasons.append("Suspicious keyword flagged")
    if prev < 10:
        reasons.append("Low previous transaction history (new/unusual pattern)")
    if raw_inputs.get("Device_Type") == "Desktop" and amount > 150:
        reasons.append("High-value transaction from Desktop (less common)")
    if not reasons:
        if prob > 0.5:
            reasons.append("Combined behavioral pattern matches known fraud signatures")
        else:
            reasons.append("Transaction pattern is consistent with legitimate activity")
    return " • ".join(reasons)


def prepare_single_row(raw: dict, encoders, scaler, feature_cols):
    """Transform one raw transaction into model-ready vector."""
    row = {
        "Transaction_Amount": raw["Transaction_Amount"],
        "Merchant_Category": raw["Merchant_Category"],
        "Payment_Method": raw["Payment_Method"],
        "Device_Type": raw["Device_Type"],
        "Location": raw["Location"],
        "Is_International": raw["Is_International"],
        "Previous_Transactions": raw["Previous_Transactions"],
        "Average_Spend": raw["Average_Spend"],
        "Account_Age_Days": raw["Account_Age_Days"],
        "Suspicious_Keyword": raw["Suspicious_Keyword"],
        "Hour": raw.get("Hour", 12),
        "Day": raw.get("Day", 15),
        "Month": raw.get("Month", 6),
        "Weekday": raw.get("Weekday", 0),
        "Is_Weekend": 1 if raw.get("Weekday", 0) >= 5 else 0,
    }
    row["Amount_to_AvgSpend_Ratio"] = row["Transaction_Amount"] / (row["Average_Spend"] + 1)
    row["Amount_to_PrevTxn_Ratio"] = row["Transaction_Amount"] / (row["Previous_Transactions"] + 1)

    # Encode
    for col, le in encoders.items():
        val = str(row[col])
        if val in le.classes_:
            row[col] = le.transform([val])[0]
        else:
            # unseen category → most frequent class (0)
            row[col] = 0

    df_row = pd.DataFrame([row])
    num_cols = [
        "Transaction_Amount", "Previous_Transactions", "Average_Spend",
        "Account_Age_Days", "Amount_to_AvgSpend_Ratio", "Amount_to_PrevTxn_Ratio",
        "Hour", "Day", "Month"
    ]
    df_row[num_cols] = scaler.transform(df_row[num_cols])
    return df_row[feature_cols]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🔍 Live Detector", "📈 Analytics", "🤖 Model Performance", "⚠️ High-Risk Table"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

use_default = st.sidebar.checkbox("Use built-in dataset", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Models: Logistic Regression • Random Forest • XGBoost  \nImbalance handled with SMOTE")

# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models…")
def get_pipeline(df_hash, df):
    X, y, encoders, scaler, feature_cols, raw = clean_and_engineer(df)
    results, X_test, y_test = train_models(X, y)
    return {
        "X": X, "y": y, "raw": raw,
        "encoders": encoders, "scaler": scaler,
        "feature_cols": feature_cols,
        "results": results,
        "X_test": X_test, "y_test": y_test
    }


df_source = None
if uploaded is not None:
    df_source = read_uploaded_file(uploaded)
    if df_source is not None:
        st.sidebar.success(f"Loaded: {uploaded.name} ({df_source.shape[0]:,} rows)")
elif use_default:
    df_source = load_default_data()
    if df_source is not None:
        st.sidebar.info(f"Built-in dataset ({df_source.shape[0]:,} rows)")

if df_source is None:
    st.warning("Please upload a dataset or enable the built-in dataset.")
    st.stop()

# Hash for cache
df_hash = pd.util.hash_pandas_object(df_source).sum()

try:
    pipeline = get_pipeline(df_hash, df_source)
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

results = pipeline["results"]
raw_df = pipeline["raw"]
encoders = pipeline["encoders"]
scaler = pipeline["scaler"]
feature_cols = pipeline["feature_cols"]
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = results[best_name]
model = best["model"]

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("<h1 style='color:#f1f5f9; margin-bottom:0.2rem;'>💳 Financial Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Real-time monitoring • ML-powered risk scoring • Interactive analytics")

    total_tx = len(raw_df)
    fraud_count = int(raw_df["Fraudulent"].sum()) if "Fraudulent" in raw_df.columns else 0
    fraud_rate = (fraud_count / total_tx * 100) if total_tx else 0
    amount_at_risk = raw_df.loc[raw_df["Fraudulent"] == 1, "Transaction_Amount"].sum() if "Fraudulent" in raw_df.columns else 0
    roc_auc = best["roc_auc"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{total_tx:,}")
    c2.metric("Fraudulent", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}%", delta_color="inverse")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Amount at Risk", f"₹{amount_at_risk:,.0f}")
    c5.metric(f"Best Model ROC-AUC", f"{roc_auc:.3f}", delta=best_name)

    st.markdown("<div class='section-header'>Quick Insights</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Fraud by Merchant Category
        if "Merchant_Category" in raw_df.columns:
            fraud_by_cat = raw_df.groupby("Merchant_Category")["Fraudulent"].agg(["sum", "count"])
            fraud_by_cat["rate"] = (fraud_by_cat["sum"] / fraud_by_cat["count"] * 100).round(1)
            fraud_by_cat = fraud_by_cat.sort_values("rate", ascending=False).reset_index()
            fig = px.bar(
                fraud_by_cat, x="Merchant_Category", y="rate",
                title="Fraud Rate by Merchant Category (%)",
                color="rate", color_continuous_scale=["#22c55e", "#f59e0b", "#EE322F"],
                labels={"rate": "Fraud Rate %", "Merchant_Category": "Category"}
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_size=14, height=340,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Fraud by Location
        if "Location" in raw_df.columns:
            fraud_by_loc = raw_df.groupby("Location")["Fraudulent"].agg(["sum", "count"])
            fraud_by_loc["rate"] = (fraud_by_loc["sum"] / fraud_by_loc["count"] * 100).round(1)
            fraud_by_loc = fraud_by_loc.sort_values("rate", ascending=False).reset_index()
            fig2 = px.bar(
                fraud_by_loc, x="Location", y="rate",
                title="Fraud Rate by Location (%)",
                color="rate", color_continuous_scale=["#22c55e", "#f59e0b", "#EE322F"],
                labels={"rate": "Fraud Rate %"}
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_size=14, height=340,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Amount distribution
    st.markdown("<div class='section-header'>Transaction Amount Distribution</div>", unsafe_allow_html=True)
    fig3 = px.histogram(
        raw_df, x="Transaction_Amount", color="Fraudulent",
        barmode="overlay", nbins=40,
        color_discrete_map={0: "#22c55e", 1: "#EE322F"},
        labels={"Fraudulent": "Is Fraud", "Transaction_Amount": "Amount (₹)"},
        title="Amount Distribution — Legitimate vs Fraud"
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=360, margin=dict(t=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: LIVE DETECTOR
# ─────────────────────────────────────────────
elif page == "🔍 Live Detector":
    st.markdown("<h1 style='color:#f1f5f9;'>🔍 Live Transaction Detector</h1>", unsafe_allow_html=True)
    st.caption("Enter transaction details → get instant fraud probability, risk level & explanation")

    with st.form("detector_form"):
        st.markdown("#### Transaction Details")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=120.0, step=10.0)
            merchant = st.selectbox("Merchant Category (Type)", [
                "Electronics", "Entertainment", "Fashion", "Food",
                "Grocery", "Health", "Travel", "Utilities"
            ])
            payment = st.selectbox("Payment Method", [
                "Credit Card", "Debit Card", "NetBanking", "PayPal", "UPI"
            ])
        with r1c2:
            device = st.selectbox("Device Type", ["Desktop", "Mobile", "POS"])
            location = st.selectbox("Location", [
                "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"
            ])
            is_intl = st.selectbox("International Transaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with r1c3:
            prev_txn = st.number_input("Previous Transactions", min_value=0, value=45, step=1)
            avg_spend = st.number_input("Average Spend (₹)", min_value=0.0, value=85.0, step=5.0)
            acct_age = st.number_input("Account Age (Days)", min_value=1, value=800, step=10)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            hour = st.slider("Hour of Day", 0, 23, 14)
        with r2c2:
            weekday = st.selectbox("Weekday", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        with r2c3:
            suspicious = st.selectbox("Suspicious Keyword", ["No", "Yes"])

        submitted = st.form_submit_button("🔎 Analyze Transaction", use_container_width=True)

    if submitted:
        raw_inputs = {
            "Transaction_Amount": amount,
            "Merchant_Category": merchant,
            "Payment_Method": payment,
            "Device_Type": device,
            "Location": location,
            "Is_International": is_intl,
            "Previous_Transactions": prev_txn,
            "Average_Spend": avg_spend,
            "Account_Age_Days": acct_age,
            "Suspicious_Keyword": suspicious,
            "Hour": hour,
            "Day": 15,
            "Month": 6,
            "Weekday": weekday,
        }

        try:
            X_new = prepare_single_row(raw_inputs, encoders, scaler, feature_cols)
            prob = float(model.predict_proba(X_new)[0, 1])
            pred = int(prob >= 0.5)

            if prob >= 0.75:
                risk = "HIGH"
            elif prob >= 0.45:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            reason = build_reason(raw_inputs, prob)

            st.markdown("---")
            if pred == 1:
                st.markdown(f"""
                <div class="fraud-card">
                    <h2 style="margin:0; color:#fecaca;">🔴 FRAUD DETECTED</h2>
                    <p style="font-size:2.2rem; font-weight:700; margin:0.4rem 0;">{prob*100:.1f}%</p>
                    <p style="margin:0; opacity:0.9;">Fraud Probability</p>
                    <p style="margin-top:0.8rem; font-size:1.1rem;">Risk Level: <strong>{risk}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <h2 style="margin:0; color:#bbf7d0;">🟢 LEGITIMATE TRANSACTION</h2>
                    <p style="font-size:2.2rem; font-weight:700; margin:0.4rem 0;">{prob*100:.1f}%</p>
                    <p style="margin:0; opacity:0.9;">Fraud Probability</p>
                    <p style="margin-top:0.8rem; font-size:1.1rem;">Risk Level: <strong>{risk}</strong></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="reason-box">
                <strong>Reason:</strong> {reason}
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Probability %", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#EE322F" if prob >= 0.5 else "#22c55e"},
                    "steps": [
                        {"range": [0, 45], "color": "#14532d"},
                        {"range": [45, 75], "color": "#713f12"},
                        {"range": [75, 100], "color": "#7f1d1d"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": 50
                    }
                },
                number={"suffix": "%", "font": {"color": "#f1f5f9"}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=280,
                margin=dict(t=40, b=10, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown("<h1 style='color:#f1f5f9;'>📈 Analytics</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "By Category", "By Location", "Over Time", "Amount vs Fraud"
    ])

    with tab1:
        if "Merchant_Category" in raw_df.columns:
            ct = pd.crosstab(raw_df["Merchant_Category"], raw_df["Fraudulent"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="Merchant_Category", var_name="Fraudulent", value_name="Percent")
            ct["Fraudulent"] = ct["Fraudulent"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(
                ct, x="Merchant_Category", y="Percent", color="Fraudulent",
                barmode="group",
                color_discrete_map={"Legitimate": "#22c55e", "Fraud": "#EE322F"},
                title="Fraud vs Legitimate % by Merchant Category"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=420)
            st.plotly_chart(fig, use_container_width=True)

            # counts
            figc = px.histogram(
                raw_df, x="Merchant_Category", color="Fraudulent",
                barmode="group",
                color_discrete_map={0: "#22c55e", 1: "#EE322F"},
                title="Transaction Counts by Category"
            )
            figc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=380)
            st.plotly_chart(figc, use_container_width=True)

    with tab2:
        if "Location" in raw_df.columns:
            fig = px.histogram(
                raw_df, x="Location", color="Fraudulent", barmode="group",
                color_discrete_map={0: "#22c55e", 1: "#EE322F"},
                title="Transactions by Location"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=400)
            st.plotly_chart(fig, use_container_width=True)

            rate = raw_df.groupby("Location")["Fraudulent"].mean().reset_index()
            rate["Fraud Rate %"] = (rate["Fraudulent"] * 100).round(2)
            fig2 = px.choropleth()  # skip real geo — simple bar instead
            fig2 = px.bar(rate, x="Location", y="Fraud Rate %",
                          color="Fraud Rate %", color_continuous_scale="Reds",
                          title="Fraud Rate % by Location")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=380)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        # Reconstruct approximate time features for visualization from raw
        temp = raw_df.copy()
        if "Transaction_Date" in temp.columns:
            temp["Transaction_Date"] = pd.to_datetime(temp["Transaction_Date"], dayfirst=True, errors="coerce")
            temp["Hour"] = temp["Transaction_Date"].dt.hour
            temp["Weekday"] = temp["Transaction_Date"].dt.day_name()
            # Hourly
            hour_fraud = temp.groupby("Hour")["Fraudulent"].agg(["sum", "count"]).reset_index()
            hour_fraud["rate"] = (hour_fraud["sum"] / hour_fraud["count"] * 100).round(2)
            fig = px.line(hour_fraud, x="Hour", y="rate", markers=True,
                          title="Fraud Rate % by Hour of Day",
                          labels={"rate": "Fraud Rate %"})
            fig.update_traces(line_color="#EE322F")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=380)
            st.plotly_chart(fig, use_container_width=True)

            # Weekday
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd = temp.groupby("Weekday")["Fraudulent"].mean().reindex(order).reset_index()
            wd["Fraud Rate %"] = (wd["Fraudulent"] * 100).round(2)
            fig2 = px.bar(wd, x="Weekday", y="Fraud Rate %",
                          color="Fraud Rate %", color_continuous_scale="Reds",
                          title="Fraud Rate % by Weekday")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Transaction_Date not available for time-based charts.")

    with tab4:
        fig = px.box(
            raw_df, x="Fraudulent", y="Transaction_Amount",
            color="Fraudulent",
            color_discrete_map={0: "#22c55e", 1: "#EE322F"},
            labels={"Fraudulent": "Is Fraud", "Transaction_Amount": "Amount (₹)"},
            title="Transaction Amount by Fraud Status",
            points="outliers"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            raw_df.sample(min(2000, len(raw_df)), random_state=42),
            x="Average_Spend", y="Transaction_Amount",
            color="Fraudulent",
            color_discrete_map={0: "#22c55e", 1: "#EE322F"},
            title="Average Spend vs Transaction Amount",
            opacity=0.6
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=420)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown("<h1 style='color:#f1f5f9;'>🤖 Model Performance</h1>", unsafe_allow_html=True)

    # Comparison table
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4)
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.info(f"**Best model selected for live predictions:** {best_name} (ROC-AUC = {best['roc_auc']:.4f})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(best["y_test"], best["y_pred"])
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
            color_continuous_scale="Reds",
            title=f"{best_name} — Confusion Matrix"
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>ROC Curve</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={r['roc_auc']:.3f})"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#64748b"),
            name="Random"
        ))
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=380,
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp, x="Importance", y="Feature", orientation="h",
            title=f"{best_name} — Feature Importance",
            color="Importance", color_continuous_scale="Reds"
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=480, margin=dict(l=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance available for tree-based models (Random Forest / XGBoost).")

# ─────────────────────────────────────────────
# PAGE: HIGH-RISK TABLE
# ─────────────────────────────────────────────
elif page == "⚠️ High-Risk Table":
    st.markdown("<h1 style='color:#f1f5f9;'>⚠️ High-Risk Transactions</h1>", unsafe_allow_html=True)
    st.caption("Transactions with high fraud probability or actual fraud label")

    # Score the whole dataset with the best model
    X_all = pipeline["X"]
    probs = model.predict_proba(X_all)[:, 1]
    display_df = raw_df.copy()
    display_df["Fraud_Probability"] = np.round(probs * 100, 1)
    display_df["Risk_Level"] = display_df["Fraud_Probability"].apply(
        lambda p: "HIGH" if p >= 75 else ("MEDIUM" if p >= 45 else "LOW")
    )

    threshold = st.slider("Minimum Fraud Probability (%)", 0, 100, 60, 5)
    filtered = display_df[display_df["Fraud_Probability"] >= threshold].sort_values(
        "Fraud_Probability", ascending=False
    )

    st.metric("Matching Transactions", f"{len(filtered):,}")

    show_cols = [c for c in [
        "Transaction_ID", "Customer_ID", "Transaction_Amount", "Merchant_Category",
        "Location", "Device_Type", "Payment_Method", "Previous_Transactions",
        "Average_Spend", "Fraudulent", "Fraud_Probability", "Risk_Level"
    ] if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].head(500),
        use_container_width=True,
        height=480
    )

    csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download High-Risk CSV",
        data=csv,
        file_name="high_risk_transactions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Financial Fraud Detection System • Streamlit + Scikit-learn + XGBoost + Plotly • SMOTE for class imbalance")
