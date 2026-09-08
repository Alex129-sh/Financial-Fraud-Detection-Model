import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Financial Fraud Detection Dashboard", layout="wide")

st.title("💳 Financial Fraud Detection - Live Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🤖 Model Performance", "🔍 Live Prediction"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success(f"File loaded: {uploaded_file.name}")
    st.sidebar.write(f"Shape: {df.shape}")

    # Try to detect target column
    possible_targets = ['Fraudulent', 'IsFraud', 'Fraud_Label', 'isFraud', 'Class']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error("Could not find target column (Fraudulent / IsFraud / Fraud_Label). Please check your dataset.")
    else:
        # ==================== TAB 1: OVERVIEW ====================
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            total = len(df)
            fraud_count = df[target_col].sum() if df[target_col].dtype != 'object' else (df[target_col] == 1).sum()
            fraud_pct = (fraud_count / total) * 100

            col1.metric("Total Transactions", f"{total:,}")
            col2.metric("Fraud Cases", f"{fraud_count:,}")
            col3.metric("Fraud Percentage", f"{fraud_pct:.2f}%")
            col4.metric("Legitimate Cases", f"{total - fraud_count:,}")

            st.subheader("Sample Data")
            st.dataframe(df.head(10))

            st.subheader("Column Information")
            st.write(df.dtypes)

        # ==================== TAB 2: VISUALIZATIONS ====================
        with tab2:
            st.header("Visualizations")

            # Target Distribution
            st.subheader("1. Fraud vs Non-Fraud Distribution")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(x=target_col, data=df, ax=ax1, palette="Set2")
            ax1.set_title("Fraud Distribution")
            st.pyplot(fig1)

            # Numerical columns for charts
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if target_col in num_cols:
                num_cols.remove(target_col)

            if len(num_cols) > 0:
                selected_num = st.selectbox("Select Numerical Column for Distribution", num_cols)
                fig2, ax2 = plt.subplots(figsize=(8,4))
                sns.histplot(data=df, x=selected_num, hue=target_col, bins=40, kde=True, ax=ax2)
                ax2.set_title(f"{selected_num} Distribution by Fraud")
                st.pyplot(fig2)

            # Categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                selected_cat = st.selectbox("Select Categorical Column", cat_cols)
                fig3, ax3 = plt.subplots(figsize=(10,5))
                sns.countplot(data=df, x=selected_cat, hue=target_col, ax=ax3)
                plt.xticks(rotation=45)
                ax3.set_title(f"Fraud by {selected_cat}")
                st.pyplot(fig3)

            # Correlation Heatmap
            st.subheader("Correlation Heatmap")
            fig4, ax4 = plt.subplots(figsize=(10,7))
            corr = df.select_dtypes(include=np.number).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
            st.pyplot(fig4)

        # ==================== TAB 3: MODEL PERFORMANCE ====================
        with tab3:
            st.header("Model Performance")

            if st.button("Train Models on Uploaded Data"):
                with st.spinner("Training models... Please wait"):

                    # Simple preprocessing
                    data = df.copy()
                    data = data.dropna()

                    # Encode categorical
                    for col in data.select_dtypes(include='object').columns:
                        if col != target_col:
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col].astype(str))

                    X = data.drop(columns=[target_col])
                    y = data[target_col]

                    # Make sure y is binary 0/1
                    if y.dtype == 'object':
                        y = y.map({'Yes':1, 'No':0, 'Fraud':1, 'Safe':0}).fillna(y)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    # SMOTE
                    smote = SMOTE(random_state=42)
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                    # Train Random Forest
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_res, y_train_res)

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:,1]

                    st.success("Model Trained Successfully!")

                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    st.subheader("ROC-AUC Score")
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots()
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

                    # Save model in session
                    st.session_state['model'] = model
                    st.session_state['features'] = X.columns.tolist()

        # ==================== TAB 4: LIVE PREDICTION ====================
        with tab4:
            st.header("Live Fraud Prediction")

            if 'model' not in st.session_state:
                st.warning("Please train the model first in the 'Model Performance' tab.")
            else:
                st.info("Enter transaction details below:")

                # Create input fields dynamically (simplified version)
                input_data = {}
                for col in st.session_state['features']:
                    if df[col].dtype in ['int64', 'float64']:
                        input_data[col] = st.number_input(f"{col}", value=0.0)
                    else:
                        input_data[col] = st.text_input(f"{col}", value="")

                if st.button("Predict Fraud"):
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode if needed (basic)
                    for col in input_df.select_dtypes(include='object').columns:
                        le = LabelEncoder()
                        # This is simplified - in real app you should save the encoders
                        input_df[col] = 0  

                    prediction = st.session_state['model'].predict(input_df)[0]
                    probability = st.session_state['model'].predict_proba(input_df)[0][1]

                    if prediction == 1:
                        st.error(f"🚨 FRAUD DETECTED! (Probability: {probability:.2%})")
                    else:
                        st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2%})")

else:
    st.info("👆 Please upload a dataset from the sidebar to begin.")
    st.markdown("""
    ### Supported Target Column Names:
    - Fraudulent
    - IsFraud
    - Fraud_Label
    - isFraud
    - Class
    """)