import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Loan Eligibility Prediction System",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Loan-Eligibility-Prediction.csv"
DB_FILE = BASE_DIR / "loan_eligibility.db"


# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            married TEXT,
            dependents TEXT,
            education TEXT,
            self_employed TEXT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_amount_term REAL,
            credit_history REAL,
            property_area TEXT,
            prediction TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_application(gender, married, dependents, education, self_employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_amount_term, credit_history, property_area, prediction):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO applications (
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area, prediction
    ))
    conn.commit()
    conn.close()


def load_saved_applications():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM applications ORDER BY id DESC", conn)
    conn.close()
    return df


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"CSV file not found at: {DATA_FILE}")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df.dropna()

    if "Customer_ID" in df.columns:
        df = df.drop("Customer_ID", axis=1)

    return df


# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    model_df = df.copy()

    encoders = {}
    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    model_df["Loan_Status"] = target_encoder.fit_transform(model_df["Loan_Status"].astype(str))
    encoders["Loan_Status"] = target_encoder

    X = model_df.drop("Loan_Status", axis=1)
    y = model_df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, encoders, acc


# -----------------------------
# APP START
# -----------------------------
init_db()
df = load_data()
model, encoders, acc = train_model(df)

st.title("🏦 Loan Eligibility Prediction and Applicant Analysis System")
st.markdown("A final-year project using **Streamlit + Machine Learning + SQLite**")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "📊 Data Analysis",
    "🤖 Prediction",
    "🗂 Saved Applications"
])


# -----------------------------
# HOME TAB
# -----------------------------
with tab1:
    st.subheader("Project Overview")
    st.write("""
    This system predicts whether a loan applicant is eligible for approval
    based on demographic, financial, and credit-related factors.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Model Accuracy", f"{acc * 100:.2f}%")

    st.subheader("Problem Statement")
    st.write("""
    Financial institutions process many loan applications and need a faster,
    more consistent way to evaluate applicant eligibility. This project uses
    machine learning to support loan approval decisions.
    """)

    st.subheader("Technologies Used")
    st.write("""
    - Streamlit for web interface
    - Pandas for data preprocessing
    - Scikit-learn for model training
    - SQLite for storing submitted application records
    """)


# -----------------------------
# DATA ANALYSIS TAB
# -----------------------------
with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(df.shape)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Status Distribution")
        st.bar_chart(df["Loan_Status"].value_counts())

    with col2:
        st.subheader("Credit History Distribution")
        st.bar_chart(df["Credit_History"].value_counts())

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Property Area Distribution")
        st.bar_chart(df["Property_Area"].value_counts())

    with col4:
        st.subheader("Education Distribution")
        st.bar_chart(df["Education"].value_counts())

    st.subheader("Applicant Income vs Loan Amount")
    scatter_df = df[["Applicant_Income", "Loan_Amount"]]
    st.scatter_chart(scatter_df)


# -----------------------------
# PREDICTION TAB
# -----------------------------
with tab3:
    st.subheader("Check Loan Eligibility")

    with st.form("loan_form"):
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().astype(str).unique()))
        married = st.selectbox("Married", sorted(df["Married"].dropna().astype(str).unique()))
        dependents_display = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", sorted(df["Education"].dropna().astype(str).unique()))
        self_employed = st.selectbox("Self Employed", sorted(df["Self_Employed"].dropna().astype(str).unique()))

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=float(df["Applicant_Income"].median()),
            step=100.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=float(df["Coapplicant_Income"].median()),
            step=100.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=float(df["Loan_Amount"].median()),
            step=1.0
        )

        loan_amount_term = st.number_input(
            "Loan Amount Term",
            min_value=0.0,
            value=float(df["Loan_Amount_Term"].median()),
            step=12.0
        )

        credit_history = st.selectbox(
            "Credit History",
            sorted(df["Credit_History"].dropna().unique())
        )

        property_area = st.selectbox(
            "Property Area",
            sorted(df["Property_Area"].dropna().astype(str).unique())
        )

        submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Loan_Amount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }])

        for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

        prediction = model.predict(input_df)[0]
        result = encoders["Loan_Status"].inverse_transform([prediction])[0]

        if result == "Y":
            st.success("Loan Likely to be Approved")
            final_result = "Approved"
        else:
            st.error("Loan Likely to be Rejected")
            final_result = "Rejected"

        save_application(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, float(credit_history), property_area, final_result
        )

        st.info("Application record saved to SQLite database.")


# -----------------------------
# SAVED APPLICATIONS TAB
# -----------------------------
with tab4:
    st.subheader("Saved Loan Applications")
    saved_df = load_saved_applications()

    if saved_df.empty:
        st.warning("No applications saved yet.")
    else:
        st.dataframe(saved_df, use_container_width=True)

        csv = saved_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Saved Applications as CSV",
            data=csv,
            file_name="saved_loan_applications.csv",
            mime="text/csv"
        )
