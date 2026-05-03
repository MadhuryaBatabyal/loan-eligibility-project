import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Eligibility Prediction System", page_icon="🏦", layout="wide")

DATA_FILE = "data/loan_data.csv"
DB_FILE = "loan_eligibility.db"


# -----------------------------
# DATABASE
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
# DATA
# -----------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "Gender": np.random.choice(["Male", "Female"], n),
            "Married": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["0", "1", "2", "3+"], n),
            "Education": np.random.choice(["Graduate", "Not Graduate"], n),
            "Self_Employed": np.random.choice(["Yes", "No"], n),
            "ApplicantIncome": np.random.randint(1500, 25000, n),
            "CoapplicantIncome": np.random.randint(0, 10000, n),
            "LoanAmount": np.random.randint(50, 400, n),
            "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], n),
            "Credit_History": np.random.choice([1.0, 0.0], n, p=[0.8, 0.2]),
            "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], n),
        })

        score = (
            0.00008 * df["ApplicantIncome"] +
            0.00005 * df["CoapplicantIncome"] -
            0.002 * df["LoanAmount"] +
            0.55 * df["Credit_History"] +
            np.where(df["Education"] == "Graduate", 0.10, 0.0) +
            np.where(df["Married"] == "Yes", 0.05, 0.0) +
            np.where(df["Property_Area"] == "Semiurban", 0.08, 0.0)
        )

        df["Loan_Status"] = np.where(score > 0.55, "Y", "N")
        return df


# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    model_df = df.copy()

    encoders = {}
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    for col in cat_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    model_df["Loan_Status"] = target_encoder.fit_transform(model_df["Loan_Status"])
    encoders["Loan_Status"] = target_encoder

    X = model_df.drop("Loan_Status", axis=1)
    y = model_df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
# HOME
# -----------------------------
with tab1:
    st.subheader("Project Overview")
    st.write("""
    This system predicts whether a loan applicant is eligible for approval
    based on demographic, financial, and credit-related details.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Model Accuracy", f"{acc * 100:.2f}%")

    st.subheader("Problem Statement")
    st.write("""
    Banks receive many loan applications and need a fast, consistent method
    to identify eligible applicants. This system helps automate the evaluation
    process using historical applicant data.
    """)

    st.subheader("Technologies Used")
    st.write("""
    - Streamlit for frontend
    - Pandas and NumPy for data handling
    - Scikit-learn for machine learning
    - SQLite for application record storage
    """)

# -----------------------------
# DATA ANALYSIS
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

    st.subheader("Applicant Income Distribution")
    st.bar_chart(df["ApplicantIncome"])

    st.subheader("Loan Amount vs Applicant Income")
    scatter_df = df[["ApplicantIncome", "LoanAmount"]]
    st.scatter_chart(scatter_df)

# -----------------------------
# PREDICTION
# -----------------------------
with tab3:
    st.subheader("Check Loan Eligibility")

    with st.form("loan_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=500.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=1500.0, step=500.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0, step=10.0)
        loan_amount_term = st.selectbox("Loan Amount Term", [120, 180, 240, 300, 360])
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }])

        for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
            input_df[col] = encoders[col].transform(input_df[col])

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
            loan_amount_term, credit_history, property_area, final_result
        )

        st.info("Application record saved to SQLite database.")

# -----------------------------
# SAVED APPLICATIONS
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
            file_name="loan_applications.csv",
            mime="text/csv"
        )
