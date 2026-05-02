import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

DATA_FILE = "student_performance.csv"
DB_FILE = "student_performance.db"


# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            attendance INTEGER,
            study_hours REAL,
            previous_grade REAL,
            extracurricular TEXT,
            internet_access TEXT,
            predicted_result TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(gender, attendance, study_hours, previous_grade,
                    extracurricular, internet_access, predicted_result):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (
            gender, attendance, study_hours, previous_grade,
            extracurricular, internet_access, predicted_result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        gender, attendance, study_hours, previous_grade,
        extracurricular, internet_access, predicted_result
    ))
    conn.commit()
    conn.close()


def load_saved_predictions():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


# -----------------------------
# LOAD OR CREATE DATA
# -----------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return df
    else:
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "gender": np.random.choice(["Male", "Female"], n),
            "attendance": np.random.randint(40, 100, n),
            "study_hours": np.random.uniform(1, 10, n).round(1),
            "previous_grade": np.random.uniform(40, 100, n).round(1),
            "extracurricular": np.random.choice(["Yes", "No"], n),
            "internet_access": np.random.choice(["Yes", "No"], n),
        })

        score = (
            0.35 * df["attendance"] +
            4.0 * df["study_hours"] +
            0.45 * df["previous_grade"] +
            np.where(df["internet_access"] == "Yes", 5, 0) +
            np.where(df["extracurricular"] == "Yes", 2, 0)
        )

        df["result"] = np.where(score >= 65, "Pass", "Fail")
        return df


# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    model_df = df.copy()

    encoders = {}
    categorical_cols = ["gender", "extracurricular", "internet_access"]

    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    model_df["result"] = target_encoder.fit_transform(model_df["result"])
    encoders["result"] = target_encoder

    X = model_df.drop("result", axis=1)
    y = model_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, encoders, acc, X_test, y_test, y_pred


# -----------------------------
# APP START
# -----------------------------
init_db()
df = load_data()
model, encoders, acc, X_test, y_test, y_pred = train_model(df)

st.title("🎓 Student Performance Prediction and Academic Analytics System")
st.markdown("A simple final-year project using **Streamlit + Machine Learning + SQLite**")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "📊 Data Analysis",
    "🤖 Prediction",
    "🗂 Saved Records"
])

# -----------------------------
# HOME TAB
# -----------------------------
with tab1:
    st.subheader("Project Overview")
    st.write("""
    This system predicts whether a student is likely to pass or fail
    based on academic and behavioral factors such as attendance,
    study hours, previous grades, extracurricular participation,
    and internet access.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Model Accuracy", f"{acc * 100:.2f}%")

    st.subheader("Problem Statement")
    st.write("""
    Educational institutions often struggle to identify students who are at academic risk.
    This project uses machine learning to analyze student-related factors and generate
    predictions that can help in early intervention.
    """)

    st.subheader("Technologies Used")
    st.write("""
    - Streamlit for web application
    - Pandas and NumPy for data handling
    - Scikit-learn for machine learning
    - SQLite for database storage
    """)

# -----------------------------
# DATA ANALYSIS TAB
# -----------------------------
with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Information")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass / Fail Distribution")
        st.bar_chart(df["result"].value_counts())

    with col2:
        st.subheader("Average Values")
        numeric_cols = df.select_dtypes(include=np.number).columns
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    st.subheader("Attendance vs Previous Grade")
    chart_df = df[["attendance", "previous_grade"]]
    st.scatter_chart(chart_df)

    st.subheader("Study Hours Distribution")
    st.bar_chart(df["study_hours"].value_counts().sort_index())

# -----------------------------
# PREDICTION TAB
# -----------------------------
with tab3:
    st.subheader("Predict Student Performance")

    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.5)
        previous_grade = st.slider("Previous Grade (%)", 0.0, 100.0, 60.0, 0.5)
        extracurricular = st.selectbox("Extracurricular Participation", ["Yes", "No"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            "gender": gender,
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_grade": previous_grade,
            "extracurricular": extracurricular,
            "internet_access": internet_access
        }])

        for col in ["gender", "extracurricular", "internet_access"]:
            input_df[col] = encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        predicted_label = encoders["result"].inverse_transform([prediction])[0]

        if predicted_label == "Pass":
            st.success(f"Predicted Result: {predicted_label}")
        else:
            st.error(f"Predicted Result: {predicted_label}")

        save_prediction(
            gender, attendance, study_hours, previous_grade,
            extracurricular, internet_access, predicted_label
        )

        st.info("Prediction saved successfully in SQLite database.")

# -----------------------------
# SAVED RECORDS TAB
# -----------------------------
with tab4:
    st.subheader("Saved Prediction Records")
    saved_df = load_saved_predictions()

    if saved_df.empty:
        st.warning("No records saved yet.")
    else:
        st.dataframe(saved_df, use_container_width=True)

        csv = saved_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Records as CSV",
            data=csv,
            file_name="saved_predictions.csv",
            mime="text/csv"
        )