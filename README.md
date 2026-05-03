# Loan Eligibility Prediction and Applicant Analysis System

This project is a Streamlit-based machine learning application that predicts whether a loan applicant is likely to be approved or rejected using applicant, financial, and credit-related details. The app loads a CSV dataset with fields such as Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status.[1]

## Objective

The objective of this project is to build a simple and interactive system that helps analyze loan applications and predict loan eligibility using machine learning. The app is designed to demonstrate data preprocessing, model training, prediction, and record storage in a user-friendly interface suitable for a final-year project.[1][2]

## Tools Used

- Python
- Streamlit for the web application interface.[2]
- Pandas for data loading and preprocessing.
- Scikit-learn for model training and prediction.
- SQLite for storing submitted application records.
- GitHub and Streamlit Community Cloud for version control and deployment.[3]

## Dataset

The project uses the `Loan-Eligibility-Prediction.csv` dataset. The dataset contains applicant demographic details, income information, credit history, property area, and the target variable `Loan_Status`, which is used for loan approval prediction.[1]

Main dataset columns:

- Customer_ID
- Gender
- Married
- Dependents
- Education
- Self_Employed
- Applicant_Income
- Coapplicant_Income
- Loan_Amount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status

## Features

- Dataset preview and analysis
- Loan eligibility prediction form
- Model accuracy display
- Saved application records using SQLite
- CSV download option for saved records

## Run the Project

1. Clone the repository.
2. Make sure the dataset is available at `data/Loan-Eligibility-Prediction.csv`.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```bash
loan-eligibility-project/
├── app.py
├── requirements.txt
├── loan_eligibility.db
├── README.md
└── data/
    └── Loan-Eligibility-Prediction.csv
```

## Output

The application predicts whether a loan is likely to be approved or rejected and stores submitted application details in a SQLite database for later viewing and download.
