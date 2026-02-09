# MLDP Depression Prediction (Streamlit + Notebook)

Educational MLDP project that trains a binary classification model to predict **Depression (0/1)** from survey-style features (e.g., pressure, satisfaction, sleep, dietary habits, financial stress).  
**This is not a medical diagnosis tool.**

---

## What’s inside this repo

### 1) Jupyter Notebook (Model Development)
- Full workflow: data loading → cleaning → EDA → model comparison → feature selection (FS) → feature engineering (FE) → hyperparameter tuning → final evaluation.
- Output logs include Accuracy, **Recall(1)** (primary metric), and classification reports.

### 2) Streamlit Web App (Deployment)
- Interactive UI with input validation and user-friendly explanations.
- Uses the trained Logistic Regression model + saved preprocessing metadata:
  - `fe1_tuned_lr.joblib`
  - `fe1_columns.joblib`
  - `fe1_cat_cols.joblib`, `fe1_num_cols.joblib`
  - `fe1_cat_options.joblib`, `fe1_num_defaults.joblib`

---

## Files

- `streamlit_app.py` — Streamlit application entry point
- `MLDP_Depression.ipynb` — notebook for development + experiments
- `student_depression_dataset.csv` — dataset used in notebook
- `unseen_holdout_depression.csv` — held-out dataset saved early for final evaluation (unseen)
- `requirements.txt` — Python dependencies
- `*.joblib` — trained model + feature column alignment + UI metadata

---

## How to run locally

### Option A: Run Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
