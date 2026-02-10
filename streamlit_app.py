import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load model + training columns
model = joblib.load("fe1_tuned_lr.joblib")
FE1_COLS = joblib.load("fe1_columns.joblib")

# Load UI metadata, from my notebook
cat_cols = joblib.load("fe1_cat_cols.joblib")
num_cols = joblib.load("fe1_num_cols.joblib")
cat_options = joblib.load("fe1_cat_options.joblib")
num_defaults = joblib.load("fe1_num_defaults.joblib")

RAW_COLS = [
    "Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?", "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness"
]

# helpers for better UX / safer inputs
RATING_COLS_0_10 = [
    "Academic Pressure", "Work Pressure", "Financial Stress",
    "Study Satisfaction", "Job Satisfaction"
]

def clamp(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        return None
    return max(lo, min(hi, x))

def sanitize_inputs(inp: dict):
    """
    Keep your current UI (which allows decimals), but sanitize to match the dataset style:
    - rating scales -> rounded ints (0-10)
    - Age -> int
    - Hours -> int (0-24)
    - CGPA -> 2dp (0-10)
    Returns: (sanitized_dict, changes_list)
    """
    out = dict(inp)
    changes = []

    # Age (int)
    if "Age" in out:
        old = out["Age"]
        v = clamp(old, 0, 100)
        if v is not None:
            v2 = int(round(v))
            out["Age"] = v2
            if float(old) != float(v2):
                changes.append(f"Age: {old} → {v2} (rounded)")

    # Work/Study Hours (int)
    if "Work/Study Hours" in out:
        old = out["Work/Study Hours"]
        v = clamp(old, 0, 24)
        if v is not None:
            v2 = int(round(v))
            out["Work/Study Hours"] = v2
            if float(old) != float(v2):
                changes.append(f"Work/Study Hours: {old} → {v2} (rounded)")

    # Rating scales (rounded int 0-10)
    for c in RATING_COLS_0_10:
        if c in out:
            old = out[c]
            v = clamp(old, 0, 10)
            if v is not None:
                v2 = int(round(v))
                out[c] = v2
                if float(old) != float(v2):
                    changes.append(f"{c}: {old} → {v2} (rounded)")

    # CGPA (2dp)
    if "CGPA" in out:
        old = out["CGPA"]
        v = clamp(old, 0, 10)
        if v is not None:
            v2 = round(v, 2)
            out["CGPA"] = v2
            if float(old) != float(v2):
                changes.append(f"CGPA: {old} → {v2:.2f} (rounded to 2 d.p.)")

    # Ensure categoricals are strings (stable encoding)
    for c in cat_cols:
        if c in out and out[c] is not None:
            out[c] = str(out[c])

    return out, changes

def risk_band(p: float):
    if p < 0.40:
        return "Low"
    elif p < 0.70:
        return "Moderate"
    return "High"

def predict_from_inputs(inp: dict):
    """Build raw DF -> FE1 -> model proba."""
    df_in = pd.DataFrame([{c: inp.get(c) for c in RAW_COLS}])
    X_in = preprocess_fe1(df_in)
    proba = float(model.predict_proba(X_in)[0, 1])
    return proba, X_in

def build_report_df(inp_original: dict, inp_sanitized: dict, mode: str, threshold: float, proba: float, pred: int):
    base = {
        "Mode": mode,
        "Threshold": round(threshold, 2),
        "Probability_Depression1": round(proba, 4),
        "Predicted_Class": pred,
        "Risk_Band": risk_band(proba),
    }
    
    row = {}
    for c in RAW_COLS:
        row[f"input_{c}"] = inp_sanitized.get(c, None)
    # Keep original too (for transparency)
    for c in RAW_COLS:
        row[f"raw_{c}"] = inp_original.get(c, None)

    row.update(base)
    return pd.DataFrame([row])

# Page config
st.set_page_config(page_title="Depression Prediction (FE1 Tuned LR)", layout="centered")

st.title("Depression Prediction (FE1 Tuned LR)")
st.caption("Educational demo for MLDP. Not a medical diagnosis tool.")

with st.expander("What this app does (and does NOT do)", expanded=False):
    st.write(
        "- This app predicts a **Depression risk class (0/1)** using your trained Logistic Regression model.\n"
        "- It is for **educational/demo purposes** only.\n"
        "- The output is **not** a medical diagnosis."
    )
    st.write(
        "If this topic feels heavy or you’re feeling overwhelmed, it can help to talk to a trusted adult or a professional."
    )


# Sidebar: Operating mode + threshold
st.sidebar.header("Model Settings")

mode = st.sidebar.radio(
    "Operating Mode",
    ["High Sensitivity (Recall-first)", "Balanced", "Conservative"],
    index=1
)

mode_to_threshold = {
    "High Sensitivity (Recall-first)": 0.30,
    "Balanced": 0.40,
    "Conservative": 0.50
}

threshold = st.sidebar.slider(
    "Decision Threshold",
    0.10, 0.90,
    float(mode_to_threshold[mode]),
    0.05
)

st.sidebar.caption(
    "Lower threshold → higher Recall (catch more positives) but more false positives.\n"
    "Higher threshold → higher Precision but more false negatives."
)

# small UI hints + export button placeholder
st.sidebar.divider()
st.sidebar.subheader("Tips")
st.sidebar.write("- Try changing **Operating Mode** to see how the prediction changes.")
st.sidebar.write("- Use **Scenario A/B** to capture screenshots for your report.")
st.sidebar.write("- Ratings are treated as 0–10 scales.")

# FE1 preprocessing 
def preprocess_fe1(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # numeric safety for FE1 columns
    df["Academic Pressure"] = pd.to_numeric(df["Academic Pressure"], errors="coerce")
    df["Work Pressure"] = pd.to_numeric(df["Work Pressure"], errors="coerce")

    # FE1 engineered feature
    df["Total_Pressure"] = df["Academic Pressure"] + df["Work Pressure"]

    # drop originals
    df = df.drop(columns=["Academic Pressure", "Work Pressure"])

    # fill missing
    dyn_cat = df.select_dtypes(include=["object", "category"]).columns
    df[dyn_cat] = df[dyn_cat].fillna("Unknown")
    df = df.fillna(df.median(numeric_only=True))

    # one-hot
    df_enc = pd.get_dummies(df, columns=dyn_cat, drop_first=True)

    # match training columns
    df_enc = df_enc.reindex(columns=FE1_COLS, fill_value=0)
    return df_enc

# Validation
def validate_inputs(inp: dict) -> list[str]:
    errs = []

    # Age sanity
    age = inp.get("Age")
    if age is None or age < 10 or age > 100:
        errs.append("Age should be between 10 and 100.")

    # Hours sanity
    hours = inp.get("Work/Study Hours")
    if hours is None or hours < 0 or hours > 24:
        errs.append("Work/Study Hours should be between 0 and 24.")

    for col in ["Academic Pressure", "Work Pressure", "Financial Stress"]:
        v = inp.get(col)
        if v is None or v < 0 or v > 10:
            errs.append(f"{col} should be between 0 and 10.")

    for col in ["Study Satisfaction", "Job Satisfaction"]:
        v = inp.get(col)
        if v is None or v < 0 or v > 10:
            errs.append(f"{col} should be between 0 and 10.")

    # CGPA sanity
    cgpa = inp.get("CGPA")
    if cgpa is None or cgpa < 0 or cgpa > 10:
        errs.append("CGPA should be between 0 and 10.")

    return errs

# UI
st.subheader("Inputs")

# quick explanation about decimals + rounding
with st.expander("Input notes (to avoid weird decimals)", expanded=False):
    st.write(
        "- **CGPA** can be a decimal (e.g., 4.07).\n"
        "- Rating scales (Pressure / Satisfaction / Stress) are typically **whole numbers**.\n"
        "- If you enter decimals for rating scales, the app will **round them** before prediction for consistency."
    )

# Session state
if "saved" not in st.session_state:
    st.session_state.saved = {}  

# Stores the most recent successful prediction 
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
    

with st.form("predict_form"):
    inputs = {}

    st.markdown("### Demographics")
    inputs["Gender"] = st.selectbox("Gender", cat_options.get("Gender", ["Unknown"]))
    inputs["Age"] = st.number_input(
        "Age",
        value=float(num_defaults.get("Age", 20.0)),
        min_value=0.0,
        step=1.0
    )

    inputs["City"] = st.selectbox("City", cat_options.get("City", ["Unknown"]))
    inputs["Profession"] = st.selectbox("Profession", cat_options.get("Profession", ["Unknown"]))
    inputs["Degree"] = st.selectbox("Degree", cat_options.get("Degree", ["Unknown"]))

    st.markdown("### Stress & Pressure")
    inputs["Academic Pressure"] = st.number_input(
        "Academic Pressure (0–10)",
        value=float(num_defaults.get("Academic Pressure", 0.0)),
        min_value=0.0
    )
    inputs["Work Pressure"] = st.number_input(
        "Work Pressure (0–10)",
        value=float(num_defaults.get("Work Pressure", 0.0)),
        min_value=0.0
    )
    inputs["Financial Stress"] = st.number_input(
        "Financial Stress (0–10)",
        value=float(num_defaults.get("Financial Stress", 0.0)),
        min_value=0.0
    )

    st.markdown("### Study / Work")
    inputs["CGPA"] = st.number_input(
        "CGPA (0–10)",
        value=float(num_defaults.get("CGPA", 0.0)),
        min_value=0.0
    )
    inputs["Work/Study Hours"] = st.number_input(
        "Work/Study Hours (0–24)",
        value=float(num_defaults.get("Work/Study Hours", 0.0)),
        min_value=0.0
    )
    inputs["Study Satisfaction"] = st.number_input(
        "Study Satisfaction (0–10)",
        value=float(num_defaults.get("Study Satisfaction", 0.0)),
        min_value=0.0
    )
    inputs["Job Satisfaction"] = st.number_input(
        "Job Satisfaction (0–10)",
        value=float(num_defaults.get("Job Satisfaction", 0.0)),
        min_value=0.0
    )

    st.markdown("### Lifestyle")
    inputs["Sleep Duration"] = st.selectbox("Sleep Duration", cat_options.get("Sleep Duration", ["Unknown"]))
    inputs["Dietary Habits"] = st.selectbox("Dietary Habits", cat_options.get("Dietary Habits", ["Unknown"]))

    st.markdown("### Background")
    inputs["Have you ever had suicidal thoughts ?"] = st.selectbox(
        "Have you ever had suicidal thoughts ?",
        cat_options.get("Have you ever had suicidal thoughts ?", ["Unknown"])
    )
    inputs["Family History of Mental Illness"] = st.selectbox(
        "Family History of Mental Illness",
        cat_options.get("Family History of Mental Illness", ["Unknown"])
    )

    st.caption(f"Current Operating Mode: **{mode}** | Threshold: **{threshold:.2f}**")
    submitted = st.form_submit_button("Predict")

# Predict + display
show_results = submitted or (st.session_state.last_pred is not None)

if show_results:
    # If user clicked Predict -> compute fresh prediction and save to session_state
    if submitted:
        sanitized_inputs, changes = sanitize_inputs(inputs)

        # Validate
        errors = validate_inputs(sanitized_inputs)
        if errors:
            st.error("Please fix the following input issues before predicting:")
            for e in errors:
                st.write(f"- {e}")
            st.stop()

        # Predict
        df_in = pd.DataFrame([{c: sanitized_inputs.get(c) for c in RAW_COLS}])
        X_in = preprocess_fe1(df_in)
        proba = float(model.predict_proba(X_in)[0, 1])

        # Save last successful prediction
        st.session_state.last_pred = {
            "raw_inputs": inputs.copy(),
            "sanitized_inputs": sanitized_inputs.copy(),
            "changes": list(changes),
            "proba": float(proba),
        }

    # Otherwise  reuse last prediction 
    last = st.session_state.last_pred
    raw_inputs = last["raw_inputs"]
    sanitized_inputs = last["sanitized_inputs"]
    changes = last["changes"]
    proba = float(last["proba"])

    pred = int(proba >= threshold)

    
    df_in = pd.DataFrame([{c: sanitized_inputs.get(c) for c in RAW_COLS}])
    X_in = preprocess_fe1(df_in)

    # Result section 
    st.markdown("## Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability (Depression=1)", f"{proba:.3f}")
    c2.metric("Threshold", f"{threshold:.2f}")
    c3.metric("Predicted Class", str(pred))
    c4.metric("Risk Band", risk_band(proba))

    st.progress(min(max(proba, 0.0), 1.0))

    if pred == 1:
        st.warning("Prediction: **Positive (class 1)** based on the current operating mode.")
    else:
        st.success("Prediction: **Negative (class 0)** based on the current operating mode.")

    if not submitted:
        st.info("Showing your **last prediction** (so the page doesn’t reset). Click **Predict** after changing inputs to update the model output.")

    # Show rounding/clamping info 
    if changes:
        with st.expander("Auto-adjustments applied (for consistency)", expanded=False):
            st.write("Some values were rounded/clamped to match typical dataset scales:")
            for c in changes:
                st.write(f"- {c}")

    # make output more meaningful
    with st.expander("Input summary (what the model actually used)", expanded=False):
        show_df = pd.DataFrame([sanitized_inputs])[RAW_COLS]
        st.dataframe(show_df, use_container_width=True)

    # Small explanation about trade-offs
    with st.expander("How to interpret this (trade-offs)", expanded=False):
        st.write(
            "- **Lower threshold** → higher **Recall** (catch more positives), but more false positives.\n"
            "- **Higher threshold** → higher **Precision**, but more missed positives (false negatives).\n"
            "- You can change the **Operating Mode** on the left to see how predictions shift."
        )

    # quick relative profile chart vs defaults (medians)
    with st.expander("Profile vs typical (quick view)", expanded=False):
        numeric_compare_cols = [
            "Academic Pressure", "Work Pressure", "Financial Stress",
            "Study Satisfaction", "Job Satisfaction", "Work/Study Hours", "CGPA"
        ]
        comp = []
        for c in numeric_compare_cols:
            comp.append({
                "Feature": c,
                "Your Value": float(sanitized_inputs.get(c, 0) or 0),
                "Typical (Median)": float(num_defaults.get(c, 0) or 0),
            })
        comp_df = pd.DataFrame(comp).set_index("Feature")
        st.bar_chart(comp_df)

    # Downloadable report
    report_df = build_report_df(raw_inputs, sanitized_inputs, mode, threshold, proba, pred)
    st.download_button(
        "Download prediction report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="depression_prediction_report.csv",
        mime="text/csv"
    )

    # Explanation: top drivers (LogReg coefficients)
    st.markdown("### Explanation (Top Drivers)")

    coef = model.coef_[0]
    contrib = X_in.iloc[0].values * coef  
    explain_df = pd.DataFrame({
        "feature": X_in.columns,
        "contribution": contrib
    }).sort_values("contribution", ascending=False)

    left, right = st.columns(2)
    with left:
        st.write("Top features increasing risk")
        st.dataframe(explain_df.head(8), use_container_width=True)
    with right:
        st.write("Top features decreasing risk")
        st.dataframe(explain_df.tail(8).sort_values("contribution"), use_container_width=True)

    # What-if analysis
    st.markdown("### What-if Analysis (try changing key factors)")

    wi = dict(sanitized_inputs)  
    w1, w2, w3 = st.columns(3)
    with w1:
        wi["Academic Pressure"] = st.slider("What-if Academic Pressure", 0, 10, int(wi["Academic Pressure"]), 1, key="wi_acad")
    with w2:
        wi["Work Pressure"] = st.slider("What-if Work Pressure", 0, 10, int(wi["Work Pressure"]), 1, key="wi_work")
    with w3:
        wi["Financial Stress"] = st.slider("What-if Financial Stress", 0, 10, int(wi["Financial Stress"]), 1, key="wi_fin")

    wi_proba, _ = predict_from_inputs(wi)
    wi_pred = int(wi_proba >= threshold)

    k1, k2, k3 = st.columns(3)
    k1.metric("Current Prob", f"{proba:.3f}")
    k2.metric("What-if Prob", f"{wi_proba:.3f}")
    k3.metric("Δ Prob", f"{(wi_proba - proba):+.3f}")

    if wi_pred == 1:
        st.warning(f"What-if Prediction: **1** (threshold {threshold:.2f})")
    else:
        st.success(f"What-if Prediction: **0** (threshold {threshold:.2f})")

    # Scenario comparison (A/B)
    st.markdown("### Scenario Comparison (A vs B)")

    with st.expander("How to use A/B (quick)", expanded=False):
        st.write(
            "1) Fill inputs → **Predict** → click **Save as Scenario A**\n"
            "2) Change some inputs or threshold → **Predict** → click **Save as Scenario B**\n"
            "3) The app shows **A vs B** probability + class change + delta."
        )

    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.button("Save as Scenario A"):
            st.session_state.saved["A"] = {
                "proba": float(proba),
                "pred": int(pred),
                "inputs": sanitized_inputs.copy(),
                "threshold": float(threshold),
                "mode": mode
            }
            st.success("Saved as Scenario A")

    with colB:
        if st.button("Save as Scenario B"):
            st.session_state.saved["B"] = {
                "proba": float(proba),
                "pred": int(pred),
                "inputs": sanitized_inputs.copy(),
                "threshold": float(threshold),
                "mode": mode
            }
            st.success("Saved as Scenario B")

    with colC:
        if st.button("Clear Scenarios"):
            st.session_state.saved = {}
            st.info("Cleared saved scenarios")

    with colD:
        if st.button("Clear Last Prediction"):
            st.session_state.last_pred = None
            st.info("Cleared last prediction (results will hide until you Predict again).")

    if "A" in st.session_state.saved and "B" in st.session_state.saved:
        A = st.session_state.saved["A"]
        B = st.session_state.saved["B"]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("A Prob", f"{A['proba']:.3f}")
        s2.metric("B Prob", f"{B['proba']:.3f}")
        s3.metric("Δ Prob (B - A)", f"{(B['proba'] - A['proba']):+.3f}")
        s4.metric("A→B Class", f"{A['pred']} → {B['pred']}")

        with st.expander("Show scenario details", expanded=False):
            st.write("Scenario A settings:", {"mode": A["mode"], "threshold": round(A["threshold"], 2)})
            st.write("Scenario B settings:", {"mode": B["mode"], "threshold": round(B["threshold"], 2)})

            diffs = []
            for c in RAW_COLS:
                if A["inputs"].get(c) != B["inputs"].get(c):
                    diffs.append((c, A["inputs"].get(c), B["inputs"].get(c)))
            if diffs:
                diff_df = pd.DataFrame(diffs, columns=["Feature", "Scenario A", "Scenario B"])
                st.write("Changed inputs:")
                st.dataframe(diff_df, use_container_width=True)
            else:
                st.write("No input differences detected (you may have only changed threshold/mode).")

    st.markdown("---")
    st.caption("Tip: Change Operating Mode / threshold to see how predictions change. Save scenarios for easy screenshots.")
