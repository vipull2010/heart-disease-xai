# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_loader import (
    load_and_prepare_data, train_models, create_explainer,
    load_artifacts, get_unscaled_data, _unwrap_pipeline
)

st.set_page_config(layout="wide", page_title="Heart Disease XAI – Hybrid")
st.title(" Heart Disease Risk – Hybrid AI with XAI")
#st.caption("Best-in-class student project: calibrated model, SHAP explanations, and what-if simulation.")

# ======== Train or Load ========
with st.spinner("Loading data and model..."):
    X_train, X_test, y_train, y_test, pipe_base, feature_names = load_and_prepare_data(".")
    try:
        artifacts = load_artifacts()  # returns namedtuple Artifacts
        pipeline = artifacts.pipeline
        calibrated = artifacts.calibrated
        feature_names = artifacts.features
        threshold = artifacts.threshold
        leaderboard = artifacts.leaderboard
        model_loaded = True
    except Exception:
        # No artifacts yet — train
        st.info("No saved model artifacts found — training now (first run may take a while).")
        tuned_pipeline, calibrated_wrapper, threshold, leaderboard = train_models(X_train, y_train, feature_names)
        pipeline = tuned_pipeline
        calibrated = calibrated_wrapper
        model_loaded = True

# Convenience: build a predict_proba wrapper to call from app code
def predict_proba_for_df(df: pd.DataFrame):
    """
    df: DataFrame with feature columns in feature_names order.
    Returns array of probabilities for class-1.
    """
    # Try calibrated wrapper first (it may accept DataFrame directly)
    try:
        if calibrated is not None:
            return calibrated.predict_proba(df)[:, 1]
    except Exception:
        pass
    # fallback: apply pipeline's imputer/scaler and raw model
    try:
        imputer = pipeline.named_steps.get("imputer", None)
        scaler = pipeline.named_steps.get("scaler", None)
        model_step = pipeline.named_steps.get("model", None)
        if imputer is not None:
            X_imp = imputer.transform(df)
        else:
            X_imp = df.values
        if scaler is not None:
            X_trans = scaler.transform(X_imp)
        else:
            X_trans = X_imp
        return model_step.predict_proba(X_trans)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Unable to produce probabilities: {e}")

# ======== Sidebar Inputs ========
st.sidebar.header("Patient Inputs")
X_unscaled = get_unscaled_data()

def slider_num(col, label):
    mn, mx = float(np.nanpercentile(X_unscaled[col], 1)), float(np.nanpercentile(X_unscaled[col], 99))
    default = float(np.nanmedian(X_unscaled[col]))
    step = 0.1 if col in ("oldpeak",) else 1.0
    return st.sidebar.slider(label, mn, mx, default, step=step)

user = {}
user["age"] = slider_num("age","Age")
user["sex"] = st.sidebar.selectbox("Sex (0=F,1=M)", [0,1], index=1)
user["cp"] = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3], index=0)
user["trestbps"] = slider_num("trestbps","Resting BP (mm Hg)")
user["chol"] = slider_num("chol","Serum Cholesterol (mg/dl)")
user["fbs"] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], index=0)
user["restecg"] = st.sidebar.selectbox("Resting ECG (0-2)", [0,1,2], index=0)
user["thalach"] = slider_num("thalach","Max Heart Rate")
user["exang"] = st.sidebar.selectbox("Exercise Induced Angina", [0,1], index=0)
user["oldpeak"] = slider_num("oldpeak","ST Depression")
user["slope"] = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0,1,2], index=0)
user["ca"] = st.sidebar.selectbox("Major Vessels Colored (0-3)", [0,1,2,3], index=0)
user["thal"] = st.sidebar.selectbox("Thal (0=normal,1=fixed,2=reversible,3=unknown)", [0,1,2,3], index=2)

input_df = pd.DataFrame([user])[feature_names]

# ======== Predict ========
proba = predict_proba_for_df(input_df)[0]
pred = int(proba >= threshold)

left, right = st.columns(2)
with left:
    st.metric("Predicted Risk (probability)", f"{proba:.2f}")
with right:
    st.metric("Decision at threshold", "High Risk" if pred==1 else "Low Risk", delta=f"thr={threshold:.2f}")

st.divider()

# ======== SHAP Explanation & What-if Simulator ========
st.subheader("🔍 Explanation & 🧪 What-if Simulator")
st.caption("See how the patient's risk is calculated (left) and how changing key factors impacts the explanation (right).")

# What-if inputs in page (not sidebar)
sim_cols = st.columns(4)
sim = user.copy()
sim["cp"] = sim_cols[0].selectbox("What-if: cp", [0,1,2,3], index=sim["cp"])
sim["trestbps"] = sim_cols[1].number_input("What-if: trestbps", value=float(sim["trestbps"]))
sim["chol"] = sim_cols[2].number_input("What-if: chol", value=float(sim["chol"]))
sim["thalach"] = sim_cols[3].number_input("What-if: thalach", value=float(sim["thalach"]))
sim_df = pd.DataFrame([sim])[feature_names]

sim_proba = predict_proba_for_df(sim_df)[0]
delta = sim_proba - proba
st.write(f"Original Risk: **{proba:.2f}** → Simulated Risk: **{sim_proba:.2f}** (Δ {delta:+.2f})")
st.divider()

# Build SHAP explainer (robust unified API) and compute explanations for both inputs
try:
    # create_explainer expects a fitted estimator (calibrated or pipeline) and X_train DataFrame
    # We pass the calibrated wrapper if available (so calibrated.probabilities are used), otherwise the pipeline
    fitted_estimator = calibrated if calibrated is not None else pipeline
    explainer, _mode = create_explainer(fitted_estimator, X_train)

    # explainer expects numpy array input (we built it that way). Use values and index 0
    shap_exp_orig = explainer(input_df[feature_names].values)
    shap_exp_sim = explainer(sim_df[feature_names].values)

    # side-by-side plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Patient")
        try:
            _shap_fig = plt.gcf()
            _ = plt.figure(figsize=(8, 4))
            import shap as _shap  # ensure shap available
            _shap.plots.waterfall(shap_exp_orig[0], show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Could not render original SHAP waterfall: {e}")

    with col2:
        st.subheader("Simulated Patient")
        try:
            _ = plt.figure(figsize=(8, 4))
            _shap.plots.waterfall(shap_exp_sim[0], show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Could not render simulated SHAP waterfall: {e}")

except Exception as e:
    st.info(f"SHAP plot will be available after first successful training. ({e})")

# === rest of app (you had Explore/Global pages) - omitted here for brevity but can be added similarly ===