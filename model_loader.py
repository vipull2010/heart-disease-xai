# model_loader.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import shap

RANDOM_STATE = 42
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

Artifacts = namedtuple("Artifacts", ["pipeline", "calibrated", "features", "threshold", "leaderboard"])

# ---- 13 features (UCI Heart) ----
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# ---------- Loaders ----------
def _read_uci_like(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, na_values=["?"], sep=None, engine="python")
    if df.shape[1] >= 13:
        df = df.iloc[:, :13].copy()
        df.columns = FEATURE_ORDER
    return df

def _load_all_sites(base="."):
    paths = [
        "cleveland.data", "hungarian.data", "switzerland.data", "long-beach-va.data",
        "processed.hungarian.data", "processed.switzerland.data", "processed.va.data",
        "reprocessed.hungarian.data", "new.data"
    ]

    frames, y_list = [], []
    for p in paths:
        full = Path(base) / p
        if not full.exists():
            continue
        try:
            df = pd.read_csv(full, header=None, na_values=["?"], sep=None, engine="python")
        except Exception:
            continue

        ncols = df.shape[1]
        if ncols >= 14:
            X = df.iloc[:, :13].copy()
            y = df.iloc[:, 13].copy()
        elif ncols == 13:
            X = df.iloc[:, :13].copy()
            y = pd.Series([np.nan] * len(X))
        else:
            continue

        X.columns = FEATURE_ORDER
        frames.append(X)
        y_list.append(y)

    if not frames:
        raise FileNotFoundError("No UCI heart-disease files found in the project folder.")

    X_all = pd.concat(frames, axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)

    # fallback labels from heart_disease.csv
    if y_all.isna().all():
        csvp = Path(base) / "heart_disease.csv"
        if csvp.exists():
            dfc = pd.read_csv(csvp)
            target_col = "target" if "target" in dfc.columns else ("num" if "num" in dfc.columns else None)
            if target_col is not None:
                y_all = pd.Series(dfc[target_col].values[:len(X_all)])

    y_bin = pd.to_numeric(y_all, errors="coerce")
    y_bin = (y_bin.fillna(0) > 0).astype(int)

    mask_validX = ~X_all.isna().all(axis=1)
    mask_validY = y_bin.notna()
    mask = mask_validX & mask_validY

    X_all = X_all.loc[mask].reset_index(drop=True)
    y_bin = y_bin.loc[mask].reset_index(drop=True)
    return X_all, y_bin

def load_and_prepare_data(base="."):
    X, y = _load_all_sites(base)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe_base = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler(with_mean=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    return X_train, X_test, y_train, y_test, pipe_base, X.columns.tolist()

# ---------- Search spaces ----------
def _rf_space():
    return {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 4, 6, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

def _lr_space():
    return {
        "model__C": np.logspace(-3, 2, 10),
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs", "liblinear"],
    }

# ---------- Utilities ----------
def _threshold_from_pr(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1s = (2 * p * r) / (p + r + 1e-9)
    best_idx = np.nanargmax(f1s)
    return float(t[max(0, best_idx - 1)])

# ---------- Training ----------
def train_models(X_train, y_train, feature_names):
    """
    Train candidate pipelines, calibrate, choose best via CV AUC,
    store both the tuned pipeline and the calibrated wrapper.
    Returns: (tuned_pipeline, calibrated_wrapper, threshold, results)
    """
    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_name, best_est, best_auc = None, None, -1.0
    # map to remember tuned pipelines for final persistence
    tuned_pipelines = {}

    for name, pipe in candidates.items():
        search_space = _lr_space() if "Logistic" in name else _rf_space()
        rs = RandomizedSearchCV(
            pipe, search_space, n_iter=20, cv=cv, scoring="roc_auc",
            n_jobs=-1, random_state=RANDOM_STATE, refit=True, verbose=0
        )
        rs.fit(X_train, y_train)
        tuned = rs.best_estimator_
        tuned_pipelines[name] = tuned

        # Calibrate probabilities using same CV scheme
        calibrated = CalibratedClassifierCV(tuned, cv=cv, method="isotonic")
        calibrated.fit(X_train, y_train)

        oof_auc = rs.best_score_
        results[name] = {"estimator": calibrated, "cv_auc": float(oof_auc)}

        if oof_auc > best_auc:
            best_auc, best_name, best_est = oof_auc, name, calibrated

    # best_est is calibrated wrapper for the best candidate
    train_proba = best_est.predict_proba(X_train)[:, 1]
    best_threshold = _threshold_from_pr(y_train, train_proba)

    # tuned pipeline we want to persist
    tuned_pipeline = tuned_pipelines.get(best_name, None)
    if tuned_pipeline is None:
        # fallback to unwrapping best_est
        tuned_pipeline = getattr(best_est, "base_estimator", None) or getattr(best_est, "estimator", None) or best_est

    # Persist both the tuned pipeline and the calibrated wrapper
    joblib.dump(tuned_pipeline, ARTIFACTS / "best_pipeline.joblib")
    joblib.dump(best_est, ARTIFACTS / "best_calibrated.joblib")
    joblib.dump(feature_names, ARTIFACTS / "features.joblib")
    joblib.dump(best_threshold, ARTIFACTS / "threshold.joblib")
    joblib.dump(results, ARTIFACTS / "model_leaderboard.joblib")

    return tuned_pipeline, best_est, best_threshold, results

# ---------- Artifacts loader ----------
def load_artifacts():
    """
    Return Artifacts(pipeline, calibrated, features, threshold, leaderboard).
    """
    pipeline = joblib.load(ARTIFACTS / "best_pipeline.joblib")
    calibrated = joblib.load(ARTIFACTS / "best_calibrated.joblib")
    features = joblib.load(ARTIFACTS / "features.joblib")
    threshold = joblib.load(ARTIFACTS / "threshold.joblib")
    leaderboard = joblib.load(ARTIFACTS / "model_leaderboard.joblib")
    return Artifacts(pipeline, calibrated, features, threshold, leaderboard)

# ---------- Explainability ----------
def _unwrap_pipeline(calibrated_or_pipeline):
    """
    Return (pipeline, calibrated_wrapper) given either a calibrated wrapper or a pipeline.
    Raises informative errors if underlying pipeline is unavailable.
    """
    calibrated_wrapper = None
    base = calibrated_or_pipeline
    if isinstance(calibrated_or_pipeline, CalibratedClassifierCV):
        calibrated_wrapper = calibrated_or_pipeline
        base = getattr(calibrated_or_pipeline, "base_estimator", None) or getattr(calibrated_or_pipeline, "estimator", None)
        if base is None:
            raise AttributeError("Cannot access underlying estimator from CalibratedClassifierCV. Save pipeline separately.")
    from sklearn.pipeline import Pipeline as _SkPipeline
    if not isinstance(base, _SkPipeline):
        raise ValueError(f"Expected a sklearn Pipeline inside the calibrated model, got: {type(base)}")
    return base, calibrated_wrapper

def create_explainer(fitted_estimator, X_background, background_sample=200):
    """
    Build a shap.Explainer anchored to a stable predict function that always
    applies imputer+scaler before model. Accepts either a calibrated wrapper
    or a pipeline (unwrapped internally).
    Returns (explainer, "array") — explainers expect numpy array input for predict_fn.
    """
    pipeline, calibrated_wrapper = _unwrap_pipeline(fitted_estimator)
    imputer = pipeline.named_steps.get("imputer", None)
    scaler = pipeline.named_steps.get("scaler", None)
    # small background from raw X_background (DataFrame)
    bg_raw = shap.sample(X_background, min(len(X_background), background_sample), random_state=RANDOM_STATE)
    feature_names = X_background.columns.tolist()

    def predict_fn_from_array(X_numpy):
        """X_numpy: (n_samples, n_features) numpy array matching feature_names order"""
        df = pd.DataFrame(X_numpy, columns=feature_names)
        # apply imputer/scaler if present
        if imputer is not None:
            X_imp = imputer.transform(df)
        else:
            X_imp = df.values
        if scaler is not None:
            X_trans = scaler.transform(X_imp)
        else:
            X_trans = X_imp
        # prefer calibrated wrapper predict_proba if available (may accept DataFrame)
        if calibrated_wrapper is not None:
            try:
                return calibrated_wrapper.predict_proba(df)[:, 1]
            except Exception:
                return calibrated_wrapper.predict_proba(X_trans)[:, 1]
        # try raw model
        model_step = pipeline.named_steps.get("model")
        if hasattr(model_step, "predict_proba"):
            return model_step.predict_proba(X_trans)[:, 1]
        # fallback to pipeline.predict_proba if available
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(df)[:, 1]
        raise RuntimeError("No predict_proba available on model/pipeline.")

    explainer = shap.Explainer(predict_fn_from_array, bg_raw.values, feature_names=feature_names)
    return explainer, "array"

def get_unscaled_data():
    X, _ = _load_all_sites(".")
    return X.copy()