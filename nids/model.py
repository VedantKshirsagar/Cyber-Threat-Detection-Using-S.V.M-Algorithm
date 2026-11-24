"""Training, evaluation and prediction with SVM (scikit-learn)."""
import time

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import data_utils as du

MODEL_FILES = {"binary": "svm_binary.joblib", "multi": "svm_multi.joblib"}
MODE_TITLES = {
    "binary": "Binary (Normal vs Attack)",
    "multi": "Multi-class (Normal, DoS, Probe, R2L, U2R)",
}


# ----------------------------------------------------------------- pipeline
def build_pipeline(classifier):
    """Preprocessing + classifier in ONE object.

    Because both steps live in a single Pipeline, training and prediction
    always apply exactly the same preprocessing (no mismatch bugs).
    """
    preprocess = ColumnTransformer([
        # text columns -> 0/1 columns (unknown values in new data are ignored)
        ("categorical", OneHotEncoder(handle_unknown="ignore"), du.CATEGORICAL),
        # numeric columns -> mean 0, std 1 (SVM needs scaled inputs)
        ("numeric", StandardScaler(), du.NUMERIC),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def make_target(df, mode):
    if mode == "binary":
        return np.where(df["category"] == "Normal", "Normal", "Attack")
    return df["category"].to_numpy()


def make_svm(mode):
    # class_weight='balanced' gives rare attack types more importance
    weight = "balanced" if mode == "multi" else None
    return SVC(kernel="rbf", C=10, gamma="scale", class_weight=weight)


# --------------------------------------------------------------- evaluation
def evaluate(pipe, X, y, labels):
    pred = pipe.predict(X)
    report = classification_report(y, pred, labels=labels, zero_division=0,
                                   output_dict=True)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": p, "recall": r, "f1": f,
        "report": report,
        "report_text": classification_report(y, pred, labels=labels,
                                             zero_division=0, digits=3),
        "confusion": confusion_matrix(y, pred, labels=labels),
        "labels": list(labels),
    }


def prepare_training_data(mode, sample_size=40000, random_state=42):
    """Single source of truth for the training data.

    Returns the dataframes and the 80/20 split. Both train_model() and
    compare_models() use this, so every number in the project comes from
    exactly the same training rows.
    """
    train_df = du.load_nsl_kdd(du.TRAIN_FILE)
    test_df = du.load_nsl_kdd(du.TEST_FILE)
    y_all = make_target(train_df, mode)
    if sample_size and sample_size < len(train_df):
        train_df, _, y_all, _ = train_test_split(
            train_df, y_all, train_size=sample_size, stratify=y_all,
            random_state=random_state)
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df[du.FEATURES], y_all, test_size=0.2, stratify=y_all,
        random_state=random_state)
    return test_df, X_tr, X_val, y_tr, y_val


def train_model(mode="binary", sample_size=40000, progress=print, save=True,
                random_state=42):
    """Train an SVM and evaluate it on a validation split AND the official test set."""
    if not du.data_available():
        raise FileNotFoundError(
            "Dataset not found. Run:  python download_data.py")
    progress("Loading dataset ...")
    if sample_size:
        progress(f"Using a stratified sample of {sample_size:,} rows "
                 f"(80% train / 20% validation) to keep training fast ...")
    test_df, X_tr, X_val, y_tr, y_val = prepare_training_data(
        mode, sample_size, random_state)

    labels = ["Normal", "Attack"] if mode == "binary" else du.CATEGORY_ORDER
    pipe = build_pipeline(make_svm(mode))
    progress(f"Training SVM on {len(X_tr):,} rows ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    seconds = time.time() - t0
    progress(f"Training finished in {seconds:.1f} s. Evaluating ...")

    bundle = {
        "pipeline": pipe,
        "mode": mode,
        "labels": labels,
        "train_rows": len(X_tr),
        "train_seconds": seconds,
        "validation": evaluate(pipe, X_val, y_val, labels),
        "test": evaluate(pipe, test_df[du.FEATURES],
                         make_target(test_df, mode), labels),
    }
    if save:
        du.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, du.MODELS_DIR / MODEL_FILES[mode])
        progress("Model saved.")
    return bundle


def load_bundle(mode):
    path = du.MODELS_DIR / MODEL_FILES[mode]
    return joblib.load(path) if path.exists() else None


def predict_dataframe(bundle, df):
    return bundle["pipeline"].predict(df[du.FEATURES])


# ------------------------------------------------------------ comparisons
def compare_models(sample_size=40000, random_state=42, progress=print):
    """Binary task: SVM vs other algorithms on the official test set."""
    # Same training rows as train_model(), so the SVM row matches exactly.
    test_df, X_tr, _, y_tr, _ = prepare_training_data(
        "binary", sample_size, random_state)
    y_test = make_target(test_df, "binary")
    candidates = {
        "SVM (RBF kernel)": SVC(kernel="rbf", C=10, gamma="scale"),
        "SVM (Linear)": LinearSVC(C=1, max_iter=5000),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100,
                                                random_state=random_state),
    }
    rows = []
    for name, clf in candidates.items():
        progress(f"Training {name} ...")
        pipe = build_pipeline(clf)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        seconds = time.time() - t0
        pred = pipe.predict(test_df[du.FEATURES])
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, pos_label="Attack", average="binary", zero_division=0)
        rows.append({"Model": name,
                     "Test accuracy": accuracy_score(y_test, pred),
                     "Attack precision": p, "Attack recall": r, "Attack F1": f,
                     "Train time (s)": seconds})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- plotting
def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Return a matplotlib Figure (colour = share of each true class)."""
    cm = np.asarray(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of actual class")
    fig.tight_layout()
    return fig
