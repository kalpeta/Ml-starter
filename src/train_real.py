from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data import load_data
from src.metrics import sweep_thresholds, precision_recall_f1
from src.error_analysis import top_errors, slice_by_feature

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

MODEL_VERSION = "v003"  # bump when you change training logic


def choose_threshold(y_val, p_val):
    rows = sweep_thresholds(y_val, p_val)
    best = max(rows, key=lambda r: r[4])  # maximize F1
    best_t = best[0]
    return best_t, rows


def train_and_eval(name: str, model, split, seed: int):
    """
    Train model on train, tune threshold on val, evaluate on test.
    Returns: dict(report), fitted_pipeline_or_model
    """
    # For linear models we want scaling; for trees we don't need it.
    if name.lower().startswith("logreg"):
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
    else:
        estimator = model

    estimator.fit(split.X_train, split.y_train)

    # Tune threshold on validation
    p_val = estimator.predict_proba(split.X_val)[:, 1]
    best_t, rows = choose_threshold(split.y_val, p_val)

    # Test metrics at chosen threshold
    p_test = estimator.predict_proba(split.X_test)[:, 1]
    yhat_test = (p_test >= best_t).astype(int)
    acc, prec, rec, f1 = precision_recall_f1(split.y_test, yhat_test)

    # Error analysis on test
    top_fp, top_fn = top_errors(split.X_test, split.y_test, p_test, best_t, k=10)

    # Slice analysis starter: feature index 0 ("mean radius" in this dataset)
    slices = slice_by_feature(
        split.X_test, split.y_test, p_test, best_t,
        split.feature_names, feature_i=0, bins=4
    )

    report = {
        "name": name,
        "best_threshold_val": float(best_t),
        "val_sweep": [{"t": r[0], "acc": r[1], "prec": r[2], "rec": r[3], "f1": r[4]} for r in rows],
        "test_metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "top_fp_indices": [int(i) for i in top_fp],
        "top_fn_indices": [int(i) for i in top_fn],
        "slice_analysis": slices,
    }
    return report, estimator


def main(seed: int = 0):
    split = load_data(seed)

    # Model 1: Logistic Regression baseline
    logreg = LogisticRegression(max_iter=5000, random_state=seed)
    rep_lr, est_lr = train_and_eval("logreg", logreg, split, seed)

    # Model 2: RandomForest baseline (simple strong model)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1
    )
    rep_rf, est_rf = train_and_eval("rf", rf, split, seed)

    # Decide "winner" by test F1 (you can change rule later)
    winner = max([rep_lr, rep_rf], key=lambda r: r["test_metrics"]["f1"])

    # Save winner as the served model
    if winner["name"] == "logreg":
        served = est_lr
    else:
        served = est_rf

    model_path = ART / f"model_{MODEL_VERSION}.joblib"
    joblib.dump(served, model_path)
    joblib.dump(served, ART / "model.joblib")

    report = {
        "model_version": MODEL_VERSION,
        "seed": seed,
        "candidates": {
            "logreg": rep_lr,
            "rf": rep_rf
        },
        "winner": {
            "name": winner["name"],
            "best_threshold_val": winner["best_threshold_val"],
            "test_metrics": winner["test_metrics"]
        }
    }

    (ART / f"report_{MODEL_VERSION}.json").write_text(json.dumps(report, indent=2))
    (ART / "report.json").write_text(json.dumps(report, indent=2))

    # Meta for API: feature count + chosen threshold
    meta = {
        "n_features": int(split.X_train.shape[1]),
        "threshold": float(winner["best_threshold_val"]),
        "model_version": MODEL_VERSION,
        "model_name": winner["name"]
    }
    (ART / f"meta_{MODEL_VERSION}.json").write_text(json.dumps(meta, indent=2))
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved served model:", model_path)
    print("Winner:", report["winner"])


if __name__ == "__main__":
    main()
