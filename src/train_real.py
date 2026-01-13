from pathlib import Path
import json
import joblib
import time
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.data import load_data
from src.metrics import sweep_thresholds, precision_recall_f1
from src.error_analysis import top_errors, slice_by_feature
from src.calibration import brier_score, reliability_bins

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

MODEL_VERSION = "v005"  # calibration added


def choose_threshold(y_val, p_val):
    rows = sweep_thresholds(y_val, p_val)
    best = max(rows, key=lambda r: r[4])  # maximize F1 on validation
    best_t = best[0]
    return best_t, rows


def measure_infer_latency_ms(estimator, X, n_runs=200):
    _ = estimator.predict_proba(X[:1])
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = estimator.predict_proba(X[:1])
    t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / n_runs


def train_and_eval(name: str, model, split, seed: int):
    if name.lower().startswith("logreg"):
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
    else:
        estimator = model

    estimator.fit(split.X_train, split.y_train)

    # Raw (uncalibrated) validation probs
    p_val_raw = estimator.predict_proba(split.X_val)[:, 1]
    best_t_raw, rows_raw = choose_threshold(split.y_val, p_val_raw)

    # ---- Calibration (fit calibrator using validation) ----
    # We calibrate the already-fitted estimator using the validation set.
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
    calibrated.fit(split.X_val, split.y_val)

    p_val_cal = calibrated.predict_proba(split.X_val)[:, 1]
    best_t_cal, rows_cal = choose_threshold(split.y_val, p_val_cal)

    # Evaluate both on test
    p_test_raw = estimator.predict_proba(split.X_test)[:, 1]
    yhat_test_raw = (p_test_raw >= best_t_raw).astype(int)
    acc_r, prec_r, rec_r, f1_r = precision_recall_f1(split.y_test, yhat_test_raw)

    p_test_cal = calibrated.predict_proba(split.X_test)[:, 1]
    yhat_test_cal = (p_test_cal >= best_t_cal).astype(int)
    acc_c, prec_c, rec_c, f1_c = precision_recall_f1(split.y_test, yhat_test_cal)

    # Calibration metrics (Brier + reliability bins) on validation
    brier_raw = brier_score(split.y_val, p_val_raw)
    brier_cal = brier_score(split.y_val, p_val_cal)

    rel_raw = reliability_bins(split.y_val, p_val_raw, n_bins=10)
    rel_cal = reliability_bins(split.y_val, p_val_cal, n_bins=10)

    # Error analysis uses calibrated probabilities (more meaningful)
    top_fp, top_fn = top_errors(split.X_test, split.y_test, p_test_cal, best_t_cal, k=10)
    slices = slice_by_feature(
        split.X_test, split.y_test, p_test_cal, best_t_cal,
        split.feature_names, feature_i=0, bins=4
    )

    latency_raw = measure_infer_latency_ms(estimator, split.X_test, n_runs=200)
    latency_cal = measure_infer_latency_ms(calibrated, split.X_test, n_runs=200)

    report = {
        "name": name,

        "raw": {
            "best_threshold_val": float(best_t_raw),
            "val_sweep": [{"t": r[0], "acc": r[1], "prec": r[2], "rec": r[3], "f1": r[4]} for r in rows_raw],
            "test_metrics": {"accuracy": acc_r, "precision": prec_r, "recall": rec_r, "f1": f1_r},
            "latency_ms_per_call_estimate": float(latency_raw),
            "brier_val": float(brier_raw),
            "reliability_val": rel_raw,
        },

        "calibrated": {
            "method": "isotonic",
            "best_threshold_val": float(best_t_cal),
            "val_sweep": [{"t": r[0], "acc": r[1], "prec": r[2], "rec": r[3], "f1": r[4]} for r in rows_cal],
            "test_metrics": {"accuracy": acc_c, "precision": prec_c, "recall": rec_c, "f1": f1_c},
            "latency_ms_per_call_estimate": float(latency_cal),
            "brier_val": float(brier_cal),
            "reliability_val": rel_cal,
        },

        "top_fp_indices": [int(i) for i in top_fp],
        "top_fn_indices": [int(i) for i in top_fn],
        "slice_analysis": slices,
    }

    # return calibrated model as the served candidate
    return report, calibrated


def pick_winner(candidates: list, recall_margin: float = 0.01):
    # candidates here are per-model reports; we compare their calibrated test metrics
    def recall(r): return r["calibrated"]["test_metrics"]["recall"]
    def f1(r): return r["calibrated"]["test_metrics"]["f1"]
    def lat(r): return r["calibrated"]["latency_ms_per_call_estimate"]

    best_recall = max(candidates, key=recall)
    top_recall = recall(best_recall)
    top = [r for r in candidates if (top_recall - recall(r)) <= recall_margin]
    if len(top) == 1:
        return top[0]

    best_f1 = max(top, key=f1)
    top_f1 = f1(best_f1)
    top2 = [r for r in top if abs(f1(r) - top_f1) < 1e-12]
    if len(top2) == 1:
        return top2[0]

    return min(top2, key=lat)


def main(seed: int = 0):
    split = load_data(seed)

    # Candidate models
    rep_lr, est_lr = train_and_eval(
        "logreg",
        LogisticRegression(max_iter=5000, random_state=seed),
        split, seed
    )

    rep_rf, est_rf = train_and_eval(
        "rf",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1
        ),
        split, seed
    )

    winner = pick_winner([rep_lr, rep_rf], recall_margin=0.01)

    served = est_lr if winner["name"] == "logreg" else est_rf

    # Save served model
    model_path = ART / f"model_{MODEL_VERSION}.joblib"
    joblib.dump(served, model_path)
    joblib.dump(served, ART / "model.joblib")

    # Winner fields (calibrated)
    win_cal = winner["calibrated"]

    report = {
        "model_version": MODEL_VERSION,
        "seed": seed,
        "selection_rule": "recall-first (calibrated), then f1, then latency",
        "candidates": {
            "logreg": rep_lr,
            "rf": rep_rf
        },
        "winner": {
            "name": winner["name"],
            "calibration_method": win_cal["method"],
            "best_threshold_val": win_cal["best_threshold_val"],
            "test_metrics": win_cal["test_metrics"],
            "latency_ms_per_call_estimate": win_cal["latency_ms_per_call_estimate"],
            "brier_val": win_cal["brier_val"]
        }
    }

    (ART / f"report_{MODEL_VERSION}.json").write_text(json.dumps(report, indent=2))
    (ART / "report.json").write_text(json.dumps(report, indent=2))

    # Meta for API: n_features + chosen threshold
    meta = {
        "n_features": int(split.X_train.shape[1]),
        "threshold": float(win_cal["best_threshold_val"]),
        "model_version": MODEL_VERSION,
        "model_name": winner["name"],
        "calibration": win_cal["method"]
    }
    (ART / f"meta_{MODEL_VERSION}.json").write_text(json.dumps(meta, indent=2))
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved served model:", model_path)
    print("Winner:", report["winner"])


if __name__ == "__main__":
    main()
