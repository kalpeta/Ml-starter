from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data import load_data
from src.metrics import sweep_thresholds, precision_recall_f1
from src.error_analysis import top_errors, slice_by_feature

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

MODEL_VERSION = "v002"  # new version for the real dataset baseline

def main(seed: int = 0):
    split = load_data(seed)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    pipe.fit(split.X_train, split.y_train)

    # Choose threshold using VALIDATION set
    p_val = pipe.predict_proba(split.X_val)[:, 1]
    rows = sweep_thresholds(split.y_val, p_val)

    # pick best F1 on validation
    best = max(rows, key=lambda r: r[4])
    best_t = best[0]

    # Final test metrics at chosen threshold
    p_test = pipe.predict_proba(split.X_test)[:, 1]
    yhat_test = (p_test >= best_t).astype(int)

    acc, prec, rec, f1 = precision_recall_f1(split.y_test, yhat_test)

    # Error analysis on test
    top_fp, top_fn = top_errors(split.X_test, split.y_test, p_test, best_t, k=10)

    # Slice analysis: pick one feature as a starter (mean radius is index 0 in this dataset)
    slices = slice_by_feature(
        split.X_test, split.y_test, p_test, best_t,
        split.feature_names, feature_i=0, bins=4
    )

    report = {
        "model_version": MODEL_VERSION,
        "best_threshold_val": best_t,
        "val_sweep": [{"t":r[0], "acc":r[1], "prec":r[2], "rec":r[3], "f1":r[4]} for r in rows],
        "test_metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "top_fp_indices": [int(i) for i in top_fp],
        "top_fn_indices": [int(i) for i in top_fn],
        "slice_analysis": slices
    }

    # Save versioned model + report + latest pointers
    model_path = ART / f"model_{MODEL_VERSION}.joblib"
    joblib.dump(pipe, model_path)
    joblib.dump(pipe, ART / "model.joblib")

    (ART / f"report_{MODEL_VERSION}.json").write_text(json.dumps(report, indent=2))
    (ART / "report.json").write_text(json.dumps(report, indent=2))

    # also save meta for API (feature count)
    meta = {"n_features": int(split.X_train.shape[1]), "threshold": best_t}
    (ART / f"meta_{MODEL_VERSION}.json").write_text(json.dumps(meta, indent=2))
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved:", model_path)
    print("Best threshold (val):", best_t)
    print("Test metrics:", report["test_metrics"])

if __name__ == "__main__":
    main()
