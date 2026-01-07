import numpy as np
from pathlib import Path
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

N_FEATURES = 8
MODEL_VERSION = 'v001'

def main(seed: int = 0):
    np.random.seed(seed)

    # Synthetic dataset (stand-in for real data)
    X, y = make_classification(
        n_samples=2000,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=1,
        class_sep=1.2,
        random_state=seed
    )

    # Stratified split (classification best practice)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Pipeline prevents leakage: scaler fit ONLY on train; reused in inference
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate at default threshold 0.5
    p = pipe.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()

    print("Confusion:", {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})
    print("accuracy :", accuracy_score(y_test, yhat))
    print("precision:", precision_score(y_test, yhat))
    print("recall   :", recall_score(y_test, yhat))
    print("f1       :", f1_score(y_test, yhat))

    # Save versioned model
    model_path = ART / f"model_{MODEL_VERSION}.joblib"
    joblib.dump(pipe, model_path)
    print("Saved model:", model_path)

    # Also write a stable 'latest' name for serving
    latest_path = ART / "model.joblib"
    joblib.dump(pipe, latest_path)
    print("Updated latest:", latest_path)

    # Save metadata (versioned + latest)
    import json
    meta = {"n_features": N_FEATURES}

    meta_path = ART / f"meta_{MODEL_VERSION}.json"
    meta_path.write_text(json.dumps(meta))
    print("Saved metadata:", meta_path)

    latest_meta = ART / "meta.json"
    latest_meta.write_text(json.dumps(meta))
    print("Updated latest metadata:", latest_meta)

if __name__ == "__main__":
    main()
