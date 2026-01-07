from pathlib import Path
import joblib
import numpy as np

ART = Path("artifacts")
MODEL_PATH = ART / "model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
    return joblib.load(MODEL_PATH)

def predict_one(x: np.ndarray):
    """
    x: shape (n_features,)  -> returns (prob, pred)
    """
    model = load_model()
    x2 = x.reshape(1, -1)
    prob = model.predict_proba(x2)[0, 1]
    pred = int(prob >= 0.5)
    return prob, pred

def main():
    # Example input (8 features because train.py uses n_features=8)
    x = np.array([0.2, -1.1, 0.5, 2.0, -0.3, 0.7, 1.5, -0.8], dtype=float)
    prob, pred = predict_one(x)
    print("x =", x.tolist())
    print("p(y=1) =", prob)
    print("pred =", pred)

if __name__ == "__main__":
    main()
