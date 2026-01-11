import json
import numpy as np
from pathlib import Path

def test_infer_smoke():
    model_path = Path("artifacts/model.joblib")
    meta_path = Path("artifacts/meta.json")

    assert model_path.exists(), "Model artifact missing. Run: python -m src.train_real"
    assert meta_path.exists(), "Metadata missing. Run: python -m src.train_real"

    meta = json.loads(meta_path.read_text())
    n_features = int(meta["n_features"])

    from src.infer import predict_one

    x = np.zeros(n_features, dtype=float)
    prob, pred = predict_one(x)

    assert 0.0 <= prob <= 1.0
    assert pred in (0, 1)