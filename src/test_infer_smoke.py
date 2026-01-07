import numpy as np
from pathlib import Path

def test_infer_smoke():
    model_path = Path("artifacts/model.joblib")
    assert model_path.exists(), "Model artifact missing. Run: python -m src.train"

    from src.infer import predict_one

    x = np.zeros(8, dtype=float)
    prob, pred = predict_one(x)

    assert 0.0 <= prob <= 1.0
    assert pred in (0, 1)