import json
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from src.api import app

def _n_features():
    meta_path = Path("artifacts/meta.json")
    assert meta_path.exists(), "Metadata missing. Run: python -m src.train_real"
    meta = json.loads(meta_path.read_text())
    return int(meta["n_features"])

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_predict_ok():
    n = _n_features()
    payload = {"features": np.zeros(n, dtype=float).tolist()}

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        out = r.json()
        assert 0.0 <= out["probability"] <= 1.0
        assert out["prediction"] in (0, 1)

def test_predict_bad_length():
    n = _n_features()
    bad = {"features": [0.0] * (n - 1)}

    with TestClient(app) as client:
        r = client.post("/predict", json=bad)
        assert r.status_code == 400
