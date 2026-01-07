from fastapi.testclient import TestClient
from src.api import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_predict_ok():
    with TestClient(app) as client:
        payload = {"features":[0.2,-1.1,0.5,2.0,-0.3,0.7,1.5,-0.8]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        out = r.json()
        assert 0.0 <= out["probability"] <= 1.0
        assert out["prediction"] in (0,1)

def test_predict_bad_length():
    with TestClient(app) as client:
        r = client.post("/predict", json={"features":[1,2,3]})
        assert r.status_code == 400