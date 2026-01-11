PY=python

.PHONY: train infer test serve

train:
	$(PY) -m src.train_real

infer:
	$(PY) -m src.test_infer_smoke

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload