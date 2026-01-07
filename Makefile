PY=python

.PHONY: train infer test serve

train:
	$(PY) -m src.train

infer:
	$(PY) -m src.infer

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload