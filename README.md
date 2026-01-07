# ML Starter (Train → Test → Serve)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Train
bash
Copy code
make train
Artifacts:

artifacts/model_v001.joblib (versioned)

artifacts/meta_v001.json (versioned)

artifacts/model.joblib (latest pointer)

artifacts/meta.json (latest pointer)

Test
bash
Copy code
make test
Run API locally
bash
Copy code
make serve
Health:

bash
Copy code
curl -s http://127.0.0.1:8000/health
Predict:

bash
Copy code
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[0.2,-1.1,0.5,2.0,-0.3,0.7,1.5,-0.8]}'
Docker
Build (ensure artifacts exist first):

bash
Copy code
make train
docker build -t ml-starter:0.1 .
Run:

bash
Copy code
docker run --rm -p 8000:8000 ml-starter:0.1