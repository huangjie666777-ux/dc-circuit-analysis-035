# DC circuit analysis

Python 3.10 and FastAPI HTTP starter. No circuit solver is implemented yet.

Install dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Start the server:

```sh
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoint: `GET /healthz`.
