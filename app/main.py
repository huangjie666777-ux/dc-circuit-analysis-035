from fastapi import FastAPI

app = FastAPI(title="DC circuit analysis")


@app.get("/healthz")
def health():
    return {"status": "ok"}
