from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.solver import CircuitError, solve_request

app = FastAPI(title="DC circuit analysis")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.exception_handler(CircuitError)
async def circuit_error_handler(_request: Request, exc: CircuitError):
    return JSONResponse(status_code=422, content={"error": exc.detail})


@app.post("/analyze")
async def analyze(request: Request):
    body = await request.json()
    return solve_request(body)
