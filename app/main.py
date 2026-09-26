"""HTTP API for DC circuit operating point analysis."""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.solver import Circuit, Component, Solution, solve

app = FastAPI(title="DC circuit operating point analysis")

KINDS = {"resistor", "voltage_source", "current_source"}


class ComponentIn(BaseModel):
    id: str
    type: str
    start: str
    end: str
    value: float


class SweepIn(BaseModel):
    source_id: str
    values: list[float] = Field(min_length=1)


class CircuitIn(BaseModel):
    nodes: list[str] = Field(min_length=1)
    ground: str
    components: list[ComponentIn] = Field(default_factory=list)
    sweep: SweepIn | None = None


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/healthz")
def health():
    return {"status": "ok"}


def _validate(payload: CircuitIn) -> list[dict[str, Any]]:
    """Semantic validation; each error locates the field and component."""
    errors: list[dict[str, Any]] = []

    if payload.ground not in payload.nodes:
        errors.append({"field": "ground",
                       "message": f"ground node '{payload.ground}' is not in nodes"})

    seen: set[str] = set()
    for i, comp in enumerate(payload.components):
        loc = f"components[{i}]"
        if comp.id in seen:
            errors.append({"field": f"{loc}.id", "component": comp.id,
                           "message": f"duplicate component id '{comp.id}'"})
        seen.add(comp.id)
        if comp.type not in KINDS:
            errors.append({"field": f"{loc}.type", "component": comp.id,
                           "message": f"unknown component type '{comp.type}'"})
        for endpoint in ("start", "end"):
            node = getattr(comp, endpoint)
            if node not in payload.nodes:
                errors.append({"field": f"{loc}.{endpoint}", "component": comp.id,
                               "message": f"unknown node '{node}'"})
        if comp.start == comp.end:
            errors.append({"field": f"{loc}.end", "component": comp.id,
                           "message": "self-loop: start and end are the same node"})
        if not math.isfinite(comp.value):
            errors.append({"field": f"{loc}.value", "component": comp.id,
                           "message": "value must be finite"})
        elif comp.type == "resistor" and comp.value <= 0:
            errors.append({"field": f"{loc}.value", "component": comp.id,
                           "message": "resistance must be a finite positive number"})

    if payload.sweep is not None:
        sweep = payload.sweep
        by_id = {c.id: c for c in payload.components}
        target = by_id.get(sweep.source_id)
        if target is None:
            errors.append({"field": "sweep.source_id",
                           "message": f"no component with id '{sweep.source_id}'"})
        elif target.type not in ("voltage_source", "current_source"):
            errors.append({"field": "sweep.source_id", "component": target.id,
                           "message": "sweep target must be an independent source"})
        for j, v in enumerate(sweep.values):
            if not math.isfinite(v):
                errors.append({"field": f"sweep.values[{j}]",
                               "message": "sweep values must be finite"})
    return errors


def _to_circuit(payload: CircuitIn, overrides: dict[str, float] | None = None) -> Circuit:
    overrides = overrides or {}
    return Circuit(
        nodes=list(payload.nodes),
        ground=payload.ground,
        components=[
            Component(id=c.id, kind=c.type, start=c.start, end=c.end,
                      value=overrides.get(c.id, c.value))
            for c in payload.components
        ],
    )


def _result(sol: Solution) -> dict[str, Any]:
    out: dict[str, Any] = {"status": sol.status}
    if sol.message:
        out["message"] = sol.message
    if sol.status == "unique":
        out["node_voltages"] = sol.node_voltages
        out["components"] = {
            cid: {"current": sol.currents[cid], "power": sol.powers[cid]}
            for cid in sol.currents
        }
        out["kcl_residual"] = sol.kcl_residual
        out["power_residual"] = sol.power_residual
    return out


@app.post("/analyze")
def analyze(payload: CircuitIn):
    errors = _validate(payload)
    if errors:
        return JSONResponse(status_code=400, content={"detail": errors})

    if payload.sweep is None:
        return _result(solve(_to_circuit(payload)))

    sweep = payload.sweep
    points = []
    for j, v in enumerate(sweep.values):
        sol = solve(_to_circuit(payload, {sweep.source_id: v}))
        entry = _result(sol)
        entry["index"] = j
        entry["value"] = v
        points.append(entry)
    return {"sweep": {"source_id": sweep.source_id, "points": points}}
