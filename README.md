# DC circuit operating point analysis

FastAPI service that computes the DC operating point of linear resistive
circuits using Modified Nodal Analysis (MNA). Supports resistors, independent
ideal voltage/current sources, floating (non-ground-referenced) sources,
bridge networks, and parameter sweeps.

## Run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Health check: `GET /healthz`
- Analysis: `POST /analyze`
- Tests: `.venv/bin/python -m pytest tests/ -q`

## JSON format

```json
{
  "nodes": ["g", "a", "b"],
  "ground": "g",
  "components": [
    {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 10},
    {"id": "R1", "type": "resistor", "start": "a", "end": "b", "value": 5},
    {"id": "I1", "type": "current_source", "start": "g", "end": "b", "value": 0.1}
  ],
  "sweep": {"source_id": "V1", "values": [1, 5, 10]}
}
```

- `nodes`: node names; `ground` must be one of them (its voltage is 0).
- `components`: each has a unique `id`, a `type` (`resistor`,
  `voltage_source`, `current_source`), `start`/`end` nodes, and `value`
  in ohms / volts / amperes.
- `sweep` (optional): vary one independent source over an ordered list of
  values; each point is solved independently and returned in input order.

### Conventions

- Element voltage = V(start) - V(end).
- Positive current flows from `start` to `end` through the element.
- Source values may be negative. Resistances must be finite and positive.
- Reported power is **absorbed** power (P = V * I); negative means the
  element delivers power.

### Validation

Duplicate ids, unknown nodes, self-loops, non-finite values and non-positive
resistances are rejected with HTTP 400; each error names the offending field
and component id.

## Response

```json
{
  "status": "unique",
  "node_voltages": {"g": 0.0, "a": 10.0, "b": 5.0},
  "components": {"V1": {"current": -1.0, "power": -10.0}, "...": {}},
  "kcl_residual": 0.0,
  "power_residual": 0.0
}
```

`status` is one of:

- `unique` — exactly one solution; node voltages, per-component currents and
  absorbed powers are returned, plus `kcl_residual` (max absolute KCL
  imbalance over all nodes) and `power_residual` (|sum of absorbed powers|)
  so conservation can be verified.
- `no_solution` — contradictory constraints (e.g. a KVL-violating loop of
  ideal voltage sources, or series current sources with different values).
- `underdetermined` — unknowns cannot be uniquely determined: floating
  subnetworks, or redundant ideal sources whose branch currents are not
  fixed even when node voltages are. No partial solution is fabricated.

For sweeps the response is `{"sweep": {"source_id": ..., "points": [...]}}`
where each point carries `index`, `value`, `status` and (when unique) the
full result; a failing point never interrupts or contaminates later points.

## Numerics

The MNA system A x = z is analysed with an SVD-based rank test: singular
values below `1e-10 * max_singular_value * max(shape)` are treated as zero.
`rank(A)` vs `rank([A|z])` distinguishes inconsistent systems from
underdetermined ones; unique systems are solved exactly (LU), never with
least-squares smoothing of contradictions. Ideal voltage sources are true
MNA branch constraints, not small resistors.
