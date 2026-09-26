# DC circuit analysis

FastAPI service for DC operating-point analysis of linear resistive
circuits, intended for electronics lab teaching. The solver uses Modified
Nodal Analysis (MNA): unknowns are non-ground node voltages plus one branch
current per independent voltage source. It does **not** use series/parallel
reduction, small-resistor voltage-source hacks, or least-squares
approximations to hide contradictions.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Tests: `.venv/bin/python -m pytest tests -q`

Endpoints:

- `GET /healthz` — health check, returns `{"status": "ok"}`.
- `POST /analyze` — solve one circuit, or a sweep if `"sweep"` is present.
  Requests are independent; nothing is stored between calls.

## Request JSON format

```json
{
  "nodes": ["gnd", "a", "b"],
  "ground": "gnd",
  "elements": [
    {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 10.0},
    {"id": "R1", "type": "resistor", "from": "a", "to": "b", "value": 1000.0},
    {"id": "R2", "type": "resistor", "from": "b", "to": "gnd", "value": 1000.0}
  ]
}
```

- `nodes`: list of unique node names; `ground` must be one of them.
- `elements`: each element has a unique `id`, a `type`
  (`resistor`, `voltage_source`, `current_source`), endpoints
  `from`/`to`, and a numeric `value` (ohm / volt / ampere).
- Optional `sweep`: `{"source_id": "V1", "values": [1.0, 2.5, 5.0]}`
  re-solves the circuit for each value of one independent source, in the
  given order. Each point is solved independently; a failing point does not
  affect the others.

### Sign conventions

- Element voltage = potential at `from` minus potential at `to`.
- Positive element current flows from `from` to `to`.
- `power` is absorbed power (`voltage * current`); negative means the
  element delivers power.
- Source values may be negative. Resistances must be finite and positive.
- Ground node voltage is exactly 0; results do not depend on the ordering
  of `nodes` or `elements`.

### Validation

Requests are rejected with HTTP 422 and an `error` message locating the
field and element (e.g. `elements[2].value (element 'R3'): ...`) for:
duplicate element ids, unknown nodes, self-loops, non-finite or non-numeric
values, and non-positive resistances.

## Response

A single analysis returns one of three statuses:

- `unique` — unique operating point. Includes `node_voltages` (all nodes,
  relative to ground), per-element `voltage`/`current`/`power`, plus
  `kcl_residual` (net element current leaving each node) and
  `power_residual` (sum of absorbed powers) so conservation can be
  checked; both are ~0 for a consistent solution.
- `no_solution` — contradictory constraints (e.g. unequal ideal voltage
  sources in parallel, or current sources violating KCL). Detected by an
  exact consistency check on the MNA system, never masked by a
  least-squares "answer".
- `underdetermined` — equations are consistent but unknowns are not
  uniquely determined: floating subnetworks, or redundant ideal voltage
  sources whose individual branch currents cannot be fixed even when all
  node voltages are unique.

A sweep returns `{"status": "sweep", "source_id": ..., "points": [...]}`
where each point carries its `value` and a full per-point result as above,
in input order.

### Numerical tolerance

Rank and consistency decisions use a relative tolerance of `1e-9` scaled
by the largest matrix/RHS magnitude (`app/solver.py`, `TOL`).

## Example

```sh
curl -s http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{
  "nodes": ["gnd", "a", "b"],
  "ground": "gnd",
  "elements": [
    {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 10.0},
    {"id": "R1", "type": "resistor", "from": "a", "to": "b", "value": 1000.0},
    {"id": "R2", "type": "resistor", "from": "b", "to": "gnd", "value": 1000.0}
  ]
}'
```
