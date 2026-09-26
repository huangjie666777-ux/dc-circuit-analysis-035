"""DC operating point analysis via Modified Nodal Analysis (MNA).

Unknowns: non-ground node voltages plus one current per independent voltage
source. No series/parallel reduction and no small-resistor tricks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Relative tolerance used for rank / consistency decisions.
TOL = 1e-9


class CircuitError(Exception):
    """Validation error; `detail` locates the offending field/element."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class Element:
    id: str
    type: str  # "resistor" | "voltage_source" | "current_source"
    n_from: str
    n_to: str
    value: float


@dataclass
class Circuit:
    nodes: list[str]
    ground: str
    elements: list[Element]


def parse_circuit(data: dict) -> Circuit:
    if not isinstance(data, dict):
        raise CircuitError("request body must be a JSON object")
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(
        isinstance(n, str) and n for n in nodes
    ):
        raise CircuitError("field 'nodes' must be a non-empty list of node names")
    if len(set(nodes)) != len(nodes):
        raise CircuitError("field 'nodes' contains duplicate node names")
    ground = data.get("ground")
    if not isinstance(ground, str) or ground not in nodes:
        raise CircuitError("field 'ground' must be one of the declared nodes")
    raw_elements = data.get("elements")
    if not isinstance(raw_elements, list):
        raise CircuitError("field 'elements' must be a list")

    node_set = set(nodes)
    seen_ids: set[str] = set()
    elements: list[Element] = []
    for i, raw in enumerate(raw_elements):
        where = "elements[%d]" % i
        if not isinstance(raw, dict):
            raise CircuitError(where + ": element must be an object")
        eid = raw.get("id")
        if not isinstance(eid, str) or not eid:
            raise CircuitError(where + ".id: element id must be a non-empty string")
        if eid in seen_ids:
            raise CircuitError(where + ".id: duplicate element id " + repr(eid))
        seen_ids.add(eid)
        etype = raw.get("type")
        if etype not in ("resistor", "voltage_source", "current_source"):
            raise CircuitError(
                where + ".type (element " + repr(eid) + "): must be "
                "'resistor', 'voltage_source' or 'current_source'"
            )
        n_from, n_to = raw.get("from"), raw.get("to")
        for field_name, node in (("from", n_from), ("to", n_to)):
            if not isinstance(node, str) or node not in node_set:
                raise CircuitError(
                    where + "." + field_name + " (element " + repr(eid)
                    + "): unknown node " + repr(node)
                )
        if n_from == n_to:
            raise CircuitError(
                where + " (element " + repr(eid) + "): self-loop, "
                "'from' and 'to' are identical"
            )
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CircuitError(
                where + ".value (element " + repr(eid) + "): value must be a number"
            )
        value = float(value)
        if not np.isfinite(value):
            raise CircuitError(
                where + ".value (element " + repr(eid) + "): value must be finite"
            )
        if etype == "resistor" and value <= 0:
            raise CircuitError(
                where + ".value (element " + repr(eid)
                + "): resistance must be a finite positive number"
            )
        elements.append(Element(eid, etype, n_from, n_to, value))
    return Circuit(nodes=nodes, ground=ground, elements=elements)


def _build_system(circuit: Circuit):
    """Assemble the MNA system A x = b."""
    other_nodes = [n for n in circuit.nodes if n != circuit.ground]
    node_idx = {n: i for i, n in enumerate(other_nodes)}
    vsources = [e for e in circuit.elements if e.type == "voltage_source"]
    vs_idx = {e.id: len(other_nodes) + k for k, e in enumerate(vsources)}
    size = len(other_nodes) + len(vsources)
    A = np.zeros((size, size))
    b = np.zeros(size)

    def stamp_conductance(n1, n2, g):
        i, j = node_idx.get(n1), node_idx.get(n2)
        if i is not None:
            A[i, i] += g
        if j is not None:
            A[j, j] += g
        if i is not None and j is not None:
            A[i, j] -= g
            A[j, i] -= g

    for e in circuit.elements:
        if e.type == "resistor":
            stamp_conductance(e.n_from, e.n_to, 1.0 / e.value)
        elif e.type == "current_source":
            # Positive current flows from n_from to n_to.
            i, j = node_idx.get(e.n_from), node_idx.get(e.n_to)
            if i is not None:
                b[i] -= e.value
            if j is not None:
                b[j] += e.value
        else:  # voltage source: v(from) - v(to) = value
            k = vs_idx[e.id]
            i, j = node_idx.get(e.n_from), node_idx.get(e.n_to)
            if i is not None:
                A[i, k] += 1.0
                A[k, i] += 1.0
            if j is not None:
                A[j, k] -= 1.0
                A[k, j] -= 1.0
            b[k] = e.value
    return A, b, node_idx, vs_idx


def _scaled_tol(A, b):
    scale = 1.0
    if A.size:
        scale = max(scale, float(np.max(np.abs(A))))
    if b.size:
        scale = max(scale, float(np.max(np.abs(b))))
    return TOL * scale


def solve_circuit(circuit: Circuit) -> dict:
    """Solve one operating point and classify the result."""
    A, b, node_idx, vs_idx = _build_system(circuit)
    n = A.shape[0]
    tol = _scaled_tol(A, b)

    if n == 0:
        x = np.zeros(0)
        rank = 0
        inconsistent = False
    else:
        rank = int(np.linalg.matrix_rank(A, tol=tol))
        if rank == n:
            x = np.linalg.solve(A, b)
            inconsistent = False
        else:
            # Singular system: distinguish contradiction (no solution) from
            # non-uniqueness by checking consistency of A x = b. The least-
            # squares solution is used ONLY for this residual test, never
            # reported as an answer.
            x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
            res = A @ x_lstsq - b
            inconsistent = bool(res.size and np.max(np.abs(res)) > tol)
            x = None

    if n > 0 and inconsistent:
        return {
            "status": "no_solution",
            "detail": "conflicting constraints (e.g. unequal ideal voltage "
            "sources in parallel or a current source violating KCL); no "
            "operating point satisfies all equations",
            "tolerance": TOL,
        }
    if rank < n:
        return {
            "status": "underdetermined",
            "detail": "system is consistent but unknowns are not uniquely "
            "determined (e.g. floating subnetwork, or redundant ideal "
            "voltage sources leaving branch currents free)",
            "tolerance": TOL,
        }

    node_voltages = {circuit.ground: 0.0}
    for node, i in node_idx.items():
        node_voltages[node] = float(x[i])

    elements_out = []
    for e in circuit.elements:
        v_across = node_voltages[e.n_from] - node_voltages[e.n_to]
        if e.type == "resistor":
            current = v_across / e.value
        elif e.type == "current_source":
            current = e.value
        else:
            current = float(x[vs_idx[e.id]])
        elements_out.append({
            "id": e.id,
            "type": e.type,
            "voltage": float(v_across),
            "current": float(current),
            "power": float(v_across * current),
        })

    # KCL residual per node: net element current leaving the node (must be 0).
    kcl = {node: 0.0 for node in circuit.nodes}
    for e, out in zip(circuit.elements, elements_out):
        kcl[e.n_from] += out["current"]
        kcl[e.n_to] -= out["current"]
    power_residual = float(sum(out["power"] for out in elements_out))

    return {
        "status": "unique",
        "node_voltages": node_voltages,
        "elements": elements_out,
        "kcl_residual": {node: float(v) for node, v in kcl.items()},
        "power_residual": power_residual,
        "tolerance": TOL,
    }


def solve_request(data: dict) -> dict:
    circuit = parse_circuit(data)
    sweep = data.get("sweep")
    if sweep is None:
        return solve_circuit(circuit)

    if not isinstance(sweep, dict):
        raise CircuitError("field 'sweep' must be an object")
    source_id = sweep.get("source_id")
    targets = [
        e for e in circuit.elements
        if e.id == source_id and e.type in ("voltage_source", "current_source")
    ]
    if not targets:
        raise CircuitError(
            "field 'sweep.source_id': no independent source with id "
            + repr(source_id)
        )
    values = sweep.get("values")
    if not isinstance(values, list) or not values:
        raise CircuitError("field 'sweep.values' must be a non-empty list of numbers")
    for k, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise CircuitError(
                "field 'sweep.values[%d]': must be a finite number" % k
            )

    target = targets[0]
    original = target.value
    points = []
    try:
        for v in values:
            target.value = float(v)
            point = solve_circuit(circuit)
            point["value"] = float(v)
            points.append(point)
    finally:
        target.value = original
    return {"status": "sweep", "source_id": source_id, "points": points}
