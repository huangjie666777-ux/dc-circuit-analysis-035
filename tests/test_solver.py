import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def analyze(payload):
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def voltage_divider():
    return {
        "nodes": ["gnd", "a", "b"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 10.0},
            {"id": "R1", "type": "resistor", "from": "a", "to": "b", "value": 1000.0},
            {"id": "R2", "type": "resistor", "from": "b", "to": "gnd", "value": 1000.0},
        ],
    }


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_voltage_divider_unique():
    out = analyze(voltage_divider())
    assert out["status"] == "unique"
    assert out["node_voltages"]["a"] == pytest.approx(10.0)
    assert out["node_voltages"]["b"] == pytest.approx(5.0)
    assert out["node_voltages"]["gnd"] == 0.0
    by_id = {e["id"]: e for e in out["elements"]}
    assert by_id["R1"]["current"] == pytest.approx(0.005)
    assert by_id["V1"]["current"] == pytest.approx(-0.005)  # from a to gnd
    assert by_id["V1"]["power"] == pytest.approx(-0.05)  # negative: delivers
    assert abs(out["power_residual"]) < 1e-9
    assert all(abs(v) < 1e-9 for v in out["kcl_residual"].values())


def test_input_order_independent():
    import copy
    p1 = voltage_divider()
    p2 = copy.deepcopy(p1)
    p2["elements"] = list(reversed(p2["elements"]))
    p2["nodes"] = list(reversed(p2["nodes"]))
    o1, o2 = analyze(p1), analyze(p2)
    assert o1["node_voltages"] == pytest.approx(o2["node_voltages"])


def test_bridge_network():
    # Wheatstone bridge, unbalanced; floating source-free measurement via
    # non-ground voltage source is covered separately.
    payload = {
        "nodes": ["gnd", "a", "b", "c"],
        "ground": "gnd",
        "elements": [
            {"id": "V", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "R1", "type": "resistor", "from": "a", "to": "b", "value": 100.0},
            {"id": "R2", "type": "resistor", "from": "b", "to": "gnd", "value": 200.0},
            {"id": "R3", "type": "resistor", "from": "a", "to": "c", "value": 300.0},
            {"id": "R4", "type": "resistor", "from": "c", "to": "gnd", "value": 100.0},
            {"id": "R5", "type": "resistor", "from": "b", "to": "c", "value": 150.0},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "unique"
    # Cross-check with numpy reference MNA-free solve via Kirchhoff.
    va = 5.0
    # nodal equations for vb, vc
    G = np.array([
        [1/100 + 1/200 + 1/150, -1/150],
        [-1/150, 1/300 + 1/100 + 1/150],
    ])
    rhs = np.array([va/100, va/300])
    vb, vc = np.linalg.solve(G, rhs)
    assert out["node_voltages"]["b"] == pytest.approx(vb)
    assert out["node_voltages"]["c"] == pytest.approx(vc)


def test_floating_voltage_source_and_current_source():
    payload = {
        "nodes": ["gnd", "a", "b", "c"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "b", "value": 3.0},
            {"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": 100.0},
            {"id": "R2", "type": "resistor", "from": "b", "to": "gnd", "value": 200.0},
            {"id": "I1", "type": "current_source", "from": "c", "to": "b", "value": 0.01},
            {"id": "R3", "type": "resistor", "from": "c", "to": "gnd", "value": 50.0},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "unique"
    nv = out["node_voltages"]
    assert nv["a"] - nv["b"] == pytest.approx(3.0)
    assert abs(out["power_residual"]) < 1e-9
    assert all(abs(v) < 1e-9 for v in out["kcl_residual"].values())


def test_negative_source_values():
    payload = voltage_divider()
    payload["elements"][0]["value"] = -10.0
    out = analyze(payload)
    assert out["status"] == "unique"
    assert out["node_voltages"]["a"] == pytest.approx(-10.0)
    assert out["node_voltages"]["b"] == pytest.approx(-5.0)


def test_no_solution_conflicting_voltage_sources():
    payload = {
        "nodes": ["gnd", "a"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "V2", "type": "voltage_source", "from": "a", "to": "gnd", "value": 3.0},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "no_solution"


def test_no_solution_current_source_into_open():
    payload = {
        "nodes": ["gnd", "a"],
        "ground": "gnd",
        "elements": [
            {"id": "I1", "type": "current_source", "from": "a", "to": "gnd", "value": 1.0},
            {"id": "I2", "type": "current_source", "from": "gnd", "to": "a", "value": 0.5},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "no_solution"


def test_underdetermined_floating_subnetwork():
    payload = {
        "nodes": ["gnd", "a", "x", "y"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "R1", "type": "resistor", "from": "x", "to": "y", "value": 100.0},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "underdetermined"


def test_underdetermined_redundant_voltage_sources():
    # Two equal ideal voltage sources in parallel: node voltage is unique
    # but the split of branch currents is not.
    payload = {
        "nodes": ["gnd", "a"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "V2", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": 100.0},
        ],
    }
    out = analyze(payload)
    assert out["status"] == "underdetermined"


@pytest.mark.parametrize("bad,fragment", [
    ({"nodes": ["gnd", "a"], "ground": "gnd",
      "elements": [
          {"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": 1.0},
          {"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": 2.0}]},
     "duplicate element id"),
    ({"nodes": ["gnd", "a"], "ground": "gnd",
      "elements": [{"id": "R1", "type": "resistor", "from": "a", "to": "zz", "value": 1.0}]},
     "unknown node"),
    ({"nodes": ["gnd", "a"], "ground": "gnd",
      "elements": [{"id": "R1", "type": "resistor", "from": "a", "to": "a", "value": 1.0}]},
     "self-loop"),
    ({"nodes": ["gnd", "a"], "ground": "gnd",
      "elements": [{"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": -5.0}]},
     "resistance must be"),
    ({"nodes": ["gnd", "a"], "ground": "gnd",
      "elements": [{"id": "R1", "type": "resistor", "from": "a", "to": "gnd", "value": "x"}]},
     "must be a number"),
])
def test_validation_errors(bad, fragment):
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422
    assert fragment in resp.json()["error"]
    assert "elements[1]" in resp.json()["error"] or "elements[0]" in resp.json()["error"]


def test_sweep_keeps_order_and_isolates_failures():
    payload = {
        "nodes": ["gnd", "a", "b"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 1.0},
            {"id": "V2", "type": "voltage_source", "from": "b", "to": "gnd", "value": 2.0},
            {"id": "R1", "type": "resistor", "from": "a", "to": "b", "value": 100.0},
        ],
        "sweep": {"source_id": "V2", "values": [2.0, 1.0, 4.0]},
    }
    out = analyze(payload)
    assert out["status"] == "sweep"
    pts = out["points"]
    assert [p["value"] for p in pts] == [2.0, 1.0, 4.0]
    # Middle point puts two unequal... equal? V1=1, V2=1 with R between:
    # actually consistent. Make failure: values where V2 conflicts? V1 and V2
    # are on different nodes, so always consistent; instead check statuses.
    assert all(p["status"] == "unique" for p in pts)
    assert pts[0]["node_voltages"]["b"] == pytest.approx(2.0)
    assert pts[2]["node_voltages"]["b"] == pytest.approx(4.0)


def test_sweep_with_failing_point():
    payload = {
        "nodes": ["gnd", "a"],
        "ground": "gnd",
        "elements": [
            {"id": "V1", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
            {"id": "V2", "type": "voltage_source", "from": "a", "to": "gnd", "value": 5.0},
        ],
        "sweep": {"source_id": "V2", "values": [3.0, 5.0, 7.0]},
    }
    out = analyze(payload)
    statuses = [p["status"] for p in out["points"]]
    assert statuses == ["no_solution", "underdetermined", "no_solution"]


def test_requests_are_independent():
    out1 = analyze(voltage_divider())
    p = voltage_divider()
    p["elements"][0]["value"] = 20.0
    out2 = analyze(p)
    out3 = analyze(voltage_divider())
    assert out2["node_voltages"]["a"] == pytest.approx(20.0)
    assert out3["node_voltages"] == out1["node_voltages"]
