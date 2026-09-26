import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(payload):
    return client.post("/analyze", json=payload)


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_series_parallel():
    # 10V source, R1=5 series with (R2=10 || R3=10) => 5+5=10 ohm, I=1A
    r = post({
        "nodes": ["g", "a", "b"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 10},
            {"id": "R1", "type": "resistor", "start": "a", "end": "b", "value": 5},
            {"id": "R2", "type": "resistor", "start": "b", "end": "g", "value": 10},
            {"id": "R3", "type": "resistor", "start": "b", "end": "g", "value": 10},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unique"
    assert body["node_voltages"]["a"] == pytest.approx(10)
    assert body["node_voltages"]["b"] == pytest.approx(5)
    assert body["components"]["V1"]["current"] == pytest.approx(-1)
    assert body["components"]["V1"]["power"] == pytest.approx(-10)  # delivers
    assert body["components"]["R1"]["power"] == pytest.approx(5)
    assert body["kcl_residual"] < 1e-9
    assert body["power_residual"] < 1e-9


def test_bridge_network():
    # Wheatstone bridge, unbalanced
    r = post({
        "nodes": ["g", "a", "b", "c"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "R1", "type": "resistor", "start": "a", "end": "b", "value": 1},
            {"id": "R2", "type": "resistor", "start": "a", "end": "c", "value": 2},
            {"id": "R3", "type": "resistor", "start": "b", "end": "g", "value": 2},
            {"id": "R4", "type": "resistor", "start": "c", "end": "g", "value": 1},
            {"id": "R5", "type": "resistor", "start": "b", "end": "c", "value": 1},
        ],
    })
    body = r.json()
    assert body["status"] == "unique"
    # Nodal check: Vb = 5*(1/1+1/1... ) verify via KCL residuals
    assert body["kcl_residual"] < 1e-9
    assert body["power_residual"] < 1e-9
    va, vb, vc = (body["node_voltages"][n] for n in "abc")
    assert va == pytest.approx(5)
    # hand-computed: Vb=20/7, Vc=15/7
    assert vb == pytest.approx(20 / 7)
    assert vc == pytest.approx(15 / 7)


def test_floating_voltage_source_and_current_source():
    # Voltage source not connected to ground + current source, negative values
    r = post({
        "nodes": ["g", "a", "b", "c"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "b", "value": -4},
            {"id": "I1", "type": "current_source", "start": "g", "end": "a", "value": 0.5},
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 8},
            {"id": "R2", "type": "resistor", "start": "b", "end": "g", "value": 8},
            {"id": "R3", "type": "resistor", "start": "b", "end": "c", "value": 1},
            {"id": "R4", "type": "resistor", "start": "c", "end": "g", "value": 1},
        ],
    })
    body = r.json()
    assert body["status"] == "unique"
    # Supernode {a,b}: Va/8 + Vb/8 + (Vb-Vc) = 0.5, Vb=Va+4, Vc=Vb/2
    assert body["node_voltages"]["a"] == pytest.approx(-8 / 3)
    assert body["node_voltages"]["b"] == pytest.approx(4 / 3)
    assert body["node_voltages"]["c"] == pytest.approx(2 / 3)
    assert body["kcl_residual"] < 1e-9
    assert body["power_residual"] < 1e-9


def test_contradictory_voltage_sources_no_solution():
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "V2", "type": "voltage_source", "start": "a", "end": "g", "value": 3},
        ],
    })
    assert r.json()["status"] == "no_solution"


def test_redundant_voltage_sources_underdetermined():
    # Same value: node voltages unique but branch currents are not.
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "V2", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 10},
        ],
    })
    assert r.json()["status"] == "underdetermined"


def test_floating_subnet_underdetermined():
    r = post({
        "nodes": ["g", "a", "x", "y"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "R1", "type": "resistor", "start": "x", "end": "y", "value": 10},
        ],
    })
    assert r.json()["status"] == "underdetermined"


def test_current_source_loop_contradiction():
    # Two current sources in series with different values violate KCL.
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "I1", "type": "current_source", "start": "g", "end": "a", "value": 1},
            {"id": "I2", "type": "current_source", "start": "a", "end": "g", "value": 2},
        ],
    })
    assert r.json()["status"] == "no_solution"


@pytest.mark.parametrize("comp,field", [
    ({"id": "R1", "type": "resistor", "start": "a", "end": "a", "value": 1}, "end"),  # self-loop
    ({"id": "R1", "type": "resistor", "start": "a", "end": "zz", "value": 1}, "end"),  # unknown node
    ({"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": -1}, "value"),  # bad R
    ({"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 0}, "value"),
])
def test_invalid_components_rejected(comp, field):
    r = post({"nodes": ["g", "a"], "ground": "g", "components": [comp]})
    assert r.status_code == 400
    assert any(field in e["field"] for e in r.json()["detail"])


@pytest.mark.parametrize("raw", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_values_rejected(raw):
    payload = ('{"nodes": ["g", "a"], "ground": "g", "components": ['
               '{"id": "R1", "type": "resistor", "start": "a", "end": "g", '
               '"value": ' + raw + '}]}')
    r = client.post("/analyze", content=payload,
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert any("value" in e["field"] for e in r.json()["detail"])


def test_duplicate_id_rejected():
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "X", "type": "resistor", "start": "a", "end": "g", "value": 1},
            {"id": "X", "type": "resistor", "start": "a", "end": "g", "value": 2},
        ],
    })
    assert r.status_code == 400
    assert r.json()["detail"][0]["component"] == "X"


def test_unknown_ground_rejected():
    r = post({"nodes": ["g", "a"], "ground": "nope", "components": []})
    assert r.status_code == 400


def test_sweep():
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 0},
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 2},
        ],
        "sweep": {"source_id": "V1", "values": [1, 2, 4]},
    })
    body = r.json()
    points = body["sweep"]["points"]
    assert [p["value"] for p in points] == [1, 2, 4]
    for p, v in zip(points, [1, 2, 4]):
        assert p["status"] == "unique"
        assert p["node_voltages"]["a"] == pytest.approx(v)
        assert p["components"]["R1"]["current"] == pytest.approx(v / 2)
        assert p["kcl_residual"] < 1e-9


def test_sweep_failure_does_not_interrupt():
    # Sweep current source into a node with no DC path: always no_solution,
    # mixed with valid points via resistor circuit; check order preserved.
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "I1", "type": "current_source", "start": "g", "end": "a", "value": 0},
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "V2", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
        ],
        "sweep": {"source_id": "I1", "values": [0, 1, 0]},
    })
    points = r.json()["sweep"]["points"]
    # I1=1 conflicts with nothing? Actually I1 into node a with two V sources:
    # currents through V1/V2 not unique -> underdetermined for all points.
    assert [p["status"] for p in points] == ["underdetermined"] * 3
    assert [p["index"] for p in points] == [0, 1, 2]


def test_sweep_mixed_statuses():
    # V2 conflicts with V1 unless swept value equals 5.
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 5},
            {"id": "V2", "type": "voltage_source", "start": "a", "end": "g", "value": 0},
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 10},
        ],
        "sweep": {"source_id": "V2", "values": [3, 5, 7]},
    })
    points = r.json()["sweep"]["points"]
    assert [p["status"] for p in points] == ["no_solution", "underdetermined", "no_solution"]


def test_sweep_target_must_be_source():
    r = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 2},
        ],
        "sweep": {"source_id": "R1", "values": [1]},
    })
    assert r.status_code == 400


def test_requests_are_independent():
    payload = {
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "V1", "type": "voltage_source", "start": "a", "end": "g", "value": 3},
        ],
    }
    first = post(payload).json()
    second = post({
        "nodes": ["g", "a"],
        "ground": "g",
        "components": [
            {"id": "R1", "type": "resistor", "start": "a", "end": "g", "value": 4},
        ],
    }).json()
    assert first["node_voltages"]["a"] == pytest.approx(3)
    # No state leaks: circuit without source has Va=0
    assert second["status"] == "unique"
    assert second["node_voltages"]["a"] == pytest.approx(0)
