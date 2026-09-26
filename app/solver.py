"""DC operating point analysis via Modified Nodal Analysis (MNA).

Conventions
-----------
- Voltage of an element = V(start) - V(end).
- Positive current flows from start to end through the element.
- Absorbed power P = V * I; negative power means the element delivers power.
- Ground node voltage is fixed to 0.

Solution status
---------------
- "unique":          the MNA system has exactly one solution.
- "no_solution":     the constraints are contradictory (e.g. a loop of ideal
                     voltage sources whose values do not satisfy KVL).
- "underdetermined": unknowns cannot be uniquely determined (e.g. floating
                     subnetworks, or redundant ideal voltage sources whose
                     branch currents are not fixed even when node voltages are).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Numerical tolerance: singular values of the MNA matrix below
# RTOL * max_singular_value * max(shape) are treated as zero for rank.
RTOL = 1e-10

STATUS_UNIQUE = "unique"
STATUS_NO_SOLUTION = "no_solution"
STATUS_UNDERDETERMINED = "underdetermined"


@dataclass
class Component:
    id: str
    kind: str  # "resistor" | "voltage_source" | "current_source"
    start: str
    end: str
    value: float


@dataclass
class Circuit:
    nodes: list[str]
    ground: str
    components: list[Component]


@dataclass
class Solution:
    status: str
    node_voltages: dict[str, float] = field(default_factory=dict)
    currents: dict[str, float] = field(default_factory=dict)
    powers: dict[str, float] = field(default_factory=dict)
    kcl_residual: float | None = None
    power_residual: float | None = None
    message: str = ""


def _matrix_rank(matrix: np.ndarray) -> int:
    """Matrix rank via SVD with a relative tolerance."""
    if matrix.size == 0:
        return 0
    svals = np.linalg.svd(matrix, compute_uv=False)
    if svals.size == 0 or svals[0] == 0.0:
        return 0
    tol = RTOL * svals[0] * max(matrix.shape)
    return int(np.count_nonzero(svals > tol))


def solve(circuit: Circuit) -> Solution:
    """Solve the DC operating point of ``circuit``."""
    nodes = [n for n in circuit.nodes if n != circuit.ground]
    node_index = {n: i for i, n in enumerate(nodes)}
    vsources = [c for c in circuit.components if c.kind == "voltage_source"]
    vs_index = {c.id: k for k, c in enumerate(vsources)}

    n_nodes = len(nodes)
    n_vs = len(vsources)
    n_unknowns = n_nodes + n_vs

    if n_unknowns == 0:
        return Solution(status=STATUS_UNIQUE,
                        node_voltages={circuit.ground: 0.0},
                        kcl_residual=0.0, power_residual=0.0)

    A = np.zeros((n_unknowns, n_unknowns))
    z = np.zeros(n_unknowns)

    for comp in circuit.components:
        a = node_index.get(comp.start)
        b = node_index.get(comp.end)
        if comp.kind == "resistor":
            g = 1.0 / comp.value
            if a is not None:
                A[a, a] += g
            if b is not None:
                A[b, b] += g
            if a is not None and b is not None:
                A[a, b] -= g
                A[b, a] -= g
        elif comp.kind == "current_source":
            # Positive current flows start -> end (leaves start, enters end).
            if a is not None:
                z[a] -= comp.value
            if b is not None:
                z[b] += comp.value
        else:  # voltage_source
            k = n_nodes + vs_index[comp.id]
            if a is not None:
                A[a, k] += 1.0
                A[k, a] += 1.0
            if b is not None:
                A[b, k] -= 1.0
                A[k, b] -= 1.0
            z[k] = comp.value

    rank_a = _matrix_rank(A)
    rank_aug = _matrix_rank(np.column_stack([A, z]))

    if rank_aug > rank_a:
        return Solution(
            status=STATUS_NO_SOLUTION,
            message="Constraints are contradictory (e.g. KVL violated by ideal "
                    "voltage sources or KCL by current sources).",
        )
    if rank_a < n_unknowns:
        return Solution(
            status=STATUS_UNDERDETERMINED,
            message="Unknowns cannot be uniquely determined (floating "
                    "subnetwork or redundant ideal sources).",
        )

    x = np.linalg.solve(A, z)

    voltages: dict[str, float] = {circuit.ground: 0.0}
    for name, idx in node_index.items():
        voltages[name] = float(x[idx])

    currents: dict[str, float] = {}
    powers: dict[str, float] = {}
    for comp in circuit.components:
        va = voltages[comp.start]
        vb = voltages[comp.end]
        if comp.kind == "resistor":
            current = (va - vb) / comp.value
        elif comp.kind == "current_source":
            current = comp.value
        else:
            current = float(x[n_nodes + vs_index[comp.id]])
        currents[comp.id] = current
        powers[comp.id] = (va - vb) * current

    # KCL residual: max absolute imbalance of currents leaving each node.
    node_pos = {n: i for i, n in enumerate(circuit.nodes)}
    kcl = np.zeros(len(circuit.nodes))
    for comp in circuit.components:
        kcl[node_pos[comp.start]] += currents[comp.id]
        kcl[node_pos[comp.end]] -= currents[comp.id]
    kcl_residual = float(np.max(np.abs(kcl))) if kcl.size else 0.0

    # Total absorbed power must sum to zero (conservation).
    power_residual = float(abs(sum(powers.values())))

    return Solution(
        status=STATUS_UNIQUE,
        node_voltages=voltages,
        currents=currents,
        powers=powers,
        kcl_residual=kcl_residual,
        power_residual=power_residual,
    )
