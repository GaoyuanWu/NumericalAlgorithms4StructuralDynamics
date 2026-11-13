"""Independent reference solutions built with PyNite (PyNiteFEA, open-source 3D FEM).

The 2D frame is modelled as a 3D PyNite frame in the X-Y plane with the
out-of-plane DOFs (DZ, RX, RY) restrained at every node. PyNite assembles its
own stiffness and (consistent) mass matrices, so agreement checks our element
formulation, coordinate transformation, assembly and boundary conditions.
"""

import numpy as np
import scipy.linalg as sla
from Pynite import FEModel3D

from structdyn_jax.model import THREE_STORY_SUPPORTS


def build_pynite(frame, member_props, with_mass=True):
    """PyNite model of a meshed :class:`structdyn_jax.Frame2D`."""
    ep = {k: np.asarray(v)[frame.ele_parent] for k, v in member_props.items()}
    model = FEModel3D()
    for k, (x, y) in enumerate(frame.coords):
        model.add_node(f"N{k}", float(x), float(y), 0.0)
    for e, (i, j) in enumerate(frame.conn):
        E, A, I, m = (float(ep[q][e]) for q in ("E", "A", "I", "m"))
        model.add_material(f"mat{e}", E=E, G=0.4 * E, nu=0.25, rho=m / A)  # rho*A = mass per length
        model.add_section(f"sec{e}", A=A, Iy=I, Iz=I, J=I)  # Iy = Iz: in-plane axis irrelevant
        model.add_member(f"E{e}", f"N{i}", f"N{j}", f"mat{e}", f"sec{e}")
    supported = {int(frame.orig_to_new[n]) for n in THREE_STORY_SUPPORTS}
    for k in range(len(frame.coords)):
        full = k in supported  # all supports in this example are fixed
        model.def_support(f"N{k}", full, full, True, True, True, full)
    if with_mass:
        model.add_member_self_weight("FY", 1.0, case="Mass")
        model.add_load_combo("Combo 1", {"Mass": 1.0})
    return model


def pynite_reduced_matrices(model, frame):
    """PyNite's own global K and M, reduced to the in-plane free DOFs in our ordering."""
    if model.solution != "Modal":  # PyNite numbers nodes / activates members during analysis
        model.analyze_modal(num_modes=4, mass_combo_name="Combo 1", mass_direction="Y", gravity=1.0, check_stability=False)
    K = model.Ke("Combo 1", sparse=False, check_stability=False)
    M = model.M("Combo 1", mass_direction="Y", gravity=1.0, sparse=False)
    idx = []
    for g in frame.active:  # our global dof g = 3*node + d, d in (x, y, rz)
        node, d = divmod(int(g), 3)
        idx.append(6 * model.nodes[f"N{node}"].ID + (0, 1, 5)[d])
    idx = np.array(idx)
    return np.asarray(K)[np.ix_(idx, idx)], np.asarray(M)[np.ix_(idx, idx)]


def modal_exact_piecewise_linear(K, M, zeta, influence, ag, dt, n_steps):
    """Exact response to a piecewise-linear ground acceleration by modal superposition.

    Each modal SDOF is advanced with the Nigam-Jennings recurrence (exact for
    piecewise-linear forcing), using *all* modes. Classical modal damping
    ``zeta`` (same ratio in every mode). Returns u with shape (n_steps+1, n).
    """
    w2, Phi = sla.eigh(K, M)
    w = np.sqrt(w2)
    gamma = Phi.T @ M @ influence  # participation factors (Phi mass-normalised)
    p = -np.outer(np.asarray(ag[: n_steps + 1]), gamma)  # modal forcing per unit modal mass

    wd = w * np.sqrt(1 - zeta**2)
    e = np.exp(-zeta * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)
    k = w2
    # Nigam-Jennings / Chopra (Table 5.2.1) coefficients
    A = e * (zeta / np.sqrt(1 - zeta**2) * s + c)
    B = e * (s / wd)
    C = (1 / k) * (2 * zeta / (w * dt) + e * (((1 - 2 * zeta**2) / (wd * dt) - zeta / np.sqrt(1 - zeta**2)) * s - (1 + 2 * zeta / (w * dt)) * c))
    D = (1 / k) * (1 - 2 * zeta / (w * dt) + e * ((2 * zeta**2 - 1) / (wd * dt) * s + 2 * zeta / (w * dt) * c))
    A2 = -e * (w / np.sqrt(1 - zeta**2) * s)
    B2 = e * (c - zeta / np.sqrt(1 - zeta**2) * s)
    C2 = (1 / k) * (-1 / dt + e * ((w / np.sqrt(1 - zeta**2) + zeta / (dt * np.sqrt(1 - zeta**2))) * s + 1 / dt * c))
    D2 = (1 / (k * dt)) * (1 - e * (zeta / np.sqrt(1 - zeta**2) * s + c))

    q = np.zeros((n_steps + 1, len(w)))
    qd = np.zeros(len(w))
    for i in range(n_steps):
        q[i + 1] = A * q[i] + B * qd + C * p[i] + D * p[i + 1]
        qd = A2 * q[i] + B2 * qd + C2 * p[i] + D2 * p[i + 1]
    return q @ Phi.T
