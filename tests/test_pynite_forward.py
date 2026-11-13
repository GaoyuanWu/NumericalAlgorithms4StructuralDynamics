"""Forward checks against PyNite (open-source FEM) and the legacy SAP2000 trace."""

import jax.numpy as jnp
import numpy as np
import pytest
from pynite_ref import build_pynite, modal_exact_piecewise_linear, pynite_reduced_matrices

import structdyn_jax as sj
from structdyn_jax.model import ROOF_NODE, element_props, natural_frequencies, three_story_frame


@pytest.fixture(scope="module", params=[2.5, 1.0], ids=["h=2.5m", "h=1m"])
def case(request):
    frame, mp = three_story_frame(request.param)
    Ka, Ma = frame.matrices(**element_props(frame, mp))
    model = build_pynite(frame, mp)
    return dict(frame=frame, mp=mp, Ka=Ka, Ma=Ma, model=model)


def test_assembled_matrices_match_pynite(case):
    Kp, Mp = pynite_reduced_matrices(case["model"], case["frame"])
    Ka, Ma = np.asarray(case["Ka"]), np.asarray(case["Ma"])
    np.testing.assert_allclose(Ka, Kp, rtol=0, atol=1e-10 * np.abs(Kp).max())
    np.testing.assert_allclose(Ma, Mp, rtol=0, atol=1e-10 * np.abs(Mp).max())


def test_static_lateral_load_matches_pynite(case):
    """1000 kN lateral load at the roof: PyNite linear static vs K^-1 f."""
    frame, mp = case["frame"], case["mp"]
    model = build_pynite(frame, mp, with_mass=False)
    roof = f"N{frame.orig_to_new[ROOF_NODE]}"
    model.add_node_load(roof, "FX", 1000.0, case="Lat")
    model.add_load_combo("Lat", {"Lat": 1.0})
    model.analyze_linear(check_statics=False)
    f = jnp.zeros(frame.n_active).at[frame.dof(ROOF_NODE, 0)].set(1000.0)
    u = np.asarray(jnp.linalg.solve(case["Ka"], f))
    for k in range(len(frame.coords)):
        node = model.nodes[f"N{k}"]
        for d, attr in enumerate(("DX", "DY", "RZ")):
            g = 3 * k + d
            ours = u[np.searchsorted(frame.active, g)] if g in set(frame.active) else 0.0
            assert ours == pytest.approx(getattr(node, attr)["Lat"], rel=1e-8, abs=1e-12)


def test_natural_frequencies_match_pynite(case):
    n_modes = 8
    model = case["model"]
    model.analyze_modal(num_modes=n_modes, mass_combo_name="Combo 1", mass_direction="Y", gravity=1.0, check_stability=False)
    f_pynite = np.sort(np.asarray(model.frequencies))[:n_modes]
    f_ours = np.asarray(natural_frequencies(case["Ka"], case["Ma"]))[:n_modes]
    np.testing.assert_allclose(f_ours, f_pynite, rtol=1e-8)


@pytest.mark.parametrize("method", ["newmark", "bathe", "wilson"])
def test_earthquake_time_history_vs_pynite_modal(case, method):
    """Kern County record, 5 % modal damping, horizontal influence vector.

    Reference: exact (Nigam-Jennings) modal superposition on PyNite's own K, M.
    Our implicit schemes are 2nd order; with h = dt/4 the roof history must
    agree to < 0.5 % of its peak (Wilson-theta is more dissipative: < 2 %).
    """
    frame = case["frame"]
    Kp, Mp = pynite_reduced_matrices(case["model"], frame)
    rec, dt = sj.load_kern_county()
    T = 6.0
    n_rec = int(round(T / dt))
    zeta = 0.05
    r = np.asarray(frame.influence(0))
    u_ref = modal_exact_piecewise_linear(Kp, Mp, zeta, r, rec, dt, n_rec)

    # same classical damping in our solver: C = M Phi diag(2 zeta w) Phi^T M
    w2, Phi = sj.model.generalized_eig(case["Ka"], case["Ma"])
    C = case["Ma"] @ Phi @ jnp.diag(2 * zeta * jnp.sqrt(w2)) @ Phi.T @ case["Ma"]
    load = sj.ground_motion_load(case["Ma"], r, rec, dt)
    h = dt / 4
    z = jnp.zeros(frame.n_active)
    u = np.asarray(sj.simulate(method, case["Ka"], case["Ma"], C, load, z, z, h, 4 * n_rec)[0])[::4]

    k = frame.dof(ROOF_NODE, 0)
    peak = np.abs(u_ref[:, k]).max()
    tol = 0.02 if method == "wilson" else 0.005
    assert np.abs(u[:, k] - u_ref[:, k]).max() < tol * peak


def test_sap2000_reference():
    """Legacy benchmark: SAP2000 modal superposition at node 11 (undamped, 20 s).

    The original project compared against this trace. We reproduce the legacy
    set-up (h = 0.5 m mesh, all-ones influence vector, first sample dropped,
    Newmark with h = dt). Asserted: peak within 2 %, and the first 5 s within
    4 % of the peak.

    Not asserted: the full 20 s history. Our fundamental period is ~0.5 %
    shorter than SAP2000's, which accumulates to a visible phase drift over
    ~80 cycles (correlation 0.79). Scaling K by 0.99 raises the correlation to
    0.995, consistent with SAP2000's default shear deformation (Euler-Bernoulli
    here) - a plausible explanation we cannot verify without the SAP model, so
    we do not calibrate to it. See docs/VERIFICATION.md.
    """
    t_sap, d_sap = sj.loads.load_sap2000_reference()
    frame, mp = three_story_frame(0.5)
    Ka, Ma = frame.matrices(**element_props(frame, mp))
    rec, dt = sj.load_kern_county(legacy_drop_first=True)
    n = frame.n_active
    load = sj.ground_motion_load(Ma, jnp.ones(n), rec, dt)
    z = jnp.zeros(n)
    u = np.asarray(sj.newmark(Ka, Ma, jnp.zeros_like(Ka), load, z, z, dt, 4000, a0=z)[0])
    roof = u[:, frame.dof(ROOF_NODE, 0)]
    peak_sap = np.abs(d_sap).max()
    assert np.abs(roof).max() == pytest.approx(peak_sap, rel=0.02)
    assert np.abs(roof - d_sap)[:1001].max() < 0.04 * peak_sap
