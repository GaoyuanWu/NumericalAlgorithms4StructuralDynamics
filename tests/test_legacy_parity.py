"""Forward parity: the JAX port must reproduce the original NumPy solver.

The legacy modules in Structure/ and TimeIntegration/ are left untouched and
used as the reference. Legacy quirks are reproduced explicitly where needed
(all-ones influence vector, Newmark a0 = 0, first record sample dropped).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import legacy_three_story, np_record_interp

import structdyn_jax as sj
from structdyn_jax.model import ROOF_NODE, element_props, three_story_frame

RTOL = 1e-9  # relative to the peak response


@pytest.mark.parametrize("h", [5, 2.5, 1, 0.5])
def test_mesh_and_matrices(legacy, h):
    ref = legacy_three_story(legacy, h)
    frame, mp = three_story_frame(h)
    np.testing.assert_array_equal(frame.coords, ref["coords"])
    np.testing.assert_array_equal(frame.conn, ref["conn"])
    np.testing.assert_array_equal(frame.orig_to_new, ref["itn"])
    np.testing.assert_array_equal(frame.ele_parent, ref["ite"])
    Ka, Ma = frame.matrices(**element_props(frame, mp))
    np.testing.assert_allclose(Ka, ref["Ka"], rtol=0, atol=1e-13 * np.abs(ref["Ka"]).max())
    np.testing.assert_allclose(Ma, ref["Ma"], rtol=0, atol=1e-13 * np.abs(ref["Ma"]).max())


@pytest.fixture(scope="module")
def system(legacy):
    h = 2.5
    ref = legacy_three_story(legacy, h)
    frame, mp = three_story_frame(h)
    Ka, Ma = frame.matrices(**element_props(frame, mp))
    n = frame.n_active
    rec, dt = sj.load_kern_county(legacy_drop_first=True)
    # Rayleigh damping so the C-terms of every scheme are exercised
    C = 0.3 * Ma + 1e-3 * Ka
    return dict(ref=ref, frame=frame, Ka=Ka, Ma=Ma, C=C, n=n, rec=rec, dt=dt)


def _legacy_run(legacy, name, Ka, Ma, C, f, n, N, h, dt_ft):
    z = np.zeros(n)
    Ka, Ma, C = map(np.asarray, (Ka, Ma, C))
    if name == "newmark":
        return legacy["Newmark"].Newmark_beta(0.5, 0.25, Ka, Ma, C, f, z, z, 0, N, h, dt_ft)[0]
    if name == "wilson":
        return legacy["Wilson"].Wilson_theta(1.4, Ka, Ma, C, f, z, z, 0, N, h, dt_ft)[0]
    if name == "bathe":
        return legacy["Bathe"].Bathe(Ka, Ma, C, f, z, z, 0, N, h, dt_ft)[0]
    return legacy["SSPRK3"].SSP_RK3(Ka, Ma, C, f, z, z, 0, N, h, dt_ft)[:n]


def _jax_run(name, Ka, Ma, C, load, n, N, h):
    z = jnp.zeros(n)
    kw = {"a0": z} if name == "newmark" else {}  # legacy Newmark starts from a0 = 0
    return np.asarray(sj.simulate(name, Ka, Ma, C, load, z, z, h, N, **kw)[0]).T


def _gm_case(s, legacy_interp=False):
    """Forcing -M r a_g(t) with the legacy all-ones r, as a NumPy function (legacy) and JAX load."""
    n = s["n"]
    Mr = np.asarray(s["Ma"]) @ np.ones(n)
    ag = np_record_interp(s["rec"], s["dt"], legacy=legacy_interp)
    f_np = lambda t: -Mr * ag(t)  # noqa: E731
    f_jax = sj.ground_motion_load(s["Ma"], jnp.ones(n), s["rec"], s["dt"])
    return f_np, f_jax


@pytest.mark.parametrize("h", [0.005, 0.0025], ids=["h=dt", "h=dt/2"])
@pytest.mark.parametrize("name", ["newmark", "wilson", "bathe"])
def test_implicit_ground_motion(legacy, system, name, h):
    """Unmodified legacy integrators vs JAX, same earthquake forcing, 5 s of record."""
    s = system
    f_np, f_jax = _gm_case(s)
    N = int(round(5.0 / h))
    u_ref = _legacy_run(legacy, name, s["Ka"], s["Ma"], s["C"], f_np, s["n"], N, h, 0)
    u = _jax_run(name, s["Ka"], s["Ma"], s["C"], f_jax, s["n"], N, h)
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=RTOL * np.abs(u_ref).max())


@pytest.mark.parametrize("name", ["newmark", "wilson", "bathe"])
def test_legacy_record_branch_only_differs_by_interpolation(legacy, system, name):
    """Documents a legacy bug instead of hiding it.

    (a) The legacy *record* branch (dt_ft > 0) is reproduced to round-off by the
        legacy *function* branch driven by a replica of its interpolation, so the
        integrator algebra in both branches is identical.
    (b) That interpolation reads sample i+1 instead of i whenever t/dt rounds up
        to an integer (a time jitter of one record sample at ~5 % of steps).
        Measured effect vs. consistent interpolation (3-story frame, h = 2.5 m,
        first 5 s): ~6.7 % of the peak displacement for Newmark. The test pins
        the effect to (0.1 %, 15 %) so a change in either code path is noticed.
    """
    s = system
    n, h, N = s["n"], 0.005, 1000
    rec2d = -np.asarray(s["Ma"]) @ np.ones((n, 1)) @ s["rec"][None, :]
    u_rec = _legacy_run(legacy, name, s["Ka"], s["Ma"], s["C"], rec2d, n, N, h, s["dt"])
    f_leg, _ = _gm_case(s, legacy_interp=True)
    u_fun = _legacy_run(legacy, name, s["Ka"], s["Ma"], s["C"], f_leg, n, N, h, 0)
    peak = np.abs(u_rec).max()
    np.testing.assert_allclose(u_fun, u_rec, rtol=0, atol=1e-12 * peak)

    f_ok, _ = _gm_case(s)
    u_ok = _legacy_run(legacy, name, s["Ka"], s["Ma"], s["C"], f_ok, n, N, h, 0)
    rel = np.abs(u_ok - u_rec).max() / peak
    assert 1e-3 < rel < 0.15, rel


@pytest.mark.parametrize("name", ["newmark", "wilson", "bathe", "ssprk3"])
def test_function_load(legacy, system, name):
    """Functional input: constant 1000 kN at original node 9, x (legacy test f1)."""
    s = system
    n = s["n"]
    k = s["frame"].dof(8, 0)
    e = np.zeros(n)
    e[k] = 1000.0
    h, N = (2e-5, 400) if name == "ssprk3" else (0.005, 400)  # explicit: tiny step for stability
    u_ref = _legacy_run(legacy, name, s["Ka"], s["Ma"], s["C"], lambda t: e, n, N, h, 0)
    u = _jax_run(name, s["Ka"], s["Ma"], s["C"], lambda t: jnp.asarray(e), n, N, h)
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=RTOL * np.abs(u_ref).max())


def test_roof_dof_mapping(system):
    s = system
    assert s["frame"].dof(ROOF_NODE, 0) == int(s["ref"]["g2b"][s["ref"]["itn_roof"] * 3])
