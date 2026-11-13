"""Backward capability: reverse-mode gradients vs central finite differences.

Parameters are log-scale design factors (see structdyn_jax.problems); the
objective is the roof response of the 3-story frame under the Kern County
record with 5 % Rayleigh damping (which itself depends on the eigenvalues, so
the eigen-derivative is exercised too).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from structdyn_jax.problems import EarthquakeProblem, finite_difference_grad, zero_params

RTOL = 1e-5
# FD step in log-parameters, with Richardson extrapolation (O(eps^4)).
# Step study (peak objective, plain central FD, relative error vs AD):
#   log_EI:     eps 1e-8 -> 6e-3, 1e-6 -> 1.5e-4, 1e-4 -> 9e-7   (round-off dominated)
#   log_height: eps 1e-3 -> 1.4e-3, 1e-4 -> 1.4e-5, 1e-5 -> 1.3e-7 (truncation dominated)
# so a single plain step cannot serve both; Richardson at eps = 1e-4 does.
EPS = 1e-4


def _check(grad, fd, rtol=RTOL):
    for k in fd:
        g, f = np.asarray(grad[k]), fd[k]
        scale = max(np.abs(f).max(), 1e-30)
        np.testing.assert_allclose(g, f, rtol=0, atol=rtol * scale, err_msg=k)


@pytest.fixture(scope="module")
def nominal():
    # small random perturbation so we are not at a symmetric special point
    rng = np.random.default_rng(0)
    p = zero_params()
    return {k: v + 0.05 * jnp.asarray(rng.standard_normal(np.shape(v))) for k, v in p.items()}


@pytest.mark.parametrize("objective", ["rms_roof", "soft_peak_roof", "peak_roof"])
def test_newmark_gradients_all_parameters(nominal, objective):
    prob = EarthquakeProblem(h_mesh=2.5, duration=3.0, method="newmark")
    f = jax.jit(getattr(prob, objective))
    g = jax.grad(f)(nominal)
    fd = finite_difference_grad(f, nominal, EPS)
    _check(g, fd)


@pytest.mark.parametrize("method,kw", [("bathe", {}), ("wilson", {"theta": 1.4}), ("ssprk3", {})])
def test_other_integrators_gradients(nominal, method, kw):
    # explicit SSP-RK3 needs a step below the stability limit of the coarse h=5 m mesh
    dt = 2e-4 if method == "ssprk3" else 0.005
    prob = EarthquakeProblem(h_mesh=5.0, duration=1.0, dt=dt, method=method, method_kwargs=kw)
    f = jax.jit(prob.rms_roof)
    g = jax.grad(f)(nominal)
    fd = finite_difference_grad(f, nominal, EPS)
    _check(g, fd)


def test_forward_and_reverse_mode_agree(nominal):
    """jacfwd (JVP) and jacrev (VJP) of the full roof history must match."""
    prob = EarthquakeProblem(h_mesh=5.0, duration=1.0)

    def hist(theta):
        return prob.roof_history({**nominal, "log_EI": theta})

    Jf = jax.jacfwd(hist)(nominal["log_EI"])
    Jr = jax.jacrev(hist)(nominal["log_EI"])
    np.testing.assert_allclose(Jf, Jr, rtol=0, atol=1e-10 * np.abs(Jr).max())


def test_gradient_wrt_horizontal_beam_coordinates_is_finite():
    """The legacy arccos-based rotation has an infinite derivative for horizontal
    members; the direction-cosine form must give finite, FD-consistent values."""
    prob = EarthquakeProblem(h_mesh=5.0, duration=1.0)
    p = zero_params()
    g = jax.grad(prob.rms_roof)(p)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())
    fd = finite_difference_grad(prob.rms_roof, p, EPS)
    np.testing.assert_allclose(g["log_height"], fd["log_height"], rtol=RTOL)
