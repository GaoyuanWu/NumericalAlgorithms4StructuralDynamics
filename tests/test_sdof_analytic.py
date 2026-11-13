"""SDOF under harmonic load vs closed form (legacy SDOF.py study): observed order of accuracy."""

import jax.numpy as jnp
import numpy as np
import pytest

import structdyn_jax as sj

k, m, p0, w = 800e3, 6000.0, 1e5, 5.5
zeta = 0.02
wn = np.sqrt(k / m)
c = 2 * zeta * m * wn


def exact(t):
    """Damped SDOF, u0 = v0 = 0, load p0 sin(w t) (Chopra eq. 3.2.3-3.2.4)."""
    wd = wn * np.sqrt(1 - zeta**2)
    b = w / wn
    den = (1 - b**2) ** 2 + (2 * zeta * b) ** 2
    C = p0 / k * (1 - b**2) / den
    D = p0 / k * (-2 * zeta * b) / den
    B = -D
    A = (zeta * wn * B - C * w) / wd
    return np.exp(-zeta * wn * t) * (A * np.sin(wd * t) + B * np.cos(wd * t)) + C * np.sin(w * t) + D * np.cos(w * t)


@pytest.mark.parametrize("method,order", [("newmark", 2), ("bathe", 2), ("wilson", 2), ("ssprk3", 3)])
def test_convergence_order(method, order):
    K, M, Cm = jnp.array([[k]]), jnp.array([[m]]), jnp.array([[c]])
    load = lambda t: jnp.array([p0 * jnp.sin(w * t)])  # noqa: E731
    T, errs, hs = 2.0, [], [2e-3, 1e-3, 5e-4]
    for h in hs:
        N = int(round(T / h))
        u = np.asarray(sj.simulate(method, K, M, Cm, load, jnp.zeros(1), jnp.zeros(1), h, N)[0][:, 0])
        errs.append(np.abs(u - exact(np.arange(N + 1) * h)).max())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 1e-2 * np.abs(exact(np.linspace(0, T, 2001))).max()
    assert np.all(rates > order - 0.25), rates
