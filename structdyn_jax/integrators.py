"""Differentiable implicit/explicit time integrators for M a + C v + K u = f(t).

JAX ports of the legacy ``TimeIntegration`` modules:

=================  ===========================  ==============================
function           legacy                       scheme
=================  ===========================  ==============================
``newmark``        ``Newmark.Newmark_beta``     Newmark-beta (gamma, beta)
``wilson``         ``Wilson.Wilson_theta``      Wilson-theta
``bathe``          ``Bathe.Bathe``              Bathe composite (trapezoid + 3-point BDF)
``ssprk3``         ``SSPRK3.SSP_RK3``           3-stage, 3rd-order SSP Runge-Kutta
=================  ===========================  ==============================

Conventions
-----------
* ``load`` is a JAX-traceable callable ``f(t) -> (n_dof,)``. Build one with
  :func:`structdyn_jax.loads.ground_motion_load` (earthquake record, linearly
  interpolated exactly as in the legacy code) or write any ``jnp`` function.
* Time stepping: ``t_i = i * h`` for ``i = 0..N-1`` (identical to the legacy
  arithmetic, so record interpolation hits the same samples).
* Outputs are **time-major**: ``u`` has shape ``(N + 1, n_dof)`` (the legacy
  code returned ``(n_dof, N + 1)``).
* The loop body is a ``jax.lax.scan``; the effective-stiffness factorization
  (Cholesky) is computed once outside the loop. Everything is differentiable
  with ``jax.grad`` / ``jax.jvp`` with respect to K, M, C, the load, the
  initial conditions and the scheme parameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def _times(N, h):
    return jnp.arange(N, dtype=jnp.float64) * h


def _initial_acceleration(K, M, C, load, u0, v0):
    return cho_solve(cho_factor(M), load(0.0) - C @ v0 - K @ u0)


def newmark(K, M, C, load, u0, v0, h, N, gamma=0.5, beta=0.25, a0=None):
    """Newmark-beta method (legacy argument names: ``delta`` = gamma).

    ``a0=None`` computes the initial acceleration from equilibrium. The legacy
    implementation always started from ``a0 = 0``; pass ``a0=jnp.zeros(n)``
    to reproduce it exactly.
    """
    c0 = 1 / (beta * h**2)
    c1 = gamma / (beta * h)
    c2 = 1 / (beta * h)
    c3 = 1 / (2 * beta) - 1
    c4 = gamma / beta - 1
    c5 = h / 2 * (gamma / beta - 2)
    c6 = h * (1 - gamma)
    c7 = gamma * h
    fac = cho_factor(K + c0 * M + c1 * C)
    if a0 is None:
        a0 = _initial_acceleration(K, M, C, load, u0, v0)

    def step(carry, t):
        u, v, a = carry
        F = load(t + h) + M @ (c0 * u + c2 * v + c3 * a) + C @ (c1 * u + c4 * v + c5 * a)
        u1 = cho_solve(fac, F)
        a1 = c0 * (u1 - u) - c2 * v - c3 * a
        v1 = v + c6 * a + c7 * a1
        return (u1, v1, a1), (u1, v1, a1)

    return _stack((u0, v0, a0), jax.lax.scan(step, (u0, v0, a0), _times(N, h))[1])


def wilson(K, M, C, load, u0, v0, h, N, theta=1.4):
    """Wilson-theta method (unconditionally stable for theta >= 1.37)."""
    th = theta * h
    c0 = 6 / th**2
    c1 = 3 / th
    c2 = 2 * c1
    c3 = th / 2
    c4 = c0 / theta
    c5 = -c2 / theta
    c6 = 1 - 3 / theta
    c7 = h / 2
    c8 = h**2 / 6
    fac = cho_factor(K + c0 * M + c1 * C)
    a0 = _initial_acceleration(K, M, C, load, u0, v0)

    def step(carry, t):
        u, v, a = carry
        f_i = load(t)
        F = f_i + theta * (load(t + h) - f_i) + M @ (c0 * u + c2 * v + 2 * a) + C @ (c1 * u + 2 * v + c3 * a)
        u_th = cho_solve(fac, F)
        a1 = c4 * (u_th - u) + c5 * v + c6 * a
        v1 = v + c7 * (a1 + a)
        u1 = u + h * v + c8 * (a1 + 2 * a)
        return (u1, v1, a1), (u1, v1, a1)

    return _stack((u0, v0, a0), jax.lax.scan(step, (u0, v0, a0), _times(N, h))[1])


def bathe(K, M, C, load, u0, v0, h, N):
    """Bathe composite method: trapezoidal rule to t+h/2, then 3-point BDF."""
    c1, c2, c3, c4 = 16 / h**2, 4 / h, 9 / h**2, 3 / h
    c5, c6, c7, c8, c9 = 8 / h, 4 / h, 12 / h**2, 3 / h**2, 1 / h
    fac1 = cho_factor(c1 * M + c2 * C + K)
    fac2 = cho_factor(c3 * M + c4 * C + K)
    a0 = _initial_acceleration(K, M, C, load, u0, v0)

    def step(carry, t):
        u, v, a = carry
        F1 = load(t + 0.5 * h) + M @ (c1 * u + c5 * v + a) + C @ (c6 * u + v)
        u_h = cho_solve(fac1, F1)
        v_h = (u_h - u) * c6 - v
        F2 = load(t + h) + M @ (c7 * u_h - c8 * u + c6 * v_h - c9 * v) + C @ (c6 * u_h - c9 * u)
        u1 = cho_solve(fac2, F2)
        v1 = c9 * u - c2 * u_h + c4 * u1
        a1 = c9 * v - c2 * v_h + c4 * v1
        return (u1, v1, a1), (u1, v1, a1)

    return _stack((u0, v0, a0), jax.lax.scan(step, (u0, v0, a0), _times(N, h))[1])


def ssprk3(K, M, C, load, u0, v0, h, N):
    """Explicit SSP-RK3 on the first-order form (conditionally stable).

    Stage times follow the legacy code: t, t + h, t + h/2. Returns
    ``(u, v, a)`` with ``a`` recovered from equilibrium.
    """
    fac = cho_factor(M)

    def rhs(u, v, t):
        return v, cho_solve(fac, load(t) - K @ u - C @ v)

    def step(carry, t):
        u, v = carry
        du, dv = rhs(u, v, t)
        u1, v1 = u + h * du, v + h * dv
        du, dv = rhs(u1, v1, t + h)
        u2, v2 = 0.75 * u + 0.25 * (u1 + h * du), 0.75 * v + 0.25 * (v1 + h * dv)
        du, dv = rhs(u2, v2, t + 0.5 * h)
        un, vn = u / 3 + 2 / 3 * (u2 + h * du), v / 3 + 2 / 3 * (v2 + h * dv)
        return (un, vn), (un, vn)

    u_hist, v_hist = jax.lax.scan(step, (u0, v0), _times(N, h))[1]
    u = jnp.concatenate([u0[None], u_hist])
    v = jnp.concatenate([v0[None], v_hist])
    ts = jnp.arange(N + 1, dtype=jnp.float64) * h
    f = jax.vmap(load)(ts)
    a = cho_solve(fac, (f - u @ K.T - v @ C.T).T).T
    return u, v, a


def _stack(init, hist):
    return tuple(jnp.concatenate([x0[None], xs]) for x0, xs in zip(init, hist))


METHODS = {"newmark": newmark, "wilson": wilson, "bathe": bathe, "ssprk3": ssprk3}


def simulate(method, K, M, C, load, u0, v0, h, N, **kwargs):
    """Dispatch by name: ``simulate("newmark", K, M, C, load, u0, v0, h, N)``."""
    return METHODS[method](K, M, C, load, u0, v0, h, N, **kwargs)
