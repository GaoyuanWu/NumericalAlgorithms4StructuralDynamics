"""Differentiable 2D Euler-Bernoulli frame elements and global assembly (JAX).

JAX port of the legacy ``Structure/getMK.py``. The element matrices use exactly
the same coefficients as the legacy code (consistent mass matrix, 6 DOFs per
element: ``[u_i, v_i, rz_i, u_j, v_j, rz_j]``), so assembled matrices agree with
the legacy NumPy implementation to round-off.

Differences from the legacy code (all deliberate):

* The element rotation uses direction cosines ``c = dx/L, s = dy/L`` instead of
  ``beta = arccos(...)``. Mathematically identical, but ``arccos`` has an
  infinite derivative for horizontal members (``d/dx arccos(x)`` at ``x = 1``),
  which would poison gradients with respect to nodal coordinates.
* Element loops are replaced by ``jax.vmap`` + scatter-add, so the assembly is
  jit-compilable and differentiable with respect to ``coords``, ``m``, ``E``,
  ``A`` and ``I``.
"""

import jax
import jax.numpy as jnp


def local_stiffness(E, I, A, L):
    """6x6 local stiffness matrix (same expression as legacy ``local_K``)."""
    r = A * L**2 / I
    z = jnp.zeros_like(L)
    one = jnp.ones_like(L)
    mat = jnp.array(
        [
            [r, z, z, -r, z, z],
            [z, 12 * one, 6 * L, z, -12 * one, 6 * L],
            [z, 6 * L, 4 * L**2, z, -6 * L, 2 * L**2],
            [-r, z, z, r, z, z],
            [z, -12 * one, -6 * L, z, 12 * one, -6 * L],
            [z, 6 * L, 2 * L**2, z, -6 * L, 4 * L**2],
        ]
    )
    return E * I * mat / (L**3)


def local_mass(m, L):
    """6x6 consistent local mass matrix (same expression as legacy ``local_M``)."""
    z = jnp.zeros_like(L)
    one = jnp.ones_like(L)
    mat = jnp.array(
        [
            [140 * one, z, z, 70 * one, z, z],
            [z, 156 * one, 22 * L, z, 54 * one, -13 * L],
            [z, 22 * L, 4 * L**2, z, 13 * L, -3 * L**2],
            [70 * one, z, z, 140 * one, z, z],
            [z, 54 * one, 13 * L, z, 156 * one, -22 * L],
            [z, -13 * L, -3 * L**2, z, -22 * L, 4 * L**2],
        ]
    )
    return m * L / 420 * mat


def rotation(c, s):
    """6x6 local->global transformation from direction cosines."""
    z = jnp.zeros_like(c)
    one = jnp.ones_like(c)
    return jnp.array(
        [
            [c, s, z, z, z, z],
            [-s, c, z, z, z, z],
            [z, z, one, z, z, z],
            [z, z, z, c, s, z],
            [z, z, z, -s, c, z],
            [z, z, z, z, z, one],
        ]
    )


def element_geometry(coords, conn):
    """Length and direction cosines of each element.

    coords: (n_nodes, 2) array; conn: (n_ele, 2) int array, **0-based**.
    """
    d = coords[conn[:, 1]] - coords[conn[:, 0]]
    L = jnp.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    return L, d[:, 0] / L, d[:, 1] / L


def element_matrices(coords, conn, m, E, A, I):
    """Global-frame element stiffness/mass matrices, each (n_ele, 6, 6)."""
    L, c, s = element_geometry(coords, conn)

    def one(m_e, E_e, A_e, I_e, L_e, c_e, s_e):
        T = rotation(c_e, s_e)
        ke = T.T @ local_stiffness(E_e, I_e, A_e, L_e) @ T
        me = T.T @ local_mass(m_e, L_e) @ T
        return ke, me

    return jax.vmap(one)(m, E, A, I, L, c, s)


def element_dofs(conn):
    """(n_ele, 6) global DOF indices: node k owns DOFs 3k, 3k+1, 3k+2."""
    conn = jnp.asarray(conn)
    base = 3 * conn
    return jnp.concatenate([base[:, :1] + jnp.arange(3), base[:, 1:] + jnp.arange(3)], axis=1)


def assemble(coords, conn, m, E, A, I):
    """Assemble global stiffness and mass matrices.

    Mirrors legacy ``getMK.getMK_FEM`` (returns ``K, M`` in that order).
    All per-element arrays (m, E, A, I) have shape (n_ele,).
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    n_dof = 3 * coords.shape[0]
    ke, me = element_matrices(coords, conn, m, E, A, I)
    dofs = element_dofs(conn)
    rows = dofs[:, :, None]
    cols = dofs[:, None, :]
    K = jnp.zeros((n_dof, n_dof)).at[rows, cols].add(ke)
    M = jnp.zeros((n_dof, n_dof)).at[rows, cols].add(me)
    return K, M
