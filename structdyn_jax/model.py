"""High-level 2D frame model: geometry/topology (static) + differentiable matrices."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cholesky, solve_triangular

from . import bc as _bc
from . import fem
from .mesh import mesh_frame


@dataclass(frozen=True)
class Frame2D:
    """A meshed 2D frame. Everything here is static (NumPy); material and
    section properties are passed separately so they can be differentiated."""

    coords: np.ndarray  # (n_nodes, 2)
    conn: np.ndarray  # (n_ele, 2), 0-based
    ele_parent: np.ndarray  # (n_ele,), original member index
    orig_to_new: np.ndarray  # (n_orig_nodes,)
    active: np.ndarray  # active global DOFs (ascending)

    @classmethod
    def build(cls, nodes, members, supports, h=0.0):
        coords, conn, o2n, parent = mesh_frame(nodes, members, h)
        active, _ = _bc.partition_dofs(len(coords), supports, o2n)
        return cls(coords, conn, parent, o2n, active)

    @property
    def n_active(self):
        return len(self.active)

    def dof(self, orig_node, direction=0):
        """Index into the active-DOF vector of (original joint, direction 0=x,1=y,2=rz)."""
        g = 3 * int(self.orig_to_new[orig_node]) + direction
        return int(np.searchsorted(self.active, g))

    def influence(self, direction=0):
        """Ground-motion influence vector: 1 on active DOFs of the given direction."""
        return jnp.asarray((self.active % 3 == direction).astype(float))

    def matrices(self, m, E, A, I, coords=None):
        """Reduced (active) K and M. Per-element arrays; differentiable."""
        coords = self.coords if coords is None else coords
        K, M = fem.assemble(coords, self.conn, m, E, A, I)
        return _bc.reduce(K, self.active), _bc.reduce(M, self.active)

    def displacement_field(self, u_active):
        """Scatter active-DOF displacements (..., n_active) to nodal (..., n_nodes, 3)."""
        u_active = jnp.asarray(u_active)
        full = jnp.zeros(u_active.shape[:-1] + (3 * len(self.coords),))
        full = full.at[..., self.active].set(u_active)
        return full.reshape(u_active.shape[:-1] + (len(self.coords), 3))


def generalized_eig(K, M):
    """Eigenvalues/vectors of K phi = w^2 M phi via Cholesky (differentiable).

    Returns (omega^2 ascending, mass-normalized mode shapes as columns).
    """
    L = cholesky(M, lower=True)
    Ainv_K = solve_triangular(L, solve_triangular(L, K, lower=True).T, lower=True)
    w2, Y = jnp.linalg.eigh(0.5 * (Ainv_K + Ainv_K.T))
    Phi = solve_triangular(L.T, Y, lower=False)
    return w2, Phi


def natural_frequencies(K, M):
    """Natural frequencies in Hz."""
    return jnp.sqrt(generalized_eig(K, M)[0]) / (2 * jnp.pi)


def rayleigh(M, K, alpha, beta):
    return alpha * M + beta * K


def rayleigh_coefficients(omega_i, omega_j, zeta):
    """(alpha, beta) giving damping ratio ``zeta`` at two circular frequencies."""
    alpha = zeta * 2 * omega_i * omega_j / (omega_i + omega_j)
    beta = zeta * 2 / (omega_i + omega_j)
    return alpha, beta


# ---------------------------------------------------------------------------
# The 3-story, 3-bay frame from the original APC523 project (MDOF_FEM.py)
# ---------------------------------------------------------------------------
THREE_STORY_NODES = np.array(
    [[0, 0], [10, 0], [20, 0], [30, 0], [0, 5], [10, 5], [20, 5], [30, 5], [10, 10], [20, 10], [10, 15], [20, 15]],
    dtype=float,
)
THREE_STORY_MEMBERS = (
    np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 6], [6, 7], [7, 8], [6, 9], [7, 10], [9, 10], [9, 11], [10, 12], [11, 12]]) - 1
)
THREE_STORY_BEAMS = [4, 5, 6, 9, 12]
THREE_STORY_SUPPORTS = {0: _bc.FIXED, 1: _bc.FIXED, 2: _bc.FIXED, 3: _bc.FIXED}
# Units: m, kN, t (so mass in t and acceleration in m/s^2)
THREE_STORY_SECTIONS = dict(col_A=1.0, beam_A=0.24, E=3.79e7, col_I=1 / 12, beam_I=0.4 * 0.6**3 / 12, rho=2.4473)
ROOF_NODE = 10  # "node 11" in the legacy (1-based) scripts


def three_story_frame(h=1.0):
    """Return (frame, member_props) for the legacy 3-story example.

    member_props is a dict of per-*original-member* arrays (13,) with keys
    m, E, A, I. Use :func:`element_props` to expand to mesh elements.
    """
    frame = Frame2D.build(THREE_STORY_NODES, THREE_STORY_MEMBERS, THREE_STORY_SUPPORTS, h)
    s = THREE_STORY_SECTIONS
    is_beam = np.isin(np.arange(len(THREE_STORY_MEMBERS)), THREE_STORY_BEAMS)
    A = np.where(is_beam, s["beam_A"], s["col_A"])
    props = dict(
        m=jnp.asarray(A * s["rho"]),
        E=jnp.full(len(A), s["E"]),
        A=jnp.asarray(A),
        I=jnp.asarray(np.where(is_beam, s["beam_I"], s["col_I"])),
    )
    return frame, props


def element_props(frame, member_props):
    """Expand per-member properties to per-element arrays (differentiable gather)."""
    return {k: jnp.asarray(v)[frame.ele_parent] for k, v in member_props.items()}
