"""Ready-made differentiable problems built on the legacy 3-story frame.

``EarthquakeProblem`` maps design parameters -> roof response under the Kern
County record. Parameters are *log-scale factors* on the nominal design, so
``theta = 0`` is the original structure and ``dJ/dtheta`` is a relative
(elasticity-like) sensitivity:

* ``log_EI``   (13,) per original member: flexural rigidity scale
* ``log_m``    (13,) per original member: mass-per-length scale
* ``log_zeta`` ()    Rayleigh damping ratio (fit at modes 1 and 2), nominal 5 %
* ``log_height`` ()  scales all y-coordinates (story heights), via the nodal
  coordinates fed to the assembly - exercises geometric differentiation.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .integrators import simulate
from .loads import ground_motion_load, load_kern_county
from .model import ROOF_NODE, element_props, generalized_eig, rayleigh_coefficients, three_story_frame


def zero_params(n_members=13):
    return dict(log_EI=jnp.zeros(n_members), log_m=jnp.zeros(n_members), log_zeta=jnp.zeros(()), log_height=jnp.zeros(()))


@dataclass
class EarthquakeProblem:
    h_mesh: float = 2.5
    duration: float = 5.0
    dt: float = 0.005
    method: str = "newmark"
    zeta: float = 0.05
    method_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.frame, self.member_props = three_story_frame(self.h_mesh)
        self.record, self.rec_dt = load_kern_county()
        self.n_steps = int(round(self.duration / self.dt))
        self.roof = self.frame.dof(ROOF_NODE, 0)
        self.r = self.frame.influence(0)

    def matrices(self, p):
        mp = dict(self.member_props)
        mp["I"] = mp["I"] * jnp.exp(p["log_EI"])
        mp["m"] = mp["m"] * jnp.exp(p["log_m"])
        coords = jnp.asarray(self.frame.coords) * jnp.array([1.0, 1.0]).at[1].set(jnp.exp(p["log_height"]))
        K, M = self.frame.matrices(**element_props(self.frame, mp), coords=coords)
        w2, _ = generalized_eig(K, M)
        w = jnp.sqrt(w2)
        alpha, beta = rayleigh_coefficients(w[0], w[1], self.zeta * jnp.exp(p["log_zeta"]))
        return K, M, alpha * M + beta * K

    def response(self, p):
        """Full displacement history (n_steps + 1, n_dof)."""
        K, M, C = self.matrices(p)
        load = ground_motion_load(M, self.r, self.record, self.rec_dt)
        z = jnp.zeros(self.frame.n_active)
        return simulate(self.method, K, M, C, load, z, z, self.dt, self.n_steps, **self.method_kwargs)[0]

    def roof_history(self, p):
        return self.response(p)[:, self.roof]

    # --- scalar objectives -------------------------------------------------
    def rms_roof(self, p):
        """Root-mean-square roof displacement (smooth)."""
        return jnp.sqrt(jnp.mean(self.roof_history(p) ** 2))

    def peak_roof(self, p):
        """Peak |roof displacement| (differentiable almost everywhere)."""
        return jnp.max(jnp.abs(self.roof_history(p)))

    def soft_peak_roof(self, p, sharpness=2000.0):
        """Smooth upper bound of the peak (log-sum-exp)."""
        u = self.roof_history(p)
        return jax.nn.logsumexp(sharpness * jnp.concatenate([u, -u])) / sharpness


def flatten(p):
    keys = sorted(p)
    return jnp.concatenate([jnp.ravel(p[k]) for k in keys]), keys


def finite_difference_grad(f, p, eps=1e-4, richardson=True):
    """Central finite differences of scalar f over every entry of the param dict.

    With ``richardson=True`` the central differences at ``eps`` and ``eps/2``
    are combined as ``(4 D(eps/2) - D(eps)) / 3``, cancelling the O(eps^2)
    truncation term. This matters because parameters differ: for stiffness
    scales the plain FD error is round-off dominated (grows as 1/eps), for the
    geometry scale it is truncation dominated (grows as eps^2).
    """

    def central(k, v, idx, e):
        vp, vm = v.copy(), v.copy()
        vp[idx] += e
        vm[idx] -= e
        return (float(f({**p, k: jnp.asarray(vp)})) - float(f({**p, k: jnp.asarray(vm)}))) / (2 * e)

    out = {}
    for k, v in p.items():
        v = np.asarray(v, dtype=float)
        g = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            d1 = central(k, v, idx, eps)
            g[idx] = (4 * central(k, v, idx, eps / 2) - d1) / 3 if richardson else d1
        out[k] = g
    return out
