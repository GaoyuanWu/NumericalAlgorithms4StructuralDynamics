"""Loads: ground-motion records and bundled data files."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

DATA = Path(__file__).parent / "data"


def load_kern_county(scale=10.0, legacy_drop_first=False):
    """Kern County (1952) ground acceleration, LA Hollywood Storage FF, dt = 0.005 s.

    ``scale=10`` reproduces the legacy scripts (record stored in units of g,
    multiplied by 10 to obtain m/s^2 with g ~ 10).

    ``legacy_drop_first=True`` drops the first sample: the legacy scripts read
    ``GM.xlsx`` with ``pandas.read_excel`` whose default header row silently
    swallowed it.
    """
    rec = np.loadtxt(DATA / "kern_county_1952_hollywood.csv")
    if legacy_drop_first:
        rec = rec[1:]
    return scale * rec, 0.005


def load_sap2000_reference():
    """SAP2000 modal time history of node 11 (horizontal), from the legacy repo."""
    a = np.loadtxt(DATA / "sap2000_node11_disp.csv", delimiter=",")
    return a[:, 0], a[:, 1]


def interp_record(record, dt):
    """Return a JAX-traceable ``ag(t)``: piecewise-linear interpolation of a record.

    The index and weight are derived from the *same* quotient ``q = t / dt``
    (``i = floor(q)``, ``w = q - i``). The legacy code used
    ``int(t/dt)`` and ``(t % dt)/dt`` separately; that is consistent in eager
    Python, but inside a compiled ``lax.scan`` XLA may evaluate ``t/dt`` as
    ``t * (1/dt)``, so ``floor`` and ``mod`` can disagree at sample boundaries
    (e.g. t = 0.05, dt = 0.005 gives floor = 10 but mod ~ dt), which silently
    shifts the record by one sample. Both formulas give the same value up to
    round-off.

    The record is padded with zeros, so the excitation is zero (free vibration)
    after it ends instead of raising an IndexError as the legacy code did.
    """
    rec = jnp.concatenate([jnp.asarray(record, dtype=jnp.float64), jnp.zeros(2)])
    last = rec.shape[0] - 2

    def ag(t):
        q = t / dt
        i = jnp.clip(jnp.floor(q), 0, last)
        w = jnp.clip(q - i, 0.0, 1.0)
        i = i.astype(jnp.int32)
        return rec[i] + (rec[i + 1] - rec[i]) * w

    return ag


def ground_motion_load(M, influence, record, dt):
    """Effective earthquake load ``f(t) = -M r a_g(t)``.

    ``influence`` (r) is 1 on DOFs that move rigidly with the ground in the
    excitation direction (horizontal translations for horizontal shaking).
    The legacy scripts used ``r = ones`` on *all* active DOFs (including
    vertical translations and rotations); pass ``jnp.ones(n)`` to reproduce it.
    """
    ag = interp_record(record, dt)
    Mr = M @ jnp.asarray(influence, dtype=jnp.float64)
    return lambda t: -Mr * ag(t)
