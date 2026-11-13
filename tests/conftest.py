import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structdyn_jax as sj  # noqa: E402  (enables float64)
from structdyn_jax.model import THREE_STORY_BEAMS, THREE_STORY_MEMBERS, THREE_STORY_NODES, THREE_STORY_SECTIONS  # noqa: E402


@pytest.fixture(scope="session")
def legacy():
    """Import the original (frozen) NumPy solver modules."""
    sys.path[:0] = [str(ROOT / "Structure"), str(ROOT / "TimeIntegration")]
    import Bathe, getBC, get1DMesh, getMK, Newmark, SSPRK3, Wilson  # noqa: E401

    return dict(Bathe=Bathe, getBC=getBC, get1DMesh=get1DMesh, getMK=getMK,
                Newmark=Newmark, SSPRK3=SSPRK3, Wilson=Wilson)


def legacy_three_story(legacy, h):
    """Build K_a, M_a of the 3-story frame exactly as legacy MDOF_FEM.py does."""
    c, b, itn, ite = legacy["get1DMesh"].get1DMesh(THREE_STORY_NODES, THREE_STORY_MEMBERS + 1, h)
    s = THREE_STORY_SECTIONS
    is_beam = np.isin(ite, THREE_STORY_BEAMS)
    A = np.where(is_beam, s["beam_A"], s["col_A"])
    I = np.where(is_beam, s["beam_I"], s["col_I"])
    K, M = legacy["getMK"].getMK_FEM(c, b, A * s["rho"], s["E"] * np.ones(len(b)), A, I)
    Mb, Kb, n_inact, g2b = legacy["getBC"].MK_Reduction(M, K, {0: 1, 1: 1, 2: 1, 3: 1}, itn)
    n = Kb.shape[0] - n_inact
    return dict(coords=c, conn=b - 1, itn=itn, ite=ite, K=K, M=M, Ka=Kb[:n, :n], Ma=Mb[:n, :n], g2b=g2b, itn_roof=itn[10])


def np_record_interp(rec, dt, legacy=False):
    """NumPy record interpolation f(t) for driving the legacy integrators' function branch.

    legacy=False: consistent interpolation (same formula as structdyn_jax.loads).
    legacy=True:  bit-for-bit replica of the legacy record branch
                  (``int(t/dt)`` with ``(t % dt)/dt``), including its boundary bug.
    """
    rec = np.concatenate([np.asarray(rec, float), np.zeros(2)])

    def ag(t):
        if legacy:
            i, w = int(t / dt), (t % dt) / dt
        else:
            q = t / dt
            i = int(np.floor(q))
            w = q - i
        return rec[i] + (rec[i + 1] - rec[i]) * w

    return ag
