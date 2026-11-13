"""Boundary conditions: partition DOFs into active / constrained (NumPy).

Port of legacy ``Structure/getBC.py``. The active-DOF ordering is the same as
the legacy code (ascending global DOF index), so ``K[active][:, active]``
equals the legacy ``K_a`` exactly.

Fix relative to legacy: a roller support constrains one DOF; the legacy code
counted it as two (``inact_num += 2``), which shifted ``act_num``.
"""

import numpy as np

FIXED, PINNED, ROLLER = 1, 2, 3
_CONSTRAINED = {FIXED: (0, 1, 2), PINNED: (0, 1), ROLLER: (1,)}  # roller: vertical only (as legacy)


def partition_dofs(n_nodes, supports, orig_to_new=None):
    """Return (active, constrained) global DOF index arrays.

    supports : dict {original node index (0-based): FIXED | PINNED | ROLLER}
    orig_to_new : optional map from original to meshed node numbering.
    """
    if orig_to_new is None:
        orig_to_new = np.arange(n_nodes)
    constrained = []
    for node, kind in supports.items():
        k = int(orig_to_new[node])
        constrained += [3 * k + d for d in _CONSTRAINED[kind]]
    constrained_set = set(constrained)
    active = np.array([i for i in range(3 * n_nodes) if i not in constrained_set], dtype=int)
    return active, np.array(constrained, dtype=int)


def reduce(mat, active):
    """Extract the active-active block of a global matrix."""
    return mat[np.ix_(active, active)]
