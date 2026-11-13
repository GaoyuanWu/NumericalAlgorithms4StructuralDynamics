"""1D meshing of 2D frames (topology only, plain NumPy).

Port of legacy ``Structure/get1DMesh.py``. Node numbering and element ordering
are identical to the legacy routine so that results can be compared
DOF-by-DOF. Indices here are **0-based** (the legacy routine returned 1-based
connectivity).

Meshing decides the *topology* (how many nodes/elements), which is discrete and
therefore not differentiated. The resulting nodal coordinates are plain arrays
that can be fed into the differentiable assembly in :mod:`structdyn_jax.fem`.
"""

import numpy as np


def mesh_frame(nodes, conn, h):
    """Subdivide each member into ``floor(L / h)`` equal elements.

    Parameters
    ----------
    nodes : (n, 2) array of original joint coordinates.
    conn : (n_members, 2) int array of member end nodes, 0-based.
    h : target mesh size. ``h == 0`` returns the original model unchanged.

    Returns
    -------
    coords : (n_new, 2) node coordinates.
    new_conn : (n_ele, 2) element connectivity, 0-based.
    orig_to_new : (n,) index of each original joint in the new numbering.
    ele_parent : (n_ele,) index of the original member each element belongs to.
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    if h == 0:
        return nodes.copy(), conn.copy(), np.arange(len(nodes)), np.arange(len(conn))

    coords = []
    new_conn = []
    ele_parent = []
    orig_to_new = -np.ones(len(nodes), dtype=int)

    def joint(k):
        if orig_to_new[k] < 0:
            orig_to_new[k] = len(coords)
            coords.append(nodes[k].copy())
        return orig_to_new[k]

    for e, (a, b) in enumerate(conn):
        du, dv = nodes[b] - nodes[a]
        n_sub = int(np.floor(np.sqrt(du**2 + dv**2) / h))
        if n_sub < 1:
            raise ValueError(f"mesh size h={h} is larger than member {e}")
        ids = []
        for j in range(n_sub + 1):
            if j == 0:
                ids.append(joint(a))
            elif j == n_sub:
                ids.append(joint(b))
            else:  # same arithmetic as the legacy routine
                coords.append(np.array([nodes[a, 0] + du / n_sub * j, nodes[a, 1] + dv / n_sub * j]))
                ids.append(len(coords) - 1)
        for j in range(n_sub):
            new_conn.append((ids[j], ids[j + 1]))
            ele_parent.append(e)

    return np.array(coords), np.array(new_conn, dtype=int), orig_to_new, np.array(ele_parent, dtype=int)
