"""structdyn_jax: a differentiable JAX port of the APC523 structural dynamics solver.

Float64 is enabled on import: structural matrices are badly conditioned
(EA/L vs EI/L^3 differ by many orders of magnitude) and the legacy-parity and
finite-difference gradient checks need double precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import bc, fem, integrators, loads, mesh, model  # noqa: E402
from .integrators import bathe, newmark, simulate, ssprk3, wilson  # noqa: E402
from .loads import ground_motion_load, load_kern_county  # noqa: E402
from .model import Frame2D, element_props, natural_frequencies, three_story_frame  # noqa: E402

__all__ = [
    "bc", "fem", "integrators", "loads", "mesh", "model",
    "newmark", "wilson", "bathe", "ssprk3", "simulate",
    "ground_motion_load", "load_kern_county",
    "Frame2D", "three_story_frame", "element_props", "natural_frequencies",
]
