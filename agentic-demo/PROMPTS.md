# Prompt log and session timeline

The user prompts are copied verbatim. The timeline entries are a summary of what the agent did, written by the agent at the end of the session. They are not quotations. [Back to the demo](README.md).

## P01: Scope and acceptance criteria

**November 13, 2026 · Verbatim user message**

> Our research group is doing a AI workshop tomorrow where each member uses a demo to demonstrate what is good/bad practice of agentic workflow.
>
> I'd like to use my this final project repo for numerical algorithms class i took in first year grad school from 5 years ago, which is an old, non-differentiable dynamics Python solver of structures.
>
> I'd like to show a demo where I use CLAUDE to adapt it to a jax and differentiable solver, with the exact forward ability, and additional backward capability. With checks against open-source solver PyNite (forward), and gradient check using finite difference. Visuals would be great to show how structures vibrate under ground motions. Open a new branch called jax-migration. Everything tested online.
>
> In addition to **code development**, Review the example and guideline of such demo provided by the PI: "https://github.com/SOLARIS-JHU/solaris-agentic-research", so that this code development procedure becomes a new example by me. Record this session, of what prompts that helped, did not help much.

**Assessment:** see [§5 of the demo](README.md#5-which-parts-of-the-prompt-helped-and-which-did-not).

## Session timeline (agent summary, in order)

1. **Read everything first.** The agent read the whole legacy repo (mesh, FEM, BCs, four integrators, two study scripts, the ground-motion and SAP2000 data). Cloning the PI repo anonymously failed because it is private. The agent attached it through the GitHub integration, then read the README, getting-started guide, demo template and worked example.
2. **Probed the tools before relying on them.** It installed JAX and PyNite 3.2 and inspected PyNite's API. PyNite turned out to use the *same* consistent-mass coefficients, which enabled a matrix-level comparison and not only displacements.
3. **Ported the code** into `structdyn_jax/`:
   - mesh and BCs in NumPy;
   - FEM assembly with `vmap` and scatter-add;
   - integrators as `lax.scan` loops with the Cholesky factorization hoisted out of the loop;
   - float64 throughout;
   - direction cosines instead of `arccos`.
4. **Smoke test.** Matrices matched to 1e-16, but the time histories were off by 2 %. The error stayed at 1e-17 for 8 steps and then jumped, which pointed to a discrete event, not round-off. The agent traced it to XLA's `floor(t/dt)` disagreeing with `mod(t, dt)` inside compiled code.
5. **Found the legacy bug.** A consistent interpolation then disagreed with the *legacy* code. The legacy Python arithmetic has the same inconsistency. Bit-for-bit reproduction proved fragile: XLA's division differed from Python's at 204 of 4000 steps even with a runtime `dt`. Decision: consistent interpolation in the port, with legacy parity defined through identical forcing. A test pins the legacy effect, which the agent had assumed was < 1 %. It measured **6.7 %** and recorded the measured value instead of keeping the assumption.
6. **PyNite cross-check.** First failure: PyNite only builds its internal state during an analysis call, so the agent ran a modal analysis before extracting matrices. Then K/M, static, modal and time-history checks all passed. The time-history reference is an exact Nigam–Jennings modal solution built only from PyNite's matrices.
7. **SAP2000 benchmark.** The peak matched; the full history did not. The agent scanned hypotheses: damping (rejected), influence vector (negligible), dropped first sample (SAP used the full record), and period scale (a 0.5 % shift explains it). It did not calibrate; it asserted only peak and early-time agreement.
8. **Gradients.**
   - First FD check failed at 1.4e-5. A step study showed opposite error behavior (round-off vs truncation) for different parameters.
   - Switched to Richardson-extrapolated FD; all 28 parameters × 4 integrators passed.
   - Added forward-vs-reverse-mode agreement and a finite geometry-gradient test.
9. **SDOF closed form.** Observed convergence orders 2, 2, 2, 3, as theory predicts.
10. **Figures.** Animation, verification plot, sensitivity map and mode shapes, using a validated colorblind-safe palette. The agent rendered and inspected them; three iterations fixed label collisions and an outlier-dominated scatter. The GIF went from 7 MB to 3.7 MB.
11. **CI and docs.** GitHub Actions on Python 3.11 and 3.12 (tests plus a figure smoke run). Wrote the README section, `docs/VERIFICATION.md` and this demo. Pushed to `jax-migration` and watched CI.

## P02: Fix an ambiguous metric

**November 13, 2026 · Verbatim user message**

> For the verification image `docs/figures/verification.png`, what does the peak mean?
> i) did you mean the peak error?
> ii) or did you mean the error when the roof's displacement goes to the peak.
> Fix the ambiguity.
>
> It might also be helpful to output .csv files that record time history of what you plotted for different solver/setup.

**What changed:**

- The agent answered "neither": every "x of peak" meant max_t|u − u_ref| / max_t|u_ref|.
- While checking, it found that the panel (a) number had been computed over all DOFs although only the roof is plotted, and that the SAP2000 "peak ratio" compared peaks at different times.
- The figure now reports two named metrics per panel, ε_max (largest error at any time) and ε_peak (error at the reference peak). Each is computed on the plotted roof DOF, marked on a new error-vs-time strip, and labelled with the time at which it occurs.
- `docs/VERIFICATION.md` gained a notation section, and every "of peak" phrase in the docs and tests now names its metric and scope.
- The plotted histories are written to `docs/figures/data/*.csv`, with the full setup in the header lines.
- New, previously hidden fact: SAP2000's peak-magnitude ratio is 0.983, but at the instant SAP2000 peaks the error is 30 % (phase drift).

## P03+: follow-ups

_Add the next prompts here verbatim, with a one-line note on what changed after each._
