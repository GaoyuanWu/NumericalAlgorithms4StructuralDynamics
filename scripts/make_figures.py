"""Generate the figures in docs/figures/ and the metrics in docs/figures/metrics.json.

    python scripts/make_figures.py            # everything (~3-5 min on a laptop CPU)
    python scripts/make_figures.py --quick    # coarser mesh / shorter GIF
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "Structure"), str(ROOT / "TimeIntegration")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import structdyn_jax as sj  # noqa: E402
from structdyn_jax.model import ROOF_NODE, element_props, generalized_eig, three_story_frame  # noqa: E402
from structdyn_jax.problems import EarthquakeProblem, finite_difference_grad, zero_params  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data"

# Reference palette (dataviz skill): categorical slots 1-2, neutral inks, blue<->red diverging.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
})


def frame_segments(coords, conn, disp=None, scale=1.0):
    xy = coords if disp is None else coords + scale * disp[:, :2]
    return np.stack([xy[conn[:, 0]], xy[conn[:, 1]]], axis=1)


def earthquake_run(h_mesh, T, zeta=0.05):
    frame, mp = three_story_frame(h_mesh)
    K, M = frame.matrices(**element_props(frame, mp))
    w2, _ = generalized_eig(K, M)
    a, b = sj.model.rayleigh_coefficients(jnp.sqrt(w2[0]), jnp.sqrt(w2[1]), zeta)
    rec, dt = sj.load_kern_county()
    N = int(round(T / dt))
    load = sj.ground_motion_load(M, frame.influence(0), rec, dt)
    z = jnp.zeros(frame.n_active)
    u = np.asarray(sj.newmark(K, M, a * M + b * K, load, z, z, dt, N)[0])
    return frame, np.asarray(frame.displacement_field(u)), rec[: N + 1], dt, np.asarray(w2)


# ---------------------------------------------------------------------------
def animation(quick):
    T = 12.0 if quick else 20.0
    frame, U, ag, dt, w2 = earthquake_run(2.5 if quick else 1.0, T)
    t = np.arange(len(ag)) * dt
    roof = U[:, frame.orig_to_new[ROOF_NODE], 0] * 1000  # mm
    write_history_csv(
        "frame_earthquake_roof.csv",
        f"Animation: 3-story frame, h_mesh = {frame_h(frame)} m, full Kern County record, horizontal influence vector, "
        "5 % Rayleigh damping (modes 1-2), Newmark gamma=0.5 beta=0.25, h = dt = 0.005 s. "
        "a_g in m/s^2 (legacy scale x10); roof = node 11 horizontal displacement relative to the ground, in mm.",
        t, {"a_g_m_s2": ag, "u_roof_mm": roof})
    scale = 250.0
    stride = 12  # 0.06 s per frame
    frames = range(0, len(t), stride)

    fig = plt.figure(figsize=(7.2, 6.4), dpi=64)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1, 1], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(-6, 36)
    ax.set_ylim(-1.5, 17)
    ax.axis("off")
    ax.add_collection(LineCollection(frame_segments(frame.coords, frame.conn), colors=GRID, linewidths=3))
    lc = LineCollection(frame_segments(frame.coords, frame.conn), colors=BLUE, linewidths=3, capstyle="round")
    ax.add_collection(lc)
    ground = ax.plot([-4, 34], [0, 0], color=INK2, lw=2)[0]
    title = ax.set_title("")
    ax.text(0.0, -0.06, f"3-story frame, Kern County 1952 record, 5 % damping · displacements ×{scale:.0f}",
            transform=ax.transAxes, color=INK2, fontsize=8.5)

    ax_g = fig.add_subplot(gs[1])
    ax_g.plot(t, ag, color=MUTED, lw=0.8)
    ax_g.set_ylabel("a$_g$ [m/s²]")
    cur_g = ax_g.axvline(0, color=INK, lw=1)
    ax_r = fig.add_subplot(gs[2])
    ax_r.plot(t, roof, color=BLUE, lw=1.0)
    ax_r.set_ylabel("roof u [mm]")
    ax_r.set_xlabel("time [s]")
    cur_r = ax_r.axvline(0, color=INK, lw=1)
    for a_ in (ax_g, ax_r):
        a_.set_xlim(0, T)

    # ground displacement is not modelled (relative displacements), so the base stays put
    def update(i):
        lc.set_segments(frame_segments(frame.coords, frame.conn, U[i], scale))
        cur_g.set_xdata([t[i], t[i]])
        cur_r.set_xdata([t[i], t[i]])
        title.set_text(f"t = {t[i]:5.2f} s   roof drift {roof[i]:+5.1f} mm")
        return lc, cur_g, cur_r, title, ground

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT / "frame_earthquake.gif", writer=PillowWriter(fps=17))
    update(int(np.argmax(np.abs(roof))))
    fig.savefig(OUT / "frame_earthquake_peak.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # mode shapes
    Kf, Mf = frame.matrices(**element_props(frame, three_story_frame(frame_h(frame))[1]))
    w2, Phi = generalized_eig(Kf, Mf)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for k, a_ in enumerate(axes):
        phi = np.asarray(frame.displacement_field(Phi[:, k]))
        s = 1.8 / np.abs(phi[:, :2]).max()
        a_.add_collection(LineCollection(frame_segments(frame.coords, frame.conn), colors=GRID, linewidths=2.5))
        a_.add_collection(LineCollection(frame_segments(frame.coords, frame.conn, phi, s), colors=BLUE, linewidths=2.5))
        a_.set_aspect("equal")
        a_.set_xlim(-5, 35)
        a_.set_ylim(-1, 19)
        a_.axis("off")
        a_.set_title(f"mode {k + 1}: T = {2 * np.pi / np.sqrt(float(w2[k])):.3f} s")
    fig.savefig(OUT / "mode_shapes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(T1=float(2 * np.pi / np.sqrt(w2[0])), peak_roof_mm=float(np.abs(roof).max()))


def frame_h(frame):
    # recover mesh size used to build the frame (number of elements -> h)
    return {13: 0.0, 18: 5, 36: 2.5, 90: 1.0, 180: 0.5}[len(frame.conn)]


# ---------------------------------------------------------------------------
def error_metrics(u, u_ref):
    """Unambiguous comparison metrics for one DOF history (both arrays same time grid).

    eps_max   = max_t |u - u_ref| / max_t |u_ref|   (largest error anywhere, normalized by the reference peak)
    eps_peak  = |u - u_ref| at t_peak / max_t |u_ref|, t_peak = argmax_t |u_ref|  (error at the reference peak)
    peak_ratio = max_t |u| / max_t |u_ref|            (the two peaks may occur at different times)
    """
    e = u - u_ref
    P = np.abs(u_ref).max()
    i = int(np.argmax(np.abs(u_ref)))
    return dict(eps_max=float(np.abs(e).max() / P), eps_peak=float(abs(e[i]) / P),
                peak_ratio=float(np.abs(u).max() / P), ref_peak_m=float(P), i_peak=i, i_err=int(np.argmax(np.abs(e))))


def write_history_csv(name, header, t, columns):
    """Write a time-history CSV to docs/figures/data/ (columns: name -> array, in mm)."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    names = ["t_s"] + list(columns)
    arr = np.column_stack([t] + [columns[c] for c in columns])
    with open(DATA_OUT / name, "w") as fh:  # '#' description lines, then a plain column-name row
        fh.write("".join(f"# {line}\n" for line in header.strip().splitlines()))
        fh.write(",".join(names) + "\n")
        np.savetxt(fh, arr, delimiter=",", fmt="%.10g")


def _compare_panel(fig, gs_slot, t, u_ref, u, lab_ref, lab, col_ref, col, title, sharex=None):
    """Top: the two histories. Bottom: the error u - u_ref vs time, with the two metric instants marked."""
    sub = gs_slot.subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
    ax = fig.add_subplot(sub[0], sharex=sharex)
    ax.plot(t, 1000 * u_ref, color=col_ref, lw=3.5 if col_ref == MUTED else 1.2, label=lab_ref)
    ax.plot(t, 1000 * u, color=col, lw=1.2 if col_ref == MUTED else 1.0, label=lab)
    ax.set_ylabel("roof u [mm]")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.tick_params(labelbottom=False)
    m = error_metrics(u, u_ref)
    ip, ie = m["i_peak"], m["i_err"]
    ax.plot(t[ip], 1000 * u_ref[ip], "o", ms=7, color=INK, mfc="none", mew=1.4, zorder=5)
    ax.annotate("reference peak", (t[ip], 1000 * u_ref[ip]), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=INK2)
    extra = f" · peak ratio max|u|/max|u$_{{ref}}$| = {m['peak_ratio']:.3f}" if abs(m["peak_ratio"] - 1) > 1e-3 else ""
    ax.set_title(f"{title}\nε$_{{max}}$ = {m['eps_max']:.2g} (orange ■, t = {t[ie]:.2f} s) · "
                 f"ε$_{{peak}}$ = {m['eps_peak']:.2g} (○, t = {t[ip]:.2f} s){extra}", loc="left", fontsize=10)
    ax2 = fig.add_subplot(sub[1], sharex=ax)
    e = 1000 * (u - u_ref)
    ex = int(np.floor(np.log10(np.abs(e).max())))
    sc = 10.0 ** ex if abs(ex) >= 3 else 1.0
    ax2.plot(t, e / sc, color=INK2, lw=0.9)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.plot(t[ie], e[ie] / sc, "s", ms=6, color=ORANGE, zorder=5)
    ax2.plot(t[ip], e[ip] / sc, "o", ms=7, color=INK, mfc="none", mew=1.4, zorder=5)
    unit = f"[10$^{{{ex}}}$ mm]" if sc != 1.0 else "[mm]"
    ax2.set_ylabel(f"u − u$_{{ref}}$\n{unit}")
    ax2.margins(y=0.15)
    return ax2, m


def verification(quick):
    from conftest import np_record_interp
    from pynite_ref import build_pynite, modal_exact_piecewise_linear, pynite_reduced_matrices

    import Newmark  # legacy

    metrics = {}
    fig = plt.figure(figsize=(8.5, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.38)

    # (a) legacy NumPy vs JAX (same forcing)
    frame, mp = three_story_frame(2.5)
    Ka, Ma = frame.matrices(**element_props(frame, mp))
    n = frame.n_active
    rec, dt = sj.load_kern_county(legacy_drop_first=True)
    C = 0.3 * Ma + 1e-3 * Ka
    N = 2000
    Mr = np.asarray(Ma) @ np.ones(n)
    ag = np_record_interp(rec, dt)
    z = np.zeros(n)
    uL = Newmark.Newmark_beta(0.5, 0.25, np.asarray(Ka), np.asarray(Ma), np.asarray(C), lambda t: -Mr * ag(t), z, z, 0, N, dt)[0].T
    uJ = np.asarray(sj.newmark(Ka, Ma, C, sj.ground_motion_load(Ma, jnp.ones(n), rec, dt), jnp.zeros(n), jnp.zeros(n), dt, N, a0=jnp.zeros(n))[0])
    k = frame.dof(ROOF_NODE, 0)
    t = np.arange(N + 1) * dt
    ax_a, m = _compare_panel(fig, gs[0], t, uL[:, k], uJ[:, k], "legacy NumPy 2021 (reference)", "JAX port", MUTED, BLUE,
                             "(a) Forward parity with the original solver (same forcing)")
    m["eps_max_all_dofs"] = float(np.abs(uL - uJ).max() / np.abs(uL).max())  # what the parity tests assert on
    metrics["a_legacy_vs_jax_roof"] = m
    write_history_csv(
        "verification_a_legacy_vs_jax.csv",
        "Panel (a): roof (node 11) horizontal displacement [mm]. 3-story frame, h_mesh = 2.5 m, Kern County record "
        "(first sample dropped, legacy), influence r = ones (legacy), C = 0.3 M + 1e-3 K, Newmark gamma=0.5 beta=0.25, "
        "h = dt = 0.005 s, a0 = 0. u_legacy: Structure/TimeIntegration NumPy code; u_jax: structdyn_jax.newmark.",
        t, {"u_legacy_mm": 1000 * uL[:, k], "u_jax_mm": 1000 * uJ[:, k], "error_jax_minus_legacy_mm": 1000 * (uJ[:, k] - uL[:, k])})

    # (b) PyNite-assembled matrices + exact modal solution
    model = build_pynite(frame, mp)
    Kp, Mp = pynite_reduced_matrices(model, frame)
    rec, dt = sj.load_kern_county()
    T = 10.0
    nr = int(round(T / dt))
    r = np.asarray(frame.influence(0))
    u_ref = modal_exact_piecewise_linear(Kp, Mp, 0.05, r, rec, dt, nr)
    w2, Phi = generalized_eig(Ka, Ma)
    Cm = Ma @ Phi @ jnp.diag(0.1 * jnp.sqrt(w2)) @ Phi.T @ Ma
    u = np.asarray(sj.newmark(Ka, Ma, Cm, sj.ground_motion_load(Ma, r, rec, dt), jnp.zeros(n), jnp.zeros(n), dt / 4, 4 * nr)[0])[::4]
    t = np.arange(nr + 1) * dt
    ax_b, m = _compare_panel(fig, gs[1], t, u_ref[:, k], u[:, k], "PyNite K, M + exact modal solution (reference)",
                             "JAX Newmark, h = dt/4", MUTED, BLUE, "(b) Independent check with PyNite")
    metrics["b_pynite_modal_vs_jax_roof"] = m
    write_history_csv(
        "verification_b_pynite_vs_jax.csv",
        "Panel (b): roof (node 11) horizontal displacement [mm]. 3-story frame, h_mesh = 2.5 m, full Kern County record, "
        "horizontal influence vector, 5 % classical modal damping in every mode. u_pynite_modal: Nigam-Jennings exact modal "
        "superposition (all modes) on PyNite-assembled K, M. u_jax: structdyn_jax.newmark, h = dt/4 (every 4th step written).",
        t, {"u_pynite_modal_mm": 1000 * u_ref[:, k], "u_jax_newmark_mm": 1000 * u[:, k], "error_jax_minus_pynite_mm": 1000 * (u[:, k] - u_ref[:, k])})

    # (c) SAP2000 trace from the original project
    t_sap, d_sap = sj.loads.load_sap2000_reference()
    frame5, mp5 = three_story_frame(0.5)
    K5, M5 = frame5.matrices(**element_props(frame5, mp5))
    n5 = frame5.n_active
    rec, dt = sj.load_kern_county(legacy_drop_first=True)
    z5 = jnp.zeros(n5)
    u5 = np.asarray(sj.newmark(K5, M5, 0 * K5, sj.ground_motion_load(M5, jnp.ones(n5), rec, dt), z5, z5, dt, 4000, a0=z5)[0])
    roof5 = u5[:, frame5.dof(ROOF_NODE, 0)]
    ax_c, m = _compare_panel(fig, gs[2], t_sap, d_sap, roof5, "SAP2000, original project (reference)", "JAX port, legacy set-up",
                             ORANGE, BLUE, "(c) SAP2000 benchmark (phase drifts after ~5 s; see VERIFICATION.md §3)")
    m["eps_max_first_5s"] = float(np.abs(roof5 - d_sap)[:1001].max() / np.abs(d_sap).max())
    m["corr_20s"] = float(np.corrcoef(roof5, d_sap)[0, 1])
    metrics["c_sap2000_vs_jax_roof"] = m
    write_history_csv(
        "verification_c_sap2000_vs_jax.csv",
        "Panel (c): roof (node 11) horizontal displacement [mm]. Legacy set-up: 3-story frame, h_mesh = 0.5 m, undamped, "
        "Kern County record (first sample dropped), influence r = ones, Newmark h = dt = 0.005 s, a0 = 0. "
        "u_sap2000: SAP2000 modal superposition from the original project (Structure/Disp.xlsx).",
        t_sap, {"u_sap2000_mm": 1000 * d_sap, "u_jax_mm": 1000 * roof5, "error_jax_minus_sap2000_mm": 1000 * (roof5 - d_sap)})
    ax_c.set_xlabel("time [s]")

    fig.text(0.0, 0.065,
             "All metrics use the roof (node 11) horizontal displacement shown in the panel, over the whole plotted record:\n"
             "ε$_{max}$ = max$_t$|u − u$_{ref}$| / max$_t$|u$_{ref}$|  (largest error at any time ■, divided by the reference peak).\n"
             "ε$_{peak}$ = |u − u$_{ref}$| at t$_{peak}$ / max$_t$|u$_{ref}$|, where t$_{peak}$ = argmax$_t$|u$_{ref}$|  "
             "(error at the instant of the reference peak ○).\n"
             "Time histories: docs/figures/data/verification_{a,b,c}_*.csv",
             fontsize=8.5, color=INK2, va="top")
    fig.savefig(OUT / "verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
def sensitivity(quick):
    prob = EarthquakeProblem(h_mesh=2.5, duration=8.0 if quick else 15.0)
    p = zero_params()
    f = jax.jit(prob.rms_roof)
    J0 = float(f(p))
    g = jax.grad(f)(p)
    g = {k: np.asarray(v) / J0 for k, v in g.items()}  # relative: d ln J / d ln(param)

    # FD check of every entry (Richardson)
    fd = {k: v / J0 for k, v in finite_difference_grad(f, p).items()}
    ad_all = np.concatenate([np.ravel(g[k]) for k in sorted(g)])
    fd_all = np.concatenate([np.ravel(fd[k]) for k in sorted(fd)])
    max_rel = float(np.abs(ad_all - fd_all).max() / np.abs(fd_all).max())

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.0, 1.1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    s = g["log_EI"]
    lim = np.abs(s).max()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    segs = frame_segments(sj.model.THREE_STORY_NODES, sj.model.THREE_STORY_MEMBERS)
    lc = LineCollection(segs, cmap=DIVERGING, norm=norm, linewidths=8, capstyle="round")
    lc.set_array(s)
    ax.add_collection(lc)
    for e, seg in enumerate(segs):
        (x0, y0), (x1, y1) = seg
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if abs(x1 - x0) > abs(y1 - y0):  # beam: label below
            ax.text(mx, my - 0.8, f"{s[e]:+.2f}", fontsize=8.5, color=INK, ha="center", va="top")
            continue
        my = min(y0, y1) + 0.3 * abs(y1 - y0)  # columns: label low, clear of the beam labels
        if mx < 15:  # left-half column: label to the left
            ax.text(mx - 0.8, my, f"{s[e]:+.2f}", fontsize=8.5, color=INK, ha="right", va="center")
        else:  # right-half column: label to the right
            ax.text(mx + 0.8, my, f"{s[e]:+.2f}", fontsize=8.5, color=INK, ha="left", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 35)
    ax.set_ylim(-1, 16.5)
    ax.axis("off")
    ax.set_title("Sensitivity of J to each member's EI\n(∂ ln J / ∂ ln EI, one reverse-mode pass)")
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.02, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)

    ax2 = fig.add_subplot(gs[1])
    names = ["story height", "damping ζ", "all EI", "all mass"]
    vals = [float(g["log_height"]), float(g["log_zeta"]), float(g["log_EI"].sum()), float(g["log_m"].sum())]
    ax2.barh(names, vals, color=[BLUE if v < 0 else ORANGE for v in vals], height=0.55)
    ax2.axvline(0, color=INK2, lw=1)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax2.set_title("Global elasticities of J\n(scale a whole group by 1 %)")
    ax2.grid(axis="y", visible=False)
    span = max(abs(min(vals)), abs(max(vals))) * 1.6
    ax2.set_xlim(-span, span)

    ax3 = fig.add_subplot(gs[2])
    rel = np.abs(ad_all - fd_all) / np.maximum(np.abs(fd_all), 1e-300)
    ax3.scatter(np.arange(len(rel)), rel, s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax3.axhline(1e-5, color=ORANGE, lw=1.2)
    ax3.text(len(rel) - 1, 1.6e-5, "test tolerance 1e-5", color=INK2, fontsize=8.5, ha="right")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-13, 1e-3)
    ax3.set_xlabel("gradient entry (28 design parameters)")
    ax3.set_ylabel("|AD − FD| / |FD|")
    ax3.set_title("Gradient check vs finite differences\n(Richardson-extrapolated)")
    fig.text(0.01, -0.03, f"J = RMS roof displacement, Kern County record, first {prob.duration:.0f} s, 5 % Rayleigh damping, h = 2.5 m mesh. "
             "Orange = increasing the parameter increases J.", color=INK2, fontsize=8.5)
    fig.savefig(OUT / "sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(J0_rms_roof_m=J0, grad_vs_fd_max_rel=max_rel, dlnJ_dln={k: np.round(v, 4).tolist() for k, v in g.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    metrics = {}
    metrics["animation"] = animation(args.quick)
    metrics["verification"] = verification(args.quick)
    metrics["sensitivity"] = sensitivity(args.quick)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
