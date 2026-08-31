r"""
Generate paper figures from saved .npz checkpoints.

Serves BOTH model families — tabular Q-learning and IPPO — because they emit
identical metric keys and identical checkpoint filenames. Only the directory
differs:

    # Q-learning  -> plots/v3
    python plot_from_checkpoints.py --checkpoint-dir checkpoints_qlearning \
        --outdir plots/v3 --layout column

    # IPPO        -> plots/v4_final
    python plot_from_checkpoints.py --checkpoint-dir checkpoints_ablation \
        --outdir plots/v4_final --layout column

Typography
----------
Every figure is drawn at exactly the size it occupies on the page — figsize ==
printed width — so LaTeX's `width=` is a 1:1 include and nothing is resampled:

    \includegraphics[width=\columnwidth]{fig01_epistemic}    % no height=

The point sizes below are therefore literal: the 8pt set here is the 8pt that
prints, and the 7pt legend prints at 7pt. Nothing is pre-compensated for a
downscale, because there is no downscale. `--audit` reports the printed sizes.

Include with `width=` ONLY. A `height=...` or `keepaspectratio` cap makes height
the binding constraint, shrinks the figure below 1:1, and takes the type under
the 7pt floor with it. `--print-width` / `--panel-print-width` declare the widths
the figures are drawn at and printed at.

Defaults follow the AAAI 2026 style (aaai2026.sty): \textwidth = 7.0in and
\columnwidth = (7.0 - 0.375)/2 = 3.3125in.

Output is vector PDF (Type 42 embedded fonts, never Type 3), so text stays
crisp at any zoom and the printer sets it rather than resampling a raster. It is
saved at the nominal figsize with NO tight bounding box: a tight bbox trims the
canvas, and LaTeX would then scale the trimmed file back up to `width=` and
enlarge the type with it. `tight_layout` already fits the content inside the
declared size, so the trim is not needed.

Printed WIDTHS are unchanged from the camera-ready originals — nothing here
shrinks a figure. Only the printed heights change (PANEL_PRINT_H and friends),
because height is the axis along which overlapping curves separate, and the
previous panels gave four curves just 1.4in of it.

Figures 05, 06, 09 and 10 are composites, and --layout says which environment
the paper puts them in:

    --layout column   `figure`  -> 3.3125in, panels stacked one per row  (paper)
    --layout wide     `figure*` -> 7.0in, panels side by side

The paper uses plain `figure`, so `column` is what it needs. Building them wide
and letting LaTeX squeeze 6.90in into a 3.28in column is a 0.48x scale that puts
8pt type at 3.8pt, well under the 7pt floor. Stacked, each panel gets the full
column at 1:1.

NOTE: the stacked figures are taller than the caps in the paper's
\includegraphics. Any `height=...,keepaspectratio` on these four must be
dropped, or height binds instead of width and shrinks them further than before.
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CONDITIONS = ["SRB", "ES", "DPF", "DERL", "AC"]
ABL_PRS    = [0.0, 0.5, 1.5, 3.0]
MC         = ["SRB", "ES", "DPF", "DERL"]

# Colours are spread across the grayscale range so the series stay separable
# when printed without colour. BT.601 luma of each: DPF 48, DERL 85, SRB 126,
# ES 160 — roughly 35+ apart, versus 5 apart for the original DPF/DERL pair.
# ES is the lightest but stays under 165 so its dotted line holds up on white.
CL = {"SRB": "#E66100", "ES": "#A0A0A0",
      "DPF": "#3B1F6B", "DERL": "#1B7837", "AC": "#D62728"}
LB = {"SRB":  "SRB",
      "ES":   "ES",
      "DPF":  "DPF",
      "DERL": "DERL (Ours)",
      "AC":   "AC"}
# Line style is the second, colour-independent encoding. `lw` and `dashes` are
# in PRINTED points and, at 1:1, are used as written.
#
# Every dash period is a different length, which is what keeps series apart
# where they coincide exactly — SRB and ES both sit at 0.0 for most of figs 1-4,
# and identical periods would stack into one indistinguishable line. Widths are
# ~1.3pt rather than the old 2.6-2.8pt: at that weight two curves 0.005 apart
# on a rate axis were touching before their means were.
ST = {"SRB":  dict(ls="--", lw=1.3, dashes=(4.5, 2.0)),
      "ES":   dict(ls=":",  lw=1.5, dashes=(1.0, 1.8)),
      "DPF":  dict(ls="-.", lw=1.3, dashes=(6.0, 1.8, 1.2, 1.8)),
      "DERL": dict(ls="-",  lw=1.7),
      "AC":   dict(ls="-.", lw=1.3, dashes=(3.0, 1.6, 1.0, 1.6))}
# Later = drawn on top. DERL is the contribution, so it is never buried.
ZO = {"SRB": 2.0, "ES": 2.1, "AC": 2.2, "DPF": 2.3, "DERL": 2.4}
# Hatching gives bars the redundant encoding that line style gives curves.
HATCH = ["", "///", "\\\\\\", "xxx"]

# AAAI 2026 page geometry (aaai2026.sty), US Letter with 0.75in side margins
TEXT_WIDTH   = 7.0                              # \textwidth
COLUMN_SEP   = 0.375                            # \columnsep
COLUMN_WIDTH = (TEXT_WIDTH - COLUMN_SEP) / 2    # 3.3125in

# Printed point sizes. Drawn 1:1, so these are literal: what is set here is
# what measures on the page. 8pt is the AAAI floor for readable axis type and
# 7pt (\scriptsize) is the reviewer's hard floor, which the legend sits on.
TARGET_PT = 8.0    # axis labels, tick labels, panel titles
LEGEND_PT = 7.0    # legends and the standalone legend strips
MIN_PT    = 7.0    # reviewer's floor (\scriptsize)

# Printed HEIGHT of the drawing area, in inches. Widths are fixed by the column
# geometry above and are not touched; height is the only free axis, and it is
# the axis the curves separate along. The previous version printed a single
# panel at 3.31 x 1.43in (2.3:1) — four curves inside 1.4in of vertical room,
# which is what made them read as one band. These give ~1.4:1 instead.
PANEL_PRINT_H  = 2.35   # figs 1-4, 7, 8: one column-wide panel
STRIP3_PRINT_H = 3.30   # figs 5, 10: full-width 3-panel strip
STRIP2_PRINT_H = 3.60   # fig 6: full-width 2-panel
BARS_PRINT_H   = 3.20   # fig 9: full-width bar chart

# --layout column: figs 5, 6, 9, 10 rebuilt to sit in ONE \columnwidth. The
# paper puts them in `figure` (not `figure*`) environments, so a 1x3 strip is
# squeezed from 6.90in into 3.28in and its 10pt type lands at 4.8pt. Stacking
# the panels instead gives each one the full column width at 1:1.
COL_STRIP3_H = 3.60     # fig 5: 3 curve panels stacked, one shared x-axis
COL_FIG10_H  = 3.90     # fig 10: bar panel + 2 curve panels, two x-axes
COL_STRIP2_H = 2.80     # fig 6: 2 panels stacked
COL_BARS_H   = 2.60     # fig 9: horizontal bars

LAYOUT = "wide"         # "wide" (figure*, 7.0in) | "column" (figure, 3.3125in)

FILL_ALPHA = 0.10  # +/-1 sigma band; 0.15 turned overlapping bands into fog

# EVERY figure is drawn 1:1 — figsize == printed width, so scale == 1 and
# nothing is resampled on the way into the page. Figures used to be drawn
# 6.5-15in wide and squeezed down by LaTeX. Fonts and linewidths were
# compensated for that squeeze, but the properties matplotlib does not express
# in scaled units were not: hatch strokes are a fixed 1.0pt, so fig09's and
# fig10's bar hatching printed at 0.5-0.8pt and read as grey wash rather than
# as texture. At 1:1 there is nothing left to compensate.

FORMATS = ["pdf"]  # vector only: text stays crisp and the type is set, not
                   # resampled. Add "png" here for a quick on-screen look.

PANEL_LEGENDS = False   # False: strip the per-panel legend, emit legend_*.pdf instead

_AUDIT = []        # (figure, width, height) collected for --audit


def sty():
    """
    Style shared by every figure. Drawn 1:1, so every size below is in printed
    points and inches — there is no scale factor to divide out.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':          TARGET_PT,
        'axes.titlesize':     TARGET_PT,
        'axes.labelsize':     TARGET_PT,
        'xtick.labelsize':    TARGET_PT,
        'ytick.labelsize':    TARGET_PT,
        'legend.fontsize':    LEGEND_PT,
        'axes.linewidth':     0.8,
        'grid.linewidth':     0.6,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.major.size':   3.5,
        'ytick.major.size':   3.5,
        'patch.linewidth':    0.6,
        # matplotlib draws hatch strokes at a fixed width that ignores the
        # figure scale; at 1:1 the value asked for is the value printed
        'hatch.linewidth':    0.9,
        'legend.handlelength': 2.6,
        'figure.dpi':         150,
        # embed real outline fonts. matplotlib's default (Type 3) is rejected
        # by most camera-ready checkers and renders poorly in some viewers
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        # the default whitegrid rules compete with 1.3pt curves for attention;
        # keep them as a readable ruler, not as foreground
        'grid.color':         '#B0B0B0',
        'grid.alpha':         0.55,
        'axes.axisbelow':     True,
    })


def st(c):
    """Condition line style. Widths and dashes are already in printed points."""
    d = dict(ST.get(c, dict(ls="-", lw=1.3)))
    d["zorder"] = ZO.get(c, 2.0)
    d.setdefault("solid_capstyle", "round")
    d.setdefault("dash_capstyle", "round")
    return d


def stacked():
    """True when the \\textwidth composites are being rebuilt for one column."""
    return LAYOUT == "column"


def composite(name, nax, print_w, wide_h, col_h, width_ratios=None, sharex=True):
    """
    Build one of the four composites that the paper typesets as a unit.

    wide   -> 1 x nax across \\textwidth
    column -> nax x 1 stacked inside \\columnwidth

    Stacked panels share an x-axis so only the bottom one pays for tick labels
    — but only when every panel IS the same axis. Pass sharex=False when one of
    them is categorical (fig10's bar panel), or its four category positions get
    forced onto the 0..50000 episode scale.

    Both layouts are drawn 1:1, so nothing is resampled into the page.
    Returns (fig, axes).
    """
    pw = COLUMN_WIDTH if stacked() else print_w
    ph = col_h if stacked() else wide_h
    sty()
    _record(name, pw, ph)
    if stacked():
        fig, axes = plt.subplots(nax, 1, figsize=(pw, ph), sharex=sharex)
    else:
        fig, axes = plt.subplots(1, nax, figsize=(pw, ph),
                                 gridspec_kw=(dict(width_ratios=width_ratios)
                                              if width_ratios else None))
    return fig, np.atleast_1d(axes)


def panel_label(ax, text):
    """
    Name a panel's metric. Stacked panels are ~0.9in tall and a rotated
    "Verification Rate" is 1.05in long at 10pt, so it would overrun the axes;
    a left-aligned title above the panel costs less height than it wastes.
    """
    if stacked():
        ax.set_title(text, loc="left", pad=2.0)
    else:
        ax.set_ylabel(text)


def episode_axis_for(axes, n_ep, shared=True):
    """
    x-axis furniture. Side by side every panel gets it. Stacked, only the last
    one does — unless the panels are not actually sharing an axis, in which case
    each still needs its own labels.
    """
    for a in (axes[-1:] if (stacked() and shared) else axes):
        _episode_axis(a, n_ep)


def _record(name, w, h):
    # saved with no tight bbox, so the figsize IS the size of the file and the
    # size on the page — nothing to measure back off the written image
    _AUDIT.append(dict(name=name, w=w, h=h))


_LEG_KW = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               columnspacing=1.2, handletextpad=0.5, borderaxespad=0.3)


def _legend(ax, ncol=2):
    """
    With PANEL_LEGENDS the legend sits above the axes (never over the data).
    Otherwise it is simply not drawn: the figure heights above are already the
    printed heights we want for the axes, so reclaiming the legend band would
    undo exactly the vertical room this change is adding. (It also drove the
    old fig05 canvas down to 3.1in, short enough that "Punishment Rate" ran off
    the top of the image.)
    """
    if PANEL_LEGENDS:
        ax.legend(ncol=ncol, **_LEG_KW)


def _legend_strip(entries, fname, print_w):
    """
    Standalone key, drawn at the width of the figure it sits above so its text
    is exactly LEGEND_PT with no rescaling. Include once above that figure, at
    the same `width=` — a strip built for \\textwidth and dropped into a
    \\columnwidth slot lands at 3.3pt.

    One row if the entries fit; otherwise as few rows as do. Nothing here is
    saved with a tight bbox, so an overflowing row would be clipped at the
    figure edge rather than spilling visibly.
    entries: list of (label, artist).
    """
    sty()                           # 1:1, so the text is exactly LEGEND_PT
    fig = plt.figure(figsize=(print_w, 1.0))
    n = len(entries)
    for rows in range(1, n + 1):
        ncol = -(-n // rows)        # ceil: fill each row before starting another
        leg = fig.legend([e[1] for e in entries], [e[0] for e in entries],
                         loc="center", ncol=ncol, frameon=False,
                         columnspacing=1.6, handletextpad=0.6, handlelength=2.6)
        fig.canvas.draw()
        box = leg.get_window_extent()
        if box.width / fig.dpi <= print_w or rows == n:
            break
        leg.remove()
    band = box.height / fig.dpi
    # keep the full print width (no tight bbox) so the strip is placed 1:1 and
    # its text is not rescaled; trim only the height down to the legend itself
    fig.set_size_inches(print_w, band * 1.3)
    _record(fname, print_w, band * 1.3)
    for e in FORMATS:
        fig.savefig(f"{fname}.{e}", format=e)
    plt.close(fig)


def _episode_axis(ax, n_ep):
    """
    Label the x-axis in units of 10^k so ticks stay short (0..5 instead of
    0..50000). The factor goes in the axis label rather than a corner offset,
    which would collide with the label at print size.
    """
    exp = int(np.floor(np.log10(max(n_ep, 1)))) if n_ep > 0 else 0
    if exp < 3:
        ax.set_xlabel("Episode")
        return
    div = 10.0 ** exp
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/div:g}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel(rf"Episode ($\times 10^{{{exp}}}$)")


def _fit_y(ax):
    """
    Fit the y-range to what is actually drawn instead of a fixed limit.
    Hardcoded limits left `truth` and `mine` using only a third of their panel,
    which squeezed the curves together; fitting spreads them out.
    """
    ax.margins(y=0.09)
    ax.autoscale_view()
    # fitting the range can leave only two ticks; keep enough to read values off
    # (6 rather than 5 — the taller panels have room for the extra gridline, and
    # a denser ruler is what lets a reader resolve two curves that run close).
    # Stacked panels are a third of that height and 6 labels overlap into an
    # unreadable stack, so they get 4.
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4 if stacked() else 6, steps=[1, 2, 2.5, 5, 10]))
    lo, hi = ax.get_ylim()
    if lo < 0:
        # these metrics are all non-negative: keep a sliver below zero so a
        # flat-at-zero series is still visible, but no misleading negative span
        ax.set_ylim(-0.025 * (hi - lo), hi)


def sm(x_2d, w=100):
    if x_2d.ndim == 1:
        x_2d = x_2d.reshape(1, -1)
    mean_val = np.mean(x_2d, axis=0)
    std_val  = np.std(x_2d,  axis=0)
    w_dyn = max(w, int(x_2d.shape[1] * 0.02))
    if len(mean_val) < w_dyn:
        return mean_val, std_val * 0.6
    window = np.hanning(w_dyn); window /= window.sum()
    pad = (w_dyn // 2, w_dyn - w_dyn // 2 - 1)
    smooth_mean = np.convolve(np.pad(mean_val, pad, mode='edge'), window, 'valid')
    smooth_std  = np.convolve(np.pad(std_val,  pad, mode='edge'), window, 'valid')
    return smooth_mean, smooth_std * 0.6


def plot_with_fill(ax, data, label, color, **kwargs):
    mean, std = sm(data)
    eps = np.arange(len(mean))
    # every band sits under every line, so a wide band never hides a mean
    ax.fill_between(eps, mean - std, mean + std, color=color,
                    alpha=FILL_ALPHA, lw=0, zorder=1.0)
    ax.plot(eps, mean, label=label, color=color, **kwargs)


def _save(fig, fname):
    """
    Write at the nominal figsize. No bbox_inches="tight": trimming would make
    the file narrower than the column, LaTeX's `width=` would scale it back up,
    and the type would print larger than it was set. tight_layout has already
    fitted the content inside the declared size.
    """
    for e in FORMATS:
        fig.savefig(f"{fname}.{e}", format=e)
    plt.close(fig)


def _curve(R, key, ylabel, fname, panel_w, n_ep, conds=None):
    """Single-panel figure. No title — the LaTeX caption carries that."""
    sty()
    _record(fname, panel_w, PANEL_PRINT_H)
    fig, ax = plt.subplots(figsize=(panel_w, PANEL_PRINT_H))
    for c in (conds or MC):
        if c in R:
            plot_with_fill(ax, R[c][key], label=LB.get(c, c),
                           color=CL.get(c, "#000"), **st(c))
    ax.set_ylabel(ylabel)
    _episode_axis(ax, n_ep)
    _fit_y(ax)
    _legend(ax, ncol=2)
    fig.tight_layout()
    _save(fig, fname)


def load_checkpoints(ckpt_dir, seeds):
    R = {}

    for nm in CONDITIONS:
        seed_results = []
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"{nm}_seed{s}.npz")
            if os.path.exists(ckpt):
                data = np.load(ckpt)
                seed_results.append({k: data[k] for k in data.files})
            else:
                print(f"  WARNING: {os.path.basename(ckpt)} missing")
        if seed_results:
            R[nm] = {k: np.vstack([r[k] for r in seed_results]) for k in seed_results[0]}
            print(f"  {nm}: {len(seed_results)}/{len(seeds)} seeds")
        else:
            print(f"  {nm}: NO checkpoints found, skipping")

    abl_pr = {}
    for pr in ABL_PRS:
        seed_results = []
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"abl_pr{pr}_seed{s}.npz")
            if os.path.exists(ckpt):
                data = np.load(ckpt)
                seed_results.append({k: data[k] for k in data.files})
        if seed_results:
            abl_pr[pr] = {k: np.vstack([r[k] for r in seed_results]) for k in seed_results[0]}
            print(f"  ablation pr={pr}: {len(seed_results)}/{len(seeds)} seeds")
    if abl_pr:
        R["abl_pr"] = abl_pr
        print(f"  ablation: {len(abl_pr)}/4 punish_reward values loaded")
    else:
        print("  ablation: no checkpoints found — fig 9 will be skipped")

    return R


def make_plots(R, od, print_w=TEXT_WIDTH, panel_w=COLUMN_WIDTH):
    os.makedirs(od, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(od)
    try:
        n_saved = _draw_all(R, print_w, panel_w)
    finally:
        os.chdir(orig_dir)
    print(f"\n  {n_saved} figures ({' + '.join(FORMATS)}) saved → {od}/")
    return n_saved


def _draw_all(R, print_w, panel_w):
    n_saved = 0
    n_ep = R[next(c for c in CONDITIONS if c in R)]["truth"].shape[1]

    # ── Figs 1-4, 7, 8: single panels, printed inside 2x2 composites ──────
    _curve(R, "truth",  "Truth Rate",  "fig01_epistemic",     panel_w, n_ep)
    _curve(R, "gather", "Gather Rate", "fig02_ethical",       panel_w, n_ep)
    _curve(R, "lie",    "Lie Rate",    "fig03_hallucination", panel_w, n_ep)
    _curve(R, "mine",   "Mine Rate",   "fig04_moral_drift",   panel_w, n_ep)
    n_saved += 4

    # ── Fig 5: emergent social dynamics, 3 panels ─────────────────────────
    fig, axes = composite("fig05_social", 3, print_w,
                          STRIP3_PRINT_H, COL_STRIP3_H)
    for c in ["DPF", "DERL"]:
        if c not in R:
            continue
        plot_with_fill(axes[0], R[c]["punish"],   label=LB[c], color=CL[c], **st(c))
        plot_with_fill(axes[1], R[c]["verify"],   label=LB[c], color=CL[c], **st(c))
        plot_with_fill(axes[2], R[c]["mean_rep"], label=LB[c], color=CL[c], **st(c))
    for a, yl in zip(axes, ["Punishment Rate", "Verification Rate",
                            "Mean Reputation"]):
        panel_label(a, yl)
        _fit_y(a)
        _legend(a, ncol=2)
    episode_axis_for(axes, n_ep)
    fig.tight_layout()
    _save(fig, "fig05_social")
    n_saved += 1

    # ── Fig 6: cooperation + oracle accuracy, 2 panels ────────────────────
    fig, axes = composite("fig06_coop_oracle", 2, print_w,
                          STRIP2_PRINT_H, COL_STRIP2_H)
    a1, a2 = axes
    for c in MC:
        if c not in R:
            continue
        plot_with_fill(a1, R[c]["coop"],       label=LB[c], color=CL[c], **st(c))
        plot_with_fill(a2, R[c]["oracle_acc"], label=LB[c], color=CL[c], **st(c))
    for a, yl in zip(axes, ["Cooperation Rate", "Oracle Accuracy"]):
        panel_label(a, yl)
        _fit_y(a)
        _legend(a, ncol=2)
    episode_axis_for(axes, n_ep)
    fig.tight_layout()
    _save(fig, "fig06_coop_oracle")
    n_saved += 1

    # ── Fig 7: cumulative reward ──────────────────────────────────────────
    sty(); _record("fig07_reward", panel_w, PANEL_PRINT_H)
    fig, ax = plt.subplots(figsize=(panel_w, PANEL_PRINT_H))
    for c in MC:
        if c not in R:
            continue
        ax.plot(np.cumsum(np.mean(R[c]["reward"], axis=0)),
                label=LB[c], color=CL[c], **st(c))
    ax.set_ylabel("Cumulative Reward")
    _episode_axis(ax, n_ep)
    _fit_y(ax)
    # y only: an x-axis offset label would collide with "Episode" at print size,
    # and plain episode counts match the other panels.
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    _legend(ax, ncol=2)
    fig.tight_layout()
    _save(fig, "fig07_reward")
    n_saved += 1

    # ── Fig 8: resource sustainability ────────────────────────────────────
    _curve(R, "res", "Active Resources", "fig08_resources", panel_w, n_ep)
    n_saved += 1

    # ── Fig 9: punishment-profitability ablation ──────────────────────────
    if "abl_pr" in R:
        abl  = R["abl_pr"]
        keys = sorted(abl.keys())
        mets = ["truth", "gather", "lie", "mine", "punish"]
        mls  = [r"Truth $\uparrow$", r"Gather $\uparrow$",
                r"Lie $\downarrow$", r"Mine $\downarrow$", "Punish"]
        pw = COLUMN_WIDTH if stacked() else print_w
        ph = COL_BARS_H if stacked() else BARS_PRINT_H
        sty(); _record("fig09_ablation_punish", pw, ph)
        fig, ax = plt.subplots(figsize=(pw, ph))
        x  = np.arange(len(mets))
        bw = 0.18
        cm = plt.cm.viridis
        n_last = max(1, abl[keys[0]]["truth"].shape[1] // 5)
        for i, k in enumerate(keys):
            vals = [np.mean(abl[k][m][:, -n_last:]) for m in mets]
            off  = (i - len(keys) / 2 + 0.5) * bw
            args = dict(label=f"pun_rew={k}",
                        color=cm(i / max(len(keys) - 1, 1)),
                        hatch=HATCH[i % len(HATCH)],
                        edgecolor="white", linewidth=0.6)
            if stacked():
                # 20 vertical bars inside 3.31in leaves each 0.11in wide and the
                # five category labels overlapping; horizontally the column's
                # scarce axis is the one the labels do not compete for
                ax.barh(x[::-1] - off, vals, bw, **args)
            else:
                ax.bar(x + off, vals, bw, **args)
        if stacked():
            ax.set_yticks(x[::-1]); ax.set_yticklabels(mls)
            ax.set(xlabel=f"Rate (last {n_last} ep)")
        else:
            ax.set_xticks(x); ax.set_xticklabels(mls)
            ax.set(ylabel=f"Rate (last {n_last} ep)")
        _legend(ax, ncol=4)
        fig.tight_layout()
        _save(fig, "fig09_ablation_punish")
        n_saved += 1
        print(f"  fig09 generated ({len(keys)}/4 punish_reward values)")
    else:
        print("  fig09 SKIPPED — no abl_pr checkpoints in this directory")

    # ── Fig 10: adversarial coalition robustness ──────────────────────────
    if "DPF" in R and "AC" in R:
        # Side by side, panel A carries four category labels and needs more
        # width than the two curve panels or "Gather"/"Mine" collide. Stacked,
        # every panel already spans the column, so the ratios do not apply.
        fig, axes = composite("fig10_collusion", 3, print_w,
                              STRIP3_PRINT_H, COL_FIG10_H,
                              width_ratios=[1.35, 1, 1], sharex=False)
        n_last = max(1, R["DPF"]["truth"].shape[1] // 5)
        s = slice(-n_last, None)
        ed, cd = R["DPF"], R["AC"]
        ms = ["truth", "lie", "gather", "mine"]
        ls = ["Truth", "Lie", "Gather", "Mine"]
        xp = np.arange(4)
        ev = [np.mean(ed[m][:, s]) for m in ms]
        cv = [np.mean(cd[m][:, s]) for m in ms]
        axes[0].bar(xp - 0.17, ev, 0.32, label="DPF", color=CL["DPF"],
                    hatch="", edgecolor="white", linewidth=0.6)
        axes[0].bar(xp + 0.17, cv, 0.32, label="AC", color=CL["AC"],
                    hatch="///", edgecolor="white", linewidth=0.6)
        axes[0].set_xticks(xp); axes[0].set_xticklabels(ls)
        panel_label(axes[0], "Rate"); _legend(axes[0], ncol=2)
        plot_with_fill(axes[1], ed["coop"], "DPF", CL["DPF"], **st("DPF"))
        plot_with_fill(axes[1], cd["coop"], "AC",  CL["AC"],  **st("AC"))
        panel_label(axes[1], "Cooperation Rate")
        _fit_y(axes[1]); _legend(axes[1], ncol=2)
        plot_with_fill(axes[2], ed["oracle_acc"], "DPF", CL["DPF"], **st("DPF"))
        plot_with_fill(axes[2], cd["oracle_acc"], "AC",  CL["AC"],  **st("AC"))
        panel_label(axes[2], "Oracle Accuracy")
        _fit_y(axes[2]); _legend(axes[2], ncol=2)
        # Panel A is categorical, so the figure cannot share x globally. The two
        # curve panels are both episode axes though, so pair them by hand: one
        # set of tick labels and one "Episode" label instead of two.
        if stacked():
            axes[1].sharex(axes[2])
            axes[1].tick_params(labelbottom=False)
            episode_axis_for(axes[2:], n_ep, shared=False)
        else:
            episode_axis_for(axes[1:], n_ep, shared=False)
        fig.tight_layout()
        _save(fig, "fig10_collusion")
        n_saved += 1

    if not PANEL_LEGENDS:
        n_saved += _draw_legend_strips(R, print_w, panel_w)

    return n_saved


def _draw_legend_strips(R, print_w, panel_w):
    """
    One key per distinct series set, to be included once above its figure:
      legend_conditions -> figs 1-4, 6, 7, 8   (Figures 2, 4, 5)
      legend_dpf_derl   -> fig05               (Figure 3)
      legend_dpf_ac     -> fig10               (Figure 7)
      legend_ablation   -> fig09               (Figure 6)

    Each strip is drawn at the width of the figure it accompanies, so both are
    included at the same `width=` and both are 1:1. Stacked, every figure is one
    column wide; side by side, the composites span \\textwidth while the single
    panels stay in a column. legend_conditions serves both kinds, so it takes
    the panel width — the narrower of the two, which still centres above a
    \\textwidth figure rather than overrunning it.
    """
    def line(c):
        return Line2D([0], [0], color=CL[c], **ST[c])

    col = COLUMN_WIDTH if stacked() else None
    wide_w  = col or print_w        # strips above figs 5, 6, 9, 10
    panel_wd = col or panel_w       # strips above the single panels
    n = 0
    conds = [c for c in MC if c in R]
    if conds:
        _legend_strip([(LB[c], line(c)) for c in conds],
                      "legend_conditions", panel_wd); n += 1
    pair = [c for c in ("DPF", "DERL") if c in R]
    if len(pair) == 2:
        _legend_strip([(LB[c], line(c)) for c in pair],
                      "legend_dpf_derl", wide_w); n += 1
    if "DPF" in R and "AC" in R:
        _legend_strip([(LB[c], line(c)) for c in ("DPF", "AC")],
                      "legend_dpf_ac", wide_w); n += 1
    if "abl_pr" in R:
        keys = sorted(R["abl_pr"].keys())
        cm = plt.cm.viridis
        ents = [(f"pun_rew={k}",
                 Patch(facecolor=cm(i / max(len(keys) - 1, 1)),
                       hatch=HATCH[i % len(HATCH)], edgecolor="white"))
                for i, k in enumerate(keys)]
        _legend_strip(ents, "legend_ablation", wide_w); n += 1
    print(f"  {n} legend strip(s) written")
    return n


def print_audit():
    """Printed geometry and point sizes. Drawn 1:1, so set == printed."""
    print(f"\n{'='*74}")
    print(f"  FONT AUDIT — {TARGET_PT:.0f}pt axis type / {LEGEND_PT:.0f}pt legend, "
          f"floor {MIN_PT:.0f}pt, drawn 1:1")
    print(f"{'='*74}")
    print(f"  {'figure':<28}{'width':>9}{'height':>9}{'axis pt':>9}"
          f"{'leg pt':>8}  ok")
    print(f"  {'-'*68}")
    ok_all = TARGET_PT >= MIN_PT and LEGEND_PT >= MIN_PT
    for r in _AUDIT:
        print(f"  {r['name']:<28}{r['w']:>8.2f}\"{r['h']:>8.2f}\""
              f"{TARGET_PT:>9.1f}{LEGEND_PT:>8.1f}  "
              f"{'yes' if ok_all else 'NO'}")
    print(f"  {'-'*68}")
    print(f"  All figures clear the {MIN_PT:.0f}pt floor: {'YES' if ok_all else 'NO'}")
    print(f"  Include each at its width above with \\includegraphics[width=...]")
    print(f"  and NO height= / keepaspectratio, or the scale drops below 1:1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--outdir",         default="plots/v4_ppo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--print-width", type=float, default=TEXT_WIDTH,
                        help=f"width (in) full-width figures are drawn AND "
                             f"printed at (default {TEXT_WIDTH}in = AAAI "
                             f"\\textwidth)")
    parser.add_argument("--panel-print-width", type=float, default=COLUMN_WIDTH,
                        help=f"width (in) a single panel is drawn AND printed "
                             f"at (default {COLUMN_WIDTH}in = AAAI "
                             f"\\columnwidth)")
    parser.add_argument("--inline-legends", action="store_true",
                        help="keep a legend on every panel instead of emitting "
                             "standalone legend_*.pdf strips")
    parser.add_argument("--layout", choices=["wide", "column"], default="wide",
                        help="geometry for figs 5, 6, 9, 10: 'wide' for a "
                             "figure* spanning \\textwidth, 'column' to rebuild "
                             "them stacked inside one \\columnwidth (what a "
                             "plain `figure` environment gives them)")
    parser.add_argument("--audit", action="store_true",
                        help="report effective printed point sizes")
    args = parser.parse_args()

    PANEL_LEGENDS = args.inline_legends
    LAYOUT = args.layout

    print(f"Loading checkpoints from {args.checkpoint_dir} ...")
    R = load_checkpoints(args.checkpoint_dir, args.seeds)

    avail = [c for c in CONDITIONS if c in R]
    if not avail:
        print("No checkpoints found. Run the training script first.")
        raise SystemExit(1)

    print(f"\nGenerating plots → {args.outdir} ...")
    make_plots(R, args.outdir, args.print_width, args.panel_print_width)

    if args.audit:
        print_audit()
