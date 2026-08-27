r"""
Generate paper figures from saved .npz checkpoints.

Serves BOTH model families — tabular Q-learning and IPPO — because they emit
identical metric keys and identical checkpoint filenames. Only the directory
differs:

    # Q-learning  -> plots/v3
    python plot_from_checkpoints.py --checkpoint-dir checkpoints_qlearning --outdir plots/v3

    # IPPO        -> plots/v4_final
    python plot_from_checkpoints.py --checkpoint-dir checkpoints_ablation --outdir plots/v4_final

Typography
----------
Figures are drawn wider than they are printed, and LaTeX's `width=` shrinks all
text with the image:

    effective_pt = source_pt * (print_width / source_width)

So fonts are set to `target_pt / scale`, making text land at TARGET_PT once the
figure is scaled into the page. `--print-width` / `--panel-print-width` declare
the widths the figures will occupy; `--audit` reports the resulting print sizes
and how much headroom each figure has above the 7pt floor.

Defaults follow the AAAI 2026 style (aaai2026.sty): \textwidth = 7.0in and
\columnwidth = (7.0 - 0.375)/2 = 3.3125in. Figure sizes are unchanged from the
originals, so the page budget is unaffected.
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
# Line style is the second, colour-independent encoding.
ST = {"SRB":  dict(ls="--", lw=2.2),
      "ES":   dict(ls=":",  lw=2.4),
      "DPF":  dict(ls="-.", lw=2.0),
      "DERL": dict(ls="-",  lw=2.8),
      "AC":   dict(ls="-.", lw=2.0)}
# Hatching gives bars the redundant encoding that line style gives curves.
HATCH = ["", "///", "\\\\\\", "xxx"]

# AAAI 2026 page geometry (aaai2026.sty), US Letter with 0.75in side margins
TEXT_WIDTH   = 7.0                              # \textwidth
COLUMN_SEP   = 0.375                            # \columnsep
COLUMN_WIDTH = (TEXT_WIDTH - COLUMN_SEP) / 2    # 3.3125in

TARGET_PT = 10.0   # body / axis labels / ticks, once printed
LEGEND_PT = 9.0    # legend, once printed
MIN_PT    = 7.0    # reviewer's floor (\scriptsize)

PANEL_LEGENDS = False   # False: strip the per-panel legend, emit legend_*.pdf instead

_K = 1.0           # current font multiplier, set by sty()
_AUDIT = []        # (figure, source_w, print_w) collected for --audit


def sty(source_w, print_w):
    """Style for a figure `source_w` inches wide that will print at `print_w`."""
    global _K
    scale = print_w / float(source_w)
    _K = 1.0 / scale
    k = _K
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':          TARGET_PT * k,
        'axes.titlesize':     TARGET_PT * k,
        'axes.labelsize':     TARGET_PT * k,
        'xtick.labelsize':    TARGET_PT * k,
        'ytick.labelsize':    TARGET_PT * k,
        'legend.fontsize':    LEGEND_PT * k,
        'axes.linewidth':     0.8 * k,
        'grid.linewidth':     0.6 * k,
        'xtick.major.width':  0.8 * k,
        'ytick.major.width':  0.8 * k,
        'xtick.major.size':   3.5 * k,
        'ytick.major.size':   3.5 * k,
        'patch.linewidth':    0.6 * k,
        'legend.handlelength': 2.6,
        'figure.dpi':         150,
    })
    return scale


def st(c):
    """Condition line style with the linewidth scaled to survive downscaling."""
    d = dict(ST.get(c, dict(ls="-", lw=2.0)))
    d["lw"] = d.get("lw", 2.0) * _K
    return d


def _record(name, source_w, print_w):
    # _K is the multiplier the fonts were actually set with (from the nominal
    # figsize); actual_w is filled in after saving, since tight bbox trims.
    _AUDIT.append(dict(name=name, nominal_w=source_w, print_w=print_w,
                       k=_K, actual_w=source_w))


def _actual_width(fname):
    """Width in inches of the file just written (tight bbox trims the nominal figsize)."""
    from PIL import Image
    with Image.open(f"{fname}.png") as im:
        return im.width / 300.0


_LEG_KW = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               columnspacing=1.2, handletextpad=0.5, borderaxespad=0.3)


def _legend(ax, ncol=2):
    """
    With PANEL_LEGENDS the legend sits above the axes (never over the data).
    Otherwise it is dropped and the canvas loses exactly the height the legend
    would have occupied — removing it alone saves nothing, because tight_layout
    simply expands the axes to refill a fixed figsize.
    """
    fig = ax.get_figure()
    if PANEL_LEGENDS:
        ax.legend(ncol=ncol, **_LEG_KW)
        return
    if getattr(fig, "_legend_dropped", False):
        return                      # one shrink per figure, not per panel
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(ncol=ncol, **_LEG_KW)
        fig.canvas.draw()
        band = leg.get_window_extent().height / fig.dpi
        leg.remove()
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, max(h - band, h * 0.5))
    fig._legend_dropped = True


def _legend_strip(entries, fname, print_w):
    """
    Standalone one-row key, drawn at its printed width so its text is exactly
    LEGEND_PT with no rescaling. Include once above a composite.
    entries: list of (label, artist).
    """
    sty(print_w, print_w)           # k = 1, so the text is exactly LEGEND_PT
    fig = plt.figure(figsize=(print_w, 1.0))
    leg = fig.legend([e[1] for e in entries], [e[0] for e in entries],
                     loc="center", ncol=len(entries), frameon=False,
                     columnspacing=1.6, handletextpad=0.6, handlelength=2.6)
    fig.canvas.draw()
    band = leg.get_window_extent().height / fig.dpi
    # keep the full print width (no tight bbox) so the strip is placed 1:1 and
    # its text is not rescaled; trim only the height down to the legend itself
    fig.set_size_inches(print_w, band * 1.3)
    for e in ["pdf", "png"]:
        fig.savefig(f"{fname}.{e}", format=e, dpi=300)
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
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
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
    ax.plot(eps, mean, label=label, color=color, **kwargs)
    ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15, lw=0)


def _save(fig, fname):
    for e in ["pdf", "png"]:
        fig.savefig(f"{fname}.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)
    for rec in _AUDIT:                     # record the width actually written
        if rec["name"] == fname:
            rec["actual_w"] = _actual_width(fname)
            break


def _curve(R, key, ylabel, fname, panel_w, n_ep, conds=None):
    """Single-panel figure. No title — the LaTeX caption carries that."""
    source_w = 6.5
    sty(source_w, panel_w)
    _record(fname, source_w, panel_w)
    fig, ax = plt.subplots(figsize=(source_w, 4))
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
    print(f"\n  {n_saved} figures (PDF + PNG) saved → {od}/")
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

    # ── Fig 5: emergent social dynamics, full-width 3-panel strip ─────────
    source_w = 15.0
    sty(source_w, print_w); _record("fig05_social", source_w, print_w)
    fig, axes = plt.subplots(1, 3, figsize=(source_w, 4))
    for c in ["DPF", "DERL"]:
        if c not in R:
            continue
        plot_with_fill(axes[0], R[c]["punish"],   label=LB[c], color=CL[c], **st(c))
        plot_with_fill(axes[1], R[c]["verify"],   label=LB[c], color=CL[c], **st(c))
        plot_with_fill(axes[2], R[c]["mean_rep"], label=LB[c], color=CL[c], **st(c))
    for a, yl in zip(axes, ["Punishment Rate", "Verification Rate",
                            "Mean Reputation"]):
        a.set_ylabel(yl)
        _episode_axis(a, n_ep)
        _fit_y(a)
        _legend(a, ncol=2)
    fig.tight_layout()
    _save(fig, "fig05_social")
    n_saved += 1

    # ── Fig 6: cooperation + oracle accuracy, full-width 2-panel ──────────
    source_w = 11.0
    sty(source_w, print_w); _record("fig06_coop_oracle", source_w, print_w)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(source_w, 4))
    for c in MC:
        if c not in R:
            continue
        plot_with_fill(a1, R[c]["coop"],       label=LB[c], color=CL[c], **st(c))
        plot_with_fill(a2, R[c]["oracle_acc"], label=LB[c], color=CL[c], **st(c))
    for a, yl in zip((a1, a2), ["Cooperation Rate", "Oracle Accuracy"]):
        a.set_ylabel(yl)
        _episode_axis(a, n_ep)
        _fit_y(a)
        _legend(a, ncol=2)
    fig.tight_layout()
    _save(fig, "fig06_coop_oracle")
    n_saved += 1

    # ── Fig 7: cumulative reward ──────────────────────────────────────────
    source_w = 6.5
    sty(source_w, panel_w); _record("fig07_reward", source_w, panel_w)
    fig, ax = plt.subplots(figsize=(source_w, 4))
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
        source_w = 9.0
        sty(source_w, print_w); _record("fig09_ablation_punish", source_w, print_w)
        fig, ax = plt.subplots(figsize=(source_w, 4.5))
        x  = np.arange(len(mets))
        bw = 0.18
        cm = plt.cm.viridis
        n_last = max(1, abl[keys[0]]["truth"].shape[1] // 5)
        for i, k in enumerate(keys):
            vals = [np.mean(abl[k][m][:, -n_last:]) for m in mets]
            ax.bar(x + (i - len(keys) / 2 + 0.5) * bw, vals, bw,
                   label=f"pun_rew={k}",
                   color=cm(i / max(len(keys) - 1, 1)),
                   hatch=HATCH[i % len(HATCH)],
                   edgecolor="white", linewidth=0.6 * _K)
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
        source_w = 14.0
        sty(source_w, print_w); _record("fig10_collusion", source_w, print_w)
        # Panel A carries four category labels, so it needs more width than the
        # two curve panels or "Gather"/"Mine" collide at print size.
        fig, axes = plt.subplots(1, 3, figsize=(source_w, 4),
                                 gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
        n_last = max(1, R["DPF"]["truth"].shape[1] // 5)
        s = slice(-n_last, None)
        ed, cd = R["DPF"], R["AC"]
        ms = ["truth", "lie", "gather", "mine"]
        ls = ["Truth", "Lie", "Gather", "Mine"]
        xp = np.arange(4)
        ev = [np.mean(ed[m][:, s]) for m in ms]
        cv = [np.mean(cd[m][:, s]) for m in ms]
        axes[0].bar(xp - 0.17, ev, 0.32, label="DPF", color=CL["DPF"],
                    hatch="", edgecolor="white", linewidth=0.6 * _K)
        axes[0].bar(xp + 0.17, cv, 0.32, label="AC", color=CL["AC"],
                    hatch="///", edgecolor="white", linewidth=0.6 * _K)
        axes[0].set_xticks(xp); axes[0].set_xticklabels(ls)
        axes[0].set(ylabel="Rate"); _legend(axes[0], ncol=2)
        plot_with_fill(axes[1], ed["coop"], "DPF", CL["DPF"], **st("DPF"))
        plot_with_fill(axes[1], cd["coop"], "AC",  CL["AC"],  **st("AC"))
        axes[1].set_ylabel("Cooperation Rate")
        _episode_axis(axes[1], n_ep); _fit_y(axes[1])
        _legend(axes[1], ncol=2)
        plot_with_fill(axes[2], ed["oracle_acc"], "DPF", CL["DPF"], **st("DPF"))
        plot_with_fill(axes[2], cd["oracle_acc"], "AC",  CL["AC"],  **st("AC"))
        axes[2].set_ylabel("Oracle Accuracy")
        _episode_axis(axes[2], n_ep); _fit_y(axes[2])
        _legend(axes[2], ncol=2)
        fig.tight_layout()
        _save(fig, "fig10_collusion")
        n_saved += 1

    if not PANEL_LEGENDS:
        n_saved += _draw_legend_strips(R, print_w)

    return n_saved


def _draw_legend_strips(R, print_w):
    """
    One key per distinct series set, to be included once above its figure:
      legend_conditions -> figs 1-4, 6, 7, 8   (Figures 2, 4, 5)
      legend_dpf_derl   -> fig05               (Figure 3)
      legend_dpf_ac     -> fig10               (Figure 7)
      legend_ablation   -> fig09               (Figure 6)
    """
    def line(c):
        return Line2D([0], [0], color=CL[c], **ST[c])

    n = 0
    conds = [c for c in MC if c in R]
    if conds:
        _legend_strip([(LB[c], line(c)) for c in conds],
                      "legend_conditions", print_w); n += 1
    pair = [c for c in ("DPF", "DERL") if c in R]
    if len(pair) == 2:
        _legend_strip([(LB[c], line(c)) for c in pair],
                      "legend_dpf_derl", print_w); n += 1
    if "DPF" in R and "AC" in R:
        _legend_strip([(LB[c], line(c)) for c in ("DPF", "AC")],
                      "legend_dpf_ac", print_w); n += 1
    if "abl_pr" in R:
        keys = sorted(R["abl_pr"].keys())
        cm = plt.cm.viridis
        ents = [(f"pun_rew={k}",
                 Patch(facecolor=cm(i / max(len(keys) - 1, 1)),
                       hatch=HATCH[i % len(HATCH)], edgecolor="white"))
                for i, k in enumerate(keys)]
        _legend_strip(ents, "legend_ablation", print_w); n += 1
    print(f"  {n} legend strip(s) written")
    return n


def print_audit():
    """Effective printed point sizes, and headroom above the 7pt floor."""
    print(f"\n{'='*88}")
    print(f"  FONT AUDIT — target {TARGET_PT:.0f}pt body / {LEGEND_PT:.0f}pt legend, "
          f"floor {MIN_PT:.0f}pt")
    print(f"{'='*88}")
    print(f"  {'figure':<26}{'saved':>8}{'print':>8}"
          f"{'in file':>9}{'-> body':>9}{'-> leg':>8}{'floor at':>10}  ok")
    print(f"  {'-'*82}")
    ok_all = True
    for r in _AUDIT:
        src_body = TARGET_PT * r["k"]          # what is embedded in the file
        src_leg  = LEGEND_PT * r["k"]
        ratio    = r["print_w"] / r["actual_w"]
        eff_body = src_body * ratio            # what it measures on the page
        eff_leg  = src_leg  * ratio
        floor_w  = r["actual_w"] * MIN_PT / src_leg
        ok = eff_leg >= MIN_PT and eff_body >= MIN_PT
        ok_all &= ok
        print(f"  {r['name']:<26}{r['actual_w']:>7.1f}\"{r['print_w']:>7.1f}\""
              f"{src_body:>9.1f}{eff_body:>9.1f}{eff_leg:>8.1f}"
              f"{floor_w:>9.1f}\"  {'yes' if ok else 'NO'}")
    print(f"  {'-'*82}")
    print(f"  All figures clear the {MIN_PT:.0f}pt floor: {'YES' if ok_all else 'NO'}")
    lo = min(r["print_w"] / r["actual_w"] * LEGEND_PT * r["k"] for r in _AUDIT)
    hi = max(r["print_w"] / r["actual_w"] * TARGET_PT * r["k"] for r in _AUDIT)
    print(f"  Body text lands at {hi:.1f}pt; smallest legend lands at {lo:.1f}pt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--outdir",         default="plots/v4_ppo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--print-width", type=float, default=TEXT_WIDTH,
                        help=f"printed width (in) of full-width figures "
                             f"(default {TEXT_WIDTH}in = AAAI \\textwidth)")
    parser.add_argument("--panel-print-width", type=float, default=COLUMN_WIDTH,
                        help=f"printed width (in) of one panel of a composite "
                             f"(default {COLUMN_WIDTH}in = AAAI \\columnwidth)")
    parser.add_argument("--inline-legends", action="store_true",
                        help="keep a legend on every panel instead of emitting "
                             "standalone legend_*.pdf strips")
    parser.add_argument("--audit", action="store_true",
                        help="report effective printed point sizes")
    args = parser.parse_args()

    PANEL_LEGENDS = args.inline_legends

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
