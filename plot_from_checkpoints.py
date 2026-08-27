"""
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
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

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

TARGET_PT = 10.0   # body / axis labels / ticks, once printed
LEGEND_PT = 9.0    # legend, once printed
MIN_PT    = 7.0    # reviewer's floor (\scriptsize)

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
    _AUDIT.append((name, source_w, print_w))


def _legend(ax, ncol=2):
    """
    Legend above the axes rather than inside it. At 10pt print size the legend
    block is large relative to a 3.4in panel, and inside the axes it covers the
    data; above, it costs height instead of hiding curves.
    """
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=ncol, frameon=False, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.3)


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


def _curve(R, key, ylabel, fname, panel_w, conds=None, ylim=None):
    """Single-panel figure. No title — the LaTeX caption carries that."""
    source_w = 6.5
    sty(source_w, panel_w)
    _record(fname, source_w, panel_w)
    fig, ax = plt.subplots(figsize=(source_w, 4))
    for c in (conds or MC):
        if c in R:
            plot_with_fill(ax, R[c][key], label=LB.get(c, c),
                           color=CL.get(c, "#000"), **st(c))
    ax.set(xlabel="Episode", ylabel=ylabel)
    if ylim:
        ax.set_ylim(ylim)
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


def make_plots(R, od, print_w=7.0, panel_w=3.4):
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

    # ── Figs 1-4, 7, 8: single panels, printed inside 2x2 composites ──────
    _curve(R, "truth",  "Truth Rate",  "fig01_epistemic",     panel_w, ylim=(-0.02, 0.65))
    _curve(R, "gather", "Gather Rate", "fig02_ethical",       panel_w, ylim=(-0.02, 1.0))
    _curve(R, "lie",    "Lie Rate",    "fig03_hallucination", panel_w, ylim=(-0.02, 1.0))
    _curve(R, "mine",   "Mine Rate",   "fig04_moral_drift",   panel_w, ylim=(-0.02, 0.5))
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
    axes[0].set(xlabel="Episode", ylabel="Punishment Rate")
    axes[1].set(xlabel="Episode", ylabel="Verification Rate")
    axes[2].set(xlabel="Episode", ylabel="Mean Reputation")
    for a in axes:
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
    a1.set(xlabel="Episode", ylabel="Cooperation Rate", ylim=(-0.02, 0.5))
    a2.set(xlabel="Episode", ylabel="Oracle Accuracy",  ylim=(-0.02, 1.05))
    _legend(a1, ncol=2); _legend(a2, ncol=2)
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
    ax.set(xlabel="Episode", ylabel="Cumulative Reward")
    # y only: an x-axis offset label would collide with "Episode" at print size,
    # and plain episode counts match the other panels.
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    _legend(ax, ncol=2)
    fig.tight_layout()
    _save(fig, "fig07_reward")
    n_saved += 1

    # ── Fig 8: resource sustainability ────────────────────────────────────
    mx = max(np.mean(R[c]["res"], axis=0).max() for c in MC if c in R)
    _curve(R, "res", "Active Resources", "fig08_resources", panel_w, ylim=(0, mx * 1.2))
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
        axes[1].set(xlabel="Episode", ylabel="Cooperation Rate", ylim=(-0.02, 0.5))
        _legend(axes[1], ncol=2)
        plot_with_fill(axes[2], ed["oracle_acc"], "DPF", CL["DPF"], **st("DPF"))
        plot_with_fill(axes[2], cd["oracle_acc"], "AC",  CL["AC"],  **st("AC"))
        axes[2].set(xlabel="Episode", ylabel="Oracle Accuracy", ylim=(-0.02, 1.05))
        _legend(axes[2], ncol=2)
        fig.tight_layout()
        _save(fig, "fig10_collusion")
        n_saved += 1

    return n_saved


def print_audit():
    """Effective printed point sizes, and headroom above the 7pt floor."""
    print(f"\n{'='*82}")
    print(f"  FONT AUDIT — target {TARGET_PT:.0f}pt body / {LEGEND_PT:.0f}pt legend, "
          f"floor {MIN_PT:.0f}pt")
    print(f"{'='*82}")
    print(f"  {'figure':<26}{'drawn':>8}{'print':>8}"
          f"{'src body':>10}{'src leg':>9}{'-> body':>9}{'-> leg':>8}"
          f"{'floor at':>10}  ok")
    print(f"  {'-'*76}")
    ok_all = True
    for name, sw, pw in _AUDIT:
        # what is literally embedded in the PDF
        src_body = TARGET_PT * sw / pw
        src_leg  = LEGEND_PT * sw / pw
        # what it measures once scaled to pw: src * (pw/sw) -> back to target
        eff_body = src_body * pw / sw
        eff_leg  = src_leg  * pw / sw
        # printed width at which the legend would hit the 7pt floor
        floor_w = pw * MIN_PT / LEGEND_PT
        ok = eff_leg >= MIN_PT and eff_body >= MIN_PT
        ok_all &= ok
        print(f"  {name:<26}{sw:>7.1f}\"{pw:>7.1f}\""
              f"{src_body:>10.1f}{src_leg:>9.1f}{eff_body:>9.1f}{eff_leg:>8.1f}"
              f"{floor_w:>9.1f}\"  {'yes' if ok else 'NO'}")
    print(f"  {'-'*76}")
    print(f"  All figures clear the {MIN_PT:.0f}pt floor: "
          f"{'YES' if ok_all else 'NO'}")
    print(f"  Legends stay >= {MIN_PT:.0f}pt as long as a figure is not printed "
          f"narrower than {MIN_PT/LEGEND_PT:.0%} of its stated width.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--outdir",         default="plots/v4_ppo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--print-width", type=float, default=7.0,
                        help="printed width (in) of the full-width multi-panel figures")
    parser.add_argument("--panel-print-width", type=float, default=3.4,
                        help="printed width (in) of a single panel inside a 2x2 composite")
    parser.add_argument("--audit", action="store_true",
                        help="report effective printed point sizes")
    args = parser.parse_args()

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
