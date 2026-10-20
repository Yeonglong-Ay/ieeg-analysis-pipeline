#!/usr/bin/env python
# stage8_spider_epochs.py
"""
Spider/radar plot in the style of Overton et al. 2025 Figure 7D, adapted:
  - SPOKES = brain regions (reward-relevant set)
  - LINES  = the 5 task epochs (fixation, bet_onset, bet_submitted,
             color_submitted, feedback)
  - VALUE  = proportion of channels in that region that are TASK-ACTIVE
             (responsive vs baseline, FDR q<0.05) in that epoch

Shows which regions are engaged in which epoch. One spider per FREQUENCY BAND
(so you can see, e.g., high-gamma engagement across regions/epochs), plus an
'any-band' spider (channel counts if responsive in >=1 band).

Reads the per-event responsive_fraction CSVs from Stage 7
(responsive_fraction_<event>.csv) and the per-channel detail for 'any-band'.

Usage:
  python stage8_spider_epochs.py --stage7-dir stage7_out --out-dir stage8_out
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
EVENTS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
REGIONS = ['orbitofrontal', 'insula', 'cingulate', 'amygdala',
           'hippocampus', 'thalamus', 'frontal']


def radar(ax, regions, series_dict, title):
    """series_dict: {label: [value per region]}; regions on spokes."""
    n = len(regions)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for i, (label, vals) in enumerate(series_dict.items()):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, lw=1.8, label=label, color=f'C{i}')
        ax.fill(angles, v, alpha=0.06, color=f'C{i}')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10, pad=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage7-dir', default='stage7_out')
    ap.add_argument('--out-dir', default='stage8_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # load per-event fraction tables
    frac = {}
    for ev in EVENTS:
        f = os.path.join(args.stage7_dir, f'responsive_fraction_{ev}.csv')
        if os.path.exists(f):
            frac[ev] = pd.read_csv(f)
    present_regions = [r for r in REGIONS
                       if any(r in set(frac[ev]['region']) for ev in frac)]

    # ---- one spider per band: spokes=regions, lines=epochs, value=% responsive
    for band in BANDS:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        series = {}
        for ev in EVENTS:
            if ev not in frac:
                continue
            vals = []
            for r in present_regions:
                row = frac[ev][(frac[ev]['region'] == r) &
                               (frac[ev]['band'] == band)]
                vals.append(float(row['pct_resp_fdr'].iloc[0]) if len(row) else 0.0)
            series[ev] = vals
        radar(ax, present_regions, series,
              f'Task-active channel % by region and epoch — {band}')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'spider_epochs_{band}.png'),
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved spider_epochs_{band}.png")

    # ---- 'any-band' spider: channel responsive in >=1 band, per region/epoch
    any_series = {}
    for ev in EVENTS:
        chf = os.path.join(args.stage7_dir, f'responsive_channels_{ev}.csv')
        if not os.path.exists(chf):
            continue
        ch = pd.read_csv(chf)
        vals = []
        for r in present_regions:
            sub = ch[ch['region'] == r]
            if len(sub) == 0:
                vals.append(0.0); continue
            # per channel: responsive in any band?
            by_chan = sub.groupby('bipolar_name')['responsive_fdr'].any()
            vals.append(100.0 * by_chan.mean())
        any_series[ev] = vals
    if any_series:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        radar(ax, present_regions, any_series,
              'Task-active channel % (responsive in ANY band) by region and epoch')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'spider_epochs_anyband.png'),
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
        print("  saved spider_epochs_anyband.png")

    print(f"\nDone. Spiders in {args.out_dir}/")


if __name__ == '__main__':
    main()
