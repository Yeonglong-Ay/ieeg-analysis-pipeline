#!/usr/bin/env python
# stage6_spider.py
"""
Spider (radar) plots of band activity per epoch.

For each epoch (bet_submitted, color_submitted, feedback), one radar plot with
6 spokes = frequency bands (delta, theta, alpha, beta, low_gamma, high_gamma).
Each reward-relevant region is a polygon = its mean z-scored band power over the
analysis window. Lets you compare the spectral 'fingerprint' of each region
within each epoch.

Regions shown: orbitofrontal, insula, cingulate, amygdala, hippocampus,
thalamus, frontal  (temporal dropped — likely auditory/sensory).

Value per band = mean over channels-in-region of (mean z-scored log power over
the analysis window), from the Stage 2 per-trial power averaged across trials.
z-scored power can be negative (below baseline); the radial axis spans the true
min..max with the zero/baseline drawn as a reference circle, so signs are honest.

Usage:
  python stage6_spider.py --stage1-dir stage1_mne_out --stage2-dir stage2_mne_out \
      --out-dir stage6_out
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
EVENTS = ['bet_submitted', 'color_submitted', 'feedback']
REGIONS = ['orbitofrontal', 'insula', 'cingulate', 'amygdala',
           'hippocampus', 'thalamus', 'frontal']


def region_of(fslabel):
    s = str(fslabel).lower()
    if 'white-matter' in s or 'unknown' in s: return None
    if 'hippocamp' in s: return 'hippocampus'
    if 'amygdala' in s: return 'amygdala'
    if 'thalamus' in s: return 'thalamus'
    if 'insula' in s or 'insular' in s or 'ins_lg' in s: return 'insula'
    if 'cingul' in s: return 'cingulate'
    if 'orbital' in s: return 'orbitofrontal'
    if 'front' in s or 'subcentral' in s: return 'frontal'
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', default='stage2_mne_out')
    ap.add_argument('--out-dir', default='stage6_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    reg = anat['anode_region'].map(region_of)
    reg = reg.where(reg.notna(), anat['cathode_region'].map(region_of))
    anat['region'] = reg
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    # angles for the 6 spokes
    angles = np.linspace(0, 2*np.pi, len(BANDS), endpoint=False).tolist()
    angles += angles[:1]     # close the polygon

    for ev in EVENTS:
        # per-trial power -> mean per region x band
        p = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{ev}.csv'))
        p['region'] = p['bipolar_name'].map(ch2region)
        # region x band mean of power_z (over trials and channels in region)
        table = {}
        for reg_name in REGIONS:
            sub = p[p['region'] == reg_name]
            if len(sub) == 0:
                continue
            vals = [sub[sub['band'] == b]['power_z'].mean() for b in BANDS]
            table[reg_name] = vals

        # radial limits from the actual data (signed, honest)
        allvals = np.array([v for v in table.values()])
        vmin = np.nanmin(allvals); vmax = np.nanmax(allvals)
        pad = 0.1 * (vmax - vmin + 1e-9)
        rmin, rmax = vmin - pad, vmax + pad

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, (reg_name, vals) in enumerate(table.items()):
            v = vals + vals[:1]                     # close polygon
            ax.plot(angles, v, lw=1.8, label=f"{reg_name}", color=f'C{i}')
            ax.fill(angles, v, alpha=0.08, color=f'C{i}')
        # zero/baseline reference circle
        if rmin < 0 < rmax:
            zero = [0]*len(angles)
            ax.plot(angles, zero, color='grey', ls='--', lw=1, alpha=0.7)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(BANDS, fontsize=10)
        ax.set_ylim(rmin, rmax)
        ax.set_title(f'Band-power fingerprint by region — {ev}\n'
                     f'(mean z-scored power; dashed grey = baseline/0)',
                     fontsize=11, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'spider_{ev}.png'), dpi=140,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  saved spider_{ev}.png")

        # also save the underlying table
        pd.DataFrame(table, index=BANDS).to_csv(
            os.path.join(args.out_dir, f'spider_{ev}_values.csv'))

    print(f"\nDone. Spider plots in {args.out_dir}/")


if __name__ == '__main__':
    main()
