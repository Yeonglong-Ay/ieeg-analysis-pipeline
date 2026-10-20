#!/usr/bin/env python
# stage4_region_summary.py
"""
STAGE 4: region-level summary of the PETHs (band power).

Groups the 167 bipolar channels into major brain regions (lumping the fine
FreeSurfer labels), then averages each channel's PETH within region to give a
per-region response for each band and event. Pure white-matter and Unknown
channels are excluded from the region means (but reported).

This is computationally trivial — it just averages the already-computed Stage 2
PETHs by region. Produces presentation-ready per-region figures.

Region grouping (coarse, from FSLabel):
  temporal, frontal, orbitofrontal, insula, cingulate, hippocampus,
  amygdala, thalamus  (white_matter / unknown -> excluded)

Bipolar channel region = region of the ANODE contact (first of the pair). A
channel is excluded if BOTH contacts are white-matter/unknown.

Inputs: Stage 1 anatomy (bipolar_anatomy.csv), Stage 2 PETHs (peth_<event>.npz).
Outputs (in --out-dir):
  region_assignment.csv       each bipolar channel -> region
  region_peth_<event>.png     per-region PETH, all bands, per event
  region_counts.txt

Usage:
  python stage4_region_summary.py --stage1-dir stage1_mne_out \
      --stage2-dir stage2_mne_out --out-dir stage4_out
"""
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
EVENTS = ['bet_submitted', 'color_submitted', 'feedback']


def region_of(fslabel):
    """Map a fine FreeSurfer label to a coarse region. Returns None to exclude
    (white matter / unknown)."""
    s = str(fslabel).lower()
    if 'white-matter' in s or 'unknown' in s:
        return None
    if 'hippocamp' in s:
        return 'hippocampus'
    if 'amygdala' in s:
        return 'amygdala'
    if 'thalamus' in s:
        return 'thalamus'
    if 'insula' in s or 'insular' in s or 'ins_lg' in s:
        return 'insula'
    if 'cingul' in s:
        return 'cingulate'
    if 'orbital' in s:
        return 'orbitofrontal'
    if 'front' in s or 'subcentral' in s:
        return 'frontal'
    if ('temp' in s or 'fusifor' in s or 'collat' in s or 'oc-temp' in s
            or 'lat_fis' in s or 'transv' in s):
        return 'temporal'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', default='stage2_mne_out')
    ap.add_argument('--out-dir', default='stage4_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    # region from the anode contact; exclude if BOTH contacts are wm/unknown
    reg_anode = anat['anode_region'].map(region_of)
    reg_cath = anat['cathode_region'].map(region_of)
    region = reg_anode.where(reg_anode.notna(), reg_cath)   # fall back to cathode
    anat['region'] = region
    anat[['bipolar_name', 'anode_region', 'cathode_region', 'region']].to_csv(
        os.path.join(args.out_dir, 'region_assignment.csv'), index=False)

    counts = anat['region'].value_counts(dropna=False)
    with open(os.path.join(args.out_dir, 'region_counts.txt'), 'w') as f:
        f.write("Channels per region (None = excluded white-matter/unknown):\n")
        f.write(counts.to_string())
    print("Channels per region:")
    print(counts.to_string())

    # order of channels in the PETH arrays matches anat row order (Stage 1/2)
    for ev in EVENTS:
        d = np.load(os.path.join(args.stage2_dir, f'peth_{ev}.npz'),
                    allow_pickle=True)
        peth = d['peth_mean']            # [chan x band x time]
        t = d['t']
        names = list(d['bipolar_name'])
        # sanity: array channel order should match anat order
        # (both come from the same Stage 1 montage table)
        n_ch = peth.shape[0]
        regions = anat['region'].values[:n_ch]

        uniq = [r for r in ['orbitofrontal', 'insula', 'cingulate',
                            'hippocampus', 'amygdala', 'thalamus',
                            'temporal', 'frontal', 'other']
                if r in set(regions)]
        # figure: rows = regions, cols = bands; each cell = region-mean PETH
        fig, axes = plt.subplots(len(uniq), len(BANDS),
                                 figsize=(2.4*len(BANDS), 1.9*len(uniq)),
                                 squeeze=False)
        for ri, reg in enumerate(uniq):
            sel = np.where(regions == reg)[0]
            for bi, band in enumerate(BANDS):
                ax = axes[ri, bi]
                # region mean +/- SEM across channels in the region
                chans = peth[sel, bi, :]                # [n_reg_chan x time]
                chans = chans[~np.all(np.isnan(chans), axis=1)]
                if len(chans):
                    m = np.nanmean(chans, axis=0)
                    sem = np.nanstd(chans, axis=0) / np.sqrt(len(chans))
                    ax.plot(t, m, color=f'C{bi}', lw=1.2)
                    ax.fill_between(t, m-sem, m+sem, color=f'C{bi}', alpha=0.25)
                ax.axvline(0, color='k', ls='--', lw=0.7)
                ax.axhline(0, color='grey', ls=':', lw=0.5)
                if ri == 0:
                    ax.set_title(band, fontsize=9)
                if bi == 0:
                    ax.set_ylabel(f"{reg}\n(n={len(sel)})", fontsize=7)
        for bi in range(len(BANDS)):
            axes[-1, bi].set_xlabel('t (s)', fontsize=8)
        fig.suptitle(f'Region-level PETH — {ev}', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'region_peth_{ev}.png'), dpi=140)
        plt.close(fig)
        print(f"  saved region_peth_{ev}.png")

    print(f"\nStage 4 done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
