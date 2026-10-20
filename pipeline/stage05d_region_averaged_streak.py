#!/usr/bin/env python
# stage5d_region_averaged_streak.py
"""
STAGE 5d: region-AVERAGED streak-length slopes, with the coarse frontal cortex
split into subregions.

Regions (each averaged): amygdala, insula, hippocampus, thalamus, temporal,
orbitofrontal, cingulate  — PLUS the coarse 'frontal' split into
superior_frontal, middle_frontal, inferior_frontal. (Orbital and cingulate stay
as their own regions; only the leftover 'frontal' is subdivided, so no overlap.)

Averaging method: compute each CHANNEL's streak-length slope (OLS of power_z on
streak_len 1-4), then AVERAGE the channel slopes within each region. This avoids
pooling correlated channels as independent (no inflation). Significance is a
one-sample t-test of the channel slopes vs 0 (only when >=3 channels).

Epoch = bet_submitted. Loss and win streaks. Reads Stage 2 per-trial power +
behaviour.

Outputs:
  region_averaged_slopes.csv      per region x band x direction: mean slope, t, p, n
  region_averaged_slopes.png      figure: regions (rows) x bands (cols), mean
                                  power vs streak length, loss (red) vs win (blue)

HONEST NOTE: channel counts vary widely (e.g. inferior_frontal ~1, cingulate ~3
vs temporal ~63). Small regions are underpowered — counts are reported; do not
over-read regions with <~4 channels.

Usage:
  python stage5d_region_averaged_streak.py --stage2-dir stage2_mne_fix \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv --out-dir stage5d_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
EVENT = 'bet_submitted'
MAX_STREAK = 4
# display order (subregions of frontal placed together)
REGION_ORDER = ['orbitofrontal', 'superior_frontal', 'middle_frontal',
                'inferior_frontal', 'cingulate', 'insula', 'amygdala',
                'hippocampus', 'thalamus', 'temporal']


def assign_region(fslabel):
    """Coarse region, BUT split the leftover frontal into sub-gyri.
    Orbital and cingulate remain their own regions."""
    s = str(fslabel).lower()
    if 'white-matter' in s or 'unknown' in s: return None
    if 'hippocamp' in s: return 'hippocampus'
    if 'amygdala' in s: return 'amygdala'
    if 'thalamus' in s: return 'thalamus'
    if 'insula' in s or 'insular' in s or 'ins_lg' in s: return 'insula'
    if 'cingul' in s: return 'cingulate'
    if 'orbital' in s: return 'orbitofrontal'
    # frontal sub-gyri (the split)
    if 'front_sup' in s: return 'superior_frontal'
    if 'front_middle' in s: return 'middle_frontal'
    if 'front_inf' in s: return 'inferior_frontal'
    if 'front' in s or 'subcentral' in s: return 'frontal_other'
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return None


def reconstruct_order(event_csvs):
    dfs = []
    for i, c in enumerate(event_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    ev = ev[(ev['matched'] == True) & (ev['trial_number'] > NUM_PRACTICE)]
    sub = ev[ev['phase'] == EVENT].reset_index(drop=True)
    sub['trial_index'] = np.arange(len(sub))
    sub['exp_trial'] = sub['trial_number'] - NUM_PRACTICE
    return sub[['trial_index', 'block', 'exp_trial']]


def load_behaviour(behav_csvs):
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    st = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_dir'] = st.str[0].map({'L': 'loss', 'W': 'win'})
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    b['streak_len'] = sl.where(sl.notna(), pd.to_numeric(st.str[1:], errors='coerce'))
    return b[['block', 'exp_trial', 'streak_dir', 'streak_len']]


def channel_slope(g):
    if len(g) < 6 or g['streak_len'].nunique() < 2:
        return np.nan
    return stats.linregress(g['streak_len'].astype(float), g['power_z']).slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_fix')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage5d_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(assign_region)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(assign_region)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)
    power = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')
    power['region'] = power['bipolar_name'].map(ch2region)

    regions_present = [r for r in REGION_ORDER if r in set(power['region'].dropna())]

    # ---- compute per-region averaged slopes + per-length means for plotting ----
    rows = []
    plotdata = {}   # (region,band,direction) -> (streak_lengths, mean_power, sem)
    for region in regions_present:
        rchans = anat.loc[anat['region'] == region, 'bipolar_name'].tolist()
        for band in BANDS:
            for direction in ['loss', 'win']:
                ch_slopes = []
                for chan in rchans:
                    g = power[(power['bipolar_name'] == chan) &
                              (power['band'] == band) &
                              (power['streak_dir'] == direction) &
                              (power['streak_len'].between(1, MAX_STREAK))]
                    sl = channel_slope(g)
                    if np.isfinite(sl):
                        ch_slopes.append(sl)
                t, p = ((stats.ttest_1samp(ch_slopes, 0))
                        if len(ch_slopes) >= 3 else (np.nan, np.nan))
                rows.append({'region': region, 'band': band,
                             'direction': direction, 'n_chan': len(rchans),
                             'n_slopes': len(ch_slopes),
                             'mean_slope': np.mean(ch_slopes) if ch_slopes else np.nan,
                             't': t, 'p': p})
                # for plotting: region-averaged mean power per streak length
                gg = power[(power['region'] == region) & (power['band'] == band) &
                           (power['streak_dir'] == direction) &
                           (power['streak_len'].between(1, MAX_STREAK))]
                if len(gg):
                    m = gg.groupby('streak_len')['power_z'].mean()
                    se = gg.groupby('streak_len')['power_z'].sem()
                    plotdata[(region, band, direction)] = (m.index.values,
                                                           m.values, se.values)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'region_averaged_slopes.csv'), index=False)
    print(f"saved region_averaged_slopes.csv")
    print("\nChannels per region:")
    for r in regions_present:
        print(f"  {r}: {df[df['region']==r]['n_chan'].iloc[0]}")

    # ---- figure: regions (rows) x bands (cols) ----
    nrow, ncol = len(regions_present), len(BANDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6*ncol, 1.8*nrow),
                             squeeze=False)
    for ri, region in enumerate(regions_present):
        for bi, band in enumerate(BANDS):
            ax = axes[ri, bi]
            for direction, col in [('loss', '#C44E52'), ('win', '#4C72B0')]:
                pd_key = (region, band, direction)
                if pd_key in plotdata:
                    x, m, se = plotdata[pd_key]
                    ax.errorbar(x, m, yerr=se, marker='o', color=col,
                                capsize=2, lw=1, markersize=3)
            ax.axhline(0, color='grey', ls=':', lw=0.5); ax.set_xticks([1,2,3,4])
            if ri == 0: ax.set_title(band, fontsize=9)
            if bi == 0:
                n = df[df['region']==region]['n_chan'].iloc[0]
                ax.set_ylabel(f"{region}\n(n={n})", fontsize=6)
    fig.suptitle('Region-averaged power vs streak length (bet submission) — '
                 'loss (red) vs win (blue); frontal split into sub-gyri',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'region_averaged_slopes.png'), dpi=140)
    plt.close(fig)
    print("saved region_averaged_slopes.png")
    print(f"\nStage 5d done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
