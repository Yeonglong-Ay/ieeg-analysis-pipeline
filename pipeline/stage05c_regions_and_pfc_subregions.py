#!/usr/bin/env python
# stage5c_regions_and_pfc_subregions.py
"""
STAGE 5c: extends the streak-length analysis two ways.

PART 1 — PER-CHANNEL streak plots for multiple REWARD REGIONS (not just
amygdala). Same idea as Stage 5b, applied to each region: one figure per
region, grid of rows=channels x cols=6 bands, power vs streak length (loss red,
win blue). Reveals within-region heterogeneity for each region.

PART 2 — PFC SUBREGION analysis (AVERAGED per subregion). Breaks the frontal
cortex into fine FreeSurfer subregions (superior/middle/inferior frontal,
orbital, anterior cingulate) and computes the streak-length slope averaged
across channels within each subregion, per band, loss vs win.

Epoch = bet_submitted. Reads Stage 2 per-trial power + behaviour.

HONEST NOTE: subregions with very few channels (e.g. inferior frontal ~1) are
underpowered — the script reports channel counts; interpret small subregions
cautiously. Per-channel p-values are uncorrected across the many channels.

Usage:
  python stage5c_regions_and_pfc_subregions.py --stage2-dir stage2_mne_fix \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv --out-dir stage5c_out
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
# reward regions to do PER-CHANNEL (coarse grouping)
REWARD_REGIONS = ['orbitofrontal', 'insula', 'cingulate', 'amygdala',
                  'hippocampus', 'thalamus']


def coarse_region(fslabel):
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
    return None


def pfc_subregion(fslabel):
    """Fine PFC subregion from the FreeSurfer label (None if not PFC)."""
    s = str(fslabel).lower()
    if 'front_sup' in s: return 'superior_frontal'
    if 'front_middle' in s: return 'middle_frontal'
    if 'front_inf' in s: return 'inferior_frontal'
    if 'orbital' in s: return 'orbitofrontal'
    if 'cingul' in s: return 'cingulate'
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


def slope_for(sub):
    """OLS slope of power_z on streak_len; returns (slope,p,n) or (nan,nan,n)."""
    if len(sub) < 6 or sub['streak_len'].nunique() < 2:
        return np.nan, np.nan, len(sub)
    res = stats.linregress(sub['streak_len'].astype(float), sub['power_z'])
    return res.slope, res.pvalue, len(sub)


def plot_perchannel(power, channels, region, out_dir):
    nrow, ncol = len(channels), len(BANDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7*ncol, 2.0*nrow),
                             squeeze=False)
    for ri, chan in enumerate(channels):
        for bi, band in enumerate(BANDS):
            ax = axes[ri, bi]
            g = power[(power['bipolar_name'] == chan) & (power['band'] == band)]
            for direction, col in [('loss', '#C44E52'), ('win', '#4C72B0')]:
                s = g[(g['streak_dir'] == direction) &
                      (g['streak_len'].between(1, MAX_STREAK))]
                if len(s) < 6 or s['streak_len'].nunique() < 2:
                    continue
                m = s.groupby('streak_len')['power_z'].mean()
                se = s.groupby('streak_len')['power_z'].sem()
                ax.errorbar(m.index, m.values, yerr=se.values, marker='o',
                            color=col, capsize=2, lw=1)
                res = stats.linregress(s['streak_len'].astype(float), s['power_z'])
                ax.plot(m.index, res.intercept + res.slope*np.array(m.index),
                        color=col, ls='--', lw=0.8)
            ax.axhline(0, color='grey', ls=':', lw=0.5); ax.set_xticks([1,2,3,4])
            if ri == 0: ax.set_title(band, fontsize=9)
            if bi == 0: ax.set_ylabel(chan, fontsize=6)
    fig.suptitle(f'{region} PER-CHANNEL power vs streak length '
                 f'(bet submission) — loss (red) vs win (blue)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'perchannel_{region}.png'), dpi=140)
    plt.close(fig)
    print(f"  saved perchannel_{region}.png ({len(channels)} channels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_fix')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage5c_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    # coarse region (anode, fallback cathode)
    anat['region'] = anat['anode_region'].map(coarse_region)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(coarse_region)
    # fine PFC subregion
    anat['pfc_sub'] = anat['anode_region'].map(pfc_subregion)
    anat.loc[anat['pfc_sub'].isna(), 'pfc_sub'] = \
        anat.loc[anat['pfc_sub'].isna(), 'cathode_region'].map(pfc_subregion)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))
    ch2pfc = dict(zip(anat['bipolar_name'], anat['pfc_sub']))

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)
    power = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')
    power['region'] = power['bipolar_name'].map(ch2region)
    power['pfc_sub'] = power['bipolar_name'].map(ch2pfc)

    # ---- PART 1: per-channel plots for each reward region ----
    print("PART 1: per-channel streak plots by region")
    for region in REWARD_REGIONS:
        chans = anat.loc[anat['region'] == region, 'bipolar_name'].tolist()
        if len(chans) < 1:
            print(f"  (skip {region}: no channels)"); continue
        plot_perchannel(power[power['region'] == region], chans, region,
                        args.out_dir)

    # ---- PART 2: averaged streak slopes per PFC subregion ----
    print("\nPART 2: PFC subregion averaged slopes")
    rows = []
    for sub_name in ['superior_frontal', 'middle_frontal', 'inferior_frontal',
                     'orbitofrontal', 'cingulate']:
        sub_chans = anat.loc[anat['pfc_sub'] == sub_name, 'bipolar_name'].tolist()
        if not sub_chans:
            continue
        for band in BANDS:
            for direction in ['loss', 'win']:
                # channel-level slopes, then average (avoids channel pooling)
                ch_slopes = []
                for chan in sub_chans:
                    g = power[(power['bipolar_name'] == chan) &
                              (power['band'] == band) &
                              (power['streak_dir'] == direction) &
                              (power['streak_len'].between(1, MAX_STREAK))]
                    sl, p, n = slope_for(g)
                    if np.isfinite(sl):
                        ch_slopes.append(sl)
                if ch_slopes:
                    t, p = (stats.ttest_1samp(ch_slopes, 0)
                            if len(ch_slopes) >= 3 else (np.nan, np.nan))
                    rows.append({'subregion': sub_name, 'band': band,
                                 'direction': direction, 'n_chan': len(sub_chans),
                                 'n_slopes': len(ch_slopes),
                                 'mean_slope': np.mean(ch_slopes),
                                 't': t, 'p': p})
    pfc = pd.DataFrame(rows)
    pfc.to_csv(os.path.join(args.out_dir, 'pfc_subregion_slopes.csv'), index=False)
    print(f"  saved pfc_subregion_slopes.csv")
    # quick text summary (loss)
    if len(pfc):
        print("\n  PFC subregion channel counts:")
        for sn in pfc['subregion'].unique():
            print(f"    {sn}: {pfc[pfc['subregion']==sn]['n_chan'].iloc[0]} channels")

    print(f"\nStage 5c done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
