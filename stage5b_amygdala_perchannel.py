#!/usr/bin/env python
# stage5b_amygdala_perchannel.py
"""
STAGE 5b: PER-CHANNEL amygdala streak-length analysis (no channel averaging).

Stage 5 averaged across amygdala channels. This shows each amygdala channel
SEPARATELY, so within-region heterogeneity is visible — some channels may scale
with losing-streak length while others don't. This also avoids the channel-
pooling inflation of Stage 5 (each channel's slope uses only its own trials).

Output: one figure, grid of rows = amygdala channels, cols = 6 bands; each cell
plots mean band power vs streak length (1-4), loss (red) vs win (blue), with a
fitted slope printed. Epoch = bet_submitted.

Reads Stage 2 per-trial power (trial_power_bet_submitted.csv), Stage 1 anatomy
(bipolar_anatomy.csv), behaviour (pilot_results).

HONEST NOTE ON STATS: per-channel is cleaner than the pooled Stage 5 (no cross-
channel pseudo-replication), but now there are 6 channels x 6 bands x 2
directions = many tests — interpret individual-channel p-values with that in
mind (they are not corrected across channels here).

Usage:
  python stage5b_amygdala_perchannel.py --stage2-dir stage2_mne_fix \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv \
      --out-dir stage5b_out
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


def region_of(fslabel):
    s = str(fslabel).lower()
    if 'amygdala' in s: return 'amygdala'
    return None   # we only need amygdala here


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_fix')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage5b_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(region_of)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(region_of)
    amyg_channels = anat.loc[anat['region'] == 'amygdala', 'bipolar_name'].tolist()
    if not amyg_channels:
        raise SystemExit("No amygdala channels found in this patient.")
    print(f"Amygdala channels ({len(amyg_channels)}): {amyg_channels}")

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)

    power = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')
    # keep only amygdala channels
    power = power[power['bipolar_name'].isin(amyg_channels)]

    # figure: rows = channels, cols = bands
    nrow, ncol = len(amyg_channels), len(BANDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7*ncol, 2.0*nrow),
                             squeeze=False)
    for ri, chan in enumerate(amyg_channels):
        for bi, band in enumerate(BANDS):
            ax = axes[ri, bi]
            g = power[(power['bipolar_name'] == chan) & (power['band'] == band)]
            for direction, col in [('loss', '#C44E52'), ('win', '#4C72B0')]:
                sub = g[(g['streak_dir'] == direction) &
                        (g['streak_len'].between(1, MAX_STREAK))]
                if len(sub) < 6 or sub['streak_len'].nunique() < 2:
                    continue
                # mean +/- SEM per streak length
                m = sub.groupby('streak_len')['power_z'].mean()
                se = sub.groupby('streak_len')['power_z'].sem()
                ax.errorbar(m.index, m.values, yerr=se.values, marker='o',
                            color=col, capsize=2, label=direction, lw=1)
                # fitted slope
                res = stats.linregress(sub['streak_len'].astype(float),
                                       sub['power_z'])
                ax.plot(m.index, res.intercept + res.slope*np.array(m.index),
                        color=col, ls='--', lw=0.8)
                # annotate slope + p (small)
                ytxt = 0.92 - (0.10 if direction == 'win' else 0)
                ax.text(0.03, ytxt, f"{direction[0]}: {res.slope:+.3f} "
                        f"(p={res.pvalue:.2f})", transform=ax.transAxes,
                        fontsize=5.5, color=col, va='top')
            ax.axhline(0, color='grey', ls=':', lw=0.5)
            ax.set_xticks([1, 2, 3, 4])
            if ri == 0:
                ax.set_title(band, fontsize=9)
            if bi == 0:
                ax.set_ylabel(chan, fontsize=6)
            if ri == nrow-1:
                ax.set_xlabel('streak length', fontsize=7)
    fig.suptitle('Amygdala PER-CHANNEL band power vs streak length '
                 '(bet submission) — loss (red) vs win (blue)', fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out_dir, 'amygdala_perchannel_streak.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")

    # also save a table of per-channel slopes
    rows = []
    for chan in amyg_channels:
        for band in BANDS:
            g = power[(power['bipolar_name'] == chan) & (power['band'] == band)]
            row = {'channel': chan, 'band': band}
            for direction in ['loss', 'win']:
                sub = g[(g['streak_dir'] == direction) &
                        (g['streak_len'].between(1, MAX_STREAK))]
                if len(sub) >= 6 and sub['streak_len'].nunique() >= 2:
                    res = stats.linregress(sub['streak_len'].astype(float),
                                           sub['power_z'])
                    row[f'{direction}_slope'] = res.slope
                    row[f'{direction}_p'] = res.pvalue
                    row[f'{direction}_n'] = len(sub)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, 'amygdala_perchannel_slopes.csv'), index=False)
    print("saved amygdala_perchannel_slopes.csv")
    print(f"\nStage 5b done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
