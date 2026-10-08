#!/usr/bin/env python
# stage9_peth_conditions.py
"""
Condition-split PETHs (peri-event time histograms) per brain region.

For every epoch and every frequency band, plots the region-averaged band-power
time course, split and overlaid by behavioral condition, so you can SEE whether
the neural response differs by what the subject did / experienced:

  splits:
    - outcome     : win vs loss (current trial)
    - streak      : in a losing streak (>=2 consecutive losses) vs not
    - bet_size    : high bet vs low bet (median split)

Reads the per-region, per-trial time courses from Stage 2
(region_trial_tc_<event>.npz) and the behavioral data (pilot_results), matched
by trial via the event identity files (ident_<event>.csv from Stage 1).

For each (epoch, band, region) it draws the two condition means +/- SEM overlaid.
Output: one multi-panel figure per (epoch, split), rows=regions, cols=bands.

Usage:
  python stage9_peth_conditions.py --stage1-dir stage1_mne_out \
      --stage2-dir stage2_mne_fix \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv \
      --out-dir stage9_fix_out
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVENTS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
STREAK_MIN = 2         # a trial is "in a losing streak" if loss run >= this


def load_behaviour(behav_csvs):
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    b['trial_number'] = b['exp_trial'] + NUM_PRACTICE      # match ident files
    b['win'] = b['correct'].astype(str).str.lower().isin(['true', '1', 'yes'])
    b['streak_type'] = b['streak_type'].astype(str).str.lower()
    b['streak_length'] = pd.to_numeric(b['streak_length'], errors='coerce')
    b['bet'] = pd.to_numeric(b['bet'], errors='coerce')
    # define condition labels
    b['in_loss_streak'] = (b['streak_type'] == 'loss') & (b['streak_length'] >= STREAK_MIN)
    bet_median = b['bet'].median()
    b['high_bet'] = b['bet'] > bet_median
    return b


def condition_masks(behav_rows):
    """Given the per-epoch behavioral rows (aligned to trials), return dict of
    split_name -> (labelA, maskA, labelB, maskB)."""
    return {
        'outcome':  ('win',  behav_rows['win'].values == True,
                     'loss', behav_rows['win'].values == False),
        'streak':   ('loss streak', behav_rows['in_loss_streak'].values == True,
                     'no streak',   behav_rows['in_loss_streak'].values == False),
        'bet_size': ('high bet', behav_rows['high_bet'].values == True,
                     'low bet',  behav_rows['high_bet'].values == False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', required=True)
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage9_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    behav = load_behaviour(args.behav)

    for ev in EVENTS:
        tc_file = os.path.join(args.stage2_dir, f'region_trial_tc_{ev}.npz')
        id_file = os.path.join(args.stage1_dir, f'ident_{ev}.csv')
        if not (os.path.exists(tc_file) and os.path.exists(id_file)):
            print(f"  (skip {ev}: missing {tc_file} or {id_file})")
            continue
        d = np.load(tc_file, allow_pickle=True)
        reg_tc = d['reg_tc']            # [band x tr x region x t]
        t = d['t']; regions = list(d['regions'])
        ident = pd.read_csv(id_file)    # per-epoch trial identity (block, trial_number)

        # align behavior to these epochs by (block, trial_number)
        merged = ident.merge(behav, on=['block', 'trial_number'], how='left')
        if len(merged) != reg_tc.shape[1]:
            print(f"  WARNING {ev}: ident rows {len(merged)} != tc trials "
                  f"{reg_tc.shape[1]}; skipping")
            continue

        masks = condition_masks(merged)
        for split_name, (labA, mA, labB, mB) in masks.items():
            nrow, ncol = len(regions), len(BANDS)
            fig, axes = plt.subplots(nrow, ncol,
                                     figsize=(2.5*ncol, 1.9*nrow), squeeze=False)
            for ri, rg in enumerate(regions):
                for bi, band in enumerate(BANDS):
                    ax = axes[ri, bi]
                    for lab, m, col in [(labA, mA, '#C44E52'),
                                        (labB, mB, '#4C72B0')]:
                        seg = reg_tc[bi, m, ri, :]          # [trials_in_cond x t]
                        seg = seg[~np.all(np.isnan(seg), axis=1)]
                        if len(seg) < 2:
                            continue
                        mean = np.nanmean(seg, axis=0)
                        sem = np.nanstd(seg, axis=0) / np.sqrt(len(seg))
                        ax.plot(t, mean, color=col, lw=1.2,
                                label=f"{lab} (n={len(seg)})")
                        ax.fill_between(t, mean-sem, mean+sem, color=col, alpha=0.2)
                    ax.axvline(0, color='k', ls='--', lw=0.7)
                    ax.axhline(0, color='grey', ls=':', lw=0.5)
                    if ri == 0:
                        ax.set_title(band, fontsize=9)
                    if bi == 0:
                        ax.set_ylabel(rg, fontsize=7)
                    if ri == 0 and bi == ncol-1:
                        ax.legend(fontsize=6, loc='upper right')
            for bi in range(ncol):
                axes[-1, bi].set_xlabel('t (s)', fontsize=8)
            fig.suptitle(f'PETH by region and band — {ev} — split: {split_name}',
                         fontsize=12)
            fig.tight_layout()
            out = os.path.join(args.out_dir, f'peth_{ev}_{split_name}.png')
            fig.savefig(out, dpi=140); plt.close(fig)
            print(f"  saved {out}")

    print(f"\nStage 9 done. Condition-split PETHs in {args.out_dir}/")


if __name__ == '__main__':
    main()
