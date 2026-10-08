#!/usr/bin/env python
# stage11_difference_plots.py
"""
Condition-DIFFERENCE plots per brain region (clearer than overlaid PETHs).

For each region x band x epoch, plots a single trace of the DIFFERENCE between
two conditions over time: (A - B). A flat line ~0 = no difference; a deflection
= the conditions differ, and you see WHEN and HOW MUCH.

Comparisons (positive = more power in the first-named condition):
  outcome : loss - win
  streak  : long streak (L3-4) - short streak (L1-2)
  bet     : high bet - low bet

Each difference trace is shown with:
  - SEM band (parametric standard error of the difference)
  - Permutation null envelope (2.5-97.5% of the shuffled-label null)
  - Significant time windows marked (CLUSTER-BASED permutation, field standard)

Statistics are done at the TRIAL level: for each region, the good channels are
averaged into ONE value per trial first (avoids treating correlated channels as
independent). Condition labels are then shuffled across trials.

Reads per-region per-trial time courses (region_trial_tc_<event>.npz from
Stage 2) + behaviour (pilot_results), matched via ident_<event>.csv.

Usage:
  python stage11_difference_plots.py --stage1-dir stage1_mne_out \
      --stage2-dir stage2_mne_fix \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv \
      --out-dir stage11_fix_out --n-perm 1000
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
MIN_PER_COND = 3        # need >= this many trials per condition to test


def load_behaviour(behav_csvs):
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    b['trial_number'] = b['exp_trial'] + NUM_PRACTICE
    b['win'] = b['correct'].astype(str).str.lower().isin(['true', '1', 'yes'])
    st_raw = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_dir'] = st_raw.str[0].map({'L': 'loss', 'W': 'win'})
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    sl_lab = pd.to_numeric(st_raw.str[1:], errors='coerce')
    b['streak_len'] = sl.where(sl.notna(), sl_lab)
    b['bet'] = pd.to_numeric(b['bet'], errors='coerce')
    b['high_bet'] = b['bet'] > b['bet'].median()
    return b


def cluster_perm_test(A, B, n_perm=1000, alpha=0.05, rng=None):
    """
    Cluster-based permutation test on two groups of per-trial time courses.
    A [nA x t], B [nB x t]. Returns:
      diff (mean A - mean B) [t], sem [t], null_lo/null_hi [t] (perm envelope),
      sig_mask [t] bool (timepoints in a significant cluster).
    Uses pointwise t as the cluster-forming statistic; cluster mass compared to
    a max-cluster null from shuffling condition labels.
    """
    rng = rng or np.random.default_rng(0)
    nA, nB = len(A), len(B)
    if nA < MIN_PER_COND or nB < MIN_PER_COND:
        t = A.shape[1] if len(A) else (B.shape[1] if len(B) else 0)
        nan = np.full(t, np.nan)
        return nan, nan, nan, nan, np.zeros(t, bool)

    diff = np.nanmean(A, 0) - np.nanmean(B, 0)
    sem = np.sqrt(np.nanvar(A, 0)/nA + np.nanvar(B, 0)/nB)

    def point_t(a, b):
        ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
        va, vb = np.nanvar(a, 0, ddof=1), np.nanvar(b, 0, ddof=1)
        return (ma - mb) / np.sqrt(va/len(a) + vb/len(b) + 1e-12)

    obs_t = point_t(A, B)
    thr = 2.0    # cluster-forming threshold on |t| (~p<0.05)

    def clusters(tvec):
        """Return list of (start,end,mass) for supra-threshold runs."""
        sig = np.abs(tvec) > thr
        out = []; i = 0; n = len(sig)
        while i < n:
            if sig[i]:
                j = i
                while j < n and sig[j]:
                    j += 1
                out.append((i, j, np.sum(np.abs(tvec[i:j]))))
                i = j
            else:
                i += 1
        return out

    obs_clusters = clusters(obs_t)

    # null: shuffle labels, record max cluster mass
    pooled = np.vstack([A, B])
    labels = np.array([0]*nA + [1]*nB)
    null_max = np.zeros(n_perm)
    null_diffs = np.empty((n_perm, pooled.shape[1]), np.float32)
    for p in range(n_perm):
        perm = rng.permutation(labels)
        pa, pb = pooled[perm == 0], pooled[perm == 1]
        null_diffs[p] = np.nanmean(pa, 0) - np.nanmean(pb, 0)
        cl = clusters(point_t(pa, pb))
        null_max[p] = max([c[2] for c in cl]) if cl else 0.0

    null_lo = np.nanpercentile(null_diffs, 2.5, axis=0)
    null_hi = np.nanpercentile(null_diffs, 97.5, axis=0)

    # a cluster is significant if its mass exceeds the 95th pct of null max
    crit = np.nanpercentile(null_max, 100*(1-alpha))
    sig_mask = np.zeros(len(obs_t), bool)
    for (s, e, mass) in obs_clusters:
        if mass > crit:
            sig_mask[s:e] = True
    return diff, sem, null_lo, null_hi, sig_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', required=True)
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage11_out')
    ap.add_argument('--n-perm', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    behav = load_behaviour(args.behav)

    for ev in EVENTS:
        tc_file = os.path.join(args.stage2_dir, f'region_trial_tc_{ev}.npz')
        id_file = os.path.join(args.stage1_dir, f'ident_{ev}.csv')
        if not (os.path.exists(tc_file) and os.path.exists(id_file)):
            print(f"  (skip {ev}: missing inputs)"); continue
        d = np.load(tc_file, allow_pickle=True)
        reg_tc = d['reg_tc']; t = d['t']; regions = list(d['regions'])
        # reg_tc is already region-averaged per trial: [band x tr x region x t]
        ident = pd.read_csv(id_file)
        merged = ident.merge(behav, on=['block', 'trial_number'], how='left')
        if len(merged) != reg_tc.shape[1]:
            print(f"  WARNING {ev}: identity mismatch; skipping"); continue

        # define the three comparisons as (name, maskA, labelA, maskB, labelB)
        win = merged['win'].values == True
        sdir = merged['streak_dir'].values
        slen = merged['streak_len'].values
        hbet = merged['high_bet'].values == True
        comparisons = {
            'outcome':  ('loss', win == False, 'win', win == True),
            'streak':   ('long streak', (slen >= 3),
                         'short streak', (slen >= 1) & (slen <= 2)),
            'bet':      ('high bet', hbet == True, 'low bet', hbet == False),
        }

        for comp, (labA, mA, labB, mB) in comparisons.items():
            nrow, ncol = len(regions), len(BANDS)
            fig, axes = plt.subplots(nrow, ncol, figsize=(2.7*ncol, 1.9*nrow),
                                     squeeze=False)
            for ri, rg in enumerate(regions):
                for bi, band in enumerate(BANDS):
                    ax = axes[ri, bi]
                    A = reg_tc[bi, mA, ri, :]     # [nA x t]
                    B = reg_tc[bi, mB, ri, :]
                    A = A[~np.all(np.isnan(A), axis=1)]
                    B = B[~np.all(np.isnan(B), axis=1)]
                    diff, sem, nlo, nhi, sig = cluster_perm_test(
                        A, B, n_perm=args.n_perm, rng=rng)
                    if np.all(np.isnan(diff)):
                        ax.text(0.5, 0.5, f'n<{MIN_PER_COND}', fontsize=6,
                                ha='center', transform=ax.transAxes)
                    else:
                        # permutation null envelope (grey), SEM band (colour)
                        ax.fill_between(t, nlo, nhi, color='0.8', alpha=0.7,
                                        label='perm null 95%')
                        ax.fill_between(t, diff-sem, diff+sem, color='#4C72B0',
                                        alpha=0.3, label='SEM')
                        ax.plot(t, diff, color='#1a1a1a', lw=1.2, label='A-B')
                        # mark significant clusters with a bar at the bottom
                        if sig.any():
                            ylo = ax.get_ylim()[0]
                            ax.plot(t[sig], np.full(sig.sum(), ylo),
                                    color='red', lw=4, solid_capstyle='butt')
                    ax.axvline(0, color='k', ls='--', lw=0.6)
                    ax.axhline(0, color='grey', ls=':', lw=0.5)
                    if ri == 0: ax.set_title(band, fontsize=9)
                    if bi == 0: ax.set_ylabel(f"{rg}\n(nA={len(A)},nB={len(B)})",
                                              fontsize=6)
                    if ri == 0 and bi == ncol-1:
                        ax.legend(fontsize=5, loc='upper right')
            for bi in range(ncol):
                axes[-1, bi].set_xlabel('t (s)', fontsize=8)
            fig.suptitle(f'Difference ({labA} - {labB}) by region and band — '
                         f'{ev}   [red bar = significant cluster, p<0.05]',
                         fontsize=11)
            fig.tight_layout()
            out = os.path.join(args.out_dir, f'diff_{ev}_{comp}.png')
            fig.savefig(out, dpi=140); plt.close(fig)
            print(f"  saved {out}")

    print(f"\nStage 11 done. Difference plots in {args.out_dir}/")


if __name__ == '__main__':
    main()
