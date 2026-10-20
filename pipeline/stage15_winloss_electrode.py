#!/usr/bin/env python
# stage15_winloss_electrode.py
"""
STAGE 15: within-electrode correlation & coherence, split by WIN vs LOSS.

Extends Stage 13c (within-lead matrices + distance decay) by computing the
same measures SEPARATELY for win-trials and loss-trials on the amygdala-
containing electrode, then comparing them.

For each epoch, each band, each measure (correlation + coherence):
  - compute the channel x channel matrix for WIN trials and for LOSS trials
  - TRIAL-MATCH: subsample the larger condition to equal N (repeated, averaged)
    so the trial-count bias (which inflates coherence at small N) is identical
    in both conditions and cannot masquerade as a win/loss difference
  - produce: win matrix | loss matrix | (win - loss) difference
  - permutation test: shuffle win/loss labels, rebuild the null difference
    distribution, mark channel pairs whose |win-loss| exceeds the null

CAVEATS (printed/README):
  - Outcome is revealed at FEEDBACK. At pre-outcome epochs (fixation/bet/colour)
    the win/loss label is an outcome not yet known — interpret those as the
    subtler "does a trial that WILL be a win differ beforehand" question.
  - Low-frequency bands show inflated correlation (smoothness artifact); read
    win/loss DIFFERENCES most reliably in high-gamma.
  - Single patient, fewer trials per condition after splitting + matching — noisy.

Usage:
  python stage15_winloss_electrode.py --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_1.csv pilot_results_2.csv \
      --out-dir stage15_out --n-perm 500 --n-subsample 50
"""
import argparse, os
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from mne_connectivity import spectral_connectivity_epochs
    HAVE_CONN = True
except Exception:
    HAVE_CONN = False

# block-aware practice handling (shared helper)
try:
    from hie_practice import practice_per_block, is_experimental, \
        exp_trial_to_trial_number
    HAVE_PRACTICE = True
except Exception:
    HAVE_PRACTICE = False
    NUM_PRACTICE = 5

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'low_gamma': (30, 70), 'high_gamma': (70, 150)}
EPOCHS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']


def is_amygdala(f):
    return 'amygdala' in str(f).lower()


def bandpass(x, lo, hi, sf, order=4):
    ny = sf / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def corr_matrix(amp):
    """amp [tr x ch x t] -> ch x ch mean correlation (diagonal NaN)."""
    n_tr, n_ch, n_t = amp.shape
    M = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            rs = []
            for tr in range(n_tr):
                xi, xj = amp[tr, i], amp[tr, j]
                if np.std(xi) > 0 and np.std(xj) > 0:
                    rs.append(np.corrcoef(xi, xj)[0, 1])
            if rs:
                M[i, j] = np.mean(rs)
    return M


def coh_matrix(data, sf, lo, hi):
    con = spectral_connectivity_epochs(
        data, method='coh', mode='multitaper', sfreq=sf,
        fmin=lo, fmax=hi, faverage=True, verbose='ERROR')
    cmat = con.get_data(output='dense')[:, :, 0]
    full = cmat + cmat.T
    np.fill_diagonal(full, np.nan)
    return full


def load_winloss(event_csvs, behav_csvs):
    """Return a dict: for each epoch, an array (len = n usable trials) of
    True(win)/False(loss), aligned to the epoch's trial order."""
    # behaviour: trial_number -> win/loss
    bdfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        bdfs.append(d)
    b = pd.concat(bdfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    if HAVE_PRACTICE:
        npb = practice_per_block(behav_csvs)
        b['trial_number'] = b.apply(
            lambda r: exp_trial_to_trial_number(r['exp_trial'], r['block'], npb),
            axis=1)
    else:
        b['trial_number'] = b['exp_trial'] + NUM_PRACTICE
        npb = None
    b['win'] = b['correct'].astype(str).str.lower().isin(['true', '1', 'yes'])
    win_lut = {(int(r['block']), int(r['trial_number'])): bool(r['win'])
               for _, r in b.iterrows()}
    return win_lut, npb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage15_out')
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--n-subsample', type=int, default=50,
                    help='repeats of the trial-matched subsampling')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['is_amyg'] = (anat['anode_region'].map(is_amygdala) |
                       anat['cathode_region'].map(is_amygdala))
    amyg_leads = anat.loc[anat['is_amyg'], 'lead'].unique()
    if len(amyg_leads) == 0:
        raise SystemExit("No amygdala lead found.")
    lead = amyg_leads[0]
    lead_ch = anat[anat['lead'] == lead].sort_values('anode_idx')
    print(f"Amygdala lead: {lead} ({len(lead_ch)} channels)")

    win_lut, _ = load_winloss(args.events, args.behav)

    for ev in EPOCHS:
        fpath = os.path.join(args.stage1_dir, f"epochs_{ev}-epo.fif")
        identf = os.path.join(args.stage1_dir, f"ident_{ev}.csv")
        if not (os.path.exists(fpath) and os.path.exists(identf)):
            print(f"  (skip {ev})"); continue
        epochs = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        ident = pd.read_csv(identf)
        sf = epochs.info['sfreq']
        names = epochs.ch_names
        lead_names = [nm for nm in lead_ch['bipolar_name'] if nm in names]
        if len(lead_names) < 3:
            print(f"  (skip {ev}: <3 lead channels)"); continue
        idx = [names.index(nm) for nm in lead_names]
        data = epochs.get_data()[:, idx, :]             # [tr x lead_ch x t]

        # win/loss label per epoch trial (aligned to ident order).
        # Build index lists explicitly (avoid object-array == True pitfalls).
        win_tr, loss_tr = [], []
        for ti, (_, r) in enumerate(ident.iterrows()):
            lab = win_lut.get((int(r['block']), int(r['trial_number'])), None)
            if lab is True:
                win_tr.append(ti)
            elif lab is False:
                loss_tr.append(ti)
        win_tr = np.array(win_tr, dtype=int)
        loss_tr = np.array(loss_tr, dtype=int)
        n_match = min(len(win_tr), len(loss_tr))
        print(f"\n=== {ev}: {len(win_tr)} win, {len(loss_tr)} loss "
              f"-> trial-matched to {n_match} each ===", flush=True)
        if n_match < 5:
            print("  too few trials to compare; skipping", flush=True); continue

        for measure in (['correlation', 'coherence'] if HAVE_CONN
                        else ['correlation']):
            for band, (lo, hi) in BANDS.items():
                # resume: skip if this figure is already done
                outp = os.path.join(args.out_dir, f'{measure}_{ev}_{band}.png')
                if os.path.exists(outp):
                    print(f"    skip {measure}_{ev}_{band} (already done)", flush=True)
                    continue
                # precompute per-trial feature once
                if measure == 'correlation':
                    amp = np.abs(hilbert(bandpass(data.astype(np.float64), lo, hi, sf),
                                         axis=-1))

                def mat_for(trials):
                    if measure == 'correlation':
                        return corr_matrix(amp[trials])
                    else:
                        return coh_matrix(data[trials], sf, lo, hi)

                # trial-matched win & loss matrices (subsample larger, repeat, average)
                win_mats, loss_mats = [], []
                for _ in range(args.n_subsample):
                    w = rng.choice(win_tr, n_match, replace=False)
                    l = rng.choice(loss_tr, n_match, replace=False)
                    win_mats.append(mat_for(w))
                    loss_mats.append(mat_for(l))
                win_M = np.nanmean(win_mats, axis=0)
                loss_M = np.nanmean(loss_mats, axis=0)
                diff_M = win_M - loss_M

                # permutation test on the difference: shuffle win/loss labels
                pooled = np.concatenate([win_tr, loss_tr])
                null_diffs = np.empty((args.n_perm,) + diff_M.shape)
                for p in range(args.n_perm):
                    perm = rng.permutation(pooled)
                    a, b2 = perm[:n_match], perm[n_match:2*n_match]
                    null_diffs[p] = mat_for(a) - mat_for(b2)
                # two-sided p per cell
                pmat = np.full(diff_M.shape, np.nan)
                for i in range(diff_M.shape[0]):
                    for j in range(diff_M.shape[1]):
                        if np.isfinite(diff_M[i, j]):
                            nd = null_diffs[:, i, j]; nd = nd[np.isfinite(nd)]
                            if len(nd):
                                pmat[i, j] = (np.sum(np.abs(nd) >= abs(diff_M[i, j])) + 1) / (len(nd) + 1)

                _plot_winloss(win_M, loss_M, diff_M, pmat, lead_names, measure,
                              band, ev, outp)
    print(f"\nStage 15 done. Outputs in {args.out_dir}/")


def _plot_winloss(win_M, loss_M, diff_M, pmat, labels, measure, band, ev, outp):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    if measure == 'coherence':
        vmax = np.nanmax([np.nanmax(win_M), np.nanmax(loss_M)]); vmin = 0
        cmap = 'viridis'
    else:
        vmax = np.nanmax([np.nanmax(np.abs(win_M)), np.nanmax(np.abs(loss_M))])
        vmin = -vmax; cmap = 'RdBu_r'
    for ax, M, ttl in [(axes[0], win_M, 'WIN'), (axes[1], loss_M, 'LOSS')]:
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax, shrink=0.6)
    # difference panel, with significance marked
    dmax = np.nanmax(np.abs(diff_M)) if np.isfinite(np.nanmax(np.abs(diff_M))) else 1
    imd = axes[2].imshow(diff_M, vmin=-dmax, vmax=dmax, cmap='RdBu_r', aspect='equal')
    axes[2].set_title('WIN - LOSS (* p<.05)', fontsize=10)
    fig.colorbar(imd, ax=axes[2], shrink=0.6)
    for i in range(diff_M.shape[0]):
        for j in range(diff_M.shape[1]):
            if np.isfinite(pmat[i, j]) and pmat[i, j] < 0.05:
                axes[2].text(j, i, '*', ha='center', va='center',
                             color='black', fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=4, rotation=90)
        ax.set_yticklabels(labels, fontsize=4)
    fig.suptitle(f'{ev} — {band} — within-electrode {measure}: win vs loss '
                 f'(trial-matched)', fontsize=12)
    fig.savefig(outp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"    saved {outp}")


if __name__ == '__main__':
    main()
