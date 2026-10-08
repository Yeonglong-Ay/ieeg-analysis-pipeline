#!/usr/bin/env python
# stage13_within_region_matrices.py
"""
STAGE 13: within-region channel x channel connectivity matrices (heatmaps).

For each brain region, produces an N x N matrix (N = number of channels in that
region) showing pairwise connectivity between the region's channels, two ways:

  (A) POWER CORRELATION — Pearson correlation of the band-power envelopes
      between each channel pair (same measure as Stage 12 Part A). Captures
      whether channels' power co-varies (includes shared/local signal — the
      right question for 'are these channels doing the same thing').

  (B) SPECTRAL COHERENCE — magnitude-squared coherence between each channel
      pair (phase/frequency based). NOTE: within a region this is strongly
      inflated by volume conduction (nearby contacts share sources), so high
      coherence here does NOT necessarily mean communication. Provided because
      the PI requested it; interpret with that caveat.

Diagonal = self (correlation/coherence of a channel with itself = 1).
Off-diagonal = the pairwise value. Bright, uniform off-diagonal = homogeneous
region; block/patchy structure = functional sub-groups within the region.

Scope (as configured): epoch = bet_submitted; all trials (no condition split);
all 6 bands; all regions. One figure per region per measure (6 band panels each).

Requires mne_connectivity for coherence (Part B). If absent, only correlation
(Part A) is produced.

Usage:
  python stage13_within_region_matrices.py --stage1-dir stage1_mne_out \
      --out-dir stage13_out --event bet_submitted
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

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'low_gamma': (30, 70), 'high_gamma': (70, 150)}
REGION_ORDER = ['orbitofrontal', 'superior_frontal', 'middle_frontal',
                'inferior_frontal', 'cingulate', 'insula', 'amygdala',
                'hippocampus', 'thalamus', 'temporal']


def region_of(fslabel):
    s = str(fslabel).lower()
    if 'white-matter' in s or 'unknown' in s: return None
    if 'hippocamp' in s: return 'hippocampus'
    if 'amygdala' in s: return 'amygdala'
    if 'thalamus' in s: return 'thalamus'
    if 'insula' in s or 'insular' in s or 'ins_lg' in s: return 'insula'
    if 'cingul' in s: return 'cingulate'
    if 'orbital' in s: return 'orbitofrontal'
    # frontal split into sub-gyri (orbital & cingulate stay separate above)
    if 'front_sup' in s: return 'superior_frontal'
    if 'front_middle' in s: return 'middle_frontal'
    if 'front_inf' in s: return 'inferior_frontal'
    if 'front' in s or 'subcentral' in s: return 'frontal_other'
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return None


def bandpass(x, lo, hi, sf, order=4):
    ny = sf / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def corr_matrix(amp_region):
    """amp_region [tr x ch x t] band-power envelopes -> N x N mean correlation.
    Per trial, correlate each channel pair over time; average across trials."""
    n_tr, n_ch, n_t = amp_region.shape
    M = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                M[i, j] = 1.0
                continue
            rs = []
            for tr in range(n_tr):
                xi, xj = amp_region[tr, i], amp_region[tr, j]
                if np.std(xi) > 0 and np.std(xj) > 0:
                    rs.append(np.corrcoef(xi, xj)[0, 1])
            if rs:
                M[i, j] = np.mean(rs)
    return M


def plot_matrices(matrices, region, chan_labels, measure, outpath):
    """matrices: dict band -> NxN. One figure, 6 band panels."""
    bands = list(BANDS)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.2*len(bands), 3.4),
                             squeeze=False)
    vmin, vmax = (0, 1) if measure == 'coherence' else (-1, 1)
    cmap = 'viridis' if measure == 'coherence' else 'RdBu_r'
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        M = matrices.get(band)
        if M is None:
            ax.axis('off'); continue
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_title(band, fontsize=9)
        ax.set_xticks(range(len(chan_labels)))
        ax.set_yticks(range(len(chan_labels)))
        ax.set_xticklabels(range(1, len(chan_labels)+1), fontsize=6)
        ax.set_yticklabels(range(1, len(chan_labels)+1), fontsize=6)
        if bi == 0:
            ax.set_ylabel('channel')
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6,
                 label=('coherence' if measure == 'coherence'
                        else 'power correlation'))
    fig.suptitle(f'{region} — within-region {measure} '
                 f'(N={len(chan_labels)} channels) — bet_submitted', fontsize=12)
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--out-dir', default='stage13_out')
    ap.add_argument('--event', default='bet_submitted')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    reg = anat['anode_region'].map(region_of)
    reg = reg.where(reg.notna(), anat['cathode_region'].map(region_of))
    anat['region'] = reg

    epochs = mne.read_epochs(
        os.path.join(args.stage1_dir, f"epochs_{args.event}-epo.fif"),
        preload=True, verbose='ERROR')
    sf = epochs.info['sfreq']
    names = epochs.ch_names
    ch_region = {nm: anat.loc[anat['bipolar_name'] == nm, 'region'].iloc[0]
                 for nm in names if (anat['bipolar_name'] == nm).any()}
    data = epochs.get_data()             # [tr x ch x t]

    for region in REGION_ORDER:
        idx = [i for i, nm in enumerate(names) if ch_region.get(nm) == region]
        if len(idx) < 2:
            print(f"  (skip {region}: <2 channels)")
            continue
        chan_labels = [names[i] for i in idx]
        region_data = data[:, idx, :]    # [tr x n_region_ch x t]

        # ---- (A) power correlation matrices, per band ----
        corr_mats = {}
        for band, (lo, hi) in BANDS.items():
            amp = np.abs(hilbert(bandpass(region_data.astype(np.float64),
                                          lo, hi, sf), axis=-1))
            corr_mats[band] = corr_matrix(amp)
        plot_matrices(corr_mats, region, chan_labels, 'correlation',
                      os.path.join(args.out_dir, f'corr_{region}.png'))

        # ---- (B) spectral coherence matrices, per band ----
        if HAVE_CONN:
            coh_mats = {}
            for band, (lo, hi) in BANDS.items():
                con = spectral_connectivity_epochs(
                    region_data, method='coh', mode='multitaper', sfreq=sf,
                    fmin=lo, fmax=hi, faverage=True, verbose='ERROR')
                cmat = con.get_data(output='dense')[:, :, 0]   # lower-triangular
                # symmetrize + unit diagonal for display
                full = cmat + cmat.T
                np.fill_diagonal(full, 1.0)
                coh_mats[band] = full
            plot_matrices(coh_mats, region, chan_labels, 'coherence',
                          os.path.join(args.out_dir, f'coh_{region}.png'))

    if not HAVE_CONN:
        print("\nNOTE: mne_connectivity not installed — coherence (B) skipped, "
              "correlation (A) produced. Install: pip install mne-connectivity")
    print(f"\nStage 13 done. Matrices in {args.out_dir}/")


if __name__ == '__main__':
    main()
