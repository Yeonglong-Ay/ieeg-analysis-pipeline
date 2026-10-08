#!/usr/bin/env python
# stage13b_amygdala_with_controls.py
"""
STAGE 13b: amygdala within-region connectivity matrix WITH same-lead neighbour
CONTROLS, and improved contrast (masked diagonal + autoscaled colour).

Two improvements over Stage 13 (per PI):
  1. CONTROL CHANNELS: include a few non-amygdala channels that sit on the SAME
     electrode lead(s) as the amygdala contacts, immediately adjacent to them
     (the 'hard' spatial control). If amygdala channels correlate more with each
     other than with these immediate neighbours, that supports the amygdala
     being a distinct functional block (beyond mere proximity/volume conduction).
  2. CONTRAST: mask the diagonal (self = 1 carries no info and stretches the
     colour scale) and AUTOSCALE the colour to the off-diagonal values, so the
     informative pairwise contrast fills the colour range.

Amygdala channels are ordered first, then the control neighbours, so the
amygdala block is visually grouped in the top-left; a divider line separates
amygdala from controls.

Both measures (correlation + coherence), all 6 bands, bet_submitted epoch.

Usage:
  python stage13b_amygdala_with_controls.py --stage1-dir stage1_mne_out \
      --out-dir stage13b_out --event bet_submitted --n-control 4
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


def is_amygdala(fslabel):
    return 'amygdala' in str(fslabel).lower()


def bandpass(x, lo, hi, sf, order=4):
    ny = sf / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def corr_matrix(amp_region):
    n_tr, n_ch, n_t = amp_region.shape
    M = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue                      # leave diagonal as NaN (masked)
            rs = []
            for tr in range(n_tr):
                xi, xj = amp_region[tr, i], amp_region[tr, j]
                if np.std(xi) > 0 and np.std(xj) > 0:
                    rs.append(np.corrcoef(xi, xj)[0, 1])
            if rs:
                M[i, j] = np.mean(rs)
    return M


def pick_controls(anat, amyg_names, n_control):
    """Find non-amygdala bipolar channels on the SAME leads as the amygdala,
    immediately adjacent in contact order. Returns a list of channel names."""
    amyg_leads = anat.loc[anat['bipolar_name'].isin(amyg_names), 'lead'].unique()
    controls = []
    for lead in amyg_leads:
        lead_ch = anat[anat['lead'] == lead].sort_values('anode_idx')
        names = lead_ch['bipolar_name'].tolist()
        is_amyg = [is_amygdala(r1) or is_amygdala(r2) for r1, r2 in
                   zip(lead_ch['anode_region'], lead_ch['cathode_region'])]
        # find the positions of amygdala channels, take the neighbours just
        # beyond the amygdala span (non-amygdala) on this lead
        amyg_pos = [k for k, a in enumerate(is_amyg) if a]
        if not amyg_pos:
            continue
        lo, hi = min(amyg_pos), max(amyg_pos)
        # neighbour just below and just above the amygdala span
        for pos in (lo - 1, hi + 1, lo - 2, hi + 2):
            if 0 <= pos < len(names) and not is_amyg[pos]:
                if names[pos] not in controls:
                    controls.append(names[pos])
    return controls[:n_control]


def plot_matrices(matrices, labels, n_amyg, measure, outpath):
    bands = list(BANDS)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.4*len(bands), 3.8),
                             squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        M = matrices.get(band)
        if M is None:
            ax.axis('off'); continue
        # AUTOSCALE to the off-diagonal (non-NaN) values for max contrast
        off = M[~np.isnan(M)]
        if len(off):
            if measure == 'coherence':
                vmin, vmax = np.nanmin(off), np.nanmax(off)
                cmap = 'viridis'
            else:
                vmax = np.nanmax(np.abs(off)); vmin = -vmax   # symmetric
                cmap = 'RdBu_r'
        else:
            vmin, vmax, cmap = 0, 1, 'viridis'
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        # divider between amygdala block and controls
        ax.axhline(n_amyg - 0.5, color='lime', lw=1.5)
        ax.axvline(n_amyg - 0.5, color='lime', lw=1.5)
        ax.set_title(band, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5, rotation=90)
        ax.set_yticklabels(labels, fontsize=5)
        fig.colorbar(im, ax=ax, shrink=0.5)
    fig.suptitle(f'Amygdala (first {n_amyg}) + same-lead controls — '
                 f'within/between {measure} (bet_submitted)\n'
                 f'diagonal masked, colour autoscaled; green line = amygdala|control',
                 fontsize=11)
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--out-dir', default='stage13b_out')
    ap.add_argument('--event', default='bet_submitted')
    ap.add_argument('--n-control', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    amyg_names = anat.loc[
        anat['anode_region'].map(is_amygdala) |
        anat['cathode_region'].map(is_amygdala), 'bipolar_name'].tolist()
    if len(amyg_names) < 2:
        raise SystemExit(f"Need >=2 amygdala channels; found {len(amyg_names)}.")
    controls = pick_controls(anat, amyg_names, args.n_control)
    print(f"Amygdala channels ({len(amyg_names)}): {amyg_names}")
    print(f"Same-lead control channels ({len(controls)}): {controls}")
    if not controls:
        print("WARNING: no same-lead neighbour controls found (amygdala may be "
              "at the lead ends, or neighbours also amygdala). Proceeding with "
              "amygdala only.")
    ordered = amyg_names + controls        # amygdala first, then controls
    n_amyg = len(amyg_names)

    epochs = mne.read_epochs(
        os.path.join(args.stage1_dir, f"epochs_{args.event}-epo.fif"),
        preload=True, verbose='ERROR')
    sf = epochs.info['sfreq']
    names = epochs.ch_names
    idx = [names.index(nm) for nm in ordered if nm in names]
    labels = [nm for nm in ordered if nm in names]
    data = epochs.get_data()[:, idx, :]      # [tr x (amyg+control) x t]

    # correlation matrices per band
    corr_mats = {}
    for band, (lo, hi) in BANDS.items():
        amp = np.abs(hilbert(bandpass(data.astype(np.float64), lo, hi, sf),
                             axis=-1))
        corr_mats[band] = corr_matrix(amp)
    plot_matrices(corr_mats, labels, n_amyg, 'correlation',
                  os.path.join(args.out_dir, 'corr_amygdala_controls.png'))

    # coherence matrices per band
    if HAVE_CONN:
        coh_mats = {}
        for band, (lo, hi) in BANDS.items():
            con = spectral_connectivity_epochs(
                data, method='coh', mode='multitaper', sfreq=sf,
                fmin=lo, fmax=hi, faverage=True, verbose='ERROR')
            cmat = con.get_data(output='dense')[:, :, 0]
            full = cmat + cmat.T
            np.fill_diagonal(full, np.nan)      # mask diagonal
            coh_mats[band] = full
        plot_matrices(coh_mats, labels, n_amyg, 'coherence',
                      os.path.join(args.out_dir, 'coh_amygdala_controls.png'))
    else:
        print("NOTE: mne_connectivity not installed — coherence skipped.")

    print(f"\nStage 13b done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
