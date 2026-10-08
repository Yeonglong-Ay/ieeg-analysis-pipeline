#!/usr/bin/env python
# stage13c_full_lead_distance.py
"""
STAGE 13c: EXTENDED within-region analysis along the FULL amygdala electrode
lead, plus a DISTANCE-DECAY quantification.

This EXTENDS stage13b (which showed amygdala + a few immediate neighbours). It
does NOT replace it — 13b's output is kept. 13c writes to its own folder so the
progression of the analysis (and our reasoning) is preserved for review.

Motivation (recorded for the PI): with contacts ~3-4 mm apart across a small
structure (~1.5 cm), we do NOT expect a sharp 'amygdala block' boundary — a
smooth fall-off of similarity with distance is expected regardless of whether
the amygdala is a discrete functional unit. So the within-region matrix is best
read DESCRIPTIVELY (how similarity varies with distance/band), not as a decisive
test of functional unity. This script makes that distance-decay explicit.

Produces, per measure (correlation + coherence), per band:
  1. Full-lead matrix (all contacts on the amygdala's lead), amygdala channels
     marked, diagonal greyed, colour autoscaled to off-diagonal.
  2. A distance-decay line plot: mean correlation/coherence vs. how many contacts
     apart the pair is (1 = adjacent, 2 = two apart, ...), per band.

Epoch = bet_submitted. Patient-agnostic (reads Stage 1 epochs + anatomy only).

Usage:
  python stage13c_full_lead_distance.py --stage1-dir stage1_mne_out \
      --out-dir stage13c_out --event bet_submitted
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
BAND_COLORS = {'delta': '#4C72B0', 'theta': '#55A868', 'alpha': '#8172B2',
               'beta': '#CCB974', 'low_gamma': '#C44E52', 'high_gamma': '#000000'}


def is_amygdala(fslabel):
    return 'amygdala' in str(fslabel).lower()


def bandpass(x, lo, hi, sf, order=4):
    ny = sf / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def corr_matrix(amp):
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


def distance_decay(M):
    """Mean off-diagonal value as a function of |i-j| (contacts apart)."""
    n = M.shape[0]
    out = {}
    for d in range(1, n):
        vals = [M[i, i + d] for i in range(n - d) if np.isfinite(M[i, i + d])]
        vals += [M[i + d, i] for i in range(n - d) if np.isfinite(M[i + d, i])]
        if vals:
            out[d] = np.mean(vals)
    return out


def plot_matrices(matrices, labels, amyg_mask, measure, outpath):
    bands = list(BANDS)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.4*len(bands), 3.9),
                             squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        M = matrices.get(band)
        if M is None:
            ax.axis('off'); continue
        off = M[~np.isnan(M)]
        if len(off):
            if measure == 'coherence':
                vmin, vmax, cmap = np.nanmin(off), np.nanmax(off), 'viridis'
            else:
                vmax = np.nanmax(np.abs(off)); vmin = -vmax; cmap = 'RdBu_r'
        else:
            vmin, vmax, cmap = 0, 1, 'viridis'
        # grey background for masked diagonal
        ax.imshow(np.ones_like(M), cmap='Greys', vmin=0, vmax=1, aspect='equal')
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        # mark amygdala span with green lines at its boundaries
        amyg_idx = np.where(amyg_mask)[0]
        if len(amyg_idx):
            lo, hi = amyg_idx.min() - 0.5, amyg_idx.max() + 0.5
            for v in (lo, hi):
                ax.axhline(v, color='lime', lw=1.2)
                ax.axvline(v, color='lime', lw=1.2)
        ax.set_title(band, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=4, rotation=90)
        ax.set_yticklabels(labels, fontsize=4)
        fig.colorbar(im, ax=ax, shrink=0.5)
    fig.suptitle(f'Full amygdala lead — within-lead {measure} (bet_submitted)\n'
                 f'green lines = amygdala span; diagonal greyed; colour autoscaled',
                 fontsize=11)
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_distance_decay(decays, measure, outpath):
    fig, ax = plt.subplots(figsize=(7, 5))
    for band, dd in decays.items():
        if not dd:
            continue
        ds = sorted(dd)
        ax.plot(ds, [dd[d] for d in ds], marker='o',
                color=BAND_COLORS.get(band, 'grey'), label=band, lw=1.5)
    ax.set_xlabel('contacts apart (|i - j|)  — proxy for distance')
    ax.set_ylabel(f'mean {measure}')
    ax.set_title(f'Distance-decay of {measure} along the amygdala lead\n'
                 f'(expected to fall off smoothly at ~3-4 mm spacing)')
    ax.axhline(0, color='grey', ls=':', lw=0.5)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--out-dir', default='stage13c_out')
    ap.add_argument('--event', default='bet_submitted')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    # find the lead(s) that contain amygdala contacts
    anat['is_amyg'] = (anat['anode_region'].map(is_amygdala) |
                       anat['cathode_region'].map(is_amygdala))
    amyg_leads = anat.loc[anat['is_amyg'], 'lead'].unique()
    print(f"Amygdala lead(s): {list(amyg_leads)}")

    epochs = mne.read_epochs(
        os.path.join(args.stage1_dir, f"epochs_{args.event}-epo.fif"),
        preload=True, verbose='ERROR')
    sf = epochs.info['sfreq']
    names = epochs.ch_names

    for lead in amyg_leads:
        lead_ch = anat[anat['lead'] == lead].sort_values('anode_idx')
        ordered = [nm for nm in lead_ch['bipolar_name'] if nm in names]
        if len(ordered) < 3:
            print(f"  (skip lead {lead}: <3 channels on lead)")
            continue
        amyg_mask = np.array([bool(lead_ch.loc[lead_ch['bipolar_name'] == nm,
                              'is_amyg'].iloc[0]) for nm in ordered])
        idx = [names.index(nm) for nm in ordered]
        data = epochs.get_data()[:, idx, :]
        print(f"\nLead {lead}: {len(ordered)} channels "
              f"({amyg_mask.sum()} amygdala, {(~amyg_mask).sum()} other)")

        # correlation
        corr_mats, corr_decays = {}, {}
        for band, (lo, hi) in BANDS.items():
            amp = np.abs(hilbert(bandpass(data.astype(np.float64), lo, hi, sf),
                                 axis=-1))
            M = corr_matrix(amp)
            corr_mats[band] = M
            corr_decays[band] = distance_decay(M)
        plot_matrices(corr_mats, ordered, amyg_mask, 'correlation',
                      os.path.join(args.out_dir, f'corr_lead_{lead}.png'))
        plot_distance_decay(corr_decays, 'correlation',
                            os.path.join(args.out_dir, f'decay_corr_lead_{lead}.png'))

        # coherence
        if HAVE_CONN:
            coh_mats, coh_decays = {}, {}
            for band, (lo, hi) in BANDS.items():
                con = spectral_connectivity_epochs(
                    data, method='coh', mode='multitaper', sfreq=sf,
                    fmin=lo, fmax=hi, faverage=True, verbose='ERROR')
                cmat = con.get_data(output='dense')[:, :, 0]
                full = cmat + cmat.T
                np.fill_diagonal(full, np.nan)
                coh_mats[band] = full
                coh_decays[band] = distance_decay(full)
            plot_matrices(coh_mats, ordered, amyg_mask, 'coherence',
                          os.path.join(args.out_dir, f'coh_lead_{lead}.png'))
            plot_distance_decay(coh_decays, 'coherence',
                                os.path.join(args.out_dir, f'decay_coh_lead_{lead}.png'))

    if not HAVE_CONN:
        print("NOTE: mne_connectivity not installed — coherence skipped.")
    print(f"\nStage 13c done. Outputs in {args.out_dir}/ (13b output preserved).")


if __name__ == '__main__':
    main()
