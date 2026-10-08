#!/usr/bin/env python
# stage14_region_connectivity.py
"""
STAGE 14: region-to-region spectral connectivity (coherence) displayed as
circular connectivity graphs (connectograms), per band, per epoch.

Pipeline:
  1. Compute channel x channel coherence (MNE spectral_connectivity_epochs),
     per band, per epoch.
  2. Aggregate to REGION x REGION by averaging coherence across all channel
     pairs spanning each region pair.
  3. Significance: permutation null (shuffle trials to break trial-locked
     coupling), recompute the region-pair coherence many times, keep region
     pairs whose observed coherence exceeds the (1 - alpha) percentile of the
     null. FDR-correct across region pairs.
  4. Draw a circular connectivity graph per band per epoch with only the
     SIGNIFICANT region-pair connections (arc width ~ coherence strength).

Coherence (not wPLI) is used: the data are bipolar-referenced, which already
substantially reduces volume conduction, so plain coherence is defensible here.
(If volume conduction were a concern, wPLI would be the conservative choice.)

NOTE: this is computationally heavy (permutation x bands x epochs) — run as a
SLURM job. Requires mne_connectivity.

Usage:
  python stage14_region_connectivity.py --stage1-dir stage1_mne_out \
      --out-dir stage14_out --n-perm 500 --alpha 0.05
"""
import argparse, os
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mne_connectivity import spectral_connectivity_epochs
try:
    from mne_connectivity.viz import plot_connectivity_circle
    HAVE_CIRCLE = True
except Exception:
    try:
        from mne.viz import plot_connectivity_circle   # older MNE location
        HAVE_CIRCLE = True
    except Exception:
        HAVE_CIRCLE = False

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'low_gamma': (30, 70), 'high_gamma': (70, 150)}
EPOCHS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
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
    if 'front_sup' in s: return 'superior_frontal'
    if 'front_middle' in s: return 'middle_frontal'
    if 'front_inf' in s: return 'inferior_frontal'
    if 'front' in s or 'subcentral' in s: return 'frontal_other'
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return None


def channel_coherence(data, sf, lo, hi):
    """data [tr x ch x t] -> ch x ch coherence matrix for the band."""
    con = spectral_connectivity_epochs(
        data, method='coh', mode='multitaper', sfreq=sf,
        fmin=lo, fmax=hi, faverage=True, verbose='ERROR')
    cmat = con.get_data(output='dense')[:, :, 0]
    return cmat + cmat.T       # symmetrize (lower-triangular -> full)


def region_matrix(chan_coh, ch_regions, regions):
    """Average channel-coherence into a region x region matrix."""
    R = len(regions)
    M = np.full((R, R), np.nan)
    ridx = {r: [i for i, cr in enumerate(ch_regions) if cr == r] for r in regions}
    for a in range(R):
        for b in range(R):
            ia, ib = ridx[regions[a]], ridx[regions[b]]
            if not ia or not ib:
                continue
            if a == b:
                # within-region: average off-diagonal pairs
                vals = [chan_coh[i, j] for i in ia for j in ia if i < j]
            else:
                vals = [chan_coh[i, j] for i in ia for j in ib]
            if vals:
                M[a, b] = np.mean(vals)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--out-dir', default='stage14_out')
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(region_of)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(region_of)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    all_rows = []
    for ev in EPOCHS:
        fpath = os.path.join(args.stage1_dir, f"epochs_{ev}-epo.fif")
        if not os.path.exists(fpath):
            print(f"  (skip {ev}: no epochs file)"); continue
        epochs = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        sf = epochs.info['sfreq']
        names = epochs.ch_names
        ch_regions = [ch2region.get(nm) for nm in names]
        data = epochs.get_data()
        # regions present with >=1 channel
        regions = [r for r in REGION_ORDER
                   if any(cr == r for cr in ch_regions)]
        print(f"\n=== {ev}: {len(regions)} regions, {data.shape[0]} trials ===")

        for band, (lo, hi) in BANDS.items():
            obs_ch = channel_coherence(data, sf, lo, hi)
            obs_M = region_matrix(obs_ch, ch_regions, regions)

            # permutation null: shuffle trials independently per channel to
            # break trial-locked coupling, recompute region matrix
            R = len(regions)
            null = np.full((args.n_perm, R, R), np.nan)
            n_tr = data.shape[0]
            for p in range(args.n_perm):
                perm_data = np.empty_like(data)
                for c in range(data.shape[1]):
                    perm_data[:, c, :] = data[rng.permutation(n_tr), c, :]
                pch = channel_coherence(perm_data, sf, lo, hi)
                null[p] = region_matrix(pch, ch_regions, regions)

            # p-value per region pair: fraction of null >= observed
            pvals = np.full((R, R), np.nan)
            for a in range(R):
                for b in range(R):
                    if np.isfinite(obs_M[a, b]):
                        nd = null[:, a, b]; nd = nd[np.isfinite(nd)]
                        if len(nd):
                            pvals[a, b] = (np.sum(nd >= obs_M[a, b]) + 1) / (len(nd) + 1)

            # FDR across the unique upper-triangle region pairs
            iu = np.triu_indices(R, k=1)
            praw = pvals[iu]
            mask = np.isfinite(praw)
            sig = np.zeros_like(praw, dtype=bool)
            if mask.sum():
                from statsmodels.stats.multitest import multipletests
                sig_m, _, _, _ = multipletests(praw[mask], alpha=args.alpha,
                                               method='fdr_bh')
                sig[np.where(mask)[0]] = sig_m

            # build significant connectivity matrix (for the circle plot)
            sig_M = np.zeros((R, R))
            for k, (a, b) in enumerate(zip(*iu)):
                if sig[k]:
                    sig_M[a, b] = sig_M[b, a] = obs_M[a, b]
                    all_rows.append({'epoch': ev, 'band': band,
                                     'region_a': regions[a], 'region_b': regions[b],
                                     'coherence': obs_M[a, b], 'p': pvals[a, b],
                                     'significant': True})

            # ---- circular connectivity graph ----
            if HAVE_CIRCLE and sig_M.any():
                fig = plt.figure(figsize=(7, 7), facecolor='white')
                try:
                    plot_connectivity_circle(
                        sig_M, regions, n_lines=None,
                        title=f'{ev} — {band} (significant coherence, FDR)',
                        facecolor='white', textcolor='black', fig=fig,
                        colormap='viridis', vmin=0,
                        vmax=float(np.nanmax(obs_M)))
                except TypeError:
                    # older signature fallback
                    plot_connectivity_circle(
                        sig_M, regions,
                        title=f'{ev} — {band}', fig=fig)
                outp = os.path.join(args.out_dir, f'circle_{ev}_{band}.png')
                fig.savefig(outp, dpi=140, facecolor='white'); plt.close(fig)
                print(f"    {band}: {int(sig_M.astype(bool).sum()/2)} "
                      f"significant connections -> {outp}")
            else:
                print(f"    {band}: {'no significant connections' if HAVE_CIRCLE else 'circle plot unavailable'}")

    # save all significant connections as a table
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.out_dir, 'significant_connections.csv'), index=False)
        print(f"\nsaved significant_connections.csv "
              f"({len(all_rows)} significant region-pair-band-epoch entries)")
    print(f"\nStage 14 done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
