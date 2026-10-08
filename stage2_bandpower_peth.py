#!/usr/bin/env python
# stage2_bandpower_peth.py
"""
STAGE 2: band power & PETH (Analysis 1).

For each bipolar channel and each of six canonical bands, compute time-resolved
power within every epoch (Hilbert envelope of the band-filtered signal), then:
  (i)  PETH  = trial-averaged power time course (mean +/- SEM), per channel/band/
       event, using each channel's own good trials (good_tc from Stage 1).
  (ii) per-trial band power = one number per trial (mean power in the analysis
       window), saved for Stage 3 (n-1 analyses).

Baseline normalization (default): z-score each trial's band-power time course
against a baseline window at the start of the epoch (configurable). This makes
PETHs comparable across channels/bands. For 'feedback', the pre-event portion
serves as a natural baseline.

Bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, low_gamma 30-70,
       high_gamma 70-150 Hz.

Inputs: the epochs_<event>.npz files from Stage 1.
Outputs (in --out-dir):
  peth_<event>.npz        PETH mean/sem [chan x band x time] + time axis
  trial_power_<event>.csv per-trial band power (long format) for Stage 3
  peth_<event>_examples.png  quick-look PETH for a few strong channels

Usage:
  python stage2_bandpower_peth.py --stage1-dir stage1_out --out-dir stage2_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = {
    'delta':      (1, 4),
    'theta':      (4, 8),
    'alpha':      (8, 13),
    'beta':       (13, 30),
    'low_gamma':  (30, 70),
    'high_gamma': (70, 150),
}
EVENTS = ['bet_submitted', 'color_submitted', 'feedback']
# analysis window (where per-trial mean power is taken), relative to event (s)
ANALYSIS_WIN = {
    'bet_submitted':   (-1.0, 0.0),
    'color_submitted': (-1.0, 0.0),
    'feedback':        (-1.5, 1.0),
}
# baseline window for z-scoring (relative to event, s). For bet/color the only
# pre-decision reference is the start of the epoch; for feedback use pre-event.
BASELINE_WIN = {
    'bet_submitted':   (-1.0, -0.8),
    'color_submitted': (-1.0, -0.8),
    'feedback':        (-1.5, -1.0),
}


def bandpass(x, lo, hi, sf, order=4):
    ny = sf / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def band_power_timecourse(ep, sf):
    """ep [trials x chans x time] -> power [band x trials x chans x time].
    Power = squared Hilbert envelope of the band-filtered signal.
    NOTE: compute in float64 — squaring iEEG amplitudes (which can be large)
    overflows float32 and produces inf/nan. We keep double precision here and
    only downcast final summaries if needed."""
    ep = ep.astype(np.float64)
    n_tr, n_ch, n_t = ep.shape
    out = np.empty((len(BANDS), n_tr, n_ch, n_t), dtype=np.float64)
    for bi, (name, (lo, hi)) in enumerate(BANDS.items()):
        filt = bandpass(ep, lo, hi, sf)                # [tr x ch x t]
        env = np.abs(hilbert(filt, axis=-1))            # analytic amplitude
        out[bi] = env ** 2                              # power (float64)
    return out


def zscore_baseline(power, t, base_win):
    """power [band x tr x ch x t]; z-score each (band,trial,chan) time course
    to its own baseline-window mean/std."""
    b0, b1 = base_win
    bmask = (t >= b0) & (t < b1)
    if bmask.sum() < 2:
        bmask = t < (t[0] + 0.2)               # fallback: first 200 ms
    mu = power[:, :, :, bmask].mean(axis=3, keepdims=True)
    sd = power[:, :, :, bmask].std(axis=3, keepdims=True) + 1e-12
    return (power - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_out')
    ap.add_argument('--out-dir', default='stage2_out')
    ap.add_argument('--n-example', type=int, default=6,
                    help='how many example channels to plot per event')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))

    for ev in EVENTS:
        f = np.load(os.path.join(args.stage1_dir, f'epochs_{ev}.npz'),
                    allow_pickle=True)
        ep = f['data']                     # [tr x ch x t]  (PADDED)
        sf = float(f['sf'])
        win = f['win']                     # analysis window (pre, post)
        pad = float(f['pad']) if 'pad' in f else 0.0
        good_tc = f['good_tc']             # [tr x ch] bool
        names = f['bipolar_name']
        n_tr, n_ch, n_t = ep.shape

        print(f"[{ev}] {n_tr} trials x {n_ch} chans x {n_t} samples (padded); "
              f"computing band power ...")
        # Filter + Hilbert on the PADDED epoch so edge artifacts live in the pad.
        power = band_power_timecourse(ep, sf)          # [band x tr x ch x t]
        logpower = np.log(power + 1e-12)

        # Trim the pad off now that filtering is done.
        n_pad = int(round(pad * sf))
        if n_pad > 0:
            logpower = logpower[:, :, :, n_pad: n_t - n_pad]
        n_core = logpower.shape[3]
        t = np.arange(n_core) / sf + win[0]            # time axis, event at 0

        powerz = zscore_baseline(logpower, t, BASELINE_WIN[ev])

        # ---- PETH: mean/sem across each channel's OWN good trials ----
        n_band = len(BANDS)
        peth_mean = np.full((n_ch, n_band, n_core), np.nan, np.float32)
        peth_sem  = np.full((n_ch, n_band, n_core), np.nan, np.float32)
        for ch in range(n_ch):
            gt = good_tc[:, ch]
            if gt.sum() < 3:
                continue
            seg = powerz[:, gt, ch, :]                 # [band x good_tr x t]
            peth_mean[ch] = seg.mean(axis=1)
            peth_sem[ch]  = seg.std(axis=1) / np.sqrt(gt.sum())
        np.savez_compressed(
            os.path.join(args.out_dir, f'peth_{ev}.npz'),
            peth_mean=peth_mean, peth_sem=peth_sem, t=t,
            bands=np.array(list(BANDS.keys())), bipolar_name=names)

        # ---- per-trial band power (mean in analysis window), for Stage 3 ----
        a0, a1 = ANALYSIS_WIN[ev]
        amask = (t >= a0) & (t < a1)
        # mean over analysis window of the z-scored power -> [band x tr x ch]
        trial_val = powerz[:, :, :, amask].mean(axis=3)
        rows = []
        for bi, band in enumerate(BANDS):
            for ch in range(n_ch):
                for tr in range(n_tr):
                    if not good_tc[tr, ch]:
                        continue
                    rows.append({
                        'event': ev, 'band': band,
                        'bipolar_name': names[ch], 'trial_index': tr,
                        'power_z': float(trial_val[bi, tr, ch]),
                    })
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, f'trial_power_{ev}.csv'), index=False)

        # ---- quick-look figure: channels with the largest high-gamma response
        hg = list(BANDS).index('high_gamma')
        resp = np.nanmax(np.abs(peth_mean[:, hg, :]), axis=1)
        top = np.argsort(np.nan_to_num(resp))[::-1][:args.n_example]
        # One ROW per channel, one COLUMN per band, each band on its own y-scale
        # so high-gamma isn't visually crushed by low-frequency swings.
        n_band = len(BANDS)
        fig, axes = plt.subplots(len(top), n_band,
                                 figsize=(2.5*n_band, 2.0*len(top)),
                                 squeeze=False)
        for r, ch in enumerate(top):
            reg = anat.iloc[ch]['anode_region'] if ch < len(anat) else ''
            for bi, band in enumerate(BANDS):
                axcell = axes[r, bi]
                m = peth_mean[ch, bi]; s = peth_sem[ch, bi]
                axcell.plot(t, m, lw=1, color='C{}'.format(bi))
                axcell.fill_between(t, m-s, m+s, alpha=0.25,
                                    color='C{}'.format(bi))
                axcell.axvline(0, color='k', ls='--', lw=0.7)
                axcell.axhline(0, color='grey', ls=':', lw=0.5)
                if r == 0:
                    axcell.set_title(band, fontsize=9)
                if bi == 0:
                    axcell.set_ylabel(f"{names[ch]}\n{reg[:18]}", fontsize=6)
        for bi in range(n_band):
            axes[-1, bi].set_xlabel('t (s)', fontsize=8)
        fig.suptitle(f'PETH by band — {ev} (rows=top high-gamma channels, '
                     f'each band own y-scale)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'peth_{ev}_examples.png'), dpi=140)
        plt.close(fig)
        print(f"  saved peth_{ev}.npz, trial_power_{ev}.csv, "
              f"peth_{ev}_examples.png")

    print(f"\nStage 2 done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
