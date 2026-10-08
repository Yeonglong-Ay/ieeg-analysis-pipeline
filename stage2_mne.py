#!/usr/bin/env python
# stage2_mne.py
"""
STAGE 2 (MNE version): band power & PETH via Morlet wavelets.

Reads the MNE Epochs from stage1_mne.py, computes time-frequency power with
Morlet wavelets, averages within the six canonical bands, applies a z-score
baseline, and produces:
  (i)  PETH  = trial-averaged band power (mean+/-SEM) per channel/band/event,
       using each channel's own good trials (good_tc from Stage 1).
  (ii) per-trial band power (mean in the analysis window) for Stage 3.

Morlet TFR handles padding/edge effects internally; we still epoched with a pad
in Stage 1, and we crop to the analysis window after TFR. Baseline: zscore vs a
pre-event window (configurable), which with the 0.5 Hz high-pass from Stage 1
should remove the earlier low-frequency artifact.

Bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, low_gamma 30-70,
       high_gamma 70-150 Hz.

Outputs (in --out-dir):
  peth_<event>.npz         PETH mean/sem [chan x band x time] + time axis
  trial_power_<event>.csv  per-trial band power (long) for Stage 3
  peth_<event>_examples.png

Usage:
  python stage2_mne.py --stage1-dir stage1_mne_out --out-dir stage2_mne_out
"""
import argparse, os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_array_morlet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'low_gamma': (30, 70), 'high_gamma': (70, 150),
}
EVENTS = {
    'fixation':        (-0.5, 0.5),
    'bet_onset':       (-0.5, 0.5),
    'bet_submitted':   (-1.0, 0.0),
    'color_submitted': (-1.0, 0.0),
    'feedback':        (-1.5, 1.0),
}
BASELINE_WIN = {
    'fixation':        (-0.5, -0.3),
    'bet_onset':       (-0.5, -0.3),
    'bet_submitted':   (-1.0, -0.8),
    'color_submitted': (-1.0, -0.8),
    'feedback':        (-1.5, -1.0),
}
# When fixation is used as the BASELINE for other events, this is the clean
# sub-window of the fixation epoch used as the reference (padding trimmed).
# Pre-flash (-0.5..0): the period just BEFORE the fixation cross appears,
# matching Overton's pre-stimulus baseline convention (most neutral reference,
# provided the inter-trial interval is long enough not to overlap the prior
# trial's feedback).
FIX_BASELINE_WIN = (-0.5, 0.0)
# frequencies to sample for the TFR (log-spaced within the overall range)
FREQS = np.logspace(np.log10(2), np.log10(150), 40)

REGION_ORDER = ['orbitofrontal', 'insula', 'cingulate', 'amygdala',
                'hippocampus', 'thalamus', 'temporal', 'frontal']
REGION_MAP = True   # enable per-region trial time-course output


def region_of(fslabel):
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


def band_indices(freqs):
    """Map each band to the indices of FREQS falling inside it."""
    idx = {}
    for b, (lo, hi) in BANDS.items():
        idx[b] = np.where((freqs >= lo) & (freqs < hi))[0]
    return idx


def log_band_power(data, sf, bidx, decim, chunk, n_cycles):
    """data [tr x ch x t] -> log band power [band x tr x ch x t_decim].
    Chunked over channels for memory. Returns power and the decim factor's
    implied length (caller supplies time axis)."""
    n_tr, n_ch, n_t = data.shape
    n_band = len(BANDS)
    # figure out decimated length from a tiny probe is overkill; MNE returns
    # ceil(n_t/decim). We allocate after first chunk to match exactly.
    out = None
    for c0 in range(0, n_ch, chunk):
        c1 = min(c0 + chunk, n_ch)
        tfr = tfr_array_morlet(data[:, c0:c1, :], sfreq=sf, freqs=FREQS,
                               n_cycles=n_cycles, output='power',
                               zero_mean=True, decim=decim, verbose='ERROR')
        logtfr = np.log(tfr + 1e-20).astype(np.float32)
        del tfr
        if out is None:
            out = np.empty((n_band, n_tr, n_ch, logtfr.shape[-1]), np.float32)
        for bi, b in enumerate(BANDS):
            out[bi, :, c0:c1, :] = logtfr[:, :, bidx[b], :].mean(axis=2)
        del logtfr
    return out


def compute_reference(mode, stage1_dir, ev_ident, sf, bidx, decim, chunk,
                      n_cycles, names):
    """Compute baseline reference mean/std per band/chan for the given mode.
    Returns (mu, sd) each shaped [band x 1 x chan x 1] for 'pretask' (shared
    across trials) OR [band x n_trial_key x chan x 1] keyed for 'fixation'
    (trial-matched). For 'self', returns (None, None) — handled inline.
    ev_ident: DataFrame of (block, trial_number) for the current event's epochs.
    """
    if mode == 'self':
        return None, None, None

    if mode == 'pretask':
        f = os.path.join(stage1_dir, 'pretask_baseline.npz')
        if not os.path.exists(f):
            raise SystemExit("pretask baseline requested but pretask_baseline.npz "
                             "not found — insufficient pre-task recording in Stage 1.")
        d = np.load(f)
        pre = d['data'][None, :, :]                 # [1 x ch x samples]
        bp = log_band_power(pre, sf, bidx, decim, chunk, n_cycles)  # [band x 1 x ch x t]
        mu = bp.mean(axis=3, keepdims=True)          # [band x 1 x ch x 1]
        sd = bp.std(axis=3, keepdims=True) + 1e-12
        return mu, sd, 'shared'

    if mode == 'fixation':
        # per-trial fixation-period band power, trial-matched by (block,trial).
        fx = mne.read_epochs(os.path.join(stage1_dir, 'epochs_fixation-epo.fif'),
                             preload=True, verbose='ERROR')
        fx_ident = pd.read_csv(os.path.join(stage1_dir, 'ident_fixation.csv'))
        fxdata = fx.get_data()
        # Compute band power on the FULL padded epoch (so filter/TFR edge
        # artifacts stay in the pad), THEN crop to the clean fixation-core
        # window before collapsing across time. (Previously the pad was NOT
        # trimmed, so the baseline was contaminated by edge artifacts and by
        # time bleeding into adjacent events — this fixes that.)
        bp = log_band_power(fxdata, sf, bidx, decim, chunk, n_cycles)
        # decimated time axis, matched to how the main events build theirs
        fx_times = fx.times[::decim]
        # guard against off-by-one between bp length and the decimated axis
        n = min(bp.shape[3], len(fx_times))
        bp = bp[:, :, :, :n]; fx_times = fx_times[:n]
        b0, b1 = FIX_BASELINE_WIN
        cmask = (fx_times >= b0) & (fx_times <= b1)
        if cmask.sum() < 2:
            raise SystemExit(f"fixation baseline window {FIX_BASELINE_WIN} "
                             f"selects <2 samples — check window/epoch.")
        bp = bp[:, :, :, cmask]           # crop to clean fixation core
        # collapse time -> per-trial fixation mean/std [band x tr x ch]
        fx_mu = bp.mean(axis=3)                       # [band x tr_fx x ch]
        fx_sd = bp.std(axis=3) + 1e-12
        # build a lookup keyed by (block, trial_number)
        key = list(zip(fx_ident['block'], fx_ident['trial_number']))
        mu_lut = {k: fx_mu[:, i, :] for i, k in enumerate(key)}
        sd_lut = {k: fx_sd[:, i, :] for i, k in enumerate(key)}
        return mu_lut, sd_lut, 'trialmatched'

    raise SystemExit(f"unknown baseline mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--out-dir', default='stage2_mne_out')
    ap.add_argument('--n-example', type=int, default=6)
    ap.add_argument('--decim', type=int, default=10,
                    help='TFR time-downsample factor (band power is smooth; '
                         '10 -> 200Hz from 2kHz, big memory savings)')
    ap.add_argument('--chunk', type=int, default=20,
                    help='channels processed per Morlet chunk (lower = less memory)')
    ap.add_argument('--baseline-mode', choices=['self', 'fixation', 'pretask'],
                    default='fixation',
                    help="baseline for z-scoring: 'self'=window within each "
                         "epoch (old); 'fixation'=each trial's fixation period "
                         "(Overton-style); 'pretask'=rest before task start")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    # channel -> region map (anode region, fallback cathode) for per-region TC
    _reg = anat['anode_region'].map(region_of)
    _reg = _reg.where(_reg.notna(), anat['cathode_region'].map(region_of))
    ch2region = dict(zip(anat['bipolar_name'], _reg))
    bidx = band_indices(FREQS)
    DECIM = args.decim          # downsample TFR time axis (band power is smooth)
    CHUNK = args.chunk          # channels processed per chunk

    for ev, win in EVENTS.items():
        epochs = mne.read_epochs(
            os.path.join(args.stage1_dir, f'epochs_{ev}-epo.fif'),
            preload=True, verbose='ERROR')
        good_tc = np.load(os.path.join(args.stage1_dir, f'good_tc_{ev}.npy'))
        sf = epochs.info['sfreq']
        names = epochs.ch_names
        data = epochs.get_data()                   # [tr x ch x t] (padded, Volts)
        n_tr, n_ch, n_t = data.shape
        n_band = len(BANDS)
        n_cycles = FREQS / 2.0

        # decimated time axis (event at 0), then crop pad -> analysis window.
        # MNE's tfr_array_morlet(decim=D) returns every D-th sample, matching
        # epochs.times[::D]. We build the axis the same way so they align.
        times = epochs.times[::DECIM]
        amask = (times >= win[0]) & (times <= win[1])
        t = times[amask]
        print(f"[{ev}] {n_tr} trials x {n_ch} chans; Morlet TFR "
              f"(decim={DECIM}, chunk={CHUNK}) ...")

        # We compute band power in CHANNEL CHUNKS to keep memory small. For each
        # chunk we get the Morlet TFR (already time-decimated by MNE), take log,
        # average within bands, crop the pad, and store just the band power.
        # Peak memory is set by one chunk's TFR, not all 167 channels at once.
        bandpow = np.empty((n_band, n_tr, n_ch, t.size), np.float32)
        for c0 in range(0, n_ch, CHUNK):
            c1 = min(c0 + CHUNK, n_ch)
            tfr = tfr_array_morlet(
                data[:, c0:c1, :], sfreq=sf, freqs=FREQS, n_cycles=n_cycles,
                output='power', zero_mean=True, decim=DECIM, verbose='ERROR')
            # safety: MNE's decimated time dim must match our times[::DECIM]
            if tfr.shape[-1] != times.size:
                raise SystemExit(
                    f"TFR time dim {tfr.shape[-1]} != expected {times.size}; "
                    f"decim alignment problem — check MNE version behavior.")
            # tfr: [tr x chunk_ch x freq x t_decim]; crop pad, log, band-average
            tfr = tfr[:, :, :, amask]
            logtfr = np.log(tfr + 1e-20).astype(np.float32)
            del tfr
            for bi, b in enumerate(BANDS):
                bandpow[bi, :, c0:c1, :] = logtfr[:, :, bidx[b], :].mean(axis=2)
            del logtfr
            print(f"    channels {c0}-{c1} done")

        # ---- baseline normalization according to --baseline-mode ----
        if args.baseline_mode == 'self':
            # old behavior: window within each epoch
            b0, b1 = BASELINE_WIN[ev]
            bm = (t >= b0) & (t < b1)
            if bm.sum() < 2:
                bm = t < (t[0] + 0.2)
            mu = bandpow[:, :, :, bm].mean(axis=3, keepdims=True)
            sd = bandpow[:, :, :, bm].std(axis=3, keepdims=True) + 1e-12
            powerz = (bandpow - mu) / sd
        elif args.baseline_mode == 'pretask':
            mu, sd, kind = compute_reference('pretask', args.stage1_dir, None,
                                             sf, bidx, DECIM, CHUNK, n_cycles, names)
            powerz = (bandpow - mu) / sd            # broadcast [band x1 xch x1]
        elif args.baseline_mode == 'fixation':
            mu_lut, sd_lut, kind = compute_reference(
                'fixation', args.stage1_dir, None, sf, bidx, DECIM, CHUNK,
                n_cycles, names)
            ev_ident = pd.read_csv(os.path.join(args.stage1_dir,
                                                f'ident_{ev}.csv'))
            powerz = np.empty_like(bandpow)
            n_missing = 0
            for tr in range(n_tr):
                k = (int(ev_ident['block'].iloc[tr]),
                     int(ev_ident['trial_number'].iloc[tr]))
                if k in mu_lut:
                    # mu_lut[k], sd_lut[k] are [band x ch]; broadcast over time
                    powerz[:, tr, :, :] = (
                        (bandpow[:, tr, :, :] - mu_lut[k][:, :, None]) /
                        sd_lut[k][:, :, None])
                else:
                    # no matching fixation trial: fall back to self-window
                    n_missing += 1
                    b0, b1 = BASELINE_WIN[ev]; bm = (t >= b0) & (t < b1)
                    if bm.sum() < 2: bm = t < (t[0] + 0.2)
                    m = bandpow[:, tr, :, bm].mean(axis=2, keepdims=True)
                    s = bandpow[:, tr, :, bm].std(axis=2, keepdims=True) + 1e-12
                    powerz[:, tr, :, :] = (bandpow[:, tr, :, :] - m) / s
            if n_missing:
                print(f"    [{ev}] {n_missing}/{n_tr} trials had no matching "
                      f"fixation baseline; used self-window fallback")
        del bandpow

        # PETH per channel using each channel's own good trials
        peth_mean = np.full((n_ch, n_band, t.size), np.nan, np.float32)
        peth_sem = np.full((n_ch, n_band, t.size), np.nan, np.float32)
        for ch in range(n_ch):
            gt = good_tc[:, ch]
            if gt.sum() < 3:
                continue
            seg = powerz[:, gt, ch, :]
            peth_mean[ch] = seg.mean(axis=1)
            peth_sem[ch] = seg.std(axis=1) / np.sqrt(gt.sum())
        np.savez_compressed(
            os.path.join(args.out_dir, f'peth_{ev}.npz'),
            peth_mean=peth_mean, peth_sem=peth_sem, t=t,
            bands=np.array(list(BANDS)), bipolar_name=np.array(names))

        # ---- per-REGION, per-trial time courses (for condition-split PETHs) ----
        # Average powerz across each region's good channels, per trial, keeping
        # the time axis. Much smaller than per-channel (band x tr x region x t),
        # and per-region is what the condition-split PETH tool needs.
        if REGION_MAP is not None:
            regions_here = [ch2region.get(nm) for nm in names]
            uniq_reg = [r for r in REGION_ORDER if r in set(regions_here)]
            reg_tc = np.full((n_band, n_tr, len(uniq_reg), t.size),
                             np.nan, np.float32)
            for ri, rg in enumerate(uniq_reg):
                ch_idx = [i for i, r in enumerate(regions_here) if r == rg]
                if not ch_idx:
                    continue
                for tr in range(n_tr):
                    # average only channels good on this trial
                    good_ch = [i for i in ch_idx if good_tc[tr, i]]
                    if good_ch:
                        reg_tc[:, tr, ri, :] = powerz[:, tr, good_ch, :].mean(axis=1)
            np.savez_compressed(
                os.path.join(args.out_dir, f'region_trial_tc_{ev}.npz'),
                reg_tc=reg_tc, t=t, bands=np.array(list(BANDS)),
                regions=np.array(uniq_reg))

        # per-trial band power (mean over analysis window)
        trial_val = powerz.mean(axis=3)             # [band x tr x ch]
        rows = []
        for bi, b in enumerate(BANDS):
            for ch in range(n_ch):
                for tr in range(n_tr):
                    if not good_tc[tr, ch]:
                        continue
                    rows.append({'event': ev, 'band': b,
                                 'bipolar_name': names[ch], 'trial_index': tr,
                                 'power_z': float(trial_val[bi, tr, ch])})
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, f'trial_power_{ev}.csv'), index=False)

        # quick-look figure (per-band own y-scale), top high-gamma channels
        hg = list(BANDS).index('high_gamma')
        resp = np.nanmax(np.abs(peth_mean[:, hg, :]), axis=1)
        top = np.argsort(np.nan_to_num(resp))[::-1][:args.n_example]
        fig, axes = plt.subplots(len(top), n_band,
                                 figsize=(2.5*n_band, 2.0*len(top)),
                                 squeeze=False)
        for r, ch in enumerate(top):
            reg = anat.iloc[ch]['anode_region'] if ch < len(anat) else ''
            for bi, b in enumerate(BANDS):
                a = axes[r, bi]
                m = peth_mean[ch, bi]; s = peth_sem[ch, bi]
                a.plot(t, m, lw=1, color=f'C{bi}')
                a.fill_between(t, m-s, m+s, alpha=0.25, color=f'C{bi}')
                a.axvline(0, color='k', ls='--', lw=0.7)
                a.axhline(0, color='grey', ls=':', lw=0.5)
                if r == 0: a.set_title(b, fontsize=9)
                if bi == 0: a.set_ylabel(f"{names[ch]}\n{str(reg)[:18]}", fontsize=6)
        for bi in range(n_band):
            axes[-1, bi].set_xlabel('t (s)', fontsize=8)
        fig.suptitle(f'PETH by band (MNE) — {ev}', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'peth_{ev}_examples.png'), dpi=140)
        plt.close(fig)
        print(f"  saved peth_{ev}.npz, trial_power_{ev}.csv, figure")

    print(f"\nStage 2 (MNE) done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
