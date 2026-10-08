#!/usr/bin/env python
# stage12_coherence.py
"""
STAGE 12: connectivity analysis (two complementary parts).

PART A — WITHIN-REGION HOMOGENEITY (validates region-averaging)
  For each region, correlate the band-power time courses between all pairs of
  channels within that region (Pearson r, averaged over trials, then over pairs).
  High r  -> channels behave similarly -> region-averaging is justified.
  Low  r  -> channels are heterogeneous -> averaging may wash out signal.
  We use CORRELATION (not wPLI) here because we WANT to detect similarity,
  including shared local signal — the question is 'do these channels do the
  same thing', not 'is there long-range communication'.

PART B — FRONTAL-LIMBIC COUPLING (synchronisation)
  Compute weighted Phase-Lag Index (wPLI) between frontal and limbic channels,
  per band. wPLI is robust to volume conduction (discards zero-lag coupling),
  so it reflects genuine long-range coupling rather than shared sources.
  (a) characterise overall coupling; (b) compare loss vs win and long vs short
  streak. HONEST CAVEAT: condition splits are trial-limited and wPLI has a
  trial-count bias, so condition comparisons are exploratory.

Bands analysed: theta, alpha, beta (theta = classic frontal-limbic; alpha/beta
= where amygdala showed streak effects). All epochs.

Requires: mne_connectivity (for wPLI). If absent, Part B is skipped with a note.

Usage:
  python stage12_coherence.py --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv \
      --out-dir stage12_out
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

EVENTS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
NUM_PRACTICE = 5
LIMBIC = ['amygdala', 'hippocampus', 'insula']
FRONTAL = ['frontal', 'orbitofrontal', 'cingulate']


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


def bandpass(x, lo, hi, sf, order=4):
    ny = sf/2.0
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, x, axis=-1)


def band_amplitude(epoch_data, lo, hi, sf):
    """epoch_data [tr x ch x t] -> Hilbert amplitude envelope, same shape."""
    filt = bandpass(epoch_data.astype(np.float64), lo, hi, sf)
    return np.abs(hilbert(filt, axis=-1))


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
    st = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_dir'] = st.str[0].map({'L': 'loss', 'W': 'win'})
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    b['streak_len'] = sl.where(sl.notna(), pd.to_numeric(st.str[1:], errors='coerce'))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage12_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    reg = anat['anode_region'].map(region_of)
    reg = reg.where(reg.notna(), anat['cathode_region'].map(region_of))
    anat['region'] = reg
    behav = load_behaviour(args.behav)

    partA_rows, partB_rows = [], []

    for ev in EVENTS:
        epo_file = os.path.join(args.stage1_dir, f'epochs_{ev}-epo.fif')
        id_file = os.path.join(args.stage1_dir, f'ident_{ev}.csv')
        if not (os.path.exists(epo_file) and os.path.exists(id_file)):
            print(f"  (skip {ev}: missing inputs)"); continue
        epochs = mne.read_epochs(epo_file, preload=True, verbose='ERROR')
        sf = epochs.info['sfreq']
        names = epochs.ch_names
        ch_region = {nm: anat.loc[anat['bipolar_name'] == nm, 'region'].iloc[0]
                     for nm in names if (anat['bipolar_name'] == nm).any()}
        data = epochs.get_data()             # [tr x ch x t]
        ident = pd.read_csv(id_file)
        merged = ident.merge(behav, on=['block', 'trial_number'], how='left')

        # crop to analysis window (remove pad) via epochs.times
        # (use full epoch; wPLI/corr computed over whole epoch is fine)

        for band, (lo, hi) in BANDS.items():
            amp = band_amplitude(data, lo, hi, sf)   # [tr x ch x t]

            # ---- PART A: within-region homogeneity (power-corr) ----
            for rg in set(v for v in ch_region.values() if v):
                idx = [i for i, nm in enumerate(names)
                       if ch_region.get(nm) == rg]
                if len(idx) < 2:
                    continue
                # correlate band-amplitude time courses between channel pairs,
                # averaged over trials, then over pairs
                rs = []
                for a_i in range(len(idx)):
                    for b_i in range(a_i+1, len(idx)):
                        ca, cb = idx[a_i], idx[b_i]
                        # per-trial correlation of the two channels' envelopes
                        tr_r = []
                        for tr in range(amp.shape[0]):
                            xa, xb = amp[tr, ca], amp[tr, cb]
                            if np.std(xa) > 0 and np.std(xb) > 0:
                                tr_r.append(np.corrcoef(xa, xb)[0, 1])
                        if tr_r:
                            rs.append(np.mean(tr_r))
                if rs:
                    partA_rows.append({
                        'event': ev, 'band': band, 'region': rg,
                        'n_chan': len(idx), 'n_pairs': len(rs),
                        'mean_within_r': float(np.mean(rs)),
                        'sd_within_r': float(np.std(rs))})

            # ---- PART B: frontal-limbic wPLI ----
            if HAVE_CONN:
                fidx = [i for i, nm in enumerate(names)
                        if ch_region.get(nm) in FRONTAL]
                lidx = [i for i, nm in enumerate(names)
                        if ch_region.get(nm) in LIMBIC]
                if len(fidx) >= 1 and len(lidx) >= 1:
                    for cond_name, mask in [
                            ('all', np.ones(len(merged), bool)),
                            ('loss', merged['win'].values == False),
                            ('win', merged['win'].values == True),
                            ('long_streak', (merged['streak_dir'].values == 'loss') &
                                            (merged['streak_len'].values >= 3)),
                            ('short_streak', (merged['streak_dir'].values == 'loss') &
                                             (merged['streak_len'].values <= 2))]:
                        sel = data[mask]
                        if len(sel) < 5:
                            continue
                        # wPLI between all channels, then average frontal-limbic pairs
                        con = spectral_connectivity_epochs(
                            sel, method='wpli', mode='multitaper',
                            sfreq=sf, fmin=lo, fmax=hi, faverage=True,
                            verbose='ERROR')
                        cmat = con.get_data(output='dense')[:, :, 0]  # [ch x ch]
                        vals = [cmat[max(f, l), min(f, l)]
                                for f in fidx for l in lidx]
                        vals = [v for v in vals if np.isfinite(v) and v != 0]
                        if vals:
                            partB_rows.append({
                                'event': ev, 'band': band, 'condition': cond_name,
                                'n_trials': int(mask.sum()),
                                'n_pairs': len(vals),
                                'mean_wpli': float(np.mean(vals)),
                                'sd_wpli': float(np.std(vals))})
        print(f"  [{ev}] done")

    # save + plot Part A
    dfA = pd.DataFrame(partA_rows)
    dfA.to_csv(os.path.join(args.out_dir, 'within_region_homogeneity.csv'),
               index=False)
    if len(dfA):
        # heatmap-ish bar: mean within-region r per region, averaged over bands/events
        summ = dfA.groupby('region')['mean_within_r'].agg(['mean', 'std', 'count'])
        fig, ax = plt.subplots(figsize=(8, 4))
        summ = summ.sort_values('mean', ascending=False)
        ax.bar(range(len(summ)), summ['mean'], yerr=summ['std'], capsize=3,
               color='#55A868')
        ax.set_xticks(range(len(summ))); ax.set_xticklabels(summ.index, rotation=30)
        ax.set_ylabel('mean within-region power correlation')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title('Within-region homogeneity (higher = channels more alike;\n'
                     'validates region-averaging)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'within_region_homogeneity.png'),
                    dpi=140)
        plt.close(fig)

    # save + plot Part B
    if HAVE_CONN and partB_rows:
        dfB = pd.DataFrame(partB_rows)
        dfB.to_csv(os.path.join(args.out_dir, 'frontal_limbic_wpli.csv'),
                   index=False)
        # plot: wPLI by condition, per band, at each epoch (all-condition first)
        for band in BANDS:
            sub = dfB[dfB['band'] == band]
            fig, ax = plt.subplots(figsize=(10, 4))
            conds = ['all', 'loss', 'win', 'long_streak', 'short_streak']
            evs = [e for e in EVENTS if e in set(sub['event'])]
            x = np.arange(len(evs)); w = 0.15
            for ci, cond in enumerate(conds):
                vals = [sub[(sub['event'] == e) & (sub['condition'] == cond)]
                        ['mean_wpli'].values for e in evs]
                vals = [v[0] if len(v) else 0 for v in vals]
                ax.bar(x + ci*w, vals, w, label=cond)
            ax.set_xticks(x + w*2); ax.set_xticklabels(evs, rotation=20, fontsize=8)
            ax.set_ylabel('frontal-limbic wPLI'); ax.set_title(
                f'Frontal-limbic coupling (wPLI) — {band}', fontsize=11)
            ax.legend(fontsize=7, ncol=5)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f'frontal_limbic_wpli_{band}.png'),
                        dpi=140)
            plt.close(fig)
    elif not HAVE_CONN:
        with open(os.path.join(args.out_dir, 'PART_B_SKIPPED.txt'), 'w') as f:
            f.write("Part B (frontal-limbic wPLI) skipped: mne_connectivity not "
                    "installed.\nInstall with:  pip install mne-connectivity\n"
                    "then re-run. Part A (within-region homogeneity) completed.")
        print("\nNOTE: mne_connectivity not found — Part B skipped. "
              "Install with: pip install mne-connectivity")

    print(f"\nStage 12 done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
