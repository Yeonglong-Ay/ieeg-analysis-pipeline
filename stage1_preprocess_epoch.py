#!/usr/bin/env python
# stage1_preprocess_epoch.py
"""
STAGE 1 of the iEEG analysis pipeline.

Loads the neural recording, builds a proper anatomical BIPOLAR montage from
electrodes.csv, notch-filters line noise, epochs to the three task events using
the photodiode-aligned event times, runs artifact detection, and rejects bad
epochs/channels. Saves clean epoched data + an anatomy table + a rejection
report for downstream stages (band power / PETH, and n-1 analyses).

Design decisions (agreed with user):
  - Use only channels present in electrodes.csv (real localized iEEG contacts);
    ignore un-localized amplifier channels.
  - Bipolar = adjacent contacts WITHIN each lead (LabelPrefix), ordered by
    ContactOrder. Never pair across leads.
  - Keep white-matter / Unknown contacts but FLAG them (exclude later if wanted).
  - For each bipolar channel record BOTH contacts' anatomy + midpoint MNI.
  - Exclude practice trials and the 4 dropped bet_submitted trials (unmatched).

Events & windows:
  bet_submitted   : -1.0 to  0.0 s
  color_submitted : -1.0 to  0.0 s
  feedback        : -1.5 to +1.0 s   (anticipation into outcome)

Outputs (in --out-dir):
  bipolar_anatomy.csv                 montage + anatomy per bipolar channel
  epochs_<event>.npz                  clean epochs: data [trials x chans x time]
  rejection_report.txt                what was rejected and why

Usage:
  python stage1_preprocess_epoch.py \
      --ns3 Hub1-...ns3 --electrodes electrodes.csv \
      --events events_final_block1.csv events_final_block2.csv \
      --out-dir stage1_out
"""
import argparse, os
import numpy as np
import pandas as pd
from neo.io import BlackrockIO
from scipy.signal import iirnotch, filtfilt

# analysis windows (relative to event, s). The data is actually epoched with
# extra PAD seconds on each side; Stage 2 filters on the padded epoch and trims
# the pad off, so filter edge artifacts never enter the analysis window.
EVENTS = {
    'bet_submitted':   (-1.0, 0.0),
    'color_submitted': (-1.0, 0.0),
    'feedback':        (-1.5, 1.0),
}
PAD = 1.0   # seconds of padding added to EACH side before filtering


# ----------------------------------------------------------------------------- #
def load_neural(ns3):
    """Load all channels as [n_channels x n_samples], plus sampling rate."""
    r = BlackrockIO(filename=ns3)
    seg = r.read_block(lazy=False).segments[0]
    sig = seg.analogsignals[0]
    data = np.asarray(sig.magnitude).T          # -> [channels x samples]
    sf = float(sig.sampling_rate)
    names = list(sig.array_annotations.get('channel_names', []))
    return data, sf, names


def build_bipolar(elec_csv):
    """From electrodes.csv build the list of bipolar pairs (within-lead,
    adjacent by ContactOrder) with anatomy for each. Returns a DataFrame."""
    e = pd.read_csv(elec_csv)
    e = e.sort_values(['LabelPrefix', 'ContactOrder']).reset_index(drop=True)
    rows = []
    for lead, grp in e.groupby('LabelPrefix', sort=False):
        grp = grp.sort_values('ContactOrder').reset_index(drop=True)
        for i in range(len(grp) - 1):
            a, b = grp.iloc[i], grp.iloc[i + 1]
            # adjacency check: consecutive ContactOrder only
            if int(b['ContactOrder']) - int(a['ContactOrder']) != 1:
                continue
            rows.append({
                'bipolar_name': f"{a['Label']}-{b['Label']}",
                'lead': lead,
                'chan_anode':   int(a['Electrode']),   # 1-based channel number
                'chan_cathode': int(b['Electrode']),
                'anode_label':  a['Label'],
                'cathode_label': b['Label'],
                'anode_region':  a['FSLabel'],
                'cathode_region': b['FSLabel'],
                'is_white_matter': bool(
                    ('White-Matter' in str(a['FSLabel'])) or
                    ('White-Matter' in str(b['FSLabel']))),
                'is_unknown': bool(
                    ('Unknown' in str(a['FSLabel'])) or
                    ('Unknown' in str(b['FSLabel']))),
                'mni_x': (a['MNI305_x'] + b['MNI305_x']) / 2.0,
                'mni_y': (a['MNI305_y'] + b['MNI305_y']) / 2.0,
                'mni_z': (a['MNI305_z'] + b['MNI305_z']) / 2.0,
            })
    return pd.DataFrame(rows)


def apply_bipolar(data, bip):
    """data [chans x samples] (0-based rows = channel_number-1).
    Returns bipolar_data [n_bipolar x samples]."""
    out = np.empty((len(bip), data.shape[1]), dtype=np.float32)
    for i, r in bip.iterrows():
        out[i] = data[r['chan_anode'] - 1] - data[r['chan_cathode'] - 1]
    return out


def notch_filter(data, sf, freqs=(60, 120, 180), q=30.0):
    """Zero-phase notch at line noise + harmonics."""
    out = data.copy()
    for f0 in freqs:
        if f0 >= sf / 2:
            continue
        b, a = iirnotch(f0, q, sf)
        out = filtfilt(b, a, out, axis=1)
    return out


def load_events(event_csvs):
    """Concatenate events_final CSVs; keep matched, non-practice events.
    Returns dict: event_phase -> array of ns3 onset times (s)."""
    dfs = []
    for i, c in enumerate(event_csvs):
        d = pd.read_csv(c)
        d['block'] = i + 1
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    # matched only (drops the 4 unmatched bet_submitted), and experimental only
    # trial_number is per-block; practice = first NUM_PRACTICE (5) trials.
    ev = ev[ev['matched'] == True].copy()
    ev = ev[ev['trial_number'] > 5]           # exclude 5 practice trials/block
    out = {}
    for phase in EVENTS:
        out[phase] = ev[ev['phase'] == phase]['ns3_onset_s'].values
    return out, ev


def epoch(data, sf, onsets, win):
    """Cut [n_trials x n_chans x n_time] epochs around onsets (s)."""
    pre, post = win
    n_pre, n_post = int(round(pre * sf)), int(round(post * sf))
    length = n_post - n_pre
    n_ch = data.shape[0]
    keep, ep = [], []
    for t in onsets:
        s0 = int(round(t * sf)) + n_pre
        s1 = s0 + length
        if s0 < 0 or s1 > data.shape[1]:
            keep.append(False); continue
        ep.append(data[:, s0:s1]); keep.append(True)
    if ep:
        arr = np.stack(ep, axis=0)             # [trials x chans x time]
    else:
        arr = np.empty((0, n_ch, length), dtype=np.float32)
    return arr, np.array(keep)


def reject_artifacts(ep, z_amp=6.0, z_var=6.0, flat_uv=1.0):
    """
    PER-CHANNEL, PER-TRIAL artifact rejection (correct for many-channel iEEG).

    A trial is marked bad *for a given channel* if that channel's signal on that
    trial is an outlier relative to that channel's own distribution across trials.
    We do NOT reject a whole trial across all channels because one channel had an
    artifact — with 167 channels that throws away almost everything. Instead each
    channel keeps its own good-trial mask, and downstream analyses average over
    each channel's good trials.

    Returns:
      good_tc  : boolean [trials x chans]  (True = usable for that chan/trial)
      bad_chan : boolean [chans]           (globally bad/flat channels)
      info     : summary dict
    """
    n_tr, n_ch, n_t = ep.shape
    if n_tr == 0:
        return np.zeros((0, n_ch), bool), np.zeros(n_ch, bool), {}

    peak = np.max(np.abs(ep), axis=2)          # [trials x chans]
    var = np.var(ep, axis=2)                    # [trials x chans]

    # flat / dead channels: near-zero variance across all trials
    chan_med_var = np.median(var, axis=0)
    bad_chan = chan_med_var < (flat_uv ** 2)

    def mad(x, axis=0):
        med = np.median(x, axis=axis, keepdims=True)
        return np.median(np.abs(x - med), axis=axis) * 1.4826

    # robust per-channel thresholds (columns = channels)
    amp_med = np.median(peak, axis=0); amp_mad = mad(peak, 0) + 1e-9
    var_med = np.median(var, axis=0);  var_mad = mad(var, 0) + 1e-9
    amp_z = (peak - amp_med) / amp_mad          # [trials x chans]
    var_z = np.abs((var - var_med) / var_mad)

    # per-channel-per-trial good mask
    good_tc = (amp_z <= z_amp) & (var_z <= z_var)
    good_tc[:, bad_chan] = False                # bad channels: no good trials

    # summary
    frac_rej_per_chan = 1.0 - good_tc.mean(axis=0)
    info = {
        'n_trials': int(n_tr), 'n_chans': int(n_ch),
        'median_good_trials_per_chan': int(np.median(good_tc.sum(axis=0))),
        'min_good_trials_per_chan': int(good_tc.sum(axis=0).min()),
        'max_good_trials_per_chan': int(good_tc.sum(axis=0).max()),
        'mean_frac_rejected': float(frac_rej_per_chan.mean()),
        'n_bad_chan': int(bad_chan.sum()),
        'bad_chan_idx': np.where(bad_chan)[0].tolist(),
    }
    return good_tc, bad_chan, info


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns3', required=True)
    ap.add_argument('--electrodes', required=True)
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage1_out')
    ap.add_argument('--z-amp', type=float, default=6.0)
    ap.add_argument('--z-var', type=float, default=6.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    print("Loading neural data ...")
    data, sf, names = load_neural(args.ns3)
    report.append(f"Neural: {data.shape[0]} channels x {data.shape[1]} samples @ {sf:.0f} Hz")

    print("Building bipolar montage from electrodes.csv ...")
    bip = build_bipolar(args.electrodes)
    bip.to_csv(os.path.join(args.out_dir, 'bipolar_anatomy.csv'), index=False)
    report.append(f"Bipolar channels: {len(bip)} (from {bip['lead'].nunique()} leads)")
    report.append(f"  white-matter bipolar: {int(bip['is_white_matter'].sum())}, "
                  f"unknown: {int(bip['is_unknown'].sum())}")

    # guard: all referenced channels must exist in the recording
    max_ch = max(bip['chan_anode'].max(), bip['chan_cathode'].max())
    if max_ch > data.shape[0]:
        raise SystemExit(f"electrodes.csv references channel {max_ch} but recording "
                         f"has only {data.shape[0]} channels.")

    print("Applying bipolar reference ...")
    bdata = apply_bipolar(data, bip)
    del data

    print("Notch filtering line noise (60/120/180) ...")
    bdata = notch_filter(bdata, sf)

    print("Loading aligned events ...")
    onsets, ev = load_events(args.events)
    for ph in EVENTS:
        report.append(f"Event '{ph}': {len(onsets[ph])} experimental (matched) onsets")

    print("Epoching + artifact rejection ...")
    for ph, win in EVENTS.items():
        # Epoch with PAD extra seconds on each side. Artifact rejection uses the
        # ANALYSIS window only (not the padded edges), so padding doesn't affect
        # rejection. Stage 2 filters on the padded data then trims the pad.
        padded_win = (win[0] - PAD, win[1] + PAD)
        ep, keep = epoch(bdata, sf, onsets[ph], padded_win)
        # For artifact rejection, look only at the analysis-window portion
        n_pad = int(round(PAD * sf))
        ep_core = ep[:, :, n_pad: ep.shape[2] - n_pad] if n_pad > 0 else ep
        good_tc, bad_chan, info = reject_artifacts(ep_core, args.z_amp, args.z_var)
        # Save ALL in-bounds PADDED epochs plus the per-channel good-trial mask.
        # Downstream (Stage 2) filters on padded data, trims PAD, uses good_tc.
        np.savez_compressed(
            os.path.join(args.out_dir, f'epochs_{ph}.npz'),
            data=ep.astype(np.float32), sf=sf, win=np.array(win),
            padded_win=np.array(padded_win), pad=PAD,
            good_tc=good_tc, bad_chan=bad_chan,
            bipolar_name=bip['bipolar_name'].values,
        )
        report.append(f"\n[{ph}]  window {win[0]}..{win[1]}s")
        report.append(
            f"  onsets in-bounds: {int(keep.sum())}/{len(keep)}")
        report.append(
            f"  per-channel good trials: median "
            f"{info.get('median_good_trials_per_chan',0)}, range "
            f"{info.get('min_good_trials_per_chan',0)}-"
            f"{info.get('max_good_trials_per_chan',0)} (of {info.get('n_trials',0)})")
        report.append(
            f"  mean fraction rejected per channel: "
            f"{info.get('mean_frac_rejected',0)*100:.1f}%; "
            f"bad/flat channels: {info.get('n_bad_chan',0)} "
            f"{info.get('bad_chan_idx',[])}")
        print(f"  {ph}: median {info.get('median_good_trials_per_chan',0)}"
              f"/{info.get('n_trials',0)} good trials per channel "
              f"({info.get('mean_frac_rejected',0)*100:.1f}% rejected)")

    with open(os.path.join(args.out_dir, 'rejection_report.txt'), 'w') as f:
        f.write("\n".join(report))
    print(f"\nStage 1 done. Outputs in {args.out_dir}/")
    print("\n".join(report))


if __name__ == '__main__':
    main()
