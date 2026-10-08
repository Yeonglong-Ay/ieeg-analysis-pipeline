#!/usr/bin/env python
# stage1_mne.py
"""
STAGE 1 (MNE version) of the iEEG analysis pipeline.

Same design decisions as the scipy version (stage1_preprocess_epoch.py), but
built on MNE-Python for robustness and to match lab convention. Kept as a
parallel implementation; the scipy version remains as an independent cross-check.

Steps:
  1. Load Hub1 iEEG via neo, wrap in mne.io.RawArray (channels 1..180 that are
     present in electrodes.csv; ignore un-localized amplifier channels).
  2. Bipolar reference with mne.set_bipolar_reference (within-lead adjacent
     pairs from electrodes.csv). Anatomy carried alongside.
  3. Notch filter line noise (60/120/180). High-pass 0.5 Hz to remove slow
     drift (this is the fix for the delta/theta artifact seen earlier).
  4. Epoch to the 3 events (bet/color/feedback) with PAD seconds extra each side.
  5. Per-channel-per-trial artifact rejection (NOT whole-epoch — correct for
     many-channel iEEG). Saves a good_tc mask [trials x chans].

Outputs (in --out-dir):
  bipolar_anatomy.csv          montage + anatomy per bipolar channel
  epochs_<event>-epo.fif       MNE Epochs (padded) for each event
  good_tc_<event>.npy          per-channel good-trial mask
  rejection_report.txt

Usage:
  python stage1_mne.py --ns3 Hub1-...ns3 --electrodes electrodes.csv \
      --events events_final_block1.csv events_final_block2.csv \
      --out-dir stage1_mne_out
"""
import argparse, os
import numpy as np
import pandas as pd
import mne
from neo.io import BlackrockIO

EVENTS = {
    'fixation':        (-0.5, 0.5),
    'bet_onset':       (-0.5, 0.5),
    'bet_submitted':   (-0.5, 0.0),
    'color_submitted': (-0.5, 0.0),
    'feedback':        (-1.5, 1.0),
}
# Self-paced epochs: drop trials whose reaction time is shorter than the
# pre-event window, so the window never bleeds into the prior task phase.
# (bet/color are self-paced; the others are experimenter-timed.)
RT_MIN = {'bet_submitted': 0.5, 'color_submitted': 0.5}
PAD = 1.0
NUM_PRACTICE = 5


def load_neural(ns3):
    r = BlackrockIO(filename=ns3)
    seg = r.read_block(lazy=False).segments[0]
    sig = seg.analogsignals[0]
    data = np.asarray(sig.magnitude).T          # [channels x samples], microvolts
    sf = float(sig.sampling_rate)
    return data, sf


def build_bipolar_table(elec_csv):
    e = pd.read_csv(elec_csv).sort_values(['LabelPrefix', 'ContactOrder'])
    rows = []
    for lead, grp in e.groupby('LabelPrefix', sort=False):
        grp = grp.sort_values('ContactOrder').reset_index(drop=True)
        for i in range(len(grp) - 1):
            a, b = grp.iloc[i], grp.iloc[i + 1]
            if int(b['ContactOrder']) - int(a['ContactOrder']) != 1:
                continue
            rows.append({
                'bipolar_name': f"{a['Label']}-{b['Label']}",
                'lead': lead,
                'anode_ch': f"ch{int(a['Electrode'])}",
                'cathode_ch': f"ch{int(b['Electrode'])}",
                'anode_idx': int(a['Electrode']) - 1,
                'cathode_idx': int(b['Electrode']) - 1,
                'anode_label': a['Label'], 'cathode_label': b['Label'],
                'anode_region': a['FSLabel'], 'cathode_region': b['FSLabel'],
                'is_white_matter': ('White-Matter' in str(a['FSLabel'])) or
                                   ('White-Matter' in str(b['FSLabel'])),
                'is_unknown': ('Unknown' in str(a['FSLabel'])) or
                              ('Unknown' in str(b['FSLabel'])),
                'mni_x': (a['MNI305_x'] + b['MNI305_x']) / 2.0,
                'mni_y': (a['MNI305_y'] + b['MNI305_y']) / 2.0,
                'mni_z': (a['MNI305_z'] + b['MNI305_z']) / 2.0,
            })
    return pd.DataFrame(rows)


def load_reaction_times(behav_csvs):
    """Per-trial RT for the self-paced epochs, keyed by (block, trial_number).
    bet RT = t_bet_response - t_bet_onset; color RT = t_color_response -
    t_color_onset. Returns dict: event -> {(block,trial_number): rt_seconds}."""
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    b['trial_number'] = b['exp_trial'] + NUM_PRACTICE
    def rt(a, z):
        return (pd.to_numeric(b[z], errors='coerce') -
                pd.to_numeric(b[a], errors='coerce'))
    keys = list(zip(b['block'], b['trial_number']))
    out = {
        'bet_submitted':   {k: v for k, v in zip(keys, rt('t_bet_onset', 't_bet_response'))},
        'color_submitted': {k: v for k, v in zip(keys, rt('t_color_onset', 't_color_response'))},
    }
    return out


def load_events(event_csvs):
    dfs = []
    for i, c in enumerate(event_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    ev = ev[ev['matched'] == True]
    ev = ev[ev['trial_number'] > NUM_PRACTICE]      # drop practice
    # keep onset times AND trial identity (block, trial_number) per event, so
    # downstream can trial-match a fixation baseline to the other events.
    onsets, ident = {}, {}
    for ph in EVENTS:
        sub = ev[ev['phase'] == ph]
        onsets[ph] = sub['ns3_onset_s'].values
        ident[ph] = sub[['block', 'trial_number']].reset_index(drop=True)
    return onsets, ident


def per_channel_reject(epochs_data, z_amp=6.0, z_var=6.0):
    """epochs_data [tr x ch x t] -> good_tc [tr x ch] (True=usable).
    Flat-channel detection is scale-invariant (relative to the median channel
    variance), so it works whether the data is in Volts or microvolts."""
    n_tr, n_ch, n_t = epochs_data.shape
    peak = np.max(np.abs(epochs_data), axis=2)
    var = np.var(epochs_data, axis=2)
    chan_med_var = np.median(var, axis=0)          # per-channel typical variance
    # a channel is 'flat' only if its variance is a tiny fraction of the
    # ACROSS-channel median (relative, unit-free) — catches truly dead channels
    # without assuming microvolt units.
    global_med = np.median(chan_med_var) + 1e-30
    bad_chan = chan_med_var < (1e-4 * global_med)

    def mad(x):
        med = np.median(x, axis=0, keepdims=True)
        return np.median(np.abs(x - med), axis=0) * 1.4826
    amp_z = (peak - np.median(peak, 0)) / (mad(peak) + 1e-30)
    var_z = np.abs((var - np.median(var, 0)) / (mad(var) + 1e-30))
    good_tc = (amp_z <= z_amp) & (var_z <= z_var)
    good_tc[:, bad_chan] = False
    return good_tc, bad_chan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns3', required=True)
    ap.add_argument('--electrodes', required=True)
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage1_mne_out')
    ap.add_argument('--z-amp', type=float, default=6.0)
    ap.add_argument('--z-var', type=float, default=6.0)
    ap.add_argument('--highpass', type=float, default=0.5)
    ap.add_argument('--pretask-s', type=float, default=20.0,
                    help='seconds of pre-task recording to save as baseline')
    ap.add_argument('--behav', nargs='+', default=None,
                    help='pilot_results CSVs, for RT-based exclusion of fast '
                         'self-paced trials (bet/color). If omitted, no RT filter.')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    print("Loading neural data ...")
    data, sf = load_neural(args.ns3)
    n_ch_raw = data.shape[0]

    print("Building bipolar table ...")
    bip = build_bipolar_table(args.electrodes)
    bip.to_csv(os.path.join(args.out_dir, 'bipolar_anatomy.csv'), index=False)
    n_used = int(max(bip['anode_idx'].max(), bip['cathode_idx'].max()) + 1)
    report.append(f"Raw channels: {n_ch_raw}; localized in use: {n_used}; "
                  f"bipolar channels: {len(bip)}")

    # Build a Raw with generic ch names ch1..chN (MNE needs names to bipolar-ref)
    ch_names = [f"ch{i+1}" for i in range(n_ch_raw)]
    info = mne.create_info(ch_names, sf, ch_types='eeg')
    # MNE expects Volts; Blackrock magnitude is microVolts -> scale to V
    raw = mne.io.RawArray(data * 1e-6, info, verbose='ERROR')

    print("Bipolar referencing ...")
    raw_bip = mne.set_bipolar_reference(
        raw, anode=list(bip['anode_ch']), cathode=list(bip['cathode_ch']),
        ch_name=list(bip['bipolar_name']), verbose='ERROR')
    # keep only the bipolar channels
    raw_bip.pick(list(bip['bipolar_name']))

    print("Notch + high-pass filtering ...")
    raw_bip.notch_filter([60, 120, 180], verbose='ERROR')
    raw_bip.filter(l_freq=args.highpass, h_freq=None, verbose='ERROR')
    report.append(f"Filters: notch 60/120/180 Hz; high-pass {args.highpass} Hz")

    onsets, ident = load_events(args.events)
    rt_lut = load_reaction_times(args.behav) if args.behav else {}
    for ph, win in EVENTS.items():
        report.append(f"Event '{ph}': {len(onsets[ph])} experimental onsets")

    # ---- pre-task baseline segment: grab up to PRETASK_S seconds of recording
    # before the FIRST event onset (task-free rest), for an absolute baseline. ----
    first_onset = min(np.min(onsets[ph]) for ph in EVENTS if len(onsets[ph]))
    pretask_end = first_onset - 1.0                # 1s guard before first event
    pretask_start = max(0.0, pretask_end - args.pretask_s)
    if pretask_end - pretask_start >= 2.0:          # need >=2s to be usable
        s0 = int(round(pretask_start * sf)); s1 = int(round(pretask_end * sf))
        pretask = raw_bip.get_data(start=s0, stop=s1)   # [ch x samples]
        np.savez_compressed(os.path.join(args.out_dir, 'pretask_baseline.npz'),
                            data=pretask.astype(np.float32), sf=sf,
                            start_s=pretask_start, end_s=pretask_end)
        report.append(f"\nPre-task baseline: {pretask_start:.1f}-{pretask_end:.1f}s "
                      f"({pretask_end-pretask_start:.1f}s) saved")
        print(f"Pre-task baseline: {pretask_end-pretask_start:.1f}s before first event")
    else:
        report.append(f"\nPre-task baseline: NOT available "
                      f"(only {pretask_end-pretask_start:.1f}s before first event)")
        print(f"WARNING: insufficient pre-task recording "
              f"({pretask_end-pretask_start:.1f}s); pre-task baseline unavailable")

    print("Epoching + per-channel artifact rejection ...")
    for ph, win in EVENTS.items():
        tmin, tmax = win[0] - PAD, win[1] + PAD
        # build MNE events array from onset times (sample indices)
        samp = np.round(np.array(onsets[ph]) * sf).astype(int)
        # drop events too close to recording edges
        good = (samp + int(tmin*sf) >= 0) & (samp + int(tmax*sf) < raw_bip.n_times)
        # For self-paced epochs, also drop trials whose RT < the pre-event
        # window, so the analysis window never bleeds into the prior phase.
        n_rt_dropped = 0
        if ph in RT_MIN and ph in rt_lut:
            rt_thresh = RT_MIN[ph]
            rt_ok = np.ones(len(samp), bool)
            for i, (_, r) in enumerate(ident[ph].iterrows()):
                key = (int(r['block']), int(r['trial_number']))
                rt_val = rt_lut[ph].get(key, np.nan)
                if not (np.isfinite(rt_val) and rt_val >= rt_thresh):
                    rt_ok[i] = False
            n_rt_dropped = int((good & ~rt_ok).sum())
            good = good & rt_ok
        evarr = np.column_stack([samp, np.zeros_like(samp), np.ones_like(samp)])
        evarr = evarr[good]
        ident_ph = ident[ph].iloc[good].reset_index(drop=True)   # keep identity
        epochs = mne.Epochs(raw_bip, evarr, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, reject=None,
                            verbose='ERROR')
        # per-channel rejection on the ANALYSIS-window core (exclude pad)
        core = epochs.copy().crop(win[0], win[1]).get_data()
        good_tc, bad_chan = per_channel_reject(core, args.z_amp, args.z_var)
        np.save(os.path.join(args.out_dir, f'good_tc_{ph}.npy'), good_tc)
        ident_ph.to_csv(os.path.join(args.out_dir, f'ident_{ph}.csv'),
                        index=False)                # trial identity per epoch
        epochs.save(os.path.join(args.out_dir, f'epochs_{ph}-epo.fif'),
                    overwrite=True, verbose='ERROR')
        med = int(np.median(good_tc.sum(0))); mn = int(good_tc.sum(0).min())
        mx = int(good_tc.sum(0).max())
        report.append(
            f"\n[{ph}] window {win} + pad {PAD}s -> tmin/tmax {tmin}/{tmax}")
        if ph in RT_MIN:
            report.append(f"  RT filter: dropped {n_rt_dropped} trials with "
                          f"RT < {RT_MIN[ph]}s (self-paced window guard)")
        report.append(
            f"  epochs: {len(epochs)} (of {len(samp)} onsets)")
        report.append(
            f"  per-channel good trials: median {med}, range {mn}-{mx}; "
            f"bad/flat chans: {int(bad_chan.sum())}")
        print(f"  {ph}: {len(epochs)} epochs, median {med} good trials/chan")

    with open(os.path.join(args.out_dir, 'rejection_report.txt'), 'w') as f:
        f.write("\n".join(report))
    print(f"\nStage 1 (MNE) done. Outputs in {args.out_dir}/")
    print("\n".join(report))


if __name__ == '__main__':
    main()
