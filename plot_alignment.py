#!/usr/bin/env python
# plot_alignment.py
"""
Make a presentation-quality figure showing that photodiode pulses align with
task events and can be epoched correctly.

Panel A: raw photodiode trace over a few trials, each pulse labeled with its
         recovered phase (from .tsv order-matching). Trial starts marked.
Panel B: an example epoch — the raw trace around one bet_submitted event with
         the −1 to 0 s analysis window shaded, showing how epoching uses the
         alignment.

Usage:
    python plot_alignment.py --ns3 NSP-...ns3 --tsv trigger_log_...tsv \
        --trials-start 11 --n-trials 3 --out alignment.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neo.io import BlackrockIO

PHOTODIODE_CH = 'ainp1'
PHASE_ORDER = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
PHASE_COLORS = {
    'fixation': '#4C72B0', 'bet_onset': '#DD8452', 'bet_submitted': '#C44E52',
    'color_submitted': '#8172B3', 'feedback': '#55A868',
}


def load_signal(ns3):
    r = BlackrockIO(filename=ns3)
    sig = r.read_block(lazy=False).segments[0].analogsignals[0]
    data = np.asarray(sig.magnitude).T
    sf = float(sig.sampling_rate)
    ch = list(sig.array_annotations.get('channel_names', []))
    idx = ch.index(PHOTODIODE_CH) if PHOTODIODE_CH in ch else 0
    return data[idx], sf


def detect_pulses(sig, sf, min_gap_ms=25, min_dur_ms=15):
    thr = sig.min()/2 + sig.max()/2
    b = (sig >= thr).astype(int)
    ups = np.where(np.diff(b) > 0)[0] + 1
    downs = np.where(np.diff(b) < 0)[0] + 1
    raw = []
    for u in ups:
        later = downs[downs > u]
        if len(later):
            raw.append([u, later[0]])
    min_gap = min_gap_ms/1000*sf
    merged = [raw[0]] if raw else []
    for on, off in raw[1:]:
        if on - merged[-1][1] <= min_gap:
            merged[-1][1] = off
        else:
            merged.append([on, off])
    min_dur = min_dur_ms/1000*sf
    return [(on/sf, off/sf) for on, off in merged if (off-on) >= min_dur], thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns3', required=True)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--trials-start', type=float, default=11.0,
                    help='time (s) to start the Panel A window')
    ap.add_argument('--n-trials', type=int, default=3)
    ap.add_argument('--out', default='alignment.png')
    args = ap.parse_args()

    sig, sf = load_signal(args.ns3)
    pulses, thr = detect_pulses(sig, sf)

    # Recover phase labels by order-matching to the .tsv
    df = pd.read_csv(args.tsv, sep='\t')
    flashes = df[df['code'].astype(str).isin(['FLASH']) |
                 df['code'].astype(str).str.startswith('DUR')].reset_index(drop=True)
    n = min(len(pulses), len(flashes))
    events = []
    for i in range(n):
        onset = pulses[i][0]
        offset = pulses[i][1]
        phase = str(flashes.iloc[i]['hex'])
        events.append({'onset': onset, 'offset': offset, 'phase': phase})
    ev = pd.DataFrame(events)

    # Panel A window: from trials-start, spanning ~n_trials
    # (estimate ~10 s per trial including spin)
    win_dur = args.n_trials * 10.0
    a0, a1 = args.trials_start, args.trials_start + win_dur

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(16, 9))

    # ---- Panel A: labeled pulse train ----
    s0, s1 = int(a0*sf), int(a1*sf)
    s1 = min(s1, len(sig))
    t = np.arange(s0, s1) / sf
    axA.plot(t, sig[s0:s1], lw=0.6, color='black')
    axA.axhline(thr, color='grey', ls=':', lw=0.8)
    hi = sig.max()
    seen = set()
    for _, e in ev.iterrows():
        if a0 <= e['onset'] < a1:
            c = PHASE_COLORS.get(e['phase'], 'grey')
            # shade the pulse
            axA.axvspan(e['onset'], e['offset'], color=c, alpha=0.35,
                        label=e['phase'] if e['phase'] not in seen else None)
            seen.add(e['phase'])
            # label above
            axA.annotate(e['phase'], xy=(e['onset'], hi*1.02),
                         rotation=45, fontsize=8, color=c, ha='left', va='bottom')
    axA.set_title('A. Photodiode pulses aligned to task events '
                  '(phase recovered from trigger log)', fontsize=12, loc='left')
    axA.set_xlabel('Time (s)'); axA.set_ylabel('Photodiode (raw)')
    axA.set_ylim(sig.min()-100, hi*1.25)
    axA.legend(loc='upper right', ncol=5, fontsize=8, framealpha=0.9)

    # ---- Panel B: example epoch around a bet_submitted event ----
    bet_events = ev[ev['phase'] == 'bet_submitted']['onset'].values
    if len(bet_events):
        anchor = bet_events[len(bet_events)//2]  # a middle one
        pre, post = 1.5, 0.5
        b0, b1 = int((anchor-pre)*sf), int((anchor+post)*sf)
        tb = np.arange(b0, b1)/sf - anchor  # time relative to event
        axB.plot(tb, sig[b0:b1], lw=0.8, color='black')
        axB.axhline(thr, color='grey', ls=':', lw=0.8)
        axB.axvline(0, color=PHASE_COLORS['bet_submitted'], lw=2,
                    label='bet_submitted (t=0)')
        # Shade the -1 to 0 s analysis window
        axB.axvspan(-1.0, 0.0, color=PHASE_COLORS['bet_submitted'], alpha=0.15,
                    label='epoch window (−1 to 0 s)')
        axB.set_title('B. Example epoch: −1 to 0 s window locked to a '
                      'bet_submitted pulse', fontsize=12, loc='left')
        axB.set_xlabel('Time relative to bet_submitted (s)')
        axB.set_ylabel('Photodiode (raw)')
        axB.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    print(f"Panel A window: {a0:.0f}–{a1:.0f}s, "
          f"{sum(a0 <= e < a1 for e in ev['onset'])} events labeled")
    print(f"Total events recovered: {len(ev)} "
          f"({'counts match' if len(pulses)==len(flashes) else 'COUNT MISMATCH'})")


if __name__ == '__main__':
    main()
