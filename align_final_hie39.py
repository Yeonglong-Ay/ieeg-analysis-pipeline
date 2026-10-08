#!/usr/bin/env python
# align_final.py
"""
Definitive alignment of task events (.tsv, ground truth) to photodiode pulses
(.ns3, precise timing) for the two-block patient recording.

Ground truth established:
  - each block .tsv: 77 of each phase, 385 flashes, 77 true spins
  - both blocks: 770 intended flashes, 154 true spins
  - .ns3: 766 clean pulses (verified stable across thresholds) → 4 dropped
  - the .ns3 had 164 gaps near 4.5 s, but only 154 are real spins; the .tsv
    resolves which events are real, so we anchor on the .tsv.

Method (per block):
  1. Split the continuous .ns3 at the between-block break (largest gap, ~74 s).
  2. Robust clock fit: map .tsv time -> .ns3 time using a first/last-anchor line,
     then refine by least squares on confident nearest matches.
  3. Greedy nearest matching, each .ns3 pulse used AT MOST ONCE. A .tsv event
     with no free pulse within tolerance = a dropped flash (flagged).
  4. Sanity checks: matched ≤ pulses, no pulse reused, residuals reported.

Output per block: trial_number | phase | tsv_time | ns3_onset_s | matched
Epoch on ns3_onset_s where matched==True; use trial_number/phase for labels.

Usage:
    python align_final.py --ns3 NSP-...ns3 \
        --tsv1 ...block1...tsv --tsv2 ...block2...tsv \
        --match-tol 0.4 --out-prefix events_final
"""
import argparse
import numpy as np
import pandas as pd
from neo.io import BlackrockIO

PHOTODIODE_CH = 'ainp1'


def detect_onsets(ns3, min_gap_ms=25, min_dur_ms=15):
    r = BlackrockIO(filename=ns3)
    s = r.read_block(lazy=False).segments[0].analogsignals[0]
    data = np.asarray(s.magnitude).T
    sf = float(s.sampling_rate)
    ch = list(s.array_annotations.get('channel_names', []))
    idx = ch.index(PHOTODIODE_CH) if PHOTODIODE_CH in ch else 0
    sig = data[idx]
    thr = sig.min()/2 + sig.max()/2
    b = (sig >= thr).astype(int)
    ups = np.where(np.diff(b) > 0)[0] + 1
    downs = np.where(np.diff(b) < 0)[0] + 1
    raw = []
    for u in ups:
        later = downs[downs > u]
        if len(later):
            raw.append([u, later[0]])
    mg = min_gap_ms/1000*sf
    merged = [raw[0]] if raw else []
    for on, off in raw[1:]:
        if on - merged[-1][1] <= mg:
            merged[-1][1] = off
        else:
            merged.append([on, off])
    md = min_dur_ms/1000*sf
    return np.array([on/sf for on, off in merged if (off-on) >= md])


def load_tsv(tsv):
    df = pd.read_csv(tsv, sep='\t')
    fl = df[df['code'].astype(str) == 'FLASH'].reset_index(drop=True)
    tsv_t = fl['timestamp'].astype(float).values
    phase = fl['hex'].astype(str).values
    label = fl['label'].astype(str).values
    # trial number: carry forward from WHEEL_SPIN 'trial=N' rows if present,
    # else index by counting fixations. We derive a per-flash trial index by
    # grouping every 5 flashes (fixation starts a trial).
    trial = np.zeros(len(fl), dtype=int)
    tnum = 0
    for i, ph in enumerate(phase):
        if ph == 'fixation':
            tnum += 1
        trial[i] = tnum
    return tsv_t, phase, label, trial


def fit_clock(tsv_t, ns3_on):
    """Linear map ns3 ≈ a*tsv + b. Anchor BOTH clocks to their own first event
    before fitting, otherwise fitting ns3 (~0-1200 s) against raw Unix epoch tsv
    (~1.78e9) gives a degenerate near-zero slope. We fit on the anchored
    (relative) times, then fold the offsets back into (a, b)."""
    tsv0 = tsv_t[0]
    ns30 = ns3_on[0]
    tr = tsv_t - tsv0          # relative task time (0 .. ~1057 s)
    nr = ns3_on - ns30         # relative neural time (0 .. ~1200 s)
    # Both clocks run in real seconds, so the rate is ≈1.0. Start there (robust
    # even if the first/last events are dropped) and refine.
    a, b = 1.0, 0.0
    # refine on confident nearest matches (in relative time). Start with a wide
    # tolerance so it can lock on even if the first event was dropped, then
    # tighten to reject bad pairs.
    for tol_refine in (2.0, 0.5, 0.2):
        pred = a * tr + b
        near_n, near_t = [], []
        for i, p in enumerate(pred):
            j = np.argmin(np.abs(nr - p))
            if abs(nr[j] - p) < tol_refine:
                near_n.append(nr[j]); near_t.append(tr[i])
        if len(near_t) > 10:
            A = np.vstack([near_t, np.ones(len(near_t))]).T
            (a, b), *_ = np.linalg.lstsq(A, near_n, rcond=None)
    # Fold anchors back so caller can use raw tsv times:
    # ns3 = a*(tsv - tsv0) + b + ns30 = a*tsv + (b + ns30 - a*tsv0)
    b_full = b + ns30 - a * tsv0
    return a, b_full


def align_block(ns3_on, tsv_t, phase, label, trial, tol, name, out_csv):
    a, b = fit_clock(tsv_t, ns3_on)
    pred = a * tsv_t + b
    used = np.zeros(len(ns3_on), dtype=bool)
    rows = []
    for i in range(len(tsv_t)):
        # nearest FREE pulse to predicted time
        d = np.abs(ns3_on - pred[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        matched = d[j] <= tol
        if matched:
            used[j] = True
        rows.append({'trial_number': int(trial[i]), 'phase': phase[i],
                     'label': label[i], 'tsv_time': tsv_t[i],
                     'ns3_onset_s': ns3_on[j] if matched else np.nan,
                     'matched': matched,
                     'resid_ms': (ns3_on[j]-pred[i])*1000 if matched else np.nan})
    out = pd.DataFrame(rows)

    nmatch = int(out['matched'].sum())
    ndrop = int((~out['matched']).sum())
    resid = out.loc[out['matched'], 'resid_ms']
    print(f"\n[{name}] tsv events={len(tsv_t)}, ns3 pulses={len(ns3_on)}")
    print(f"  clock fit: slope={a:.6f} (≈1.0 if clocks same rate)")
    print(f"  matched={nmatch}, dropped(no pulse)={ndrop}, "
          f"pulses used={used.sum()}/{len(ns3_on)}")
    # sanity checks
    assert used.sum() == nmatch, "pulse-reuse bug!"
    assert nmatch <= len(ns3_on), "matched more than pulses!"
    print(f"  residual after fit: mean {resid.mean():+.1f}ms, "
          f"std {resid.std():.1f}ms, max|{resid.abs().max():.1f}|ms  ✓ small = good")
    if ndrop:
        print(f"  Dropped flashes (trial, phase):")
        for _, r in out[~out['matched']].iterrows():
            print(f"    trial {r['trial_number']:>3}  {r['phase']}")
    out.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns3', required=True)
    ap.add_argument('--tsv1', required=True)
    ap.add_argument('--tsv2', required=True)
    ap.add_argument('--match-tol', type=float, default=0.4)
    ap.add_argument('--out-prefix', default='events_final')
    args = ap.parse_args()

    onsets = detect_onsets(args.ns3)
    print(f"Total .ns3 pulses: {len(onsets)}")

    t1, p1, l1, tr1 = load_tsv(args.tsv1)
    t2, p2, l2, tr2 = load_tsv(args.tsv2)
    n1_tsv, n2_tsv = len(t1), len(t2)
    print(f".tsv flash counts: block1={n1_tsv}, block2={n2_tsv} "
          f"(total {n1_tsv+n2_tsv} vs {len(onsets)} pulses)")

    # Split by COUNT, not by largest gap. Block 1's log has n1_tsv flashes, so
    # the first n1_tsv pulses are block 1 and the remainder block 2. This is
    # robust when the block break is NOT the largest gap (e.g., when
    # within-block interruptions rival the break, and block 2 is incomplete).
    # We look for a gap near the expected boundary to sanity-check, allowing for
    # a few dropped pulses.
    gaps = np.diff(onsets)
    # candidate boundary: around index n1_tsv-1 (0-based), search a small window
    lo = max(0, n1_tsv - 6); hi = min(len(gaps), n1_tsv + 5)
    if hi > lo:
        local = lo + int(np.argmax(gaps[lo:hi]))
        split = local
        print(f"Boundary search near pulse {n1_tsv}: largest gap in "
              f"[{lo},{hi}) at pulse {split} ({gaps[split]:.1f}s)")
    else:
        split = n1_tsv - 1
    b1, b2 = onsets[:split+1], onsets[split+1:]
    print(f"Split: block1={len(b1)} pulses (tsv {n1_tsv}), "
          f"block2={len(b2)} pulses (tsv {n2_tsv})")

    o1 = align_block(b1, t1, p1, l1, tr1, args.match_tol, 'BLOCK 1',
                     f'{args.out_prefix}_block1.csv')
    o2 = align_block(b2, t2, p2, l2, tr2, args.match_tol, 'BLOCK 2',
                     f'{args.out_prefix}_block2.csv')

    tot_match = int(o1['matched'].sum() + o2['matched'].sum())
    print(f"\n=== TOTAL matched {tot_match} / {len(onsets)} pulses; "
          f"{770-tot_match} events dropped ===")
    print("Epoch on ns3_onset_s where matched==True.")


if __name__ == '__main__':
    main()
