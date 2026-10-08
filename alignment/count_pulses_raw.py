#!/usr/bin/env python
# count_pulses_raw.py
"""
Carefully count photodiode pulses in the .ns3, independently of any alignment
or block-splitting, and test how sensitive the count is to detection settings.

Purpose: verify whether "766 pulses" is a real number or an artifact of the
threshold / debounce choices. Reports:
  - raw threshold-crossing count at several threshold levels (no debounce)
  - count after debounce merging at several min-gap settings
  - the distribution of pulse widths and inter-pulse gaps
  - how many pulses fall in the two time regions (before/after the big break)
so we can see the count is stable (or not) before trusting it.

Usage:
    python count_pulses_raw.py --ns3 NSP-...ns3
"""
import argparse
import numpy as np
from neo.io import BlackrockIO

PHOTODIODE_CH = 'ainp1'


def load(ns3):
    r = BlackrockIO(filename=ns3)
    s = r.read_block(lazy=False).segments[0].analogsignals[0]
    data = np.asarray(s.magnitude).T
    sf = float(s.sampling_rate)
    ch = list(s.array_annotations.get('channel_names', []))
    idx = ch.index(PHOTODIODE_CH) if PHOTODIODE_CH in ch else 0
    return data[idx], sf, (ch[idx] if ch else str(idx))


def raw_crossings(sig, thr):
    b = (sig >= thr).astype(int)
    ups = np.where(np.diff(b) > 0)[0] + 1
    downs = np.where(np.diff(b) < 0)[0] + 1
    return ups, downs


def pair_pulses(ups, downs, sf):
    out = []
    for u in ups:
        later = downs[downs > u]
        if len(later):
            out.append((u, later[0]))
    return out


def debounce(pairs, sf, min_gap_ms, min_dur_ms):
    if not pairs:
        return []
    mg = min_gap_ms/1000*sf
    merged = [list(pairs[0])]
    for on, off in pairs[1:]:
        if on - merged[-1][1] <= mg:
            merged[-1][1] = off
        else:
            merged.append([on, off])
    md = min_dur_ms/1000*sf
    return [(on, off) for on, off in merged if (off-on) >= md]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns3', required=True)
    args = ap.parse_args()

    sig, sf, chname = load(args.ns3)
    lo, hi = float(sig.min()), float(sig.max())
    dur = len(sig)/sf
    print(f"Channel '{chname}': {len(sig)} samples @ {sf:.0f}Hz = {dur:.1f}s")
    print(f"Signal range: min={lo:.0f}, max={hi:.0f}, mid={lo/2+hi/2:.0f}")

    print("\n=== 1. RAW threshold-crossing counts (NO debounce) ===")
    print("   (if the count is stable across thresholds, pulses are clean squares)")
    for frac in (0.20, 0.35, 0.50, 0.65, 0.80):
        thr = lo + frac*(hi-lo)
        ups, downs = raw_crossings(sig, thr)
        print(f"   threshold {frac:.0%} ({thr:6.0f}): "
              f"{len(ups):4d} rising, {len(downs):4d} falling")

    print("\n=== 2. Paired pulse count at mid-threshold, varying debounce ===")
    thr = lo/2 + hi/2
    ups, downs = raw_crossings(sig, thr)
    pairs = pair_pulses(ups, downs, sf)
    print(f"   raw pairs (no debounce): {len(pairs)}")
    for gap in (0, 5, 10, 15, 25, 40):
        d = debounce(pairs, sf, gap, min_dur_ms=0)
        print(f"   debounce merge <{gap:2d}ms gap: {len(d)} pulses")
    print("   (then also dropping short blips):")
    for mindur in (5, 10, 15, 20):
        d = debounce(pairs, sf, 25, min_dur_ms=mindur)
        print(f"   merge<25ms + drop <{mindur:2d}ms dur: {len(d)} pulses")

    print("\n=== 3. Pulse widths & gaps (debounce 25ms/15ms) ===")
    d = debounce(pairs, sf, 25, 15)
    widths = np.array([(off-on)/sf*1000 for on, off in d])
    onsets = np.array([on/sf for on, off in d])
    gaps = np.diff(onsets)
    print(f"   pulses: {len(d)}")
    print(f"   width  ms: min {widths.min():.0f}, median {np.median(widths):.0f}, "
          f"max {widths.max():.0f}")
    print(f"   width histogram:")
    for lo_e, hi_e in [(0,20),(20,50),(50,90),(90,150),(150,10000)]:
        c = np.sum((widths>=lo_e)&(widths<hi_e))
        print(f"     {lo_e:4d}-{hi_e:<5d}ms: {c}")

    print("\n=== 4. Pulses before/after the largest gap (block boundary) ===")
    big = int(np.argmax(gaps))
    print(f"   largest gap: {gaps[big]:.1f}s after pulse #{big} "
          f"(at t={onsets[big]:.1f}s)")
    print(f"   pulses before gap: {big+1}")
    print(f"   pulses after gap:  {len(d)-(big+1)}")
    print(f"   → total: {len(d)}")
    # also show the 5 largest gaps (interruptions vs break)
    order = np.argsort(gaps)[-6:][::-1]
    print("   6 largest gaps (to distinguish break from interruptions):")
    for i in order:
        print(f"     {gaps[i]:6.1f}s  after pulse #{i} (t={onsets[i]:.1f}s)")


if __name__ == '__main__':
    main()
