#!/usr/bin/env python
# self_paced_durations.py
"""
Compute the self-paced epoch durations (reaction times) from the behavioural
pilot_results CSVs, using the logged event timestamps.

Self-paced epochs have no preset duration — they last as long as the subject
takes. We recover them from the timestamps recorded per trial:
  bet decision   = t_bet_response  - t_bet_onset
  color choice   = t_color_response - t_color_onset

Reports mean, median, SD, and range (median is often more representative since
reaction times are right-skewed). Experimental trials only.

Usage:
  python self_paced_durations.py pilot_results_A.csv pilot_results_B.csv
"""
import sys
import numpy as np
import pandas as pd


def summarize(x, name):
    x = x[np.isfinite(x)]
    x = x[(x > 0) & (x < 30)]        # drop nonsensical/huge values
    if len(x) == 0:
        print(f"  {name}: no valid values"); return
    print(f"  {name}:  n={len(x)}  mean={x.mean():.2f}s  median={np.median(x):.2f}s "
          f" SD={x.std():.2f}s  range={x.min():.2f}-{x.max():.2f}s")


def main(csvs):
    dfs = []
    for i, c in enumerate(csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    print(f"Experimental trials: {len(b)}")

    # duration columns from timestamps
    for col in ['t_bet_onset', 't_bet_response', 't_color_onset',
                't_color_response']:
        if col not in b.columns:
            print(f"WARNING: column {col} not found — available: "
                  f"{[c for c in b.columns if c.startswith('t_')]}")

    bet = pd.to_numeric(b['t_bet_response'], errors='coerce') - \
          pd.to_numeric(b['t_bet_onset'], errors='coerce')
    color = pd.to_numeric(b['t_color_response'], errors='coerce') - \
            pd.to_numeric(b['t_color_onset'], errors='coerce')

    print("\nSelf-paced epoch durations:")
    summarize(bet.values, 'bet decision  (onset -> submit)')
    summarize(color.values, 'color choice  (onset -> submit)')

    # also report the combined decision period if useful
    total = pd.to_numeric(b['t_color_response'], errors='coerce') - \
            pd.to_numeric(b['t_bet_onset'], errors='coerce')
    summarize(total.values, 'total decision (bet onset -> color submit)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python self_paced_durations.py <pilot_results...csv> ...")
        sys.exit(1)
    main(sys.argv[1:])
