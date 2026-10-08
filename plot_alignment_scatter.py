#!/usr/bin/env python
# plot_alignment_scatter.py
"""
Alignment scatter for the PI: task-log event time (.tsv) on x, neural pulse
time (.ns3) on y, one point per matched event, colored by phase. If the two
independent clocks agree, all points fall on a straight line (slope 1). The
response-driven events (bet_submitted, color_submitted) have random timing, so
their falling on the same line is strong proof the alignment is genuine (not a
coincidence of fixed inter-event intervals).

Reads the events_final CSVs from align_final.py. Both blocks are anchored to
their own start so they overlay on one common line.

Usage:
    python plot_alignment_scatter.py \
        --csv1 events_final_block1.csv --csv2 events_final_block2.csv \
        --out alignment_scatter.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHASE_COLORS = {
    'fixation': '#4C72B0', 'bet_onset': '#DD8452', 'bet_submitted': '#C44E52',
    'color_submitted': '#8172B3', 'feedback': '#55A868',
}
PHASE_ORDER = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv1', required=True)
    ap.add_argument('--csv2', required=True)
    ap.add_argument('--out', default='alignment_scatter.png')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, csv, name in [(axes[0], args.csv1, 'BLOCK 1'),
                          (axes[1], args.csv2, 'BLOCK 2')]:
        df = pd.read_csv(csv)
        m = df[df['matched'] == True].copy()
        # anchor both axes to their own first matched event so the line starts at 0
        t0 = m['tsv_time'].min()
        n0 = m['ns3_onset_s'].min()
        m['tsv_rel'] = m['tsv_time'] - t0
        m['ns3_rel'] = m['ns3_onset_s'] - n0

        for ph in PHASE_ORDER:
            sub = m[m['phase'] == ph]
            ax.scatter(sub['tsv_rel'], sub['ns3_rel'], s=12,
                       color=PHASE_COLORS[ph], label=ph, alpha=0.8)

        # identity line (slope 1)
        lim = max(m['tsv_rel'].max(), m['ns3_rel'].max())
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6,
                label='slope = 1 (perfect agreement)')

        # fit for the annotation
        a, b = np.polyfit(m['tsv_rel'], m['ns3_rel'], 1)
        resid_ms = (m['ns3_rel'] - (a*m['tsv_rel']+b)) * 1000
        ax.set_title(f'{name}: task-log vs neural time\n'
                     f'slope={a:.5f}, residual std={resid_ms.std():.1f} ms, '
                     f'n={len(m)}', fontsize=11)
        ax.set_xlabel('task-log event time (.tsv), s from block start')
        ax.set_ylabel('neural pulse time (.ns3), s from block start')
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Alignment: independent task-log and neural clocks agree '
                 '(points on the line)', fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    # Also print a zoomed check: residuals for the response-driven events only
    for csv, name in [(args.csv1, 'BLOCK 1'), (args.csv2, 'BLOCK 2')]:
        df = pd.read_csv(csv)
        m = df[df['matched'] == True].copy()
        t0 = m['tsv_time'].min(); n0 = m['ns3_onset_s'].min()
        m['tr'] = m['tsv_time']-t0; m['nr'] = m['ns3_onset_s']-n0
        a, b = np.polyfit(m['tr'], m['nr'], 1)
        m['resid_ms'] = (m['nr'] - (a*m['tr']+b))*1000
        print(f"\n{name} residual std by phase (response-driven = strong check):")
        for ph in PHASE_ORDER:
            s = m[m['phase'] == ph]['resid_ms']
            tag = ' <- response-driven' if ph in ('bet_submitted','color_submitted') else ''
            print(f"  {ph:16s}: std {s.std():5.1f} ms{tag}")


if __name__ == '__main__':
    main()
