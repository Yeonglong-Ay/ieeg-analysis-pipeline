#!/usr/bin/env python
# stage10_peth_streak_length.py
"""
Streak-length PETHs (graded streak effect), per brain region.

Instead of a binary streak/no-streak split, this shows the neural response
graded by STREAK LENGTH (1,2,3,4), separately for LOSING and WINNING streaks,
so you can see whether the response scales with how long a streak has run.

Trial's streak length = its position in the current run (streak_type=loss,
streak_length=3 -> the 3rd consecutive loss), matching the behavioural /
Stage 5 definition.

Two figure styles per epoch (both requested):
  A) SEPARATE: one figure for losing streaks (lines L1..L4) and one for winning
     streaks (lines W1..W4). rows=regions, cols=bands.
  B) COMBINED: one figure with all 8 lines (L1..L4 in reds, W1..W4 in blues).

Reads per-region per-trial time courses (region_trial_tc_<event>.npz from
Stage 2) + behaviour (pilot_results), matched via ident_<event>.csv.

HONEST CAVEAT: streak-length cells are thin (roughly L1~25, L2~21, L3~12, L4~12;
wins similar or fewer). Split per region/band, some lines rest on very few
trials and are noisy — n is printed in each legend; interpret long streaks
(L3/L4, W3/W4) cautiously.

Usage:
  python stage10_peth_streak_length.py --stage1-dir stage1_mne_out \
      --stage2-dir stage2_mne_fix \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv \
      --out-dir stage10_fix_out
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVENTS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
MAX_STREAK = 4
MIN_TRIALS = 2          # need at least this many trials to draw a line

# colour gradients: losses = reds (light->dark = short->long), wins = blues
LOSS_COLORS = ['#f4a8a8', '#e06666', '#cc0000', '#7a0000']   # L1..L4
WIN_COLORS  = ['#a8c8f4', '#6699e0', '#0044cc', '#00227a']   # W1..W4

# short-vs-long grouping (declutters the combined view: 4 lines, not 8)
SHORTLONG_COLORS = {
    'loss short (L1-2)': '#e06666',
    'loss long (L3-4)':  '#7a0000',
    'win short (W1-2)':  '#6699e0',
    'win long (W3-4)':   '#00227a',
}


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
    # streak_type is coded like 'L3' (loss, length 3) or 'W2' (win, length 2),
    # NOT the word 'loss'/'win'. Parse direction (first letter) and length.
    st_raw = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_dir'] = st_raw.str[0].map({'L': 'loss', 'W': 'win'})   # else NaN
    # length: prefer the explicit streak_length column; fall back to the digit
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    sl_from_label = pd.to_numeric(st_raw.str[1:], errors='coerce')
    b['streak_len'] = sl.where(sl.notna(), sl_from_label)
    return b


def draw_lines(ax, t, reg_tc, bi, ri, specs):
    """specs: list of (label, mask, color). Plots mean+/-SEM per spec."""
    for lab, m, col in specs:
        seg = reg_tc[bi, m, ri, :]
        seg = seg[~np.all(np.isnan(seg), axis=1)]
        if len(seg) < MIN_TRIALS:
            continue
        mean = np.nanmean(seg, axis=0)
        sem = np.nanstd(seg, axis=0) / np.sqrt(len(seg))
        ax.plot(t, mean, color=col, lw=1.1, label=f"{lab} (n={len(seg)})")
        ax.fill_between(t, mean-sem, mean+sem, color=col, alpha=0.15)
    ax.axvline(0, color='k', ls='--', lw=0.7)
    ax.axhline(0, color='grey', ls=':', lw=0.5)


def draw_lines_noribbon(ax, t, reg_tc, bi, ri, specs):
    """Like draw_lines but NO SEM ribbon — for decluttered multi-line views."""
    for lab, m, col in specs:
        seg = reg_tc[bi, m, ri, :]
        seg = seg[~np.all(np.isnan(seg), axis=1)]
        if len(seg) < MIN_TRIALS:
            continue
        mean = np.nanmean(seg, axis=0)
        ax.plot(t, mean, color=col, lw=1.4, label=f"{lab} (n={len(seg)})")
    ax.axvline(0, color='k', ls='--', lw=0.7)
    ax.axhline(0, color='grey', ls=':', lw=0.5)


def make_figure(t, reg_tc, regions, merged, specs_fn, title, outpath,
                draw_fn=draw_lines):
    nrow, ncol = len(regions), len(BANDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6*ncol, 1.9*nrow),
                             squeeze=False)
    for ri, rg in enumerate(regions):
        for bi, band in enumerate(BANDS):
            ax = axes[ri, bi]
            draw_fn(ax, t, reg_tc, bi, ri, specs_fn(merged))
            if ri == 0:
                ax.set_title(band, fontsize=9)
            if bi == 0:
                ax.set_ylabel(rg, fontsize=7)
            if ri == 0 and bi == ncol-1:
                ax.legend(fontsize=5, loc='upper right', ncol=2)
    for bi in range(ncol):
        axes[-1, bi].set_xlabel('t (s)', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140); plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', required=True)
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage10_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    behav = load_behaviour(args.behav)

    for ev in EVENTS:
        tc_file = os.path.join(args.stage2_dir, f'region_trial_tc_{ev}.npz')
        id_file = os.path.join(args.stage1_dir, f'ident_{ev}.csv')
        if not (os.path.exists(tc_file) and os.path.exists(id_file)):
            print(f"  (skip {ev}: missing inputs)"); continue
        d = np.load(tc_file, allow_pickle=True)
        reg_tc = d['reg_tc']; t = d['t']; regions = list(d['regions'])
        ident = pd.read_csv(id_file)
        merged = ident.merge(behav, on=['block', 'trial_number'], how='left')
        if len(merged) != reg_tc.shape[1]:
            print(f"  WARNING {ev}: identity mismatch; skipping"); continue

        sl = merged['streak_len'].values
        st = merged['streak_dir'].values

        # spec builders
        def loss_specs(_m):
            return [(f'L{k}', (st == 'loss') & (sl == k), LOSS_COLORS[k-1])
                    for k in range(1, MAX_STREAK+1)]
        def win_specs(_m):
            return [(f'W{k}', (st == 'win') & (sl == k), WIN_COLORS[k-1])
                    for k in range(1, MAX_STREAK+1)]
        def shortlong_specs(_m):
            # 4 lines: short/long × loss/win, pooling L1-2, L3-4, etc.
            return [
                ('loss short (L1-2)', (st == 'loss') & (sl >= 1) & (sl <= 2),
                 SHORTLONG_COLORS['loss short (L1-2)']),
                ('loss long (L3-4)',  (st == 'loss') & (sl >= 3) & (sl <= 4),
                 SHORTLONG_COLORS['loss long (L3-4)']),
                ('win short (W1-2)',  (st == 'win') & (sl >= 1) & (sl <= 2),
                 SHORTLONG_COLORS['win short (W1-2)']),
                ('win long (W3-4)',   (st == 'win') & (sl >= 3) & (sl <= 4),
                 SHORTLONG_COLORS['win long (W3-4)']),
            ]

        # A) separate loss / win figures (4 streak-length lines each, with SEM)
        make_figure(t, reg_tc, regions, merged, loss_specs,
                    f'PETH by losing-streak length (L1-L4) — {ev}',
                    os.path.join(args.out_dir, f'peth_{ev}_lossstreak.png'))
        make_figure(t, reg_tc, regions, merged, win_specs,
                    f'PETH by winning-streak length (W1-W4) — {ev}',
                    os.path.join(args.out_dir, f'peth_{ev}_winstreak.png'))
        # B) decluttered combined: short vs long, loss vs win (4 lines, NO ribbon)
        make_figure(t, reg_tc, regions, merged, shortlong_specs,
                    f'PETH short vs long streaks — losses (red) vs wins (blue) '
                    f'— {ev}',
                    os.path.join(args.out_dir, f'peth_{ev}_streak_shortlong.png'),
                    draw_fn=draw_lines_noribbon)

    print(f"\nStage 10 done. Streak-length PETHs in {args.out_dir}/")


if __name__ == '__main__':
    main()
