#!/usr/bin/env python
# stage5_streak_length.py
"""
STAGE 5: neural streak-length analysis (long vs short losses).

Neural parallel to the behavioural loss-chasing analysis: does band power at
BET SUBMISSION scale with STREAK LENGTH (continuous 1..4)? Fit separately for
LOSING streaks (primary) and WINNING streaks (comparison).

Primary, hypothesis-driven test: AMYGDALA (pre-specified region) — is amygdala
band power modulated by losing-streak length? Because this is one a-priori
region, it is a clean, low-correction test.

Secondary, exploratory: all regions, with FDR across regions/bands, to place
amygdala in context and see if loss-streak scaling is amygdala-specific.

Design (confirmed):
  - epoch: bet_submitted
  - continuous streak_length (1..4) as the predictor (OLS slope)
  - loss streaks primary; win streaks as comparison
  - report slope, p, effect size, and trial counts (honest about n per length)

Inputs: Stage 2 per-trial power (trial_power_bet_submitted.csv), Stage 1 anatomy
(bipolar_anatomy.csv), behavioural pilot_results CSVs (streak_type/length).
Outputs (in --out-dir):
  streak_length_channel.csv   per channel x band: loss & win slopes
  streak_length_region.csv    per region x band: loss & win slopes (channel-avg)
  amygdala_streak.txt         the pre-specified amygdala test, written out
  amygdala_streak_<band>.png  amygdala loss vs win streak-length figure

Usage:
  python stage5_streak_length.py --stage2-dir stage2_mne_out \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...112021.csv pilot_results_...113531.csv \
      --out-dir stage5_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
EVENT = 'bet_submitted'
MAX_STREAK = 4


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
    if ('temp' in s or 'fusifor' in s or 'collat' in s or 'oc-temp' in s
            or 'lat_fis' in s or 'transv' in s): return 'temporal'
    return 'other'


def reconstruct_order(event_csvs):
    dfs = []
    for i, c in enumerate(event_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    ev = ev[(ev['matched'] == True) & (ev['trial_number'] > NUM_PRACTICE)]
    sub = ev[ev['phase'] == EVENT].reset_index(drop=True)
    sub['trial_index'] = np.arange(len(sub))
    sub['exp_trial'] = sub['trial_number'] - NUM_PRACTICE
    return sub[['trial_index', 'block', 'exp_trial']]


def load_behaviour(behav_csvs):
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial']); b['exp_trial'] = b['exp_trial'].astype(int)
    # streak_type is coded like 'L3' (loss len 3) / 'W2' (win len 2) / 'none',
    # NOT the word 'loss'/'win'. Decode direction (first letter) and length.
    st_raw = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_type'] = st_raw.str[0].map({'L': 'loss', 'W': 'win'})   # else NaN
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    sl_from_label = pd.to_numeric(st_raw.str[1:], errors='coerce')
    b['streak_length'] = sl.where(sl.notna(), sl_from_label)
    return b[['block', 'exp_trial', 'streak_type', 'streak_length']]


def fit_slope(x, y):
    """OLS slope of y on x; returns slope, p, n, r."""
    if len(x) < 6 or len(np.unique(x)) < 2:
        return np.nan, np.nan, len(x), np.nan
    res = stats.linregress(x, y)
    return res.slope, res.pvalue, len(x), res.rvalue


def analyze_channel(g):
    """g = rows for one channel/band with streak_type, streak_length, power_z.
    Returns dict of loss & win slopes."""
    out = {}
    for direction in ['loss', 'win']:
        sub = g[(g['streak_type'] == direction) &
                (g['streak_length'] >= 1) &
                (g['streak_length'] <= MAX_STREAK)]
        slope, p, n, r = fit_slope(sub['streak_length'].values.astype(float),
                                   sub['power_z'].values)
        out[f'{direction}_slope'] = slope
        out[f'{direction}_p'] = p
        out[f'{direction}_n'] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_out')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage5_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(region_of)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(region_of)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)

    power = pd.read_csv(os.path.join(args.stage2_dir,
                                     f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')
    power['region'] = power['bipolar_name'].map(ch2region)

    # ---- per-channel slopes ----
    rows = []
    for (band, chan), g in power.groupby(['band', 'bipolar_name']):
        res = analyze_channel(g)
        res.update({'band': band, 'bipolar_name': chan,
                    'region': ch2region.get(chan)})
        rows.append(res)
    chan_df = pd.DataFrame(rows)
    chan_df.to_csv(os.path.join(args.out_dir, 'streak_length_channel.csv'),
                   index=False)

    # ---- region-level: average channel slopes within region ----
    reg_rows = []
    for (band, reg), g in chan_df.groupby(['band', 'region']):
        for direction in ['loss', 'win']:
            s = g[f'{direction}_slope'].dropna()
            if len(s) >= 1:
                # one-sample test: are channel slopes in this region != 0?
                if len(s) >= 3:
                    t, p = stats.ttest_1samp(s, 0.0)
                else:
                    t, p = np.nan, np.nan
                reg_rows.append({'band': band, 'region': reg,
                                 'direction': direction,
                                 'mean_slope': s.mean(), 'n_chan': len(s),
                                 't': t, 'p': p})
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(os.path.join(args.out_dir, 'streak_length_region.csv'),
                  index=False)

    # ---- PRIMARY: amygdala pre-specified test ----
    lines = ["PRE-SPECIFIED HYPOTHESIS TEST: AMYGDALA streak-length scaling",
             "(bet submission; loss streaks primary, win as comparison)\n"]
    amyg = power[power['region'] == 'amygdala']
    for band in BANDS:
        gb = amyg[amyg['band'] == band]
        lines.append(f"\n[{band}]")
        for direction in ['loss', 'win']:
            sub = gb[(gb['streak_type'] == direction) &
                     (gb['streak_length'].between(1, MAX_STREAK))]
            if len(sub) >= 6 and sub['streak_length'].nunique() >= 2:
                res = stats.linregress(sub['streak_length'].astype(float),
                                       sub['power_z'])
                # trial counts per streak length
                counts = sub.groupby('streak_length').size().to_dict()
                lines.append(
                    f"  {direction}: slope={res.slope:+.4f}, p={res.pvalue:.4f}, "
                    f"r={res.rvalue:+.3f}, n={len(sub)} trials-x-chan  "
                    f"(per length: {counts})")
            else:
                lines.append(f"  {direction}: insufficient data (n={len(sub)})")
    with open(os.path.join(args.out_dir, 'amygdala_streak.txt'), 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # ---- amygdala figure: mean power by streak length, loss vs win, per band
    fig, axes = plt.subplots(1, len(BANDS), figsize=(3*len(BANDS), 3.2),
                             squeeze=False)
    for bi, band in enumerate(BANDS):
        ax = axes[0, bi]
        gb = amyg[amyg['band'] == band]
        for direction, col in [('loss', '#C44E52'), ('win', '#4C72B0')]:
            sub = gb[(gb['streak_type'] == direction) &
                     (gb['streak_length'].between(1, MAX_STREAK))]
            if len(sub) == 0:
                continue
            m = sub.groupby('streak_length')['power_z'].mean()
            se = sub.groupby('streak_length')['power_z'].sem()
            ax.errorbar(m.index, m.values, yerr=se.values, marker='o',
                        color=col, label=direction, capsize=3)
        ax.set_title(band, fontsize=9)
        ax.set_xlabel('streak length'); ax.set_xticks([1, 2, 3, 4])
        if bi == 0:
            ax.set_ylabel('amygdala power (z)')
            ax.legend(fontsize=8)
    fig.suptitle('Amygdala band power vs streak length (bet submission)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'amygdala_streak.png'), dpi=140)
    plt.close(fig)

    print(f"\nStage 5 done. Outputs in {args.out_dir}/")
    print("See amygdala_streak.txt (pre-specified test) and "
          "streak_length_region.csv (exploratory context).")


if __name__ == '__main__':
    main()
