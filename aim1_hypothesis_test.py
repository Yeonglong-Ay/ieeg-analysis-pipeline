#!/usr/bin/env python
# aim1_hypothesis_test.py
"""
AIM 1 HYPOTHESIS TEST.

Hypothesis: High-gamma activity in FRONTAL and LIMBIC regions tracks losing-
streak length, and this is SPECIFIC to those regions (not other regions).

Three components, each tested:
  (1) SENSITIVITY  — frontal/limbic high-gamma streak slope > 0
  (2) SPECIFICITY  — frontal/limbic slope significantly GREATER than the
                     'other' region group (the crucial specificity test; a
                     direct group comparison, not 'significant here / not there')
  (3) BAND SPEC.   — within frontal/limbic, high-gamma slope greater than the
                     lower-band slopes

Unit of analysis = per-channel streak slope (OLS of power_z on streak_len 1-4,
loss streaks), computed per channel to avoid pooling correlated channels.
Groups compared at the channel level.

Region groups (as specified):
  FRONTAL/LIMBIC = amygdala, hippocampus, insula, orbitofrontal, cingulate,
                   superior_frontal, middle_frontal, inferior_frontal
  OTHER          = temporal, thalamus

Epoch = bet_submitted. Reads Stage 2 per-trial power + behaviour.

HONEST CAVEATS (printed): n=1 patient; 'not other regions' in one subject may
reflect limited power, not true absence; uneven channel coverage; the group
comparison is the right design but specificity needs multi-patient replication.

Usage:
  python aim1_hypothesis_test.py --stage2-dir stage2_mne_fix \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv --out-dir aim1_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
EVENT = 'bet_submitted'
MAX_STREAK = 4

FRONTAL_LIMBIC = ['amygdala', 'hippocampus', 'insula', 'orbitofrontal',
                  'cingulate', 'superior_frontal', 'middle_frontal',
                  'inferior_frontal']
OTHER = ['temporal', 'thalamus']


def assign_region(fslabel):
    s = str(fslabel).lower()
    if 'white-matter' in s or 'unknown' in s: return None
    if 'hippocamp' in s: return 'hippocampus'
    if 'amygdala' in s: return 'amygdala'
    if 'thalamus' in s: return 'thalamus'
    if 'insula' in s or 'insular' in s or 'ins_lg' in s: return 'insula'
    if 'cingul' in s: return 'cingulate'
    if 'orbital' in s: return 'orbitofrontal'
    if 'front_sup' in s: return 'superior_frontal'
    if 'front_middle' in s: return 'middle_frontal'
    if 'front_inf' in s: return 'inferior_frontal'
    if 'front' in s or 'subcentral' in s: return 'frontal_other'
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return None


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
    st = b['streak_type'].astype(str).str.upper().str.strip()
    b['streak_dir'] = st.str[0].map({'L': 'loss', 'W': 'win'})
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    b['streak_len'] = sl.where(sl.notna(), pd.to_numeric(st.str[1:], errors='coerce'))
    return b[['block', 'exp_trial', 'streak_dir', 'streak_len']]


def channel_slope(g):
    if len(g) < 6 or g['streak_len'].nunique() < 2:
        return np.nan
    return stats.linregress(g['streak_len'].astype(float), g['power_z']).slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_fix')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='aim1_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(assign_region)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(assign_region)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    def group_of(region):
        if region in FRONTAL_LIMBIC: return 'frontal_limbic'
        if region in OTHER: return 'other'
        return None   # frontal_other / unassigned excluded from the 2-group test

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)
    power = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')
    power['region'] = power['bipolar_name'].map(ch2region)
    power['group'] = power['region'].map(group_of)

    # per-channel LOSS slope, for every band
    chan_rows = []
    for (chan, band), g in power.groupby(['bipolar_name', 'band']):
        loss = g[(g['streak_dir'] == 'loss') & (g['streak_len'].between(1, MAX_STREAK))]
        sl = channel_slope(loss)
        chan_rows.append({'channel': chan, 'band': band,
                          'region': ch2region.get(chan),
                          'group': group_of(ch2region.get(chan)),
                          'loss_slope': sl})
    ch = pd.DataFrame(chan_rows)
    ch.to_csv(os.path.join(args.out_dir, 'aim1_channel_slopes.csv'), index=False)

    # ---- build the printed summary ----
    L = []
    def out(s=''): L.append(s); print(s)

    out("="*70)
    out("AIM 1 HYPOTHESIS TEST")
    out("High-gamma in frontal/limbic tracks loss-streak length, specifically")
    out("="*70)

    # group membership + channel counts
    out("\nREGION GROUPS (channels available):")
    for grp, regions in [('FRONTAL/LIMBIC', FRONTAL_LIMBIC), ('OTHER', OTHER)]:
        out(f"  {grp}:")
        for r in regions:
            n = anat[anat['region'] == r].shape[0]
            if n > 0:
                out(f"    {r}: {n} channels")
        tot = anat[anat['region'].isin(regions)].shape[0]
        out(f"    -> group total: {tot} channels")
    unassigned = anat[anat['region'].map(group_of).isna()]
    out(f"  (excluded from 2-group test: {len(unassigned)} channels "
        f"in {sorted(set(unassigned['region'].dropna()))})")

    hg = ch[ch['band'] == 'high_gamma']
    fl = hg[hg['group'] == 'frontal_limbic']['loss_slope'].dropna()
    ot = hg[hg['group'] == 'other']['loss_slope'].dropna()

    out("\n" + "-"*70)
    out("TEST 1 — SENSITIVITY: frontal/limbic high-gamma loss-streak slope > 0")
    out("-"*70)
    if len(fl) >= 3:
        t, p = stats.ttest_1samp(fl, 0)
        out(f"  n channels = {len(fl)}")
        out(f"  mean slope = {fl.mean():+.4f}")
        out(f"  one-sample t = {t:+.3f}, p = {p:.4f}")
        out(f"  -> {'SUPPORTS' if (p<0.05 and fl.mean()>0) else 'does NOT support'} "
            f"sensitivity (slope > 0)")
    else:
        out(f"  insufficient channels (n={len(fl)})")

    out("\n" + "-"*70)
    out("TEST 2 — SPECIFICITY: frontal/limbic slope > other-region slope")
    out("  (the key test; direct group comparison)")
    out("-"*70)
    if len(fl) >= 3 and len(ot) >= 3:
        t, p = stats.ttest_ind(fl, ot, equal_var=False)
        out(f"  frontal/limbic: n={len(fl)}, mean slope={fl.mean():+.4f}")
        out(f"  other:          n={len(ot)}, mean slope={ot.mean():+.4f}")
        out(f"  two-sample t = {t:+.3f}, p = {p:.4f}")
        out(f"  -> {'SUPPORTS' if (p<0.05 and fl.mean()>ot.mean()) else 'does NOT support'} "
            f"specificity (frontal/limbic > other)")
    else:
        out(f"  insufficient channels (frontal/limbic n={len(fl)}, other n={len(ot)})")

    out("\n" + "-"*70)
    out("TEST 3 — BAND SPECIFICITY: within frontal/limbic, high-gamma > other bands")
    out("-"*70)
    hg_fl = fl
    for band in ['delta', 'theta', 'alpha', 'beta', 'low_gamma']:
        other_band = ch[(ch['band'] == band) &
                        (ch['group'] == 'frontal_limbic')]['loss_slope'].dropna()
        if len(hg_fl) >= 3 and len(other_band) >= 3:
            t, p = stats.ttest_ind(hg_fl, other_band, equal_var=False)
            out(f"  high_gamma ({hg_fl.mean():+.3f}) vs {band} "
                f"({other_band.mean():+.3f}): t={t:+.2f}, p={p:.4f}")

    out("\n" + "="*70)
    out("CAVEATS")
    out("="*70)
    out("  - Single patient (n=1): 'not other regions' may reflect limited")
    out("    power, not true absence. Specificity needs multi-patient replication.")
    out("  - Per-channel slopes avoid channel pooling, but channel counts are")
    out("    uneven across regions (see group membership above).")
    out("  - Exploratory; p-values not corrected across the three tests.")
    out("  - The two-group comparison (Test 2) is the correct design for a")
    out("    specificity claim, but a single-subject result is suggestive.")

    with open(os.path.join(args.out_dir, 'aim1_summary.txt'), 'w') as f:
        f.write("\n".join(L))
    print(f"\n[saved aim1_summary.txt and aim1_channel_slopes.csv in {args.out_dir}/]")


if __name__ == '__main__':
    main()
