#!/usr/bin/env python
# aim1_hypothesis_test_v2.py
"""
AIM 1 HYPOTHESIS TEST (v2 — sign-agnostic).

Hypothesis: High-gamma activity in FRONTAL and LIMBIC regions TRACKS losing-
streak length (in EITHER direction), specific to those regions vs others.

KEY FIX over v1: 'tracking' is direction-agnostic. A channel can track streak by
INCREASING or DECREASING high-gamma with streak length — both count. Averaging
signed slopes would cancel opposite-direction channels and falsely hide tracking.
So we test tracking two sign-agnostic ways:
  (a) FRACTION of channels with a significant (two-sided) streak slope
  (b) MEAN ABSOLUTE slope (|slope|), compared to a permutation null

Tests:
  TEST 1 (sensitivity): do frontal/limbic channels track streak more than chance?
  TEST 2 (specificity): is tracking greater in frontal/limbic than 'other' regions?
                        (direct group comparison — both fraction and |slope|)
  TEST 3 (band):        within frontal/limbic, is high-gamma tracking > other bands?
  DIRECTION analysis:   among tracking channels, do FRONTAL vs LIMBIC differ in
                        the SIGN of their slopes (opposite directions)?

Unit = per-channel streak slope (OLS of power_z on streak_len 1-4, loss streaks).
Per-channel significance from the OLS p-value (two-sided). Group differences via
permutation (shuffle group labels) to avoid distributional assumptions.

Groups: FRONTAL/LIMBIC = amygdala, hippocampus, insula, orbitofrontal, cingulate,
superior/middle/inferior frontal.  OTHER = temporal, thalamus.

Epoch = bet_submitted. HONEST CAVEATS printed (n=1, coverage, exploratory).

Usage:
  python aim1_hypothesis_test_v2.py --stage2-dir stage2_mne_fix \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...csv pilot_results_...csv --out-dir aim1v2_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
NUM_PRACTICE = 5
EVENT = 'bet_submitted'
MAX_STREAK = 4
ALPHA = 0.05
N_PERM = 5000

FRONTAL = ['orbitofrontal', 'cingulate', 'superior_frontal', 'middle_frontal',
           'inferior_frontal']
LIMBIC = ['amygdala', 'hippocampus', 'insula']
FRONTAL_LIMBIC = FRONTAL + LIMBIC
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


def channel_slope_p(g):
    """Return (slope, two-sided p) for one channel, or (nan, nan)."""
    if len(g) < 6 or g['streak_len'].nunique() < 2:
        return np.nan, np.nan
    res = stats.linregress(g['streak_len'].astype(float), g['power_z'])
    return res.slope, res.pvalue


def perm_diff_test(a, b, stat_fn, n_perm=N_PERM, rng=None):
    """Permutation test: is stat_fn(a) - stat_fn(b) beyond chance?
    Shuffle group labels. Returns (observed_diff, p_two_sided)."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a), np.asarray(b)
    obs = stat_fn(a) - stat_fn(b)
    pooled = np.concatenate([a, b]); n_a = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = stat_fn(perm[:n_a]) - stat_fn(perm[n_a:])
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_fix')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='aim1v2_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    anat['region'] = anat['anode_region'].map(assign_region)
    anat.loc[anat['region'].isna(), 'region'] = \
        anat.loc[anat['region'].isna(), 'cathode_region'].map(assign_region)
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    def group_of(r):
        if r in FRONTAL_LIMBIC: return 'frontal_limbic'
        if r in OTHER: return 'other'
        return None

    order = reconstruct_order(args.events)
    behav = load_behaviour(args.behav)
    power = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{EVENT}.csv'))
    power = power.merge(order, on='trial_index', how='left')
    power = power.merge(behav, on=['block', 'exp_trial'], how='left')

    # per-channel slope + p, per band (loss streaks)
    rows = []
    for (chan, band), g in power.groupby(['bipolar_name', 'band']):
        loss = g[(g['streak_dir'] == 'loss') & (g['streak_len'].between(1, MAX_STREAK))]
        sl, p = channel_slope_p(loss)
        reg = ch2region.get(chan)
        rows.append({'channel': chan, 'band': band, 'region': reg,
                     'group': group_of(reg), 'macro': ('frontal' if reg in FRONTAL
                     else 'limbic' if reg in LIMBIC else 'other' if reg in OTHER
                     else None),
                     'slope': sl, 'p': p,
                     'tracks': (np.isfinite(p) and p < ALPHA)})
    ch = pd.DataFrame(rows)
    ch.to_csv(os.path.join(args.out_dir, 'aim1v2_channel_slopes.csv'), index=False)

    L = []
    def out(s=''): L.append(s); print(s)

    out("="*70)
    out("AIM 1 HYPOTHESIS TEST (v2 — sign-agnostic)")
    out("High-gamma in frontal/limbic TRACKS loss-streak (either direction),")
    out("specific to those regions.")
    out("="*70)

    out("\nREGION GROUPS (channels):")
    for gname, regions in [('FRONTAL', FRONTAL), ('LIMBIC', LIMBIC), ('OTHER', OTHER)]:
        out(f"  {gname}:")
        for r in regions:
            n = anat[anat['region'] == r].shape[0]
            if n: out(f"    {r}: {n}")
    out(f"  FRONTAL/LIMBIC total: {anat[anat['region'].isin(FRONTAL_LIMBIC)].shape[0]}")
    out(f"  OTHER total: {anat[anat['region'].isin(OTHER)].shape[0]}")

    # focus on high-gamma
    hg = ch[(ch['band'] == 'high_gamma')].dropna(subset=['slope'])
    fl = hg[hg['group'] == 'frontal_limbic']
    ot = hg[hg['group'] == 'other']

    frac = lambda d: np.mean(d['tracks'].values) if len(d) else np.nan
    mabs = lambda s: np.mean(np.abs(s))

    out("\n" + "-"*70)
    out("TEST 1 — SENSITIVITY (sign-agnostic): do frontal/limbic track streak?")
    out("-"*70)
    out(f"  frontal/limbic high-gamma channels: n = {len(fl)}")
    out(f"  (a) fraction tracking (p<{ALPHA}, two-sided): "
        f"{fl['tracks'].sum()}/{len(fl)} = {frac(fl)*100:.0f}%")
    out(f"      expected by chance ~= {ALPHA*100:.0f}%")
    out(f"  (b) mean |slope| = {mabs(fl['slope'].values):.4f}")
    # binomial test: is the tracking fraction above chance?
    if len(fl):
        k = int(fl['tracks'].sum())
        binom_p = stats.binomtest(k, len(fl), ALPHA, alternative='greater').pvalue
        out(f"      binomial test (fraction > chance): p = {binom_p:.4f}")

    out("\n" + "-"*70)
    out("TEST 2 — SPECIFICITY: is tracking greater in frontal/limbic than OTHER?")
    out("  (direct group comparison, permutation test)")
    out("-"*70)
    out(f"  frontal/limbic: n={len(fl)}, frac tracking={frac(fl)*100:.0f}%, "
        f"mean|slope|={mabs(fl['slope'].values):.4f}")
    out(f"  other:          n={len(ot)}, frac tracking={frac(ot)*100:.0f}%, "
        f"mean|slope|={mabs(ot['slope'].values):.4f}")
    if len(fl) >= 3 and len(ot) >= 3:
        # permutation on fraction tracking
        obs_f, p_f = perm_diff_test(fl['tracks'].astype(float).values,
                                    ot['tracks'].astype(float).values,
                                    np.mean, rng=rng)
        out(f"  (a) fraction difference = {obs_f*100:+.0f}%, perm p = {p_f:.4f}")
        # permutation on mean |slope|
        obs_s, p_s = perm_diff_test(np.abs(fl['slope'].values),
                                    np.abs(ot['slope'].values), np.mean, rng=rng)
        out(f"  (b) mean|slope| difference = {obs_s:+.4f}, perm p = {p_s:.4f}")

    out("\n" + "-"*70)
    out("TEST 3 — BAND: within frontal/limbic, high-gamma tracking > other bands?")
    out("-"*70)
    for band in ['delta', 'theta', 'alpha', 'beta', 'low_gamma']:
        bd = ch[(ch['band'] == band) & (ch['group'] == 'frontal_limbic')].dropna(subset=['slope'])
        if len(bd) >= 3 and len(fl) >= 3:
            obs_s, p_s = perm_diff_test(np.abs(fl['slope'].values),
                                        np.abs(bd['slope'].values), np.mean, rng=rng)
            out(f"  high_gamma |slope|={mabs(fl['slope'].values):.3f} vs {band} "
                f"|slope|={mabs(bd['slope'].values):.3f}: diff={obs_s:+.3f}, p={p_s:.4f}")

    out("\n" + "-"*70)
    out("DIRECTION — do FRONTAL vs LIMBIC track in opposite directions? (high-gamma)")
    out("-"*70)
    fr = hg[hg['macro'] == 'frontal']
    li = hg[hg['macro'] == 'limbic']
    out(f"  frontal: n={len(fr)}, mean SIGNED slope={fr['slope'].mean():+.4f}, "
        f"{(fr['slope']>0).sum()} up / {(fr['slope']<0).sum()} down")
    out(f"  limbic:  n={len(li)}, mean SIGNED slope={li['slope'].mean():+.4f}, "
        f"{(li['slope']>0).sum()} up / {(li['slope']<0).sum()} down")
    if len(fr) >= 3 and len(li) >= 3:
        t, p = stats.ttest_ind(fr['slope'], li['slope'], equal_var=False)
        out(f"  signed-slope difference (frontal vs limbic): t={t:+.3f}, p={p:.4f}")
        out(f"  -> {'opposite/different directions' if p<ALPHA else 'no significant directional difference'}")

    out("\n" + "="*70)
    out("CAVEATS")
    out("="*70)
    out("  - n=1 patient: 'not other regions' may be limited power, not true")
    out("    absence; specificity needs multi-patient replication.")
    out("  - Tracking is sign-agnostic (fraction + |slope|) so opposite-direction")
    out("    channels don't cancel. Direction analyzed separately above.")
    out("  - Uneven channel coverage across regions; exploratory; p not corrected")
    out("    across the multiple tests.")

    with open(os.path.join(args.out_dir, 'aim1v2_summary.txt'), 'w') as f:
        f.write("\n".join(L))
    print(f"\n[saved aim1v2_summary.txt and aim1v2_channel_slopes.csv]")


if __name__ == '__main__':
    main()
