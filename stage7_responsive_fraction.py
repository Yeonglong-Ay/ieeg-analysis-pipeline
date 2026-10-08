#!/usr/bin/env python
# stage7_responsive_fraction.py
"""
Responsive-fraction analysis (representativeness check).

For each channel and band, test whether the event-window band power differs
from baseline. The per-trial values from Stage 2 are already baseline-z-scored,
so 'differs from baseline' = the channel's mean power_z differs from 0 across
trials (one-sample t-test). Then, per REGION and BAND, report the FRACTION of
channels that are individually responsive.

This answers: is a region's averaged response representative of the region, or
driven by a minority of channels? (e.g. '4/6 amygdala channels responsive in
high-gamma' vs '1/6').

Definition: responsive = mean(power_z over trials) significantly != 0
  - report BOTH uncorrected (p<0.05) and FDR-corrected (q<0.05, across all
    channel x band tests within an event)

Outputs (in --out-dir):
  responsive_fraction_<event>.csv   region x band: n_responsive / n_total (+ %)
  responsive_fraction_<event>.png   bar chart of % responsive, region x band
  responsive_channels_<event>.csv   per-channel responsiveness detail

Usage:
  python stage7_responsive_fraction.py --stage1-dir stage1_mne_out \
      --stage2-dir stage2_mne_out --out-dir stage7_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except Exception:
    HAVE_SM = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
EVENTS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']
REGIONS = ['orbitofrontal', 'insula', 'cingulate', 'amygdala',
           'hippocampus', 'thalamus', 'frontal']   # temporal dropped


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
    if any(k in s for k in ['temp', 'fusifor', 'collat', 'oc-temp',
                            'lat_fis', 'transv']): return 'temporal'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--stage2-dir', default='stage2_mne_out')
    ap.add_argument('--out-dir', default='stage7_out')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    anat = pd.read_csv(os.path.join(args.stage1_dir, 'bipolar_anatomy.csv'))
    reg = anat['anode_region'].map(region_of)
    reg = reg.where(reg.notna(), anat['cathode_region'].map(region_of))
    anat['region'] = reg
    ch2region = dict(zip(anat['bipolar_name'], anat['region']))

    for ev in EVENTS:
        p = pd.read_csv(os.path.join(args.stage2_dir, f'trial_power_{ev}.csv'))
        p['region'] = p['bipolar_name'].map(ch2region)

        # per channel x band: one-sample t-test of power_z vs 0
        rows = []
        for (band, chan), g in p.groupby(['band', 'bipolar_name']):
            vals = g['power_z'].dropna().values
            if len(vals) >= 5:
                t, pval = stats.ttest_1samp(vals, 0.0)
            else:
                t, pval = np.nan, np.nan
            rows.append({'event': ev, 'band': band, 'bipolar_name': chan,
                         'region': ch2region.get(chan), 'n_trials': len(vals),
                         'mean_z': np.nanmean(vals) if len(vals) else np.nan,
                         't': t, 'p': pval})
        ch = pd.DataFrame(rows)

        # FDR across all channel x band tests within this event
        m = ch['p'].notna()
        ch['q'] = np.nan
        if HAVE_SM and m.sum() > 0:
            ch.loc[m, 'q'] = multipletests(ch.loc[m, 'p'], method='fdr_bh')[1]
        ch['responsive_unc'] = ch['p'] < args.alpha
        ch['responsive_fdr'] = ch['q'] < args.alpha
        # signed: increase vs decrease (direction from mean_z), matching Overton
        # et al. Fig 3 (power increases vs decreases relative to baseline)
        ch['increased_fdr'] = ch['responsive_fdr'] & (ch['mean_z'] > 0)
        ch['decreased_fdr'] = ch['responsive_fdr'] & (ch['mean_z'] < 0)
        ch['increased_unc'] = ch['responsive_unc'] & (ch['mean_z'] > 0)
        ch['decreased_unc'] = ch['responsive_unc'] & (ch['mean_z'] < 0)
        ch.to_csv(os.path.join(args.out_dir,
                               f'responsive_channels_{ev}.csv'), index=False)

        # region x band fraction responsive (signed)
        frac_rows = []
        for reg_name in REGIONS:
            for band in BANDS:
                sub = ch[(ch['region'] == reg_name) & (ch['band'] == band)]
                n_tot = len(sub)
                if n_tot == 0:
                    continue
                frac_rows.append({
                    'region': reg_name, 'band': band, 'n_total': n_tot,
                    'n_resp_fdr': int(sub['responsive_fdr'].sum()),
                    'pct_resp_fdr': 100*sub['responsive_fdr'].sum()/n_tot,
                    'n_increased_fdr': int(sub['increased_fdr'].sum()),
                    'pct_increased_fdr': 100*sub['increased_fdr'].sum()/n_tot,
                    'n_decreased_fdr': int(sub['decreased_fdr'].sum()),
                    'pct_decreased_fdr': 100*sub['decreased_fdr'].sum()/n_tot,
                    'n_resp_unc': int(sub['responsive_unc'].sum()),
                    'pct_resp_unc': 100*sub['responsive_unc'].sum()/n_tot})
        frac = pd.DataFrame(frac_rows)
        frac.to_csv(os.path.join(args.out_dir,
                                 f'responsive_fraction_{ev}.csv'), index=False)

        # signed bar chart: % increased (up) vs % decreased (down), per region,
        # grouped by band — matching Overton et al. Fig 3A
        present = [r for r in REGIONS if r in set(frac['region'])]
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(present))
        w = 0.13
        for bi, band in enumerate(BANDS):
            inc = [frac[(frac['region'] == r) & (frac['band'] == band)]
                   ['pct_increased_fdr'].values for r in present]
            inc = [v[0] if len(v) else 0 for v in inc]
            dec = [frac[(frac['region'] == r) & (frac['band'] == band)]
                   ['pct_decreased_fdr'].values for r in present]
            dec = [-(v[0] if len(v) else 0) for v in dec]   # negative = down
            ax.bar(x + bi*w, inc, w, color=f'C{bi}',
                   label=band if True else None)
            ax.bar(x + bi*w, dec, w, color=f'C{bi}', alpha=0.55)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x + w*2.5)
        ax.set_xticklabels([f"{r}\n(n={frac[frac['region']==r]['n_total'].iloc[0]})"
                            for r in present], fontsize=9)
        ax.set_ylabel('% decreased   ←   |   →   % increased  (FDR q<0.05)')
        ax.set_title(f'Task-active channels: power increase/decrease by region '
                     f'and band — {ev}', fontsize=12)
        ax.legend(fontsize=8, ncol=6, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir,
                                 f'responsive_fraction_{ev}.png'), dpi=140)
        plt.close(fig)
        print(f"[{ev}] saved signed fraction table + bar chart")
        top = frac.sort_values('pct_resp_fdr', ascending=False).head(6)
        for _, r in top.iterrows():
            print(f"    {r['region']:14s} {r['band']:10s} "
                  f"resp {r['n_resp_fdr']}/{r['n_total']} "
                  f"(↑{r['n_increased_fdr']} ↓{r['n_decreased_fdr']})")

    print(f"\nDone. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
