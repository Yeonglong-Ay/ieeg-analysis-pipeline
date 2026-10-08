#!/usr/bin/env python
# stage3_nminus1.py
"""
STAGE 3: trial-to-trial (n-1) analyses of band power (Analysis 2).

Uses the per-trial band power from Stage 2 (trial_power_<event>.csv) plus the
behavioural data (pilot_results CSVs) to relate each trial's neural response to
the PREVIOUS trial, three ways, per channel x band x event:

  (a) autocorrelation : corr( response_n , response_{n-1} )
  (b) prior-outcome    : response_n as a function of trial (n-1) OUTCOME
                         (win/loss) and prior bet  -> the loss-chasing link
  (c) difference       : mean and test of ( response_n - response_{n-1} )

Block boundaries are respected: the first experimental trial of each block has
no n-1 (we do not carry the previous trial across the break).

Trial-index mapping: Stage 1 concatenated block1 then block2 events, dropped
practice, then split per phase. We reconstruct that exact order here so each
power value maps to its (block, experimental-trial, prior outcome/bet).

Outputs (in --out-dir):
  nminus1_results.csv    per channel x band x event: stats for (a),(b),(c)
  nminus1_summary.txt    FDR-corrected hits, grouped by region
  (optional) figures for the strongest effects

Usage:
  python stage3_nminus1.py --stage2-dir stage2_mne_out \
      --stage1-dir stage1_mne_out \
      --events events_final_block1.csv events_final_block2.csv \
      --behav pilot_results_...112021.csv pilot_results_...113531.csv \
      --out-dir stage3_out
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

EVENTS = ['bet_submitted', 'color_submitted', 'feedback']
NUM_PRACTICE = 5


def reconstruct_event_order(event_csvs):
    """Rebuild the per-phase event ordering exactly as Stage 1 did, so
    trial_index -> (block, trial_number). Returns dict phase -> DataFrame with
    columns [trial_index, block, trial_number] in the order used."""
    dfs = []
    for i, c in enumerate(event_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    ev = ev[ev['matched'] == True]
    ev = ev[ev['trial_number'] > NUM_PRACTICE]
    out = {}
    for ph in EVENTS:
        sub = ev[ev['phase'] == ph].reset_index(drop=True)
        sub['trial_index'] = np.arange(len(sub))
        # experimental trial number within block (1..72)
        sub['exp_trial'] = sub['trial_number'] - NUM_PRACTICE
        out[ph] = sub[['trial_index', 'block', 'trial_number', 'exp_trial']]
    return out


def load_behaviour(behav_csvs):
    """Load pilot_results, keep experimental trials, index by (block, exp_trial).
    Returns DataFrame with block, exp_trial, bet, wheel_outcome, correct,
    prior_win (of n-1 within block), prior_bet."""
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c); d['block'] = i + 1
        dfs.append(d)
    b = pd.concat(dfs, ignore_index=True)
    b = b[b['phase'] == 'experimental'].copy()
    # experimental_trial is 1..72 within block
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'], errors='coerce')
    b = b.dropna(subset=['exp_trial'])
    b['exp_trial'] = b['exp_trial'].astype(int)
    b = b.sort_values(['block', 'exp_trial']).reset_index(drop=True)
    # 'correct' True = win (predicted colour matched outcome)
    b['win'] = b['correct'].astype(str).str.lower().isin(['true', '1', 'yes'])
    # prior trial (n-1) within the SAME block only
    b['prior_win'] = b.groupby('block')['win'].shift(1)
    b['prior_bet'] = b.groupby('block')['bet'].shift(1)
    return b[['block', 'exp_trial', 'bet', 'win', 'prior_win', 'prior_bet']]


def analyze(power_df, order_df, behav, ev):
    """For one event, run (a)(b)(c) per channel x band."""
    # attach block/exp_trial to each power row via trial_index
    p = power_df.merge(order_df, on='trial_index', how='left')
    p = p.merge(behav, on=['block', 'exp_trial'], how='left')
    results = []
    for (band, chan), g in p.groupby(['band', 'bipolar_name']):
        g = g.sort_values(['block', 'exp_trial'])
        # build n and n-1 arrays WITHIN block (no crossing break)
        rn, rprev, prior_win, prior_bet = [], [], [], []
        for blk, gb in g.groupby('block'):
            gb = gb.sort_values('exp_trial')
            vals = gb['power_z'].values
            et = gb['exp_trial'].values
            pw = gb['prior_win'].values
            pb = gb['prior_bet'].values
            # consecutive only: n-1 must be exp_trial-1 and present
            idx = {int(e): k for k, e in enumerate(et)}
            for k, e in enumerate(et):
                if (e - 1) in idx:
                    rn.append(vals[k]); rprev.append(vals[idx[e-1]])
                    prior_win.append(pw[k]); prior_bet.append(pb[k])
        rn = np.array(rn, float); rprev = np.array(rprev, float)
        prior_win = np.array(prior_win, float); prior_bet = np.array(prior_bet, float)
        n = len(rn)
        row = {'event': ev, 'band': band, 'bipolar_name': chan, 'n_pairs': n}
        if n >= 8:
            # (a) autocorrelation
            r_ac, p_ac = stats.pearsonr(rprev, rn)
            row['autocorr_r'] = r_ac; row['autocorr_p'] = p_ac
            # (b) prior outcome: response_n ~ prior_win (loss vs win)
            m = ~np.isnan(prior_win)
            if m.sum() >= 8 and len(np.unique(prior_win[m])) == 2:
                win_grp = rn[m][prior_win[m] == 1]
                loss_grp = rn[m][prior_win[m] == 0]
                if len(win_grp) >= 3 and len(loss_grp) >= 3:
                    t_b, p_b = stats.ttest_ind(loss_grp, win_grp,
                                               equal_var=False)
                    row['prior_loss_minus_win'] = loss_grp.mean() - win_grp.mean()
                    row['prior_outcome_t'] = t_b; row['prior_outcome_p'] = p_b
                # prior bet correlation
                mb = m & ~np.isnan(prior_bet)
                if mb.sum() >= 8:
                    r_pb, p_pb = stats.pearsonr(prior_bet[mb], rn[mb])
                    row['prior_bet_r'] = r_pb; row['prior_bet_p'] = p_pb
            # (c) difference response_n - response_{n-1}
            diff = rn - rprev
            t_d, p_d = stats.ttest_1samp(diff, 0.0)
            row['mean_diff'] = diff.mean(); row['diff_t'] = t_d; row['diff_p'] = p_d
        results.append(row)
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2-dir', default='stage2_mne_out')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--events', nargs='+', required=True)
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage3_out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    order = reconstruct_event_order(args.events)
    behav = load_behaviour(args.behav)

    all_res = []
    for ev in EVENTS:
        pf = os.path.join(args.stage2_dir, f'trial_power_{ev}.csv')
        if not os.path.exists(pf):
            print(f"  (skip {ev}: {pf} not found)"); continue
        power_df = pd.read_csv(pf)
        res = analyze(power_df, order[ev], behav, ev)
        all_res.append(res)
        print(f"[{ev}] analyzed {res['bipolar_name'].nunique()} channels x "
              f"{res['band'].nunique()} bands")
    res = pd.concat(all_res, ignore_index=True)

    # FDR correction across all tests, per analysis type
    for pcol, qcol in [('autocorr_p', 'autocorr_q'),
                       ('prior_outcome_p', 'prior_outcome_q'),
                       ('prior_bet_p', 'prior_bet_q'),
                       ('diff_p', 'diff_q')]:
        if pcol in res and HAVE_SM:
            mask = res[pcol].notna()
            q = np.full(len(res), np.nan)
            if mask.sum() > 0:
                q[mask.values] = multipletests(res.loc[mask, pcol],
                                               method='fdr_bh')[1]
            res[qcol] = q

    res.to_csv(os.path.join(args.out_dir, 'nminus1_results.csv'), index=False)

    # summary: significant prior-outcome effects (the loss-chasing link)
    lines = ["n-1 ANALYSIS SUMMARY (FDR q<0.05)\n"]
    if 'prior_outcome_q' in res:
        sig = res[(res['prior_outcome_q'] < 0.05)].sort_values('prior_outcome_p')
        lines.append(f"Prior-outcome (loss vs win) significant: {len(sig)} "
                     f"channel-band-event tests")
        for _, r in sig.head(30).iterrows():
            lines.append(f"  {r['event']:14s} {r['band']:10s} {r['bipolar_name']:14s} "
                         f"loss-win={r['prior_loss_minus_win']:+.3f} "
                         f"q={r['prior_outcome_q']:.4f}")
    for label, qc, pc in [('autocorr', 'autocorr_q', 'autocorr_r'),
                          ('prior_bet', 'prior_bet_q', 'prior_bet_r'),
                          ('difference', 'diff_q', 'mean_diff')]:
        if qc in res:
            s = res[res[qc] < 0.05]
            lines.append(f"\n{label}: {len(s)} significant (FDR q<0.05)")
    with open(os.path.join(args.out_dir, 'nminus1_summary.txt'), 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nStage 3 done. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
