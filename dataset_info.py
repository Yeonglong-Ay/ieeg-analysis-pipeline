#!/usr/bin/env python
# dataset_info.py
"""
General dataset info sheet for a single patient (reusable across patients).

Reads the patient's files and writes a markdown summary with:
  - Practice structure & trial counts per block
  - Per-epoch usable trial counts (from Stage 1 ident files, if present)
  - Self-paced epoch durations (bet decision, color choice reaction times)
  - Channels per region (from Stage 1 bipolar_anatomy.csv, if present) + leads
  - Betting behaviour summary (range, mean/median bet)
  - Photodiode flash counts (from events_final)
  - Basic recording info

Works for any patient — pass the data folder and the behaviour CSVs. Sections
whose inputs are missing are skipped with a note (e.g. if Stage 1 not yet run,
the region-channel section is skipped).

Usage:
  python dataset_info.py --data-dir . \
      --behav pilot_results_A.csv pilot_results_B.csv \
      --events events_final_block1.csv events_final_block2.csv \
      --electrodes electrodes.csv \
      --stage1-dir stage1_mne_out \
      --label SUBJECT_ID \
      --out dataset_info_SUBJECT.md
"""
import argparse, os
import numpy as np
import pandas as pd

EPOCHS = ['fixation', 'bet_onset', 'bet_submitted', 'color_submitted', 'feedback']


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


def fmt_stats(x, unit='s'):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    x = x[(x > 0) & (x < 60)]
    if len(x) == 0:
        return "no valid values"
    return (f"n={len(x)}, mean={x.mean():.2f}{unit}, median={np.median(x):.2f}{unit}, "
            f"SD={x.std():.2f}{unit}, range={x.min():.2f}-{x.max():.2f}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--events', nargs='+', default=None)
    ap.add_argument('--electrodes', default='electrodes.csv')
    ap.add_argument('--stage1-dir', default='stage1_mne_out')
    ap.add_argument('--label', default='patient')
    ap.add_argument('--out', default='dataset_info.md')
    args = ap.parse_args()

    L = []           # markdown lines
    def add(s=''): L.append(s)

    add(f"# Dataset info — {args.label}\n")

    # ---- Behaviour: practice structure, trial counts, betting ----
    bdfs = []
    for i, c in enumerate(args.behav):
        p = os.path.join(args.data_dir, c)
        d = pd.read_csv(p); d['block'] = i + 1
        bdfs.append(d)
    b = pd.concat(bdfs, ignore_index=True)
    add("## Trial structure\n")
    add("| Block | Practice | Experimental |")
    add("|---|---|---|")
    for blk, g in b.groupby('block'):
        n_prac = int((g['phase'].astype(str).str.lower() == 'practice').sum())
        n_exp = int((g['phase'].astype(str).str.lower() == 'experimental').sum())
        add(f"| {blk} | {n_prac} | {n_exp} |")
    tot_exp = int((b['phase'].astype(str).str.lower() == 'experimental').sum())
    add(f"\n**Total experimental trials: {tot_exp}**\n")

    be = b[b['phase'].astype(str).str.lower() == 'experimental'].copy()

    # ---- Per-epoch usable trial counts (from Stage 1 ident files) ----
    add("## Usable trials per epoch (after alignment + artifact rejection)\n")
    s1 = os.path.join(args.data_dir, args.stage1_dir)
    if os.path.isdir(s1):
        add("| Epoch | Usable trials |")
        add("|---|---|")
        for ev in EPOCHS:
            f = os.path.join(s1, f'ident_{ev}.csv')
            if os.path.exists(f):
                add(f"| {ev} | {len(pd.read_csv(f))} |")
            else:
                add(f"| {ev} | (not found) |")
        add("")
    else:
        add(f"_(Stage 1 output `{args.stage1_dir}` not found — run Stage 1 for "
            f"per-epoch counts.)_\n")

    # ---- Self-paced epoch durations (reaction times) ----
    add("## Self-paced epoch durations (reaction times)\n")
    def rt(a, z):
        if a in be.columns and z in be.columns:
            return (pd.to_numeric(be[z], errors='coerce') -
                    pd.to_numeric(be[a], errors='coerce')).values
        return np.array([])
    bet_rt = rt('t_bet_onset', 't_bet_response')
    col_rt = rt('t_color_onset', 't_color_response')
    add(f"- **Bet decision** (bet onset → submit): {fmt_stats(bet_rt)}")
    add(f"- **Color choice** (color onset → submit): {fmt_stats(col_rt)}")
    add("\n_(Median is more representative than mean — reaction times are "
        "right-skewed.)_\n")

    # ---- Betting behaviour ----
    add("## Betting behaviour\n")
    if 'bet' in be.columns:
        bet = pd.to_numeric(be['bet'], errors='coerce').dropna()
        add(f"- Bet values used: {sorted(bet.unique().tolist())}")
        add(f"- Mean bet: {bet.mean():.2f}; median: {bet.median():.2f}; "
            f"range: {bet.min():.0f}-{bet.max():.0f}")
        vc = bet.value_counts().sort_index()
        add(f"- Distribution: " +
            ", ".join(f"${int(k)}×{v}" for k, v in vc.items()))
    add("")

    # ---- Channels per region (from Stage 1 bipolar anatomy) ----
    add("## Channels per region\n")
    anat_f = os.path.join(s1, 'bipolar_anatomy.csv')
    if os.path.exists(anat_f):
        anat = pd.read_csv(anat_f)
        reg = anat['anode_region'].map(region_of)
        reg = reg.where(reg.notna(), anat['cathode_region'].map(region_of))
        anat['region'] = reg
        add(f"- **Total bipolar channels: {len(anat)}** "
            f"(from {anat['lead'].nunique()} leads)")
        add("\n| Region | Bipolar channels |")
        add("|---|---|")
        for rg, n in anat['region'].value_counts().items():
            add(f"| {rg} | {n} |")
        n_excl = anat['region'].isna().sum()
        add(f"\n_(White-matter/unknown excluded: {n_excl})_\n")
    else:
        add(f"_(Stage 1 anatomy `{anat_f}` not found — run Stage 1 for region "
            f"channel counts.)_\n")

    # ---- Photodiode flash counts (from events_final) ----
    if args.events:
        add("## Photodiode flashes (events_final)\n")
        add("| Block | Flash events | Matched | Dropped |")
        add("|---|---|---|---|")
        for i, c in enumerate(args.events):
            p = os.path.join(args.data_dir, c)
            if os.path.exists(p):
                e = pd.read_csv(p)
                nmatch = int(e['matched'].sum()) if 'matched' in e else len(e)
                ndrop = len(e) - nmatch
                add(f"| {i+1} | {len(e)} | {nmatch} | {ndrop} |")
        add("")

    out_path = os.path.join(args.data_dir, args.out)
    with open(out_path, 'w') as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[written to {out_path}]")


if __name__ == '__main__':
    main()
