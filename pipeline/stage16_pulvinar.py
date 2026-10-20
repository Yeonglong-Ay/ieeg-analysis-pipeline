#!/usr/bin/env python
# stage16_pulvinar.py
"""
STAGE 16: target-nucleus analysis restricted to anatomically-verified contacts.

Written for the pulvinar lead (RPUL), but generalises to any lead/label pair
via --lead and --target-label (e.g. --lead RANT --target-label Thalamus for the
anterior-thalamic control).

WHY THIS IS A SEPARATE STAGE, NOT A RE-RUN OF STAGE 5/9/11
---------------------------------------------------------
Those stages pool contacts into coarse regions. Every 'Thalamus-Proper'
contact lands in one `thalamus` bucket, so pulvinar (RPUL1-5, y~-30) gets
averaged with anterior thalamus (RANT1-4, y~-7) and the distinction is lost.
This stage selects contacts by anatomy FIRST and never pools across nuclei.

It also applies the channel-averaging fix that Stage 5 still needs: all
statistics are computed on ONE value per trial (channels averaged within the
target first), so the n entering every test is the number of trials, not
trials x channels. This is the honest unit and it is why p-values here will be
larger -- and more defensible -- than Stage 5's.

THREE PARTS
-----------
  A. RESPONSE PROFILE (no condition split)
     Does the target respond to task events at all? Per epoch x band, the
     mean normalised power time course with a permutation CI. For pulvinar
     this is the localisation check: a short-latency high-gamma response at
     fixation / bet_onset is physiological evidence the contacts are in a
     visually-responsive structure.

  B. CONDITION DIFFERENCES (cluster-based permutation)
     long (L3-4) vs short (L1-2) losing streak, and win vs loss.
     See INTERPRETABILITY below -- not all epoch x contrast cells mean the
     same thing, and the script labels each one in the figure.

  C. STREAK REGRESSION (Stage 5 logic, honest unit)
     Trial-mean power in a window regressed on streak length 1-4, separately
     for losing and winning streaks, per band per epoch. FDR across the
     band x epoch grid. This is the direct test of 'does power scale with
     accumulated losses'.

INTERPRETABILITY (printed into every figure; read before interpreting)
---------------------------------------------------------------------
  * win vs loss is ONLY defined at feedback -- the outcome is unknown at every
    earlier event.
  * win vs loss AT FEEDBACK IS VISUALLY CONFOUNDED for a visual structure like
    pulvinar: the win and loss feedback screens are different images. A
    difference there may be a response to different pictures, not to value.
    Figures tag this cell CONFOUNDED.
  * long vs short streak at bet_submitted is the CLEAN contrast for pulvinar:
    the screen is identical regardless of loss history, so any difference
    cannot be visual.
  * Low-frequency bands (delta/theta) are reported but carry a smoothness
    inflation artifact; high-gamma is the band to trust.
  * Trial counts per condition are printed on every panel. With n~96 trials
    a short second block, the long-streak cells can be very thin. A cell
    with <MIN_PER_COND trials is skipped, not plotted as if tested.

Usage
-----
  # 1. ALWAYS run this first -- reports channels/trials/columns, computes nothing
  python stage16_pulvinar.py --stage1-dir stage1_mne_out \
      --behav pilot_results_*.csv --inspect

  # 2. Then the real run
  python stage16_pulvinar.py --stage1-dir stage1_mne_out \
      --behav pilot_results_block1.csv pilot_results_block2.csv \
      --out-dir stage16_pulvinar_out --lead RPUL --target-label Thalamus \
      --n-perm 1000
"""
import argparse, os, glob, sys, re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'low_gamma': (30, 70), 'high_gamma': (70, 150)}
BAND_ORDER = list(BANDS)
EVENT_ORDER = ['fixation', 'bet_onset', 'bet_submitted',
               'color_submitted', 'feedback']
MIN_PER_COND = 5          # below this a condition cell is skipped, not tested
TRUSTED_BAND = 'high_gamma'
# Bands whose envelope this implementation cannot estimate reliably on these
# epoch lengths. A 4th-order Butterworth + Hilbert needs several seconds of
# clean signal either side at 1-8 Hz, and stage1's pad is ~1 s, so residual
# edge contamination survives the crop at delta/theta. stage2 uses Morlet
# wavelets, which are properly localised at low frequency, so ITS low bands are
# fine -- these are not. Read beta and above here.
UNRELIABLE_BANDS = ('delta', 'theta')
MAX_STREAK = 4
# palette copied from stage5/9/11 so figures sit alongside the existing ones
C_LOSS = '#C44E52'      # loss / long streak
C_WIN  = '#4C72B0'      # win / short streak
C_DIFF = '#1a1a1a'      # difference trace
CLUSTER_T_THRESH = 2.0  # same fixed |t| cluster-forming threshold as stage11
# Stage 1 epochs carry ~1 s of padding on each side so that filter/TFR edge
# artifacts live in the pad. These are stage2_mne.py's EVENTS windows -- the
# clean core the envelope MUST be cropped to after filtering. Without this the
# delta/theta envelopes are dominated by filtfilt edge transients (a smooth
# bowl centred on the epoch, with |z| in the tens).
EVENT_WINDOWS = {
    'fixation':        (-0.5, 0.5),
    'bet_onset':       (-0.5, 0.5),
    'bet_submitted':   (-1.0, 0.0),
    'color_submitted': (-1.0, 0.0),
    'feedback':        (-1.5, 1.0),
}
FIX_BASELINE_WIN = (-0.5, 0.0)   # stage2's pre-flash fixation reference

# Which streak variable is correct at each event.
# 'prior' = run before this trial's outcome (the state at bet time).
# 'incl'  = run including this trial's outcome (what the patient faces at
#           feedback). Using 'prior' at feedback understates the streak by one
#           on every continuing trial.
STREAK_VAR = {'fixation': 'prior', 'bet_onset': 'prior',
              'bet_submitted': 'prior', 'color_submitted': 'prior',
              'feedback': 'incl'}

# Which (event, contrast) cells are interpretable, and why not when they aren't.
INTERP = {
    ('feedback', 'outcome'):       ('CONFOUNDED',
                                    'win/loss feedback screens differ visually'),
    ('bet_submitted', 'streak'):   ('CLEAN',
                                    'identical screen; differs only in history'),
    ('feedback', 'streak'):        ('OK',
                                    'streak known; outcome valence not held constant'),
    ('bet_onset', 'streak'):       ('OK', 'streak known at this event'),
    ('fixation', 'streak'):        ('OK', 'streak known at this event'),
    ('color_submitted', 'streak'): ('OK', 'streak known at this event'),
}


# =====================================================================
# ======  THE TWO FUNCTIONS THAT MUST MATCH YOUR EXISTING PIPELINE  ====
# ======  Verify with --inspect before trusting any result below.   ====
# =====================================================================

def load_epochs(stage1_dir, event):
    """Return (data [n_trials x n_chan x n_times], ch_names, sfreq, times).

    Expected layout: stage1_mne_out/epochs_<event>-epo.fif
    """
    import mne
    path = os.path.join(stage1_dir, f'epochs_{event}-epo.fif')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    ep = mne.read_epochs(path, preload=True, verbose='ERROR')
    return (ep.get_data(), list(ep.ch_names),
            float(ep.info['sfreq']), ep.times.copy())


def load_ident(stage1_dir, event):
    """Per-epoch trial identity: one row per surviving epoch, in epoch order.

    CONFIRMED: stage1_mne_out/ident_<event>.csv with columns block,trial_number
    where trial_number == pilot_results global_trial. This is what makes the
    behaviour merge exact rather than positional, so trials dropped by
    artifact rejection or RT exclusion cannot silently shift the alignment.
    """
    path = os.path.join(stage1_dir, f'ident_{event}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    return pd.read_csv(path)


def load_behavior(behav_csvs):
    """One row per experimental trial, keyed by (block, global_trial).

    Streak decoding copied from stage5_streak_length.py so the two stages
    agree by construction: streak_type is coded 'L3'/'W2'/'none', NOT the
    words loss/win -- the encoding that previously inverted a Stage 5 result.

    TWO streak variables are returned, because they are not the same thing:
      prior_type/prior_len : the run BEFORE this trial's outcome. This is the
          state at BET TIME and is what Stage 5 uses at bet_submitted.
      incl_type/incl_len   : the run INCLUDING this trial's outcome. This is
          the accumulated loss total the patient faces at FEEDBACK.
    Using the prior streak at feedback would understate the streak by one on
    every continuing trial, so the script picks per epoch (see STREAK_VAR).

    The practice-trial count is DERIVED from the file (phase column), never
    hardcoded. Note stage5_streak_length.py hardcodes NUM_PRACTICE = 5, which
    is wrong for any block without practice trials; some sessions
    have none. See the warning this prints.
    """
    dfs = []
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c)
        d['block'] = i + 1
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)

    blocks = sorted(raw['block'].unique())
    n_prac = {int(b): int((raw['phase'] != 'experimental')[raw['block'] == b].sum())
              for b in blocks}
    print(f'  [practice] per-block practice trials (derived): {n_prac}')
    odd = {b: v for b, v in n_prac.items() if v != 5}
    if odd:
        print(f'  [practice] *** blocks not having 5 practice trials: {odd}. '
              'stage5_streak_length.py hardcodes NUM_PRACTICE=5 and will '
              'misalign those blocks. Check whether stage1 made the same '
              'assumption -- if it did, the EPOCHS are misaligned too. ***')

    b = raw[raw['phase'] == 'experimental'].copy()
    b['global_trial'] = pd.to_numeric(b['global_trial'],
                                      errors='coerce').astype('Int64')
    b['exp_trial'] = pd.to_numeric(b['experimental_trial'],
                                   errors='coerce').astype('Int64')

    st = b['streak_type'].astype(str).str.upper().str.strip()
    b['prior_type'] = st.str[0].map({'L': 'loss', 'W': 'win'})
    sl = pd.to_numeric(b['streak_length'], errors='coerce')
    sl_lab = pd.to_numeric(st.str[1:], errors='coerce')
    b['prior_len'] = sl.where(sl.notna(), sl_lab)

    b['outcome'] = (b['predetermined_outcome'].astype(str)
                    .str.upper().str[0].map({'W': 'win', 'L': 'loss'}))

    inc_t, inc_l = [], []
    for pt, pl, oc in zip(b['prior_type'], b['prior_len'], b['outcome']):
        if pd.notna(pt) and pt == oc and pd.notna(pl):
            inc_t.append(oc); inc_l.append(min(int(pl) + 1, MAX_STREAK))
        else:
            inc_t.append(oc); inc_l.append(1)
    b['incl_type'] = inc_t
    b['incl_len'] = inc_l

    # independent cross-check: rebuild the run structure from the outcome
    # sequence alone and confirm it matches the decoded columns
    chk, run_c, run_n, prev_blk = [], None, 0, None
    for blk, oc in zip(b['block'], b['outcome']):
        if blk != prev_blk:          # streaks reset at the start of each block
            run_c, run_n = None, 0
            prev_blk = blk
        run_n = run_n + 1 if oc == run_c else 1
        run_c = oc
        chk.append(min(run_n, MAX_STREAK))
    if chk != list(b['incl_len']):
        n_bad = sum(a != c for a, c in zip(chk, b['incl_len']))
        print(f'  [streak] *** {n_bad} trials where the file\'s streak columns '
              'disagree with the run structure rebuilt from the outcome '
              'sequence. Inspect before trusting any streak result. ***')
    else:
        print('  [streak] file columns agree with sequence-derived runs: OK')

    out = b[['block', 'global_trial', 'exp_trial', 'outcome',
             'prior_type', 'prior_len', 'incl_type', 'incl_len']].copy()
    out.attrs['n_practice_by_block'] = n_prac
    return out


# =====================================================================
# ======================  anatomy selection  ==========================
# =====================================================================

def bipolar_parts(ch_name):
    """'RPUL1-RPUL2' -> ('RPUL', 1, 2). Returns None if not parseable."""
    m = re.match(r'^([A-Za-z]+)(\d+)\s*-\s*([A-Za-z]+)?(\d+)$', ch_name.strip())
    if not m:
        return None
    pre, a, pre2, b = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if pre2 and pre2 != pre:
        return None          # cross-lead pair, not a within-lead bipolar
    return pre, a, b


def select_channels(stage1_dir, lead, target_label):
    """Bipolar channels on `lead` with BOTH ends carrying target_label.

    Reads stage1_mne_out/bipolar_anatomy.csv (columns: bipolar_name, lead,
    anode_label, cathode_label, anode_region, cathode_region, mni_*), so the
    channel naming never has to be guessed -- it comes from the same file the
    rest of the pipeline uses.

    Requiring BOTH ends is deliberate. A channel spanning thalamus and white
    matter is not a thalamic channel, and including it is how a 'pulvinar'
    analysis quietly becomes a white-matter analysis.
    """
    path = os.path.join(stage1_dir, 'bipolar_anatomy.csv')
    anat = pd.read_csv(path)
    sub = anat[anat['lead'].astype(str) == lead].copy()
    if sub.empty:
        raise ValueError(f'lead {lead!r} not in {path}. '
                         f'Leads present: {sorted(anat["lead"].unique())}')
    tl = target_label.lower()
    both = (sub['anode_region'].astype(str).str.lower().str.contains(tl) &
            sub['cathode_region'].astype(str).str.lower().str.contains(tl))
    keep = list(sub.loc[both, 'bipolar_name'])
    drop = list(sub.loc[~both, 'bipolar_name'])
    coords = {r['bipolar_name']: (r.get('mni_x'), r.get('mni_y'), r.get('mni_z'))
              for _, r in sub.iterrows()}
    return keep, drop, coords


# =====================================================================
# ======================  band power  =================================
# =====================================================================

def crop_to_core(x, times, event, decim=1):
    """Trim stage1's padding to stage2's analysis window, then decimate.

    MUST be applied AFTER filtering, never before: the point of the pad is to
    absorb the filter's edge transients so they can be discarded here.
    """
    lo, hi = EVENT_WINDOWS.get(event, (times[0], times[-1]))
    m = (times >= lo) & (times <= hi)
    if m.sum() < 10:
        raise ValueError(f'{event}: crop window {(lo, hi)} leaves {m.sum()} '
                         f'samples of {len(times)}')
    return x[..., m][..., ::decim], times[m][::decim]


def band_envelope(data, lo, hi, sfreq):
    """data [tr x ch x t] -> analytic amplitude in [lo,hi], same shape."""
    ny = sfreq / 2.0
    hi = min(hi, ny * 0.95)
    b, a = butter(4, [lo / ny, hi / ny], btype='band')
    filt = filtfilt(b, a, data, axis=-1)
    return np.abs(hilbert(filt, axis=-1))


def fixation_reference(stage1_dir, kept_names, base=FIX_BASELINE_WIN):
    """Per-trial, per-channel log-band-power mean/sd from the FIXATION epoch.

    Mirrors stage2_mne.py --baseline-mode fixation (which produced
    stage2_mne_fix): the reference is each trial's own pre-flash fixation
    window, i.e. the period BEFORE the fixation cross appears. That is a
    task-neutral reference, unlike an event's own pre-event window, which at
    bet_submitted sits mid-bet and at feedback sits mid-spin.

    Returns {band: (mu, sd)} with arrays [n_fix_trials x n_chan], plus the
    (block, trial_number) key per fixation trial so other events can look up
    their matching baseline.
    """
    d, names, sf, t = load_epochs(stage1_dir, 'fixation')
    ii = [names.index(c) for c in kept_names if c in names]
    sub = d[:, ii, :]
    idt = load_ident(stage1_dir, 'fixation')
    ref = {}
    for bn, (lo, hi) in BANDS.items():
        env = band_envelope(sub, lo, hi, sf)
        # filter on the padded epoch, THEN discard the pad (as stage2 does)
        env, tc = crop_to_core(env, t, 'fixation')
        m = (tc >= base[0]) & (tc <= base[1])
        if m.sum() < 2:
            raise ValueError(f'fixation baseline window {base} selects '
                             f'<2 samples of the cropped core')
        lg = np.log(np.maximum(env, 1e-20))
        ref[bn] = (lg[..., m].mean(axis=-1), lg[..., m].std(axis=-1) + 1e-12)
    key = {(int(b), int(tn)): i for i, (b, tn)
           in enumerate(zip(idt['block'], idt['trial_number']))}
    return ref, key


def normalise_to_fixation(env, ref_mu, ref_sd, rows, key):
    """z-score log band power against each trial's matched fixation baseline.

    env [tr x ch x t]; rows is the event's ident table (block, trial_number).
    Trials with no matching fixation trial fall back to their own pre-event
    window, and the count is reported -- same fallback stage2 uses.
    """
    lg = np.log(np.maximum(env, 1e-20))
    out = np.empty_like(lg)
    n_missing = 0
    for tr in range(lg.shape[0]):
        k = (int(rows['block'].iloc[tr]), int(rows['trial_number'].iloc[tr]))
        j = key.get(k)
        if j is None:
            n_missing += 1
            mu = lg[tr, :, :].mean(axis=-1, keepdims=True)
            sd = lg[tr, :, :].std(axis=-1, keepdims=True) + 1e-12
            out[tr] = (lg[tr] - mu) / sd
        else:
            out[tr] = (lg[tr] - ref_mu[j][:, None]) / ref_sd[j][:, None]
    return out, n_missing


def normalise(env, times, base=(-0.5, 0.0)):
    """Log then z-score each trial x channel against its own pre-event window.

    Matches Stage 2's logic (log power, baseline-normalised). Note this is a
    WITHIN-EPOCH baseline: for fixation it is the true pre-task baseline, for
    later events it is the pre-event period, which is itself task-active.
    Stage 2 used the fixation window for everything; if you need strict
    comparability with Stage 2, pass the fixation baseline in instead.
    """
    lg = np.log(np.maximum(env, 1e-20))
    m = (times >= base[0]) & (times < base[1])
    if m.sum() < 3:
        m = times < np.median(times)
    mu = lg[..., m].mean(axis=-1, keepdims=True)
    sd = lg[..., m].std(axis=-1, keepdims=True)
    return (lg - mu) / np.maximum(sd, 1e-12)


# =====================================================================
# ======================  statistics  =================================
# =====================================================================

def cluster_perm(a, b, n_perm, rng, alpha=0.05):
    """Cluster-based permutation on two groups of 1-D traces.

    Returns (diff, sem, null_lo, null_hi, sig_mask, cluster_pvals) -- the same
    quantities stage11_difference_plots.py returns, using the same fixed |t|
    cluster-forming threshold and the same max-cluster-mass null, so pulvinar
    and region-level results are computed identically.
    """
    na, nb = len(a), len(b)
    pooled = np.concatenate([a, b], axis=0)

    def tstat(x, y):
        mx, my = np.nanmean(x, 0), np.nanmean(y, 0)
        vx, vy = np.nanvar(x, 0, ddof=1), np.nanvar(y, 0, ddof=1)
        return (mx - my) / np.sqrt(vx / len(x) + vy / len(y) + 1e-12)

    def clusters(t):
        supra = np.abs(t) > CLUSTER_T_THRESH
        out, start = [], None
        for i, s in enumerate(supra):
            if s and start is None:
                start = i
            elif not s and start is not None:
                out.append((start, i, float(np.abs(t[start:i]).sum())))
                start = None
        if start is not None:
            out.append((start, len(supra), float(np.abs(t[start:]).sum())))
        return out

    diff = np.nanmean(a, 0) - np.nanmean(b, 0)
    sem = np.sqrt(np.nanvar(a, 0) / na + np.nanvar(b, 0) / nb)
    obs = clusters(tstat(a, b))

    null_max = np.empty(n_perm)
    null_diffs = np.empty((n_perm, pooled.shape[1]), np.float32)
    for p in range(n_perm):
        perm = rng.permutation(pooled)
        pa, pb = perm[:na], perm[na:]
        null_diffs[p] = np.nanmean(pa, 0) - np.nanmean(pb, 0)
        cl = clusters(tstat(pa, pb))
        null_max[p] = max((c[2] for c in cl), default=0.0)

    null_lo = np.nanpercentile(null_diffs, 2.5, axis=0)
    null_hi = np.nanpercentile(null_diffs, 97.5, axis=0)
    crit = np.nanpercentile(null_max, 100 * (1 - alpha))
    sig = np.zeros(pooled.shape[1], dtype=bool)
    pvals = []
    for s, e, mass in obs:
        pvals.append(((s, e), float((null_max >= mass).mean())))
        if mass > crit:
            sig[s:e] = True
    return diff, sem, null_lo, null_hi, sig, pvals


def fdr(pvals, q=0.05):
    """Benjamini-Hochberg. Returns boolean mask of rejected hypotheses."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    rej = np.zeros(p.shape, bool)
    if ok.sum() == 0:
        return rej
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        rej[order[:kmax + 1]] = True
    return rej


# =====================================================================
# ======================  main  =======================================
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', required=True)
    ap.add_argument('--electrodes', default=None,
                    help='unused; anatomy comes from stage1_mne_out/bipolar_anatomy.csv')
    ap.add_argument('--behav', nargs='+', required=True)
    ap.add_argument('--out-dir', default='stage16_pulvinar_out')
    ap.add_argument('--lead', default='RPUL')
    ap.add_argument('--target-label', default='Thalamus')
    ap.add_argument('--events', nargs='+', default=EVENT_ORDER)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--decim', type=int, default=10,
                    help='temporal decimation after filtering (2000 Hz / 10 '
                         '= 200 Hz). Matches the spirit of stage2 DECIM; '
                         'reduces plot density and compute.')
    ap.add_argument('--baseline-mode', choices=['fixation', 'self'],
                    default='fixation',
                    help="'fixation' (default) matches stage2_mne_fix: z-score "
                         "against each trial's pre-flash fixation window. "
                         "'self' uses each epoch's own pre-event window.")
    ap.add_argument('--inspect', action='store_true',
                    help='report channels/trials/columns and exit')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------- behaviour ----------
    beh = load_behavior(args.behav)
    print(f'\n[behaviour] {len(beh)} trials from {len(args.behav)} file(s)')
    print('  outcome counts :', beh['outcome'].value_counts().to_dict())
    for sv in ('prior', 'incl'):
        tal = (beh.dropna(subset=[f'{sv}_type'])
                  .groupby([f'{sv}_type', f'{sv}_len']).size()
                  .unstack(fill_value=0))
        print(f'  streak tally [{sv}] (rows loss/win, cols length 1-4):')
        print('    ' + tal.to_string().replace('\n', '\n    '))
    print('  >>> CHECK the tallies against your designed sequence. "prior" is '
          'used at bet events, "incl" at feedback.')

    # ---------- channels ----------
    keep, dropped, coords = select_channels(
        args.stage1_dir, args.lead, args.target_label)
    print(f'\n[anatomy] lead {args.lead}, region contains '
          f'{args.target_label!r}')
    print(f'  bipolar KEPT    ({len(keep)}): {keep}')
    for ch in keep:
        x, y, z = coords.get(ch, (None, None, None))
        if x is not None:
            print(f'      {ch:16s} MNI({float(x):6.1f},{float(y):6.1f},'
                  f'{float(z):6.1f})')
    print(f'  bipolar dropped ({len(dropped)}): {dropped}')
    if not keep:
        sys.exit('No bipolar channel has BOTH ends in the target region. '
                 'Check --lead / --target-label against bipolar_anatomy.csv.')

    if args.inspect:
        print('\n[inspect] epochs + behaviour merge:')
        for ev in args.events:
            try:
                d, nm, sf, t = load_epochs(args.stage1_dir, ev)
                idt = load_ident(args.stage1_dir, ev)
                npb = beh.attrs.get('n_practice_by_block', {})
                idt = idt.copy()
                idt['exp_trial'] = (idt['trial_number']
                                    - idt['block'].map(
                                        lambda b: npb.get(int(b), 0)))
                mg = idt.merge(beh, on=['block', 'exp_trial'], how='left')
                miss = int(mg['outcome'].isna().sum())
                sv = STREAK_VAR.get(ev, 'prior')
                lt = mg[mg[f'{sv}_type'] == 'loss']
                nlong = int(lt[f'{sv}_len'].isin([3, 4]).sum())
                nshort = int(lt[f'{sv}_len'].isin([1, 2]).sum())
                flag = '  <-- ROWS MISMATCH' if len(idt) != d.shape[0] else ''
                print(f'  {ev:17s} {d.shape[0]:4d} epochs, {len(idt):4d} ident '
                      f'rows{flag}, {d.shape[1]:4d} chan, '
                      f't=[{t[0]:+.2f},{t[-1]:+.2f}]s, sf={sf:g}')
                print(f'  {"":17s} unmerged={miss}, streak_var={sv}, '
                      f'short(L1-2)={nshort} long(L3-4)={nlong}, '
                      f'win={int((mg["outcome"]=="win").sum())} '
                      f'loss={int((mg["outcome"]=="loss").sum())}')
            except Exception as e:
                print(f'  {ev:17s} FAILED: {e}')
        print('\n[inspect] nothing computed. Verify the three blocks above '
              '(streak tally, kept channels, trial counts), then re-run '
              'without --inspect.')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    kept_names = keep

    # ---------- load + reduce every epoch to one trace per trial ----------
    # power[event][band] = [n_trials x n_times], channels averaged per trial.
    fix_ref, fix_key = (None, None)
    if args.baseline_mode == 'fixation':
        fix_ref, fix_key = fixation_reference(args.stage1_dir, keep)
        print(f'  [baseline] fixation mode: reference built from '
              f'{len(fix_key)} fixation trials (window -0.5 to 0 s, pre-flash)')

    power, tvec, trialinfo = {}, {}, {}
    for ev in args.events:
        d, names, sf, t = load_epochs(args.stage1_dir, ev)
        missing = [c for c in kept_names if c not in names]
        if missing:
            print(f'  [{ev}] dropped by stage1 rejection: {missing}')
        ii = [names.index(c) for c in kept_names if c in names]
        if not ii:
            raise ValueError(f'no target channels survive in {ev}')
        idt = load_ident(args.stage1_dir, ev)
        if len(idt) != d.shape[0]:
            raise ValueError(
                f'{ev}: {d.shape[0]} epochs but {len(idt)} ident rows. '
                'Alignment cannot be trusted; fix before interpreting.')
        npb = beh.attrs.get('n_practice_by_block', {})
        idt = idt.copy()
        idt['exp_trial'] = (idt['trial_number']
                            - idt['block'].map(lambda b: npb.get(int(b), 0)))
        mg = idt.merge(beh, on=['block', 'exp_trial'], how='left')
        n_un = int(mg['outcome'].isna().sum())
        if n_un:
            bad = mg[mg['outcome'].isna()][['block', 'trial_number']]
            raise ValueError(
                f'{ev}: {n_un} epochs have no matching behaviour row, so the '
                f'epoch-to-trial alignment is wrong. Unmatched (block, '
                f'trial_number): {list(map(tuple, bad.values))}. This usually '
                'means stage1 and the behaviour file disagree about the '
                'practice-trial count for a block. Fix upstream -- do NOT '
                'interpret results computed on a partial merge.')
        trialinfo[ev] = mg.reset_index(drop=True)
        sub = d[:, ii, :]
        power[ev], tvec[ev] = {}, None
        for bn, (lo, hi) in BANDS.items():
            env = band_envelope(sub, lo, hi, sf)
            env, tc = crop_to_core(env, t, ev, decim=args.decim)
            tvec[ev] = tc
            if args.baseline_mode == 'fixation':
                z, n_miss = normalise_to_fixation(
                    env, fix_ref[bn][0], fix_ref[bn][1], idt, fix_key)
                if n_miss:
                    print(f'    [{ev}/{bn}] {n_miss} trials had no matching '
                          'fixation baseline; used self-window fallback')
            else:
                z = normalise(env, tc)
            power[ev][bn] = z.mean(axis=1)      # <-- channel average per trial
        print(f'  [{ev}] {sub.shape[0]} trials x {sub.shape[1]} chan')

    n_tr = {ev: power[ev][BAND_ORDER[0]].shape[0] for ev in args.events}

    def masks_for(ev):
        """Condition masks for this epoch, from the ident-merged behaviour."""
        b = trialinfo[ev]
        sv = STREAK_VAR.get(ev, 'prior')
        loss = (b['outcome'] == 'loss').values
        isL = (b[f'{sv}_type'] == 'loss').values
        sl = pd.to_numeric(b[f'{sv}_len'], errors='coerce').values
        # Order is (A, B) and every difference is A - B, so A is the
        # first-named condition in the figure title. Conventions copied from
        # stage11: positive = more power in loss / in the long streak.
        return {
            'outcome': (('loss', loss),
                        ('win', ~loss & ~b['outcome'].isna().values)),
            'streak':  (('long_L3-4',  isL & np.isin(sl, [3, 4])),
                        ('short_L1-2', isL & np.isin(sl, [1, 2]))),
        }

    # ================= PART A: response profile =================
    fig, axes = plt.subplots(len(args.events), len(BAND_ORDER),
                             figsize=(3.0 * len(BAND_ORDER),
                                      2.2 * len(args.events)),
                             squeeze=False)
    for r, ev in enumerate(args.events):
        for c, bn in enumerate(BAND_ORDER):
            ax = axes[r][c]
            X, t = power[ev][bn], tvec[ev]
            m, se = X.mean(0), X.std(0) / max(np.sqrt(len(X)), 1)
            ax.plot(t, m, lw=1.2, color=C_DIFF)
            ax.fill_between(t, m - se, m + se, alpha=.25, color=C_DIFF)
            ax.axvline(0, color='k', ls='--', lw=0.7)
            ax.axhline(0, color='grey', ls=':', lw=0.5)
            if r == 0:
                ax.set_title(bn + ('\n(unreliable)' if bn in UNRELIABLE_BANDS
                                   else ''), fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{ev}\n(n={len(X)})', fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f'{args.lead} [{args.target_label}] {len(kept_names)} bipolar ch '
                 f'| response profile, all trials (mean +/- SEM, z vs '
                 f'{"trial fixation baseline" if args.baseline_mode == "fixation" else "pre-event window"})',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .97])
    pA = os.path.join(args.out_dir, f'A_response_{args.lead}.png')
    fig.savefig(pA, dpi=140); plt.close(fig)
    print(f'\n[A] {pA}')

    # ================= PART B: condition differences =================
    rows = []
    for contrast in ('streak', 'outcome'):
        evs = ['feedback'] if contrast == 'outcome' else args.events
        evs = [e for e in evs if e in args.events]
        if not evs:
            continue
        fig, axes = plt.subplots(len(evs), len(BAND_ORDER),
                                 figsize=(3.0 * len(BAND_ORDER),
                                          2.4 * len(evs)), squeeze=False)
        for r, ev in enumerate(evs):
            (na_, ma), (nb_, mb) = masks_for(ev)[contrast]
            tag, why = INTERP.get((ev, contrast), ('OK', ''))
            for c, bn in enumerate(BAND_ORDER):
                ax = axes[r][c]
                X, t = power[ev][bn], tvec[ev]
                A, B = X[ma], X[mb]
                if len(A) < MIN_PER_COND or len(B) < MIN_PER_COND:
                    ax.text(.5, .5, f'n<{MIN_PER_COND}\n{len(A)} vs {len(B)}',
                            ha='center', va='center', fontsize=7,
                            transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    rows.append(dict(contrast=contrast, event=ev, band=bn,
                                     nA=len(A), nB=len(B), interp=tag,
                                     result='skipped_low_n'))
                    continue
                diff, sem, nlo, nhi, sig, pvs = cluster_perm(
                    A, B, args.n_perm, rng)
                # same layering as stage11: grey permutation null envelope,
                # blue SEM band, dark difference trace, red significance bar
                ax.fill_between(t, nlo, nhi, color='0.8', alpha=0.7,
                                label='perm null 95%')
                ax.fill_between(t, diff - sem, diff + sem, color=C_WIN,
                                alpha=0.3, label='SEM')
                ax.plot(t, diff, color=C_DIFF if tag != 'CONFOUNDED' else '0.45',
                        lw=1.2, label='A-B')
                ax.axvline(0, color='k', ls='--', lw=0.6)
                ax.axhline(0, color='grey', ls=':', lw=0.5)
                if sig.any():
                    ylo = ax.get_ylim()[0]
                    ax.plot(t[sig], np.full(sig.sum(), ylo), color='red',
                            lw=4, solid_capstyle='butt')
                if r == 0 and c == len(BAND_ORDER) - 1:
                    ax.legend(fontsize=5, loc='upper right')
                best = min([p for _, p in pvs], default=np.nan)
                rows.append(dict(contrast=contrast, event=ev, band=bn,
                                 nA=len(A), nB=len(B), interp=tag,
                                 result='tested', min_cluster_p=best,
                                 any_sig=bool(sig.any())))
                if r == 0:
                    ax.set_title(bn + (' (unreliable)'
                                       if bn in UNRELIABLE_BANDS else ''),
                                 fontsize=9)
                if c == 0:
                    ax.set_ylabel(f'{ev}\n[{tag}]\n{len(A)} vs {len(B)}',
                                  fontsize=7)
                ax.tick_params(labelsize=6)
        lab = f'{na_} - {nb_}'
        note = ('  *** grey/CONFOUNDED: feedback screens differ visually -- '
                'a difference here may be visual, not value ***'
                if contrast == 'outcome' else
                '  CLEAN cell = bet_submitted (screen identical; '
                'only loss history differs)')
        fig.suptitle(f'{args.lead} [{args.target_label}] | {lab} | '
                     f'cluster permutation (n_perm={args.n_perm}, red bar p<.05)\n'
                     + note, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .95])
        pB = os.path.join(args.out_dir, f'B_{contrast}_{args.lead}.png')
        fig.savefig(pB, dpi=140); plt.close(fig)
        print(f'[B] {pB}')

    # ================= PART C: streak regression =================
    crows, plotdata = [], {}
    for ev in args.events:
        b = trialinfo[ev]
        sv = STREAK_VAR.get(ev, 'prior')
        t = tvec[ev]
        wmask = t >= 0 if (t >= 0).sum() > 3 else np.ones_like(t, bool)
        for bn in BAND_ORDER:
            y_all = power[ev][bn][:, wmask].mean(axis=1)   # one value per trial
            for stype in ('L', 'W'):
                want = 'loss' if stype == 'L' else 'win'
                sel = (b[f'{sv}_type'] == want).values
                x = pd.to_numeric(b[f'{sv}_len'], errors='coerce').values[sel]
                y = y_all[sel]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                keep_s = (x >= 1) & (x <= MAX_STREAK)
                x, y = x[keep_s], y[keep_s]
                if len(x):
                    s = pd.DataFrame({'L': x, 'y': y}).groupby('L')['y']
                    plotdata[(ev, bn, stype)] = (s.mean().index.values,
                                                 s.mean().values,
                                                 s.sem().values)
                if len(np.unique(x)) < 2 or len(x) < MIN_PER_COND:
                    crows.append(dict(event=ev, band=bn, streak=stype,
                                      n=int(len(x)), slope=np.nan, p=np.nan))
                    continue
                lr = stats.linregress(x, y)
                crows.append(dict(event=ev, band=bn, streak=stype, n=int(len(x)),
                                  slope=float(lr.slope), p=float(lr.pvalue),
                                  r=float(lr.rvalue)))
    C = pd.DataFrame(crows)
    for stype in ('L', 'W'):
        m = C['streak'] == stype
        C.loc[m, 'fdr_sig'] = fdr(C.loc[m, 'p'].values)
    C.to_csv(os.path.join(args.out_dir, f'C_streak_regression_{args.lead}.csv'),
             index=False)

    # ---- figure in the stage5/5d idiom: power vs streak length, loss vs win
    evs_c = [e for e in EVENT_ORDER if e in args.events]
    fig, axes = plt.subplots(len(evs_c), len(BAND_ORDER),
                             figsize=(2.6 * len(BAND_ORDER),
                                      1.9 * len(evs_c)), squeeze=False)
    for ri, ev in enumerate(evs_c):
        for bi, bn in enumerate(BAND_ORDER):
            ax = axes[ri][bi]
            for stype, col, lab in (('L', C_LOSS, 'loss'),
                                    ('W', C_WIN, 'win')):
                if (ev, bn, stype) not in plotdata:
                    continue
                xs, ms, ses = plotdata[(ev, bn, stype)]
                ax.errorbar(xs, ms, yerr=ses, marker='o', color=col,
                            capsize=2, lw=1, markersize=3, label=lab)
                sub = C[(C['event'] == ev) & (C['band'] == bn) &
                        (C['streak'] == stype)]
                if len(sub) and np.isfinite(sub['slope'].iloc[0]):
                    sl_ = sub['slope'].iloc[0]
                    ic = ms.mean() - sl_ * xs.mean()
                    ax.plot(xs, ic + sl_ * np.asarray(xs), color=col,
                            ls='--', lw=0.8)
            ax.axhline(0, color='grey', ls=':', lw=0.5)
            ax.set_xticks([1, 2, 3, 4])
            if ri == 0:
                ax.set_title(bn, fontsize=9)
            if bi == 0:
                ax.set_ylabel(f'{ev}\n({STREAK_VAR.get(ev, "prior")})',
                              fontsize=6)
            if ri == 0 and bi == len(BAND_ORDER) - 1:
                ax.legend(fontsize=6, loc='upper right')
            if ri == len(evs_c) - 1:
                ax.set_xlabel('streak length', fontsize=7)
    fig.suptitle(f'{args.lead} [{args.target_label}] power vs streak length '
                 f'- loss (red) vs win (blue); channel-averaged per trial',
                 fontsize=11)
    fig.tight_layout()
    pC1 = os.path.join(args.out_dir, f'C_streak_curves_{args.lead}.png')
    fig.savefig(pC1, dpi=140); plt.close(fig)
    print(f'[C] {pC1}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), squeeze=False)
    for k, stype in enumerate(('L', 'W')):
        ax = axes[0][k]
        piv = (C[C['streak'] == stype]
               .pivot(index='event', columns='band', values='slope')
               .reindex(index=[e for e in EVENT_ORDER if e in args.events],
                        columns=BAND_ORDER))
        v = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1
        im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
        ax.set_xticks(range(len(BAND_ORDER)))
        ax.set_xticklabels(BAND_ORDER, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        sub = C[C['streak'] == stype]
        for _, rr in sub.iterrows():
            if bool(rr.get('fdr_sig', False)):
                yy = list(piv.index).index(rr['event'])
                xx = BAND_ORDER.index(rr['band'])
                ax.text(xx, yy, '*', ha='center', va='center', fontsize=13)
        ax.set_title(f'{"LOSING" if stype=="L" else "WINNING"} streak slope '
                     f'(* = FDR q<.05)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle(f'{args.lead} [{args.target_label}] | power ~ streak length, '
                 f'channel-averaged per trial (honest n = trials)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .93])
    pC = os.path.join(args.out_dir, f'C_streak_{args.lead}.png')
    fig.savefig(pC, dpi=140); plt.close(fig)
    print(f'[C] {pC}')

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, f'B_condition_stats_{args.lead}.csv'),
        index=False)

    # ---------- headline, honestly scoped ----------
    print('\n' + '=' * 68)
    print(f'NOTE: {UNRELIABLE_BANDS} are plotted but NOT interpretable in this')
    print('      stage (Butterworth+Hilbert edge contamination at low freq on')
    print('      ~1 s padding). Read beta / low_gamma / high_gamma.')
    print(f'HEADLINE  ({TRUSTED_BAND} only -- the band to trust)')
    hg = C[(C['band'] == TRUSTED_BAND) & (C['streak'] == 'L')]
    for _, rr in hg.iterrows():
        star = ' *FDR' if bool(rr.get('fdr_sig', False)) else ''
        if np.isfinite(rr['p']):
            print(f"  {rr['event']:17s} slope={rr['slope']:+.4f} "
                  f"p={rr['p']:.3f} n={rr['n']}{star}")
    bs = pd.DataFrame(rows)
    clean = bs[(bs['event'] == 'bet_submitted') & (bs['contrast'] == 'streak')
               & (bs['band'] == TRUSTED_BAND)]
    if len(clean):
        print(f"  CLEAN contrast (long-short @ bet_submitted, {TRUSTED_BAND}): "
              f"{clean.iloc[0].to_dict()}")
    print('  Reminder: n=2 exploratory. The interpretable streak cell is '
          'bet_submitted; win/loss at feedback is visually confounded for a '
          'visual structure.')
    print('=' * 68)


if __name__ == '__main__':
    main()
