#!/usr/bin/env python
# make_demo_data.py
"""
Generate SYNTHETIC demo data for the iEEG analysis pipeline.

Creates entirely FAKE data (random numbers in the correct format) so the
pipeline can be demonstrated WITHOUT any real patient data. No real neural or
behavioural data are used or included.

Produces, under demo_data/:
  - electrodes.csv : fake electrode localization
  - events_final_block1.csv, events_final_block2.csv : fake aligned events
  - pilot_results_demo_1.csv, pilot_results_demo_2.csv : fake behaviour

Usage:  python make_demo_data.py
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
os.makedirs('demo_data', exist_ok=True)

leads = {'LA': 'Right-Amygdala', 'LH': 'Left-Hippocampus',
         'LF': 'ctx_lh_G_front_middle', 'LO': 'ctx_rh_G_orbital',
         'LI': 'Left-Insula'}
rows = []
elec = 1
for lead, region in leads.items():
    for c in range(1, int(rng.integers(6, 11)) + 1):
        rows.append({'Label': f'{lead}{c}', 'LabelPrefix': lead,
                     'ContactOrder': c, 'Electrode': elec, 'FSLabel': region,
                     'mni_x': round(rng.uniform(-40, 40), 1),
                     'mni_y': round(rng.uniform(-40, 40), 1),
                     'mni_z': round(rng.uniform(-30, 30), 1)})
        elec += 1
pd.DataFrame(rows).to_csv('demo_data/electrodes.csv', index=False)
print(f"electrodes.csv: {len(rows)} contacts")

NUM_PRACTICE = 5


def _trial(phase, exp_trial, t, streak_type='none', streak_length=0,
           prior='', win=None):
    return {'participant_id': 'DEMO', 'phase': phase,
            'experimental_trial': exp_trial,
            'predetermined_outcome': (prior if phase == 'experimental' else ''),
            'color_choice': rng.choice(['red', 'black']),
            'wheel_outcome': rng.choice(['red', 'black']),
            'correct': bool(win) if win is not None else rng.random() > 0.5,
            'streak_type': streak_type, 'streak_length': streak_length,
            'prior_outcome': prior, 'bet': int(rng.choice([1, 2, 4, 8])),
            't_fixation': round(t, 3), 't_bet_onset': round(t + 1.0, 3),
            't_bet_response': round(t + 1.0 + rng.uniform(0.5, 3), 3),
            't_color_onset': round(t + 3.0, 3),
            't_color_response': round(t + 3.0 + rng.uniform(0.5, 2), 3),
            't_spin_start': round(t + 5.0, 3), 't_feedback': round(t + 7.0, 3)}


def make_behaviour(n_exp, fname, start):
    recs = []
    t = start
    for i in range(NUM_PRACTICE):
        recs.append(_trial('practice', np.nan, t)); t += rng.uniform(6, 10)
    sd, sl = None, 0
    for i in range(1, n_exp + 1):
        win = rng.random() > 0.5
        o = 'W' if win else 'L'
        sl = sl + 1 if sd == o else 1
        sd = o
        recs.append(_trial('experimental', i, t, f'{o}{sl}', sl, o, win))
        t += rng.uniform(6, 12)
    pd.DataFrame(recs).to_csv(f'demo_data/{fname}', index=False)
    print(f"{fname}: {n_exp} experimental trials")


make_behaviour(40, 'pilot_results_demo_1.csv', 1000.0)
make_behaviour(24, 'pilot_results_demo_2.csv', 2000.0)


def make_events(behav_file, fname):
    b = pd.read_csv(f'demo_data/{behav_file}')
    recs = []
    for _, r in b.iterrows():
        if r['phase'] == 'practice':
            continue
        tn = int(r['experimental_trial']) + NUM_PRACTICE
        for ph, tcol in [('fixation', 't_fixation'), ('bet_onset', 't_bet_onset'),
                         ('bet_submitted', 't_bet_response'),
                         ('color_submitted', 't_color_response'),
                         ('feedback', 't_feedback')]:
            recs.append({'trial_number': tn, 'phase': ph,
                         'ns3_onset_s': round(r[tcol] + rng.normal(0, 0.001), 4),
                         'matched': True})
    pd.DataFrame(recs).to_csv(f'demo_data/{fname}', index=False)
    print(f"{fname}: {len(recs)} events")


make_events('pilot_results_demo_1.csv', 'events_final_block1.csv')
make_events('pilot_results_demo_2.csv', 'events_final_block2.csv')
print("\nSynthetic demo data created — entirely fake, no real patient data.")
