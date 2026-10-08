#!/usr/bin/env python
# hie_practice.py
"""
Shared helper: auto-detect the number of PRACTICE
trials per block from the behavioural pilot_results, so trial filtering and
trial-number <-> experimental-trial mapping are correct even when blocks differ
(e.g. when block 1 has practice trials but block 2 does not).

This replaces the hard-coded global NUM_PRACTICE=5 assumption. Works for any
patient/block structure (including when both blocks have practice).

Key facts about the data layout:
  - pilot_results has a 'phase' column with 'practice'/'experimental' rows and a
    'block' we assign by file order. Practice trials per block = count of
    'practice' rows in that block.
  - events_final 'trial_number' counts ALL trials in a block (practice first,
    then experimental). So experimental trials in a block are those with
    trial_number > (n_practice in that block).
  - pilot_results 'experimental_trial' numbers experimental trials 1..N within
    the block (practice excluded). So the event trial_number of an experimental
    trial = experimental_trial + n_practice(block).
"""
import pandas as pd


def practice_per_block(behav_csvs):
    """Return dict {block:int -> n_practice:int} by counting practice-phase
    rows per block in the pilot_results CSVs (file order = block order)."""
    counts = {}
    for i, c in enumerate(behav_csvs):
        d = pd.read_csv(c)
        blk = i + 1
        if 'phase' in d.columns:
            counts[blk] = int((d['phase'].astype(str).str.lower() == 'practice').sum())
        else:
            counts[blk] = 0
    return counts


def exp_trial_to_trial_number(exp_trial, block, n_practice_by_block):
    """Map experimental_trial (1..N within block) -> events_final trial_number,
    per block: trial_number = exp_trial + n_practice(block)."""
    return exp_trial + n_practice_by_block.get(int(block), 0)


def is_experimental(trial_number, block, n_practice_by_block):
    """True if an events_final trial_number is an EXPERIMENTAL trial in its
    block (i.e., beyond that block's practice trials)."""
    return trial_number > n_practice_by_block.get(int(block), 0)
