# iEEG Analysis Pipeline — Decision-Making & Loss-Chasing

A multi-stage Python pipeline for analyzing intracranial EEG (iEEG) recorded
during a risky decision-making task, studying the neural correlates of
loss-chasing behavior.

**Note on data:** This repository contains analysis **code only**. No patient
data is included. A synthetic demo dataset generator (`make_demo_data.py`) is
provided so the pipeline's data interfaces can be run without real data.

## Overview

The pipeline takes raw iEEG recordings and task events and produces
preprocessed neural features, statistical analyses relating neural activity to
behavior, and connectivity analyses. Built with MNE-Python, NumPy/SciPy,
pandas, and statsmodels; runs on a high-performance computing cluster (SLURM).

## Pipeline stages

- **Stage 1** — Preprocessing: bipolar re-referencing, notch + high-pass
  filtering, epoching, per-channel-per-trial artifact rejection (robust MAD).
- **Stage 2** — Time-frequency decomposition (Morlet wavelets) → band power
  across 6 bands; baseline normalization.
- **Stage 3** — Trial-to-trial (n-1) outcome analysis, FDR-corrected.
- **Stage 4** — Region-level response summaries.
- **Stage 5 / 5b–5d** — Streak-length regression (does band power scale with
  losing-streak length?), per channel, region, and PFC subregion.
- **Stage 6–8** — Spectral fingerprint and responsive-fraction visualizations.
- **Stage 9–11** — Condition-split time courses and difference plots with
  cluster-based permutation testing.
- **Stage 12** — Within-region homogeneity + frontal-limbic coupling.
- **Stage 13 / 13b / 13c** — Within-region connectivity matrices, controls,
  distance-decay characterization.
- **Stage 14** — Region-to-region spectral connectivity with permutation
  significance, visualized as circular connectivity graphs.

Additional: behavioral loss-chasing analysis, dataset summaries, photodiode
alignment, and an Aim-1 hypothesis test.

## Methods highlights

- Signal processing: bipolar referencing, notch/high-pass filtering, Morlet
  time-frequency decomposition, Hilbert envelopes.
- Statistics: regression, FDR correction, cluster-based permutation testing,
  permutation-based connectivity significance, robust (MAD) outlier rejection.
- Connectivity: coherence, weighted phase-lag index (wPLI), within- vs.
  between-region analyses.

## Requirements

- Python 3.10+
- MNE-Python, mne-connectivity
- NumPy, SciPy, pandas, statsmodels, matplotlib

## Demo data

python make_demo_data.py

Generates synthetic data (fake electrode, behavioral, and event files) in the
pipeline's expected format, so the data interfaces can be run without real data.

## Note

Developed for research use. Analysis code only — contains no patient data.
