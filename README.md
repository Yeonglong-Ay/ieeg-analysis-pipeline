# iEEG Analysis Pipeline — Decision-Making & Loss-Chasing

A multi-stage Python pipeline for analyzing intracranial EEG (iEEG) recorded
during a risky decision-making task, studying the neural correlates of
loss-chasing behavior.

**Note on data:** This repository contains analysis **code only**. No patient
data is included. A synthetic demo dataset generator (`make_demo_data.py`) is
provided so the pipeline's data interfaces can be run without real data.

## Repository structure

ieeg-analysis-pipeline/
├── pipeline/ Core analysis stages (1–14) + shared helper
├── analysis/ Hypothesis tests, behavioral loss-chasing, dataset summaries
├── alignment/ Photodiode-based neural–behavioral alignment
├── plotting/ Alignment visualization
├── make_demo_data.py Synthetic demo-data generator
└── demo_data/ Synthetic (fake) data for running the pipeline


## Pipeline stages (`pipeline/`)

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

## Other scripts

- `analysis/` — Aim-1 hypothesis tests, behavioral loss-chasing analysis,
  dataset-info summaries, self-paced timing.
- `alignment/` — Photodiode-to-log alignment and pulse counting.
- `plotting/` — Alignment scatter/diagnostic plots.

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
pipeline's expected format, under `demo_data/`, so the data interfaces can be
run without real data.

## Example usage

Run stages from within the `pipeline/` folder (so the shared helper imports
resolve):

cd pipeline
python stage1_mne.py --ns3 <recording.ns3>
--electrodes ../demo_data/electrodes.csv
--events ../demo_data/events_final_block1.csv ../demo_data/events_final_block2.csv
--behav ../demo_data/pilot_results_demo_1.csv ../demo_data/pilot_results_demo_2.csv
--out-dir stage1_out


Downstream stages take the previous stage's output directory via `--stage*-dir`.

## Note

Developed for research use. Analysis code only — contains no patient data.
