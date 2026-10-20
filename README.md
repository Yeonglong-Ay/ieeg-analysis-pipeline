# iEEG Analysis Pipeline — Decision-Making & Loss-Chasing

A multi-stage Python pipeline for analyzing intracranial EEG (iEEG) recorded
during a risky decision-making task, studying the neural correlates of
loss-chasing behavior.

**Note on data:** This repository contains analysis **code only**. No patient
data is included. A synthetic demo dataset generator (`make_demo_data.py`) is
provided so the pipeline's data interfaces can be run without real data.

## Repository structure

```
ieeg-analysis-pipeline/
  pipeline/          Core analysis stages (1-16) + shared helper
  analysis/          Hypothesis tests, loss-chasing, dataset summaries
  alignment/         Photodiode-based alignment
  plotting/          Alignment visualization
  make_demo_data.py  Synthetic demo-data generator
  demo_data/         Synthetic (fake) data
```

## Pipeline stages (`pipeline/`)

- **Stage 1** — Preprocessing: bipolar re-referencing, notch + high-pass
  filtering, epoching with padding, per-channel-per-trial artifact rejection
  (robust MAD). Writes per-event epochs, a per-event trial-identity table, and a
  bipolar anatomy table.
- **Stage 2** — Time-frequency decomposition (Morlet wavelets) → band power
  across 6 bands; baseline normalization (per-event or per-trial fixation
  reference). Band power is computed on the padded epoch and then cropped, so
  filter/TFR edge transients stay in the pad.
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
- **Stage 15 / 15b** — Within-electrode correlation and coherence along a single
  depth lead, split by outcome (15) and by losing-streak length (15b),
  trial-matched with permutation significance.
- **Stage 16** — Anatomy-restricted single-nucleus analysis. Selects contacts by
  contact-level anatomy rather than coarse region, requiring **both** ends of a
  bipolar channel to lie in the target structure, and never pools across
  neighbouring nuclei. Produces (A) an event-response profile, (B) condition
  differences with cluster-based permutation, and (C) streak-length regression.
  Generalises to any structure via `--lead` / `--target-label`.

## Methodological notes

A few choices in this pipeline are deliberate and worth stating, since they
differ from the obvious default:

- **Anatomy before averaging.** Coarse region labels can merge functionally
  distinct nuclei (e.g. several thalamic nuclei share one FreeSurfer label), and
  assigning a bipolar channel's region from one contact with a fallback to the
  other can admit channels that straddle a tissue boundary. Stage 16 selects on
  both contacts and keeps nuclei separate.
- **Trials, not trial × channel, as the unit.** Where channels are combined,
  Stage 16 averages them within each trial before any statistic, so the sample
  size is the number of trials. Pooling trial-channel pairs treats correlated
  channels as independent and deflates p-values.
- **Padding is cropped after filtering, never before.** Narrow-band filtering of
  a short epoch produces edge transients that can exceed the signal; the pad
  exists to absorb them and is discarded afterwards. Low-frequency bands need
  substantially more clean signal on either side than high-frequency bands.
- **Chance-level baselines for mass-univariate output.** Significant-cell counts
  in a connectivity matrix are interpreted against the number of pairs tested,
  and agreement between two independent measures is required before an effect is
  treated as real.
- **Shared-contact pairs.** Adjacent bipolar channels share a physical contact
  and are therefore mechanically correlated; these pairs are identified and
  excluded or flagged rather than read as connectivity.
- **Event interpretability.** Outcome contrasts are only defined at feedback;
  contrasts at events that coincide with a motor response or an auditory cue are
  flagged as confounded rather than silently analysed.

## Other scripts

- `analysis/` — Aim-1 hypothesis tests, behavioral loss-chasing analysis,
  dataset-info summaries, self-paced timing.
- `alignment/` — Photodiode-to-log alignment and pulse counting.
- `plotting/` — Alignment scatter/diagnostic plots.

## Methods highlights

- Signal processing: bipolar referencing, notch/high-pass filtering, Morlet
  time-frequency decomposition, Hilbert envelopes.
- Statistics: regression, FDR correction, cluster-based permutation testing,
  permutation-based connectivity significance, robust (MAD) outlier rejection,
  trial-matched subsampling.
- Connectivity: coherence, weighted phase-lag index (wPLI), power-envelope
  correlation, within- vs. between-region analyses.

## Requirements

- Python 3.10+
- MNE-Python, mne-connectivity
- NumPy, SciPy, pandas, statsmodels, matplotlib

## Demo data

```
python make_demo_data.py
```

Generates synthetic data (fake electrode, behavioral, and event files) in the
pipeline's expected format, under `demo_data/`, so the data interfaces can be
run without real data.

## Example usage

Run stages from within the `pipeline/` folder (so the shared helper imports
resolve):

```
cd pipeline
python stage1_mne.py --ns3 <recording.ns3> \
    --electrodes ../demo_data/electrodes.csv \
    --events ../demo_data/events_final_block1.csv ../demo_data/events_final_block2.csv \
    --behav ../demo_data/pilot_results_demo_1.csv ../demo_data/pilot_results_demo_2.csv \
    --out-dir stage1_out
```

Downstream stages take the previous stage's output directory via `--stage*-dir`.

Stage 16 reads Stage 1 output directly and selects its channels from the
bipolar anatomy table:

```
python stage16_pulvinar.py --stage1-dir stage1_out \
    --behav ../demo_data/pilot_results_demo_1.csv ../demo_data/pilot_results_demo_2.csv \
    --lead <LEAD> --target-label <REGION_SUBSTRING> \
    --out-dir stage16_out --inspect
```

`--inspect` reports the channels selected, the per-block practice-trial counts,
the condition tallies and the behaviour merge, and computes nothing — run it
before any full analysis.

## Note

Developed for research use. Analysis code only — contains no patient data.
