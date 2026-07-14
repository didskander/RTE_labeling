# RTE_labeling

This repository contains code and reference files for relay-protection event labeling and analysis.

## Project overview

The project combines:
- manual labeling data for protection events,
- scripts for inspecting and plotting events,
- and an automatic labeling pipeline for classifying fault and switching behavior.

The automatic labeling workflow analyzes three-phase current and voltage signals, detects event start times, computes RMS-based features, and assigns labels such as:
- `SLG`
- `LL`
- `LLG`
- `LLL`
- `LLLG`
- `energization`
- `steady_state_live`
- `uncertain_window`

## Manual labels

The manual dataset includes event classes such as:
- `1-P-SC`
- `2-P-SC`
- `2-PG-SC`
- `3-P-SC`
- `Switch On`
- `Switch Off`
- `Normal`
- `Other`

These labels are used as a reference for understanding and comparing event behavior.

## Repository structure

- `src/` — source code for labeling, feature extraction, models, and training
- `scripts_n8n/` — helper scripts for inspection and plotting
- `plots/` and `output_n8n/plots/` — generated event visualizations
- `input_n8n/` — input labeling files
- `docs/` — related background documents and references

## Notes

Large raw event data files are intentionally excluded from version control.
Generated plots and manual labels are included for analysis and review.
