# RTE Error Detection

This module is part of a master's thesis on relay protection using
real-world RTE event recordings.

## Goal

The first task is binary event detection:

- `clear_event`: a clear and sustained disturbance is visible
- `clear_normal`: a clearly normal and steady waveform
- `uncertain`: ambiguous, weak, noisy, or unverified event

Only `clear_event` and `clear_normal` are used in the first supervised
training experiments. `uncertain` events are excluded from training and
kept for later review.

## Planned workflow

1. Review and label a small high-confidence subset.
2. Extract or load event features and waveform windows.
3. Train a feature-based baseline.
4. Train a small 1D CNN on raw waveform windows.
5. Use model uncertainty to prioritize the next events for review.

## Current status

- Dataset: approximately 12,000 RTE recorded events
- Initial manual labels: approximately 180 events
- First ML task: clear event detection, not fault-type classification
- Detailed fault labels will be studied in a later project stage.

## Data policy

RTE recordings, manual labels, generated models, and local outputs are
not included in this repository.

## Run

From the project root:

```powershell
python error_detection/src/prepare_labels.py
python error_detection/src/inspect_dataset.py
```

Training scripts will be added after label review and dataset validation.