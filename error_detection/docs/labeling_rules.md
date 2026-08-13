# Error Detection Labeling Rules

## Objective

This stage detects whether an RTE record contains a clear and meaningful
electrical disturbance. It does not yet classify the detailed fault type.

## Labels

### clear_event

Use this label when there is clear and sustained evidence of a disturbance.

Typical evidence:
- Persistent change in one or more phase currents
- Clear voltage sag, interruption, or sustained imbalance
- Coherent voltage and current change around the event onset
- Residual-current change that agrees with phase-current behaviour

### clear_normal

Use this label when the three-phase current and voltage signals are steady
and no meaningful sustained disturbance is visible.

### uncertain

Use this label when:
- The change is weak or dominated by noise
- The detected start time is not plausible
- It may be a switching transient, energization, or fault
- Current and voltage evidence contradict each other
- The event cannot be defended confidently after inspection

## Training rule

The first supervised models use only:
- clear_event = 1
- clear_normal = 0

Uncertain events are excluded from training, validation, and test metrics.

## Supporting features

Residual current is calculated as:

I_res(t) = Ia(t) + Ib(t) + Ic(t)

It is supporting evidence of imbalance or possible ground-return behaviour.
It is not used alone to prove that an event is a ground fault.