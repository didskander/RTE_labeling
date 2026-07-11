import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\didsk\Desktop\Relay-protection\src\data\rte_events\DATA_S.npz"
OUT_DIR = r"C:\Users\didsk\Desktop\Relay-protection\plots"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 6400
CYCLE = 128
V_STEP = 18.310
I_STEP = 4.314
V_NOM = 90000 / np.sqrt(2)  # ~63,640 V RMS nominal

MIN_TOTAL_CURRENT_FOR_GND = 300.0
MIN_FAULT_CURRENT = 80.0
DEAD_VOLTAGE_TH = 5000.0
LIVE_VOLTAGE_TH = 20000.0
LOW_CURRENT_TH = 120.0
SMALL_SAG_TH = 0.15

N_PLOTS = 6  # events 0..5

raw = np.load(DATA_PATH)
DATA_S = raw[list(raw.keys())[0]].astype(np.float64)
t = np.arange(21000) / FS

def rms(x):
    return np.sqrt(np.mean(x**2)) if len(x) > 0 else 0.0

def cycle_rms(sig):
    n_cycles = len(sig) // CYCLE
    return np.array([
        np.sqrt(np.mean(sig[k * CYCLE:(k + 1) * CYCLE] ** 2))
        for k in range(n_cycles)
    ])

# ── STEP 1: FIND EVENT START ──────────────────────────────────────────────────
def find_fault_start(event):
    signals = [event[ch] * (V_STEP if ch < 3 else I_STEP) for ch in range(6)]

    # method 1: derivative spike
    derivs = [np.abs(np.diff(savgol_filter(s, 21, 3))) * FS for s in signals]
    combined = np.max(np.stack(derivs), axis=0)

    pre = combined[:5 * CYCLE]
    base = np.median(pre)
    mad = np.median(np.abs(pre - base))
    thr = base + 5 * mad + 1e-9

    hits = np.where(combined > thr)[0]
    if len(hits) > 0:
        return int(hits[0])

    # method 2: dead line becomes live
    vr = np.stack([cycle_rms(event[ch] * V_STEP) for ch in range(3)])
    for k in range(1, vr.shape[1]):
        prev_dead = np.max(vr[:, k - 1]) < DEAD_VOLTAGE_TH
        now_live = np.min(vr[:, k]) > LIVE_VOLTAGE_TH
        if prev_dead and now_live:
            return k * CYCLE

    # method 3: event already active at t=0
    ir = np.stack([cycle_rms(event[3 + ch] * I_STEP) for ch in range(3)])
    first_max = np.max(ir[:, 0])
    whole_max = np.max(ir)

    if first_max > MIN_FAULT_CURRENT and first_max > 0.7 * whole_max:
        return 0

    return None

# ── STEP 2: CLASSIFY WINDOW ───────────────────────────────────────────────────
def classify(event, fs):
    pre_end = 5 * CYCLE

    pre_i = [event[3 + ch][:pre_end] * I_STEP for ch in range(3)]
    pre_v = [event[ch][:pre_end] * V_STEP for ch in range(3)]

    pre_ri = np.array([rms(x) for x in pre_i])
    pre_rv = np.array([rms(x) for x in pre_v])

    if fs is None:
        all_i = [event[3 + ch] * I_STEP for ch in range(3)]
        all_v = [event[ch] * V_STEP for ch in range(3)]

        all_ri = np.array([rms(x) for x in all_i])
        all_rv = np.array([rms(x) for x in all_v])

        if np.max(all_rv) > LIVE_VOLTAGE_TH and np.max(all_ri) > MIN_FAULT_CURRENT:
            sag = [max(0, 1 - all_rv[ch] / V_NOM) for ch in range(3)]
            n = int((all_ri > 0.5 * max(all_ri.max(), 1e-9)).sum())
            return "steady_state_live", all_ri, all_rv, sag, 0.0, n

        return "uncertain_window", np.zeros(3), np.zeros(3), [0, 0, 0], 0.0, 0

    end = min(fs + 1 * CYCLE, event.shape[1])
    if end <= fs:
        return "uncertain_window", np.zeros(3), np.zeros(3), [0, 0, 0], 0.0, 0

    i = [event[3 + ch][fs:end] * I_STEP for ch in range(3)]
    v = [event[ch][fs:end] * V_STEP for ch in range(3)]

    ri = np.array([rms(x) for x in i])
    rv = np.array([rms(x) for x in v])

    n = int((ri > 0.5 * max(ri.max(), 1e-9)).sum())

    zs = np.mean(np.abs(i[0] + i[1] + i[2]))
    ti = np.mean(np.abs(i[0]) + np.abs(i[1]) + np.abs(i[2])) + 1e-9
    gnd = zs / ti if ri.sum() >= MIN_TOTAL_CURRENT_FOR_GND else 0.0

    sag = [max(0, 1 - rv[ch] / V_NOM) for ch in range(3)]

    late_start = min(fs + 2 * CYCLE, max(fs, event.shape[1] - CYCLE))
    late_end = min(late_start + CYCLE, event.shape[1])

    if late_end > late_start:
        late_i = [event[3 + ch][late_start:late_end] * I_STEP for ch in range(3)]
        late_ri = np.array([rms(x) for x in late_i])
    else:
        late_ri = ri.copy()

    pre_dead = np.max(pre_rv) < DEAD_VOLTAGE_TH
    post_live = np.min(rv) > LIVE_VOLTAGE_TH
    small_sag = max(sag) < SMALL_SAG_TH
    transient_then_low = np.max(ri) > MIN_FAULT_CURRENT and np.max(late_ri) < LOW_CURRENT_TH

    if pre_dead and post_live and small_sag and transient_then_low:
        return "energization", ri, rv, sag, gnd, n

    if ri.max() < MIN_FAULT_CURRENT:
        return "uncertain_window", ri, rv, sag, gnd, n

    if n == 1:
        label = "SLG"
    elif n == 2 and gnd < 0.10:
        label = "LL"
    elif n == 2 and gnd >= 0.10:
        label = "LLG"
    elif n == 3 and gnd < 0.10:
        label = "LLL"
    else:
        label = "LLLG"

    return label, ri, rv, sag, gnd, n

# ── STEP 3: PLOT ──────────────────────────────────────────────────────────────
colors = ['#636EFA', '#EF553B', '#00CC96']

for ev in range(min(N_PLOTS, len(DATA_S))):
    event = DATA_S[ev]

    fault_samp = find_fault_start(event)
    fault_time = None if fault_samp is None else fault_samp / FS

    label, ri, rv, sag, gnd, n = classify(event, fault_samp)

    i_phys = [event[3 + ch] * I_STEP for ch in range(3)]
    v_phys = [event[ch] * V_STEP for ch in range(3)]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Current (A) — phases elevated: {n} | ground ratio: {gnd:.2f}",
            f"Voltage (V) — sag: v1={sag[0]:.0%} v2={sag[1]:.0%} v3={sag[2]:.0%}"
        ],
        vertical_spacing=0.15
    )

    for ch in range(3):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=i_phys[ch],
                name=f"i{ch+1} RMS={ri[ch]:.0f}A",
                line=dict(color=colors[ch])
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=v_phys[ch],
                name=f"v{ch+1} RMS={rv[ch]/1000:.1f}kV",
                line=dict(color=colors[ch]),
                showlegend=True
            ),
            row=2, col=1
        )

    if fault_time is not None:
        for row in [1, 2]:
            fig.add_vline(
                x=fault_time,
                line_color="red",
                line_dash="dash",
                annotation_text=f"event@{fault_time:.3f}s",
                row=row, col=1
            )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_layout(
        title=f"Event {ev} → Classified: {label}",
        height=700,
        legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center')
    )

    path = os.path.join(OUT_DIR, f"event_{ev}.html")
    fig.write_html(path)

    start_txt = f"{fault_time:.3f}s" if fault_time is not None else "None"
    print(
        f"Event {ev}: fault at t={start_txt} | label={label} | "
        f"n_phases={n} | gnd={gnd:.3f} | "
        f"i_rms={ri.round(0)} | v_sag={[f'{s:.0%}' for s in sag]}"
    )

