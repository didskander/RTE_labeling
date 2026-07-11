import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\didsk\Desktop\Relay-protection\src\data\rte_events\DATA_S.npz"
OUT_DIR   = r"C:\Users\didsk\Desktop\Relay-protection\plots"
os.makedirs(OUT_DIR, exist_ok=True)

FS      = 6400
CYCLE   = 128
V_STEP  = 18.310
I_STEP  = 4.314
V_NOM   = 90000 / np.sqrt(2)   # ~63,640 V RMS nominal


raw    = np.load(DATA_PATH)
DATA_S = raw[list(raw.keys())[0]].astype(np.float64)
t      = np.arange(21000) / FS   # time axis in seconds


def find_fault_start(event):
    signals = [event[ch] * (V_STEP if ch < 3 else I_STEP) for ch in range(6)]
    # smooth each signal with a n-sample window to kill sensor noise
    derivs  = [np.abs(np.diff(savgol_filter(s, 13, 3))) * FS for s in signals]
    # worst spike across all 6 channels at each sample
    combined = np.max(np.stack(derivs), axis=0)


    pre = combined[:2 * CYCLE]
    base = np.median(pre)
    mad  = np.median(np.abs(pre - base))
    thr  = base + 2 * mad + 1e-9

    hits = np.where(combined > thr)[0]
    return int(hits[0]) if len(hits) > 0 else 0



def classify(event, fs):
    end = min(fs + 2 * CYCLE, 21000)
    i = [event[3+ch][fs:end] * I_STEP for ch in range(3)]
    v = [event[ch][fs:end]   * V_STEP for ch in range(3)]

    ri = np.array([np.sqrt(np.mean(x**2)) for x in i])
    rv = np.array([np.sqrt(np.mean(x**2)) for x in v])

    
    n = int((ri > 0.60 * ri.max()).sum())

    # zero sequence: if i1+i2+i3 ≠ 0 → ground is involved
    zs = np.mean(np.abs(i[0] + i[1] + i[2]))
    ti = np.mean(np.abs(i[0]) + np.abs(i[1]) + np.abs(i[2])) + 1e-9
    gnd = zs / ti   # > 0.10 = ground fault

    # voltage sag per phase (how much below nominal?)
    sag = [max(0, 1 - rv[ch] / V_NOM) for ch in range(3)]

    if   n == 1:                       label = "SLG"
    elif n == 2 and gnd < 0.08:        label = "LL"
    elif n == 2 and gnd >= 0.08:       label = "LLG"
    elif n == 3 and gnd < 0.08:        label = "LLL"
    else:                              label = "LLLG"

    return label, ri, rv, sag, gnd, n



colors = ['#636EFA', '#EF553B', '#00CC96']

for ev in range(5):
    event       = DATA_S[ev]
    fault_samp  = find_fault_start(event)
    fault_time  = fault_samp / FS
    label, ri, rv, sag, gnd, n = classify(event, fault_samp)

    i_phys = [event[3+ch] * I_STEP for ch in range(3)]
    v_phys = [event[ch]   * V_STEP for ch in range(3)]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Current (A) — phases elevated: {n}  |  ground ratio: {gnd:.2f}",
            f"Voltage (V) — sag: v1={sag[0]:.0%}  v2={sag[1]:.0%}  v3={sag[2]:.0%}"
        ],
        vertical_spacing=0.15
    )

    for ch in range(3):
        fig.add_trace(go.Scatter(
            x=t, y=i_phys[ch],
            name=f"i{ch+1}  RMS={ri[ch]:.0f}A",
            line=dict(color=colors[ch])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t, y=v_phys[ch],
            name=f"v{ch+1}  RMS={rv[ch]/1000:.1f}kV",
            line=dict(color=colors[ch]),
            showlegend=True
        ), row=2, col=1)

    # red line = fault start
    for row in [1, 2]:
        fig.add_vline(x=fault_time, line_color="red",
                      line_dash="dash", annotation_text=f"fault@{fault_time:.3f}s",
                      row=row, col=1)

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_layout(
        title=f"Event {ev}  →  Classified: <b>{label}</b>",
        height=700,
        legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center')
    )

    path = os.path.join(OUT_DIR, f"event_{ev}.html")
    fig.write_html(path)
    print(f"Event {ev}: fault at t={fault_time:.3f}s | label={label} | "
          f"n_phases={n} | gnd={gnd:.3f} | "
          f"i_rms={ri.round(0)} | v_sag={[f'{s:.0%}' for s in sag]}")

