import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = Path(r"C:\Users\didsk\Desktop\Relay-protection\src\data\manual_label_RTE.xlsx")
DATA_PATH = r"C:\Users\didsk\Desktop\Relay-protection\src\data\rte_events\DATA_S.npz"
OUT_PATH = BASE_DIR / "manual_label_RTE_with_features_fixed.xlsx"

FS = 6400
CYCLE = 128
V_STEP = 18.310
I_STEP = 4.314

SMOOTH_WINDOW = 11
SMOOTH_POLY = 3
PRE_CYCLES = 5
BASE_SCALE = 4.0
MIN_RUN = 3

raw = np.load(DATA_PATH)
DATA_S = raw[list(raw.keys())[0]].astype(np.float64)


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def extract_phases(fault_type, phase_text):
    txt = f"{norm(fault_type)} {norm(phase_text)}"

    if "3-p-sc" in txt or "3 phase" in txt or "3-phase" in txt:
        return ["a", "b", "c"]
    if "a-b" in txt or "b-a" in txt:
        return ["a", "b"]
    if "b-c" in txt or "c-b" in txt:
        return ["b", "c"]
    if "c-a" in txt or "a-c" in txt:
        return ["c", "a"]

    singles = []
    for p in ["a", "b", "c"]:
        if re.search(rf"\b{p}\b", txt):
            singles.append(p)

    return singles[:1] if len(singles) == 1 else []


def phase_channels(phases):
    mapping = {"a": (3, 0), "b": (4, 1), "c": (5, 2)}
    out = []
    for p in phases:
        out.extend(mapping[p])

    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    return uniq if uniq else [0, 1, 2, 3, 4, 5]


def first_run(mask, min_run):
    run = 0
    for i, val in enumerate(mask):
        if val:
            run += 1
            if run >= min_run:
                return i - min_run + 1
        else:
            run = 0
    return None


def rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def derivative_envelope(event, channels):
    scores = []
    for ch in channels:
        scale = V_STEP if ch < 3 else I_STEP
        sig = event[ch] * scale
        sig = savgol_filter(sig, SMOOTH_WINDOW, SMOOTH_POLY)
        scores.append(np.abs(np.diff(sig)) * FS)
    return np.max(np.stack(scores), axis=0)


def detect_start(event, fault_type, phase_text):
    chs = phase_channels(extract_phases(fault_type, phase_text))
    env = derivative_envelope(event, chs)
    pre = env[:PRE_CYCLES * CYCLE]
    base = np.median(pre)
    mad = np.median(np.abs(pre - base)) + 1e-12
    thr = base + BASE_SCALE * mad
    idx = first_run(env > thr, MIN_RUN)
    if idx is not None:
        return idx

    env = derivative_envelope(event, [0, 1, 2, 3, 4, 5])
    pre = env[:PRE_CYCLES * CYCLE]
    base = np.median(pre)
    mad = np.median(np.abs(pre - base)) + 1e-12
    thr = base + BASE_SCALE * mad
    idx = first_run(env > thr, MIN_RUN)
    if idx is not None:
        return idx

    vr = np.stack([
        np.array([
            rms(event[ch][k * CYCLE:(k + 1) * CYCLE] * V_STEP)
            for k in range(len(event[ch]) // CYCLE)
        ])
        for ch in range(3)
    ])
    for k in range(1, vr.shape[1]):
        prev_dead = np.max(vr[:, k - 1]) < 5000.0
        now_live = np.min(vr[:, k]) > 20000.0
        if prev_dead and now_live:
            return k * CYCLE

    ir0 = np.array([rms(event[3 + ch][:CYCLE] * I_STEP) for ch in range(3)])
    ir_all = np.array([rms(event[3 + ch] * I_STEP) for ch in range(3)])
    if np.max(ir0) > 80.0 and np.max(ir0) > 0.7 * np.max(ir_all):
        return 0

    return 0


def zero_seq_features(event, start_sample):
    ia = event[3][start_sample:] * I_STEP
    ib = event[4][start_sample:] * I_STEP
    ic = event[5][start_sample:] * I_STEP

    z = ia + ib + ic
    t = np.arange(len(z)) / FS

    ratio = float(np.mean(np.abs(z)) / (np.mean(np.abs(ia) + np.abs(ib) + np.abs(ic)) + 1e-9))
    integral = float(trapezoid(np.abs(z), t))

    return ratio, integral


df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

col_map = {str(c).strip().lower(): c for c in df.columns}
sample_col = col_map["sample id"]
ft_col = col_map["fault type"]
ph_col = col_map["phase"]

start_times = []
ratios = []
integrals = []

for _, row in df.iterrows():
    sid = row[sample_col]

    if pd.isna(sid) or int(sid) < 0 or int(sid) >= len(DATA_S):
        start_times.append("")
        ratios.append("")
        integrals.append("")
        continue

    event = DATA_S[int(sid)]

    st = detect_start(event, row[ft_col], row[ph_col])
    zr, zi = zero_seq_features(event, st)

    start_times.append(st / FS)
    ratios.append(zr)
    integrals.append(zi)

df["fault_start_time_s"] = start_times
df["zero_seq_ratio"] = ratios
df["zero_seq_integral"] = integrals

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Sheet1")

print(f"Saved to: {OUT_PATH}")