import os
import base64
import json
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\rte_events\DATA_S.npz"
OUTPUT_DIR = r"C:\Users\didsk\Desktop\Relay-protection\error_detection\data\processed"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "labeling.html")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels_first100_manual.csv")

N_EVENTS_TO_PROCESS = 100
FS = 6400

CONVERT_FROM_COUNTS = False
VOLTAGE_STEP = 18.310
CURRENT_STEP = 4.314


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "DATA_S" in data.files:
            arr = data["DATA_S"]
        elif "arr_0" in data.files:
            arr = data["arr_0"]
        else:
            arr = data[data.files[0]]

    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 3 or arr.shape[1] != 6:
        raise ValueError(f"Expected shape (n_events, 6, n_samples), got {arr.shape}")

    return arr


def convert_event_units(event):
    event = event.astype(float).copy()
    if CONVERT_FROM_COUNTS:
        event[0:3] *= VOLTAGE_STEP
        event[3:6] *= CURRENT_STEP
    return event


# =============================================================================
# PLOTTING
# =============================================================================
def plot_event_to_png(event_id, event):
    v1, v2, v3 = event[0], event[1], event[2]
    i1, i2, i3 = event[3], event[4], event[5]
    i0 = (i1 + i2 + i3) / 3.0

    t = np.arange(event.shape[1]) / FS

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, v1, label="V1", linewidth=0.8)
    axes[0].plot(t, v2, label="V2", linewidth=0.8)
    axes[0].plot(t, v3, label="V3", linewidth=0.8)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, i1, label="I1", linewidth=0.8)
    axes[1].plot(t, i2, label="I2", linewidth=0.8)
    axes[1].plot(t, i3, label="I3", linewidth=0.8)
    axes[1].set_ylabel("Current [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, i0, label="I0 = (I1+I2+I3)/3", color="black", linewidth=0.9)
    axes[2].set_ylabel("Zero-seq current [A]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(
        f"Event {event_id} | Nominal voltage ≈ 90 kV",
        fontsize=12
    )

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)

    buf.seek(0)
    img_bytes = buf.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


# =============================================================================
# HTML GENERATION
# =============================================================================
def build_html(events_data, output_html):
    # events_data: list of {"id": int, "img": base64, ...}
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # Pre-render labels from existing CSV if exists
    existing_labels = {}
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        for _, row in df.iterrows():
            eid = int(row["event_id"])
            existing_labels[eid] = row["label"]

    # Build JSON structure for events
    events_json = []
    for ev in events_data:
        eid = ev["id"]
        events_json.append({
            "id": eid,
            "img": f"data:image/png;base64,{ev['img']}",
            "label": existing_labels.get(eid, "")
        })

    events_json_str = json.dumps(events_json, separators=(",", ":"))

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Manual Event Labeling</title>
<style>
body {{ font-family: Arial, sans-serif; background: #f2f2f2; }}
.container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background: #fff; }}
h1 {{ font-size: 1.5em; }}
.event-box {{ margin-bottom: 20px; }}
.event-img {{ width: 100%; border: 1px solid #ccc; }}
.buttons {{ margin-top: 10px; }}
button {{
  padding: 8px 16px;
  font-size: 14px;
  margin-right: 8px;
  cursor: pointer;
}}
.btn-fault {{ background: #e53935; color: #fff; border: none; }}
.btn-normal {{ background: #43a047; color: #fff; border: none; }}
.btn-uncertain {{ background: #fb8c00; color: #fff; border: none; }}
.btn-download {{ background: #1e88e5; color: #fff; border: none; }}
.nav {{ margin-top: 10px; }}
.nav span {{ margin-right: 10px; font-weight: bold; }}
#progress {{ font-weight: bold; margin-top: 10px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Manual Event Labeling (100 events)</h1>
  <div class="nav">
    <span id="index">Event: 0 / {N_EVENTS_TO_PROCESS - 1}</span>
  </div>
  <div id="progress"></div>
  <div id="event-container" class="event-box">
    <!-- event will be injected here -->
  </div>
  <div class="buttons">
    <button class="btn-fault" onclick="setLabel('FAULT')">FAULT (F)</button>
    <button class="btn-normal" onclick="setLabel('NORMAL')">NORMAL (N)</button>
    <button class="btn-uncertain" onclick="setLabel('UNCERTAIN')">UNCERTAIN (U)</button>
    <button onclick="prevEvent()">Back (B)</button>
  </div>
  <div style="margin-top: 20px;">
    <button class="btn-download" onclick="downloadCSV()">Download CSV</button>
  </div>
</div>

<script>
const events = {events_json};
let current = 0;

function renderEvent() {{
  const ev = events[current];
  const container = document.getElementById("event-container");
  if (!ev) {{
    container.innerHTML = "<p>No more events.</p>";
    return;
  }}

  // Restore previous label if any
  const label = ev.label || "";

  container.innerHTML = `
    <div>Event ${{ev.id + 1}} / {N_EVENTS_TO_PROCESS}</div>
    <img class="event-img" src="${{ev.img}}" alt="Event ${{ev.id}}" />
  `;

  document.getElementById("index").textContent =
    `Event: ${{current + 1}} / {N_EVENTS_TO_PROCESS}`;

  const labeled = events.filter(e => e.label).length;
  document.getElementById("progress").textContent =
  `Labeled: ${{labeled}} / ${{events.length}}`;
}}

function setLabel(label) {{
  events[current].label = label;
  if (current < events.length - 1) {{
    current += 1;
  }} else {{
    alert("All events labeled. You can download CSV now.");
  }}
  renderEvent();
}}

function prevEvent() {{
  if (current > 0) {{
    current -= 1;
    renderEvent();
  }}
}}

function downloadCSV() {{
  let csv = "event_id,label,confidence,fault_start_sample,score\\n";
  events.forEach(ev => {{
    const label = ev.label || "";
    let confidence = label ? 1.0 : "";
    let score = "";
    if (label === "FAULT") {{ score = 1.0; }}
    else if (label === "NORMAL") {{ score = 0.0; }}
    else if (label === "UNCERTAIN") {{ score = 0.5; }}
    csv += `${{ev.id}},"${{label}}",${{confidence}},,${{score}}\\n`;
  }});

  const blob = new Blob([csv], {{type: "text/csv"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labels_first100_manual.csv";
  a.click();
}}

document.addEventListener("keydown", function(e) {{
  if (e.key === "f" || e.key === "F") {{
    setLabel("FAULT");
  }} else if (e.key === "n" || e.key === "N") {{
    setLabel("NORMAL");
  }} else if (e.key === "u" || e.key === "U") {{
    setLabel("UNCERTAIN");
  }} else if (e.key === "b" || e.key === "B") {{
    prevEvent();
  }}
}});

renderEvent();
</script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated: {output_html}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    data = load_data_npz(DATA_PATH)
    n_events = min(N_EVENTS_TO_PROCESS, len(data))

    events_data = []
    for i in range(n_events):
        event = convert_event_units(data[i])
        img_b64 = plot_event_to_png(i, event)
        events_data.append({"id": i, "img": img_b64})

    build_html(events_data, OUTPUT_HTML)
    print(f"Open {OUTPUT_HTML} in your browser to label.")


if __name__ == "__main__":
    main()