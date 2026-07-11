import argparse
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def moving_mean(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def first_sustained_true(mask, min_run=12):
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if ok else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def detect_suggested_index(t, V1, V2, V3, I1, I2, I3):
    I_mag = moving_mean(np.abs(I1) + np.abs(I2) + np.abs(I3), 25)
    V_rms = np.sqrt(moving_mean(V1**2 + V2**2 + V3**2, 25) / 3.0)
    V_diff = moving_mean(np.abs(np.diff(V_rms, prepend=V_rms[0])), 25)

    baseline_end = max(200, min(1200, len(t) // 12))

    I_base = I_mag[:baseline_end]
    Vd_base = V_diff[:baseline_end]

    I_med = float(np.median(I_base))
    I_mad = float(np.median(np.abs(I_base - I_med)) + 1e-9)

    Vd_med = float(np.median(Vd_base))
    Vd_mad = float(np.median(np.abs(Vd_base - Vd_med)) + 1e-9)

    I_thr = I_med + max(8.0 * I_mad, 5.0)
    Vd_thr = Vd_med + max(10.0 * Vd_mad, np.max(Vd_base) * 3.0)

    idx_i = first_sustained_true(I_mag > I_thr, min_run=12)
    idx_v = first_sustained_true(V_diff > Vd_thr, min_run=12)

    candidates = [i for i in [idx_i, idx_v] if i is not None and i > 10]

    if candidates:
        idx = min(candidates)
        left = max(0, idx - 40)
        local_score = (
            I_mag[left:idx + 1] / (I_thr + 1e-9)
            + V_diff[left:idx + 1] / (Vd_thr + 1e-9)
        )
        hits = np.where(local_score > 1.0)[0]
        if len(hits):
            return left + int(hits[0])
        return int(idx)

    return int(np.argmax(np.abs(I1 + I2 + I3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--initial_label", type=str, default="")
    parser.add_argument("--fault_type", type=str, default="")
    parser.add_argument("--phase", type=str, default="")
    parser.add_argument("--label_comment", type=str, default="")
    parser.add_argument("--resume_url", type=str, required=True)
    parser.add_argument(
        "--base_path",
        type=str,
        default=r"C:\Users\didsk\Desktop\Relay-protection\src\data\rte_events"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=r"C:\Users\didsk\Desktop\Relay-protection\output\plots"
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    initial_label = args.initial_label
    fault_type = args.fault_type
    phase = args.phase
    label_comment = args.label_comment
    resume_url = args.resume_url

    base = Path(args.base_path)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data_s = np.load(base / "DATA_S.npz")["DATA_S"]
    event = data_s[sample_id]

    t = np.linspace(0, (21000 - 1) / 6400, 21000)

    V1 = event[0] * 18.310
    V2 = event[1] * 18.310
    V3 = event[2] * 18.310

    I1 = event[3] * 4.314
    I2 = event[4] * 4.314
    I3 = event[5] * 4.314

    I0_sum = I1 + I2 + I3
    zero_sequence_value_A = float(np.max(np.abs(I0_sum)))
    zero_sequence_ratio = float(
        np.mean(np.abs(I0_sum)) /
        (np.mean(np.abs(I1) + np.abs(I2) + np.abs(I3)) + 1e-9)
    )

    # TEMP suggestion logic:
    # If you still have your old exact formula, we can drop it back in here.
    suggested_index = detect_suggested_index(t, V1, V2, V3, I1, I2, I3)
    suggested_time_s = float(t[suggested_index])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Voltages: V1, V2, V3",
            "Currents: I1, I2, I3",
            "Zero-sequence current: I1 + I2 + I3",
        ),
    )

    fig.add_trace(go.Scatter(x=t, y=V1, mode="lines", name="V1", line=dict(color="royalblue", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=V2, mode="lines", name="V2", line=dict(color="orangered", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=V3, mode="lines", name="V3", line=dict(color="mediumseagreen", width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=I1, mode="lines", name="I1", line=dict(color="royalblue", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=I2, mode="lines", name="I2", line=dict(color="orangered", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=I3, mode="lines", name="I3", line=dict(color="mediumseagreen", width=2, dash="dot")), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=I0_sum, mode="lines", name="zero_sequence", line=dict(color="deeppink", width=2)), row=3, col=1)

    # Suggested line = red dashed, not draggable
    for r in [1, 2, 3]:
        fig.add_vline(
            x=suggested_time_s,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=r,
            col=1,
        )

    # Review line = orange dashed, draggable
    for r in [1, 2, 3]:
        fig.add_shape(
            type="line",
            x0=suggested_time_s,
            x1=suggested_time_s,
            y0=0,
            y1=1,
            xref=f"x{'' if r == 1 else r}",
            yref="paper",
            line=dict(color="orange", width=3, dash="dash"),
            editable=True,
        )

    fig.update_layout(
        height=820,
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=50, r=25, t=60, b=45),
        legend=dict(orientation="v"),
        title=f"Sample {sample_id}",
    )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=1)
    fig.update_yaxes(title_text="Zero-sequence (A)", row=3, col=1)

    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "scrollZoom": True,
            "displayModeBar": True,
            "editable": True,
        },
        div_id="eventPlot"
    )

    plot_file = plot_dir / f"sample_{sample_id}.html"

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sample {sample_id}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 10px 14px;
      color: #24364b;
      background: #fff;
    }}
    .meta {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .submeta {{
      font-size: 13px;
      margin-bottom: 6px;
      line-height: 1.35;
    }}
    .review {{
      margin: 8px 0 10px 0;
      padding: 8px 10px;
      border: 1px solid #d8dee4;
      border-radius: 6px;
      background: #fafbfc;
      font-size: 13px;
    }}
    .review h3 {{
      margin: 0 0 6px 0;
      font-size: 15px;
    }}
    .review textarea {{
      width: 100%;
      max-width: 420px;
      min-height: 38px;
      font-size: 13px;
    }}
    .btn-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 6px;
    }}
    button {{
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    .hint {{
      color: #555;
      margin-top: 3px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="meta">
    Sample {sample_id} | initial_label={escape(initial_label)} | fault_type={escape(fault_type)} | phase={escape(phase)}
  </div>

  <div class="submeta">
    suggested_index = {suggested_index}
    &nbsp;&nbsp;&nbsp; suggested_time = <span id="shownSuggested">{suggested_time_s:.6f}</span> s
    &nbsp;&nbsp;&nbsp; zero_sequence_value = {zero_sequence_value_A:.2f} A
    &nbsp;&nbsp;&nbsp; zero_sequence_ratio = {zero_sequence_ratio:.6f} (-)
  </div>

  <div class="submeta">
    comment = {escape(label_comment)}
  </div>

  <div class="review">
    <h3>Manual review</h3>

    <form method="POST" action="{escape(resume_url, quote=True)}">
        <input type="hidden" name="sample_id" value="{sample_id}">
        <input type="hidden" name="shown_time_s" value="{suggested_time_s}">
        <input type="hidden" name="zero_sequence_value_A" value="{zero_sequence_value_A}">
        <input type="hidden" name="zero_sequence_ratio" value="{zero_sequence_ratio}">
        <input type="hidden" name="plot_file" value="{escape(str(plot_file), quote=True)}">

        <input type="hidden" name="initial_label" value="{escape(initial_label or '', quote=True)}">
        <input type="hidden" name="fault_type" value="{escape(fault_type or '', quote=True)}">
        <input type="hidden" name="phase" value="{escape(phase or '', quote=True)}">
        <input type="hidden" name="label_comment" value="{escape(label_comment or '', quote=True)}">

      <div><b>Selected review time:</b> <span id="selectedTime">{suggested_time_s:.6f}</span> s</div>
      <div class="hint">If the suggested time is wrong, drag the ORANGE dashed line to the correct position, then click "Wrong -> Save corrected time".</div>
      <div class="hint">If the suggested time is correct, do not move anything and click "Correct -> Next event".</div>

      <div style="margin-top:6px;">
        <label><b>Review note</b></label><br>
        <textarea name="review_note" rows="2">{escape(label_comment)}</textarea>
      </div>

      <div class="btn-row">
        <button type="submit" name="time_correct" value="true">Correct -> Next event</button>
        <button type="submit" name="time_correct" value="false">Wrong -> Save corrected time</button>
      </div>
    </form>
  </div>

  {plot_div}

  <script>
    (function() {{
      const gd = document.getElementById('eventPlot');
      const correctedInput = document.getElementById('corrected_time_s');
      const selectedTimeEl = document.getElementById('selectedTime');

      function setSelectedTime(x) {{
        const val = Number(x);
        if (!Number.isFinite(val)) return;
        correctedInput.value = val.toFixed(6);
        selectedTimeEl.textContent = val.toFixed(6);
      }}

      function readReviewLine() {{
        const shapes = (gd.layout && gd.layout.shapes) ? gd.layout.shapes : [];
        if (shapes.length >= 6) {{
          return shapes[3].x0;
        }}
        return null;
      }}

      setTimeout(function() {{
        const x = readReviewLine();
        if (x !== null) setSelectedTime(x);
      }}, 400);

      gd.on('plotly_relayout', function() {{
        const x = readReviewLine();
        if (x !== null) setSelectedTime(x);
      }});
    }})();
  </script>
</body>
</html>
"""

    plot_file.write_text(html_text, encoding="utf-8")
    print(str(plot_file))


if __name__ == "__main__":
    main()