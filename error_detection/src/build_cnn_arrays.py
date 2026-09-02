from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd




# File created by merge_binary_metadata.py
COMBINED_METADATA_CSV = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\Final_dataset\combined_metadata_binary.csv"
)

# Folder where the CNN-ready .npy / .npz arrays will be written
OUTPUT_DIR = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\Final_dataset\cnn_arrays"
)



# First CNN: use one cubicle and its six electrical signals.
# The output channel order will be:
# UA, UB, UC, IA, IB, IC
SELECTED_CUBICLE = r"Cub_2\pex_MainBus1_MainLn1-2A"

# Keep 1 initially:
# 4801 time points at 9600 Hz.
#
# If training later is too slow or memory is insufficient, use 2:
# 4801 -> approximately 2401 time points.
DOWNSAMPLE_FACTOR = 1

# Validation values found in your source files.
EXPECTED_TIME_SAMPLES = 4801
EXPECTED_SIGNAL_COLUMNS = 66
EXPECTED_TIME_START_S = 1.0
EXPECTED_TIME_END_S = 1.5

# Expected final waveform shape after channel selection.
N_SELECTED_CHANNELS = 6




def detect_format(first_header_line: str) -> tuple[str, str]:
    """
    Original result<N>.csv:
        separator = comma
        decimal = point

    New steady_state_<N>.csv:
        separator = semicolon
        decimal = comma
    """
    if first_header_line.count(";") > first_header_line.count(","):
        return ";", ","

    return ",", "."


def convert_number(value: str, decimal_symbol: str) -> float:
    """Convert PowerFactory numeric text to a Python float."""
    value = value.strip().strip('"')
    value = value.replace("D", "E").replace("d", "e")

    if decimal_symbol == ",":
        value = value.replace(",", ".")

    return float(value)


def read_powerfactory_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads a PowerFactory export with two header rows.

    Returns:
        waveform:
            DataFrame containing time_s and all numeric signal columns.

        channel_metadata:
            DataFrame describing each signal:
            column, cubicle, measurement, phase, unit.
    """
    lines = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines()

    if len(lines) < 3:
        raise ValueError(
            f"{path.name}: expected two header lines plus data."
        )

    separator, decimal_symbol = detect_format(lines[0])

    object_headers = [
        x.strip().strip('"')
        for x in lines[0].split(separator)
    ]

    quantity_headers = [
        x.strip().strip('"')
        for x in lines[1].split(separator)
    ]

    if len(object_headers) != len(quantity_headers):
        raise ValueError(
            f"{path.name}: header-row mismatch: "
            f"{len(object_headers)} vs {len(quantity_headers)}."
        )

    n_columns = len(object_headers)

    output_columns = ["time_s"]
    channel_rows = []

    for col_idx, (cubicle, signal_description) in enumerate(
        zip(object_headers[1:], quantity_headers[1:]),
        start=1,
    ):
        match = re.fullmatch(
            r"c:(Isec|Usec):([ABC]) in ([AV])",
            signal_description.strip().strip('"'),
        )

        if match is None:
            raise ValueError(
                f"{path.name}: unknown signal description in column "
                f"{col_idx}: {signal_description!r}"
            )

        measurement, phase, unit = match.groups()

        unique_column = (
            f"{cubicle}__{measurement}_{phase}_{unit}"
        )

        output_columns.append(unique_column)

        channel_rows.append(
            {
                "column": unique_column,
                "cubicle": cubicle,
                "measurement": measurement,
                "phase": phase,
                "unit": unit,
            }
        )

    numeric_rows = []

    for line_number, line in enumerate(lines[2:], start=3):
        values = [
            x.strip().strip('"')
            for x in line.split(separator)
        ]

        if len(values) != n_columns:
            raise ValueError(
                f"{path.name}: line {line_number} contains "
                f"{len(values)} columns; expected {n_columns}."
            )

        try:
            numeric_row = [
                convert_number(value, decimal_symbol)
                for value in values
            ]
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: invalid numeric text at line "
                f"{line_number}."
            ) from exc

        numeric_rows.append(numeric_row)

    if not numeric_rows:
        raise ValueError(
            f"{path.name}: no waveform rows found."
        )

    waveform = pd.DataFrame(
        numeric_rows,
        columns=output_columns,
    )

    return waveform, pd.DataFrame(channel_rows)




def validate_waveform(
    waveform: pd.DataFrame,
    channel_metadata: pd.DataFrame,
    path: Path,
) -> dict:
    """Validate time, signal count, and numerical content."""

    n_signal_columns = waveform.shape[1] - 1

    if n_signal_columns != EXPECTED_SIGNAL_COLUMNS:
        raise ValueError(
            f"{path.name}: got {n_signal_columns} signal columns; "
            f"expected {EXPECTED_SIGNAL_COLUMNS}."
        )

    if len(waveform) != EXPECTED_TIME_SAMPLES:
        raise ValueError(
            f"{path.name}: got {len(waveform)} time samples; "
            f"expected {EXPECTED_TIME_SAMPLES}."
        )

    time_s = waveform["time_s"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(time_s)):
        raise ValueError(
            f"{path.name}: time contains NaN or infinity."
        )

    if not np.all(np.diff(time_s) > 0):
        raise ValueError(
            f"{path.name}: time is not strictly increasing."
        )

    if not np.isclose(
        time_s[0],
        EXPECTED_TIME_START_S,
        atol=1e-5,
    ):
        raise ValueError(
            f"{path.name}: time starts at {time_s[0]}, "
            f"expected approximately {EXPECTED_TIME_START_S}."
        )

    if not np.isclose(
        time_s[-1],
        EXPECTED_TIME_END_S,
        atol=1e-5,
    ):
        raise ValueError(
            f"{path.name}: time ends at {time_s[-1]}, "
            f"expected approximately {EXPECTED_TIME_END_S}."
        )

    values = waveform.iloc[:, 1:].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{path.name}: signals contain NaN or infinity."
        )

    cubicle_rows = channel_metadata[
        channel_metadata["cubicle"] == SELECTED_CUBICLE
    ]

    if cubicle_rows.empty:
        available = channel_metadata["cubicle"].drop_duplicates().tolist()

        raise ValueError(
            f"{path.name}: selected cubicle not found:\n"
            f"{SELECTED_CUBICLE}\n\n"
            f"Available cubicles:\n{available}"
        )

    expected_channels = {
        ("Usec", "A"),
        ("Usec", "B"),
        ("Usec", "C"),
        ("Isec", "A"),
        ("Isec", "B"),
        ("Isec", "C"),
    }

    actual_channels = set(
        zip(
            cubicle_rows["measurement"],
            cubicle_rows["phase"],
        )
    )

    missing = expected_channels - actual_channels

    if missing:
        raise ValueError(
            f"{path.name}: selected cubicle is missing "
            f"channels {sorted(missing)}."
        )

    dt = np.diff(time_s)

    return {
        "n_time_samples": int(len(waveform)),
        "n_signal_columns": int(n_signal_columns),
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
        "median_dt_s": float(np.median(dt)),
        "sampling_hz": float(1.0 / np.median(dt)),
        "n_cubicles": int(
            channel_metadata["cubicle"].nunique()
        ),
    }


def extract_six_channels(
    waveform: pd.DataFrame,
    channel_metadata: pd.DataFrame,
) -> np.ndarray:
    """
    Output shape:
        (time_steps, 6)

    Output order:
        UA, UB, UC, IA, IB, IC
    """
    cubicle_rows = channel_metadata[
        channel_metadata["cubicle"] == SELECTED_CUBICLE
    ]

    lookup = {
        (row.measurement, row.phase): row.column
        for row in cubicle_rows.itertuples(index=False)
    }

    channel_order = [
        ("Usec", "A"),
        ("Usec", "B"),
        ("Usec", "C"),
        ("Isec", "A"),
        ("Isec", "B"),
        ("Isec", "C"),
    ]

    selected_columns = [
        lookup[channel]
        for channel in channel_order
    ]

    x = waveform[selected_columns].to_numpy(
        dtype=np.float32
    )

    if DOWNSAMPLE_FACTOR > 1:
        x = x[::DOWNSAMPLE_FACTOR]

    if x.shape[1] != N_SELECTED_CHANNELS:
        raise ValueError(
            f"Expected six channels, got tensor shape {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Selected tensor contains NaN or infinity."
        )

    # Reject only clearly unusable all-constant channels.
    channel_std = x.std(axis=0)

    if np.any(channel_std < 1e-12):
        bad_channels = np.where(channel_std < 1e-12)[0].tolist()

        raise ValueError(
            f"Selected waveform has constant channel(s): "
            f"{bad_channels}."
        )

    return x



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COMBINED_METADATA_CSV.exists():
        raise FileNotFoundError(
            "Combined metadata file not found:\n"
            f"{COMBINED_METADATA_CSV}"
        )

    metadata = pd.read_csv(COMBINED_METADATA_CSV)

    required_columns = {
        "sample_id",
        "source_dataset",
        "raw_filename",
        "raw_file_path",
        "original_class",
        "binary_label",
        "binary_target",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "combined_metadata_binary.csv is missing:\n"
            f"{sorted(missing_columns)}"
        )

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(int)
    metadata["binary_target"] = metadata[
        "binary_target"
    ].astype(int)

    if metadata["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in metadata."
        )

    if not set(metadata["binary_target"]).issubset({0, 1}):
        raise ValueError(
            "binary_target must contain only 0=normal and 1=error."
        )

    metadata = metadata.sort_values(
        "sample_id"
    ).reset_index(drop=True)

    print("=" * 80)
    print("CNN ARRAY BUILD")
    print("=" * 80)
    print(f"Metadata samples: {len(metadata)}")
    print(f"Selected cubicle: {SELECTED_CUBICLE}")
    print(f"Downsample factor: {DOWNSAMPLE_FACTOR}")

    tensors = []
    accepted_rows = []
    excluded_rows = []

    expected_tensor_shape = None
    total = len(metadata)

    for position, (_, row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):
        source_path = Path(row["raw_file_path"])
        sample_id = int(row["sample_id"])

        try:
            if not source_path.exists():
                raise FileNotFoundError(
                    "Raw waveform file does not exist."
                )

            waveform, channel_metadata = read_powerfactory_csv(
                source_path
            )

            qc = validate_waveform(
                waveform=waveform,
                channel_metadata=channel_metadata,
                path=source_path,
            )

            x = extract_six_channels(
                waveform=waveform,
                channel_metadata=channel_metadata,
            )

            if expected_tensor_shape is None:
                expected_tensor_shape = x.shape

            if x.shape != expected_tensor_shape:
                raise ValueError(
                    f"Tensor shape is {x.shape}; expected "
                    f"{expected_tensor_shape}."
                )

            accepted = row.to_dict()

            accepted.update(qc)

            accepted["selected_cubicle"] = SELECTED_CUBICLE
            accepted["cnn_time_steps"] = int(x.shape[0])
            accepted["cnn_channels"] = int(x.shape[1])

            tensors.append(x)
            accepted_rows.append(accepted)

            print(
                f"[{position:5d}/{total}] OK   "
                f"ID={sample_id:<5d} "
                f"class={row['original_class']:<28} "
                f"file={source_path.name}"
            )

        except Exception as exc:
            excluded_rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": row["source_dataset"],
                    "original_class": row["original_class"],
                    "binary_label": row["binary_label"],
                    "raw_filename": row["raw_filename"],
                    "raw_file_path": str(source_path),
                    "reason": str(exc),
                }
            )

            print(
                f"[{position:5d}/{total}] FAIL "
                f"ID={sample_id:<5d} "
                f"class={row['original_class']:<28} "
                f"file={source_path.name} | {exc}"
            )

    if not tensors:
        raise RuntimeError(
            "No waveform passed validation. Nothing was saved."
        )

    X = np.stack(tensors).astype(np.float32)

    accepted_metadata = pd.DataFrame(
        accepted_rows
    ).reset_index(drop=True)

    y = accepted_metadata["binary_target"].to_numpy(
        dtype=np.int64
    )

    sample_ids = accepted_metadata["sample_id"].to_numpy(
        dtype=np.int64
    )

    # Final integrity check: X, y, and metadata must align exactly.
    if len(X) != len(y) or len(X) != len(accepted_metadata):
        raise RuntimeError(
            "Internal alignment error between X, y, and metadata."
        )

    # Save individual arrays.
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y_binary.npy", y)
    np.save(OUTPUT_DIR / "sample_id.npy", sample_ids)

    # Save one compressed combined training dataset.
    np.savez_compressed(
        OUTPUT_DIR / "powerfactory_binary_cnn_dataset.npz",
        X=X,
        y_binary=y,
        sample_id=sample_ids,
    )

    # Save accepted metadata in the same order as X.
    accepted_metadata.to_csv(
        OUTPUT_DIR / "cnn_sample_metadata.csv",
        index=False,
    )

    # Save all rejected waveform files and reasons.
    excluded_df = pd.DataFrame(
        excluded_rows,
        columns=[
            "sample_id",
            "source_dataset",
            "original_class",
            "binary_label",
            "raw_filename",
            "raw_file_path",
            "reason",
        ],
    )

    excluded_df.to_csv(
        OUTPUT_DIR / "cnn_excluded_waveforms.csv",
        index=False,
    )

    # Counts before and after waveform validation.
    before_counts = (
        metadata.groupby(
            ["original_class", "binary_label"]
        )
        .size()
        .reset_index(name="before_qc")
    )

    after_counts = (
        accepted_metadata.groupby(
            ["original_class", "binary_label"]
        )
        .size()
        .reset_index(name="after_qc")
    )

    class_counts = before_counts.merge(
        after_counts,
        on=["original_class", "binary_label"],
        how="outer",
    ).fillna(0)

    class_counts["before_qc"] = class_counts[
        "before_qc"
    ].astype(int)

    class_counts["after_qc"] = class_counts[
        "after_qc"
    ].astype(int)

    class_counts["excluded"] = (
        class_counts["before_qc"]
        - class_counts["after_qc"]
    )

    class_counts = class_counts.sort_values(
        ["binary_label", "original_class"]
    )

    class_counts.to_csv(
        OUTPUT_DIR / "cnn_class_counts_before_after_qc.csv",
        index=False,
    )

    binary_counts = (
        accepted_metadata.groupby(
            ["binary_label", "binary_target"]
        )
        .size()
        .reset_index(name="n_samples")
        .sort_values("binary_target")
    )

    binary_counts.to_csv(
        OUTPUT_DIR / "cnn_binary_class_counts.csv",
        index=False,
    )

    summary = {
        "input_metadata_csv": str(COMBINED_METADATA_CSV),
        "selected_cubicle": SELECTED_CUBICLE,
        "channel_order": [
            "UA_V",
            "UB_V",
            "UC_V",
            "IA_A",
            "IB_A",
            "IC_A",
        ],
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "n_input_metadata_rows": int(len(metadata)),
        "n_accepted_waveforms": int(len(accepted_metadata)),
        "n_excluded_waveforms": int(len(excluded_df)),
        "X_shape": [int(value) for value in X.shape],
        "y_shape": [int(value) for value in y.shape],
        "binary_class_counts": {
            str(row["binary_label"]): int(row["n_samples"])
            for _, row in binary_counts.iterrows()
        },
    }

    with open(
        OUTPUT_DIR / "cnn_dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Accepted waveforms: {len(accepted_metadata)}")
    print(f"Excluded waveforms: {len(excluded_df)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nBinary classes after waveform QC:")
    print(binary_counts.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "X.npy")
    print(OUTPUT_DIR / "y_binary.npy")
    print(OUTPUT_DIR / "sample_id.npy")
    print(OUTPUT_DIR / "powerfactory_binary_cnn_dataset.npz")
    print(OUTPUT_DIR / "cnn_sample_metadata.csv")
    print(OUTPUT_DIR / "cnn_excluded_waveforms.csv")
    print(OUTPUT_DIR / "cnn_class_counts_before_after_qc.csv")
    print(OUTPUT_DIR / "cnn_binary_class_counts.csv")
    print(OUTPUT_DIR / "cnn_dataset_summary.json")


if __name__ == "__main__":
    main()