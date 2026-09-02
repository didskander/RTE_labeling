from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Original EvEMTBench metadata file
SETTINGS_CLEAN_CSV = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\labels\settings_clean.csv"
)

# Folder containing original files:
ORIGINAL_RESULTS_DIR = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\Simulated_data"
)

# Folder containing new files:
STEADY_STATE_RESULTS_DIR = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\Ld2_2000_batch"
)

# Folder where NEW metadata/label files will be saved
OUTPUT_DIR = Path(
    r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\Final_dataset"
)



# Number selected from the new steady-state simulations.
N_STEADY_STATE_TO_SELECT = 672

# Original event-file naming:
ORIGINAL_FILE_PATTERN = re.compile(
    r"^result(\d+)\.csv$",
    flags=re.IGNORECASE,
)

# New steady-state naming:
STEADY_STATE_FILE_PATTERN = re.compile(
    r"^steady_state_(\d+)\.csv$",
    flags=re.IGNORECASE,
)



BINARY_LABEL_MAP = {
    # ----------------------------- ERROR / FAULT -----------------------------
    "flt_1phg_incipient": "error",
    "flt_1phg_incipient_w_arc": "error",
    "flt_1phg_shc": "error",
    "flt_1phg_shc_w_arc": "error",
    "flt_2ph_shc": "error",
    "flt_2phg_shc": "error",
    "flt_3ph_shc": "error",

    # --------------------------- NORMAL / OPERATION --------------------------
    "switch_cap_off": "normal",
    "switch_cap_on": "normal",
    "switch_inrushHV": "normal",
    "switch_inrushLV": "normal",
    "switch_load_off": "normal",
    "switch_load_on": "normal",
    "switch_ohl_off": "normal",
    "switch_ohl_on": "normal",

    # ------------------------- NEW NORMAL STEADY STATE -----------------------
    "steady_state": "normal",
}

BINARY_TARGET_MAP = {
    "normal": 0,
    "error": 1,
}



def discover_original_result_files(root: Path):
    """
    Finds result<N>.csv files recursively.

    Returns:
        result_index:
            {simulation_index: full_file_path}

        exclusions:
            duplicate-ID files that are not used
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Original results folder not found:\n{root}"
        )

    result_index = {}
    exclusions = []

    for path in root.rglob("*.csv"):
        match = ORIGINAL_FILE_PATTERN.match(path.name)

        if match is None:
            continue

        simulation_index = int(match.group(1))

        if simulation_index in result_index:
            exclusions.append(
                {
                    "reason": "duplicate_original_result_id",
                    "source_dataset": "original",
                    "original_sim_idx": simulation_index,
                    "raw_filename": path.name,
                    "raw_file_path": str(path),
                    "details": (
                        "Another result file with the same numeric ID "
                        f"was already kept: {result_index[simulation_index]}"
                    ),
                }
            )
            continue

        result_index[simulation_index] = path

    return result_index, exclusions


def discover_steady_state_files(root: Path):
    """
    Finds steady_state_<N>.csv files recursively.

    Returns:
        DataFrame sorted by DPL index extracted from filename.

    The DPL index is NOT the final CNN dataset ID.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Steady-state results folder not found:\n{root}"
        )

    records = []
    exclusions = []
    seen_indices = {}

    for path in root.rglob("*.csv"):
        match = STEADY_STATE_FILE_PATTERN.match(path.name)

        if match is None:
            continue

        dpl_run_index = int(match.group(1))

        if dpl_run_index in seen_indices:
            exclusions.append(
                {
                    "reason": "duplicate_steady_state_run_index",
                    "source_dataset": "steady_state",
                    "original_sim_idx": np.nan,
                    "raw_filename": path.name,
                    "raw_file_path": str(path),
                    "details": (
                        "Another steady-state file with the same DPL index "
                        f"was already kept: {seen_indices[dpl_run_index]}"
                    ),
                }
            )
            continue

        seen_indices[dpl_run_index] = path

        records.append(
            {
                "steady_state_run_index": dpl_run_index,
                "raw_filename": path.name,
                "raw_file_path": str(path),
            }
        )

    steady_df = pd.DataFrame(records)

    if steady_df.empty:
        raise RuntimeError(
            "No steady_state_<number>.csv files were found in:\n"
            f"{root}"
        )

    steady_df = steady_df.sort_values(
        "steady_state_run_index"
    ).reset_index(drop=True)

    return steady_df, exclusions


def choose_evenly_spaced_steady_state_samples(
    steady_df: pd.DataFrame,
    n_select: int,
):
    """
    Select exactly n_select samples evenly across the available
    DPL run-index range.

    This avoids taking only low-load or only high-load steady states.
    """
    n_available = len(steady_df)

    if n_available < n_select:
        raise ValueError(
            f"Only {n_available} steady-state files were found, "
            f"but {n_select} are required."
        )

    # Select evenly spaced row positions over the sorted available list.
    positions = np.linspace(
        0,
        n_available - 1,
        num=n_select,
        dtype=int,
    )

    positions = np.unique(positions)

    # Safety check: ensure exactly N unique rows.
    if len(positions) != n_select:
        raise RuntimeError(
            f"Expected {n_select} unique evenly spaced positions, "
            f"but got {len(positions)}."
        )

    selected = steady_df.iloc[positions].copy()
    selected = selected.reset_index(drop=True)

    selected["selection_method"] = (
        "evenly_spaced_across_sorted_available_steady_state_files"
    )

    selected["selection_rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  
    if not SETTINGS_CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"settings_clean.csv not found:\n{SETTINGS_CLEAN_CSV}"
        )

    settings = pd.read_csv(SETTINGS_CLEAN_CSV)

    required_columns = {
        "general/sim_idx",
        "events/event_type",
    }

    missing_columns = required_columns - set(settings.columns)

    if missing_columns:
        raise ValueError(
            "settings_clean.csv does not contain required columns:\n"
            f"{sorted(missing_columns)}"
        )

    settings = settings.copy()

    settings["general/sim_idx"] = settings[
        "general/sim_idx"
    ].astype(int)

    settings["events/event_type"] = settings[
        "events/event_type"
    ].astype(str)

    # Check that every original class has a binary mapping.
    original_classes = set(settings["events/event_type"].unique())
    mapped_classes = set(BINARY_LABEL_MAP.keys())

    unknown_classes = sorted(original_classes - mapped_classes)

    if unknown_classes:
        raise ValueError(
            "These classes from settings_clean.csv are missing "
            "from BINARY_LABEL_MAP:\n"
            f"{unknown_classes}"
        )

    original_result_index, original_exclusions = (
        discover_original_result_files(ORIGINAL_RESULTS_DIR)
    )

    print("=" * 80)
    print("ORIGINAL DATA")
    print("=" * 80)
    print(f"Rows in settings_clean.csv : {len(settings)}")
    print(f"result<N>.csv files found  : {len(original_result_index)}")

    original_records = []
    exclusions = list(original_exclusions)

    for _, row in settings.iterrows():
        sim_idx = int(row["general/sim_idx"])
        original_class = str(row["events/event_type"])

        source_path = original_result_index.get(sim_idx)

        if source_path is None:
            exclusions.append(
                {
                    "reason": "missing_original_result_file",
                    "source_dataset": "original",
                    "original_sim_idx": sim_idx,
                    "raw_filename": "",
                    "raw_file_path": "",
                    "details": (
                        "This simulation exists in settings_clean.csv "
                        "but result<N>.csv was not found."
                    ),
                }
            )
            continue

        binary_label = BINARY_LABEL_MAP[original_class]

        original_records.append(
            {
                # Keep original IDs unchanged.
                "sample_id": sim_idx,

                "source_dataset": "original",

                "raw_filename": source_path.name,
                "raw_file_path": str(source_path),

                "original_sim_idx": sim_idx,
                "steady_state_run_index": np.nan,

                "original_class": original_class,
                "binary_label": binary_label,
                "binary_target": BINARY_TARGET_MAP[binary_label],

                # Useful event metadata, retained if present.
                "event_start_s": row.get("events/event_start", np.nan),
                "event_target": row.get("events/event_target", np.nan),
                "fault_location_percent": row.get(
                    "events/event_flt_target_line_location",
                    np.nan,
                ),
                "short_circuit_resistance_ohm": row.get(
                    "events/event_flt_shc_resistance",
                    np.nan,
                ),
                "high_impedance_resistance_ohm": row.get(
                    "events/event_flt_hif_resistance",
                    np.nan,
                ),
                "incipient_fault_duration_s": row.get(
                    "events/event_iflt_duration",
                    np.nan,
                ),

                "steady_state_selection_rank": np.nan,
                "steady_state_selection_method": "",
            }
        )

    original_df = pd.DataFrame(original_records)

 
    steady_df, steady_exclusions = discover_steady_state_files(
        STEADY_STATE_RESULTS_DIR
    )

    exclusions.extend(steady_exclusions)

    selected_steady_df = choose_evenly_spaced_steady_state_samples(
        steady_df=steady_df,
        n_select=N_STEADY_STATE_TO_SELECT,
    )

    print("\n" + "=" * 80)
    print("STEADY-STATE DATA")
    print("=" * 80)
    print(f"All discovered steady-state files : {len(steady_df)}")
    print(f"Selected steady-state files       : {len(selected_steady_df)}")
    print(
        "Selected DPL run-index range      : "
        f"{selected_steady_df['steady_state_run_index'].min()} "
        f"to {selected_steady_df['steady_state_run_index'].max()}"
    )

    # Save selection before merging.
    selected_steady_df.to_csv(
        OUTPUT_DIR / "selected_steady_state_672.csv",
        index=False,
    )

    steady_records = []

    for offset, (_, row) in enumerate(selected_steady_df.iterrows()):
        new_sample_id = 10000 + offset

        steady_records.append(
            {
                # New internal IDs are 10000 to 10671.
                "sample_id": new_sample_id,

                "source_dataset": "steady_state",

                "raw_filename": row["raw_filename"],
                "raw_file_path": row["raw_file_path"],

                "original_sim_idx": np.nan,
                "steady_state_run_index": int(
                    row["steady_state_run_index"]
                ),

                "original_class": "steady_state",
                "binary_label": "normal",
                "binary_target": 0,

                # No fault/switching event exists in a steady-state record.
                "event_start_s": np.nan,
                "event_target": "Ld2",
                "fault_location_percent": np.nan,
                "short_circuit_resistance_ohm": np.nan,
                "high_impedance_resistance_ohm": np.nan,
                "incipient_fault_duration_s": np.nan,

                "steady_state_selection_rank": int(
                    row["selection_rank"]
                ),
                "steady_state_selection_method": row[
                    "selection_method"
                ],
            }
        )

    steady_metadata_df = pd.DataFrame(steady_records)

 
    combined_metadata = pd.concat(
        [original_df, steady_metadata_df],
        ignore_index=True,
    )

    combined_metadata = combined_metadata.sort_values(
        "sample_id"
    ).reset_index(drop=True)

    # Final integrity checks.
    if combined_metadata["sample_id"].duplicated().any():
        duplicates = combined_metadata.loc[
            combined_metadata["sample_id"].duplicated(keep=False),
            [
                "sample_id",
                "source_dataset",
                "raw_filename",
                "raw_file_path",
            ],
        ]

        raise ValueError(
            "Duplicate final sample IDs found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if not set(combined_metadata["binary_label"]).issubset(
        BINARY_TARGET_MAP
    ):
        raise ValueError(
            "Unexpected binary class in combined metadata."
        )

  
    combined_metadata.to_csv(
        OUTPUT_DIR / "combined_metadata_binary.csv",
        index=False,
    )

    mapping_rows = []

    for original_class, binary_label in sorted(
        BINARY_LABEL_MAP.items()
    ):
        mapping_rows.append(
            {
                "original_class": original_class,
                "binary_label": binary_label,
                "binary_target": BINARY_TARGET_MAP[binary_label],
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)

    mapping_df.to_csv(
        OUTPUT_DIR / "binary_label_mapping.csv",
        index=False,
    )

   
    exclusions_df = pd.DataFrame(
        exclusions,
        columns=[
            "reason",
            "source_dataset",
            "original_sim_idx",
            "raw_filename",
            "raw_file_path",
            "details",
        ],
    )

    exclusions_df.to_csv(
        OUTPUT_DIR / "excluded_samples.csv",
        index=False,
    )

    original_class_counts = (
        combined_metadata
        .groupby(["original_class", "binary_label"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["binary_label", "original_class"])
    )

    original_class_counts.to_csv(
        OUTPUT_DIR / "combined_original_class_counts.csv",
        index=False,
    )

    binary_class_counts = (
        combined_metadata
        .groupby(["binary_label", "binary_target"])
        .size()
        .reset_index(name="n_samples")
        .sort_values("binary_target")
    )

    binary_class_counts.to_csv(
        OUTPUT_DIR / "combined_binary_class_counts.csv",
        index=False,
    )


    summary = {
        "n_original_metadata_rows": int(len(settings)),
        "n_original_result_files_found": int(
            len(original_result_index)
        ),
        "n_original_samples_in_combined_metadata": int(
            len(original_df)
        ),
        "n_steady_state_files_found": int(len(steady_df)),
        "n_steady_state_samples_selected": int(
            len(steady_metadata_df)
        ),
        "n_total_combined_samples": int(len(combined_metadata)),
        "n_excluded_or_duplicate_files": int(
            len(exclusions_df)
        ),
        "steady_state_selection_method": (
            "evenly spaced across sorted available steady-state "
            "filename indices"
        ),
        "selected_steady_state_run_index_min": int(
            selected_steady_df["steady_state_run_index"].min()
        ),
        "selected_steady_state_run_index_max": int(
            selected_steady_df["steady_state_run_index"].max()
        ),
        "binary_class_counts": {
            str(row["binary_label"]): int(row["n_samples"])
            for _, row in binary_class_counts.iterrows()
        },
        "original_class_counts": {
            str(row["original_class"]): int(row["n_samples"])
            for _, row in original_class_counts.iterrows()
        },
    }

    with open(
        OUTPUT_DIR / "combined_dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)


    print("\n" + "=" * 80)
    print("FINAL COMBINED METADATA REPORT")
    print("=" * 80)

    print(
        f"Original samples included       : {len(original_df)}"
    )

    print(
        f"Steady-state samples included   : {len(steady_metadata_df)}"
    )

    print(
        f"Final combined sample count     : {len(combined_metadata)}"
    )

    print(
        f"Excluded/duplicate records log  : {len(exclusions_df)}"
    )

    print("\nBinary class counts:")
    print(binary_class_counts.to_string(index=False))

    print("\nOriginal class counts:")
    print(original_class_counts.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "combined_metadata_binary.csv")
    print(OUTPUT_DIR / "binary_label_mapping.csv")
    print(OUTPUT_DIR / "selected_steady_state_672.csv")
    print(OUTPUT_DIR / "combined_binary_class_counts.csv")
    print(OUTPUT_DIR / "combined_original_class_counts.csv")
    print(OUTPUT_DIR / "excluded_samples.csv")
    print(OUTPUT_DIR / "combined_dataset_summary.json")


if __name__ == "__main__":
    main()