"""
Script to generate paired location samples from the India Post pincode directory.
Filters Head Offices (H.O / metro-urban) as controls and Branch Offices (B.O / rural)
as counterfactuals, sampled across distinct states, and saves pairs to data/location_samples.json.
"""
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def resolve_csv_path(filepath: str) -> str:
    """
    Resolve the pincode CSV path, handling potential Windows double extension (.csv.csv).
    """
    if os.path.exists(filepath):
        return filepath
    if os.path.exists(f"{filepath}.csv"):
        return f"{filepath}.csv"
    alt_path = os.path.join("data", "raw", "pincode_directory.csv")
    if os.path.exists(alt_path):
        return alt_path
    if os.path.exists(f"{alt_path}.csv"):
        return f"{alt_path}.csv"
    raise FileNotFoundError(f"Pincode directory CSV not found at: {filepath}")


def load_pincode_directory(csv_path: str = "data/raw/pincode_directory.csv") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the India Post pincode directory with robust encoding and column resolution.
    """
    resolved_path = resolve_csv_path(csv_path)
    encodings_to_try = ["latin1", "cp1252", "utf-8", "iso-8859-1"]

    df = None
    used_encoding = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(resolved_path, encoding=enc)
            used_encoding = enc
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not read {resolved_path} with supported encodings {encodings_to_try}")

    print(f"Loaded {len(df):,} rows from {resolved_path} (encoding: {used_encoding}).")

    # Map normalized column names (lowercase, no spaces/underscores) to actual DataFrame column names
    col_lookup = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}

    required_keys = {
        "officename": col_lookup.get("officename"),
        "pincode": col_lookup.get("pincode"),
        "officetype": col_lookup.get("officetype"),
        "districtname": col_lookup.get("districtname") or col_lookup.get("district"),
        "statename": col_lookup.get("statename"),
    }

    missing = [k for k, v in required_keys.items() if v is None]
    if missing:
        raise KeyError(f"Could not find required columns {missing} in CSV. Available columns: {list(df.columns)}")

    return df, required_keys


def filter_office_type(df: pd.DataFrame, type_col: str, target_type: str) -> pd.DataFrame:
    """
    Filter DataFrame by officeType, handling both dotted ('H.O', 'B.O') and undotted ('HO', 'BO') notations.
    """
    normalized_target = target_type.replace(".", "").strip().upper()
    col_series = df[type_col].astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
    return df[col_series == normalized_target]


def sample_office_rows(
    df_filtered: pd.DataFrame,
    state_col: str,
    states: List[str],
    col_mapping: Dict[str, str],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample exactly one row for each specified state, returning standardized location dictionaries.
    """
    rng = random.Random(seed)
    sampled_records = []

    for i, state in enumerate(states):
        state_df = df_filtered[df_filtered[state_col] == state]
        if state_df.empty:
            raise ValueError(f"No matching records found for state: {state}")

        sample_idx = rng.choice(state_df.index.tolist())
        row = state_df.loc[sample_idx]

        # Extract standardized fields
        pincode_val = row[col_mapping["pincode"]]
        try:
            pincode_clean = int(pincode_val)
        except (ValueError, TypeError):
            pincode_clean = str(pincode_val).strip()

        sampled_records.append({
            "pincode": pincode_clean,
            "officename": str(row[col_mapping["officename"]]).strip(),
            "districtname": str(row[col_mapping["districtname"]]).strip(),
            "statename": str(row[col_mapping["statename"]]).strip(),
        })

    return sampled_records


def build_location_pairs(
    csv_path: str = "data/raw/pincode_directory.csv",
    output_path: str = "data/location_samples.json",
    n_pairs: int = 10,
    same_state: bool = True,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """
    Build paired control (H.O - Head Office) and counterfactual (B.O - Branch Office) samples.

    :param csv_path: Path to pincode directory CSV.
    :param output_path: Path to output JSON file.
    :param n_pairs: Number of pairs to generate (default 10).
    :param same_state: If True, each pair shares the same state (holding state macroeconomics constant).
                       If False, H.O and B.O states are sampled independently.
    :param seed: Random seed for reproducibility.
    :return: List of paired location dictionaries.
    """
    df, col_mapping = load_pincode_directory(csv_path)

    type_col = col_mapping["officetype"]
    state_col = col_mapping["statename"]

    # Filter H.O (Head Office / metro-urban) and B.O (Branch Office / rural)
    ho_df = filter_office_type(df, type_col, "H.O")
    bo_df = filter_office_type(df, type_col, "B.O")

    print(f"Found {len(ho_df):,} H.O records and {len(bo_df):,} B.O records.")

    rng = random.Random(seed)

    if same_state:
        # Identify common states that have both H.O and B.O
        common_states = sorted(list(set(ho_df[state_col].unique()) & set(bo_df[state_col].unique())))
        if len(common_states) < n_pairs:
            raise ValueError(f"Only {len(common_states)} common states available, cannot sample {n_pairs} distinct states.")
        ho_states = rng.sample(common_states, n_pairs)
        bo_states = list(ho_states)
    else:
        # Independently sample distinct states for H.O and B.O
        all_ho_states = sorted(list(ho_df[state_col].unique()))
        all_bo_states = sorted(list(bo_df[state_col].unique()))
        if len(all_ho_states) < n_pairs or len(all_bo_states) < n_pairs:
            raise ValueError(f"Insufficient distinct states for requested {n_pairs} pairs.")
        ho_states = rng.sample(all_ho_states, n_pairs)
        bo_states = rng.sample(all_bo_states, n_pairs)

    # Sample rows for each state
    ho_rows = sample_office_rows(ho_df, state_col, ho_states, col_mapping, seed=(seed + 1 if seed else None))
    bo_rows = sample_office_rows(bo_df, state_col, bo_states, col_mapping, seed=(seed + 2 if seed else None))

    # Pair them up sequentially
    pairs: List[Dict[str, Any]] = []
    for idx in range(n_pairs):
        ho_sample = ho_rows[idx]
        bo_sample = bo_rows[idx]

        pair_dict = {
            "pair_id": f"location_pair_{idx + 1:02d}",
            "control": {
                "pincode": ho_sample["pincode"],
                "officename": ho_sample["officename"],
                "districtname": ho_sample["districtname"],
                "statename": ho_sample["statename"],
            },
            "counterfactual": {
                "pincode": bo_sample["pincode"],
                "officename": bo_sample["officename"],
                "districtname": bo_sample["districtname"],
                "statename": bo_sample["statename"],
            },
        }
        pairs.append(pair_dict)

    # Save to JSON
    if output_path:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(pairs)} location pairs to: {output_path}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate location control/counterfactual pairs from India Post pincode directory.")
    parser.add_argument("--csv", default="data/raw/pincode_directory.csv", help="Path to input pincode CSV.")
    parser.add_argument("--output", default="data/location_samples.json", help="Path to output JSON file.")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of pairs to sample across distinct states.")
    parser.add_argument("--independent-states", action="store_true", help="Sample H.O and B.O states independently.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")

    args = parser.parse_args()

    pairs = build_location_pairs(
        csv_path=args.csv,
        output_path=args.output,
        n_pairs=args.n_pairs,
        same_state=not args.independent_states,
        seed=args.seed,
    )

    print("\n" + "=" * 90)
    print(f"                      GENERATED {len(pairs)} LOCATION PAIRS")
    print("=" * 90)
    print(f"{'Pair ID':<18} | {'Control (H.O - Metro/Urban)':<34} | {'Counterfactual (B.O - Rural)':<34}")
    print("-" * 90)

    for p in pairs:
        ctrl = p["control"]
        cf = p["counterfactual"]
        ctrl_str = f"{ctrl['officename']} ({ctrl['pincode']}, {ctrl['statename']})"
        cf_str = f"{cf['officename']} ({cf['pincode']}, {cf['statename']})"
        print(f"{p['pair_id']:<18} | {ctrl_str[:34]:<34} | {cf_str[:34]:<34}")

    print("=" * 90)


if __name__ == "__main__":
    main()
