# -*- coding: UTF-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import pandas as pd

# Import tes steps (assume pipeline.py est dans src/ comme les autres)
from extract_step import run_extract
from structure_step import run_structure
from time_step import run_time_standardize
from smiles_step import run_smiles_lookup, smart_split_chem_list


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}")


def _ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def _build_smiles_map(smiles_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """
    Key: (Role, OriginalNameNormalized) -> SMILES or 'Not Found'
    """
    def norm(s: str) -> str:
        return str(s).strip()

    out: dict[tuple[str, str], str] = {}

    # tolerant columns
    required_cols = {"Role", "Original", "SMILES", "Status"}
    missing = required_cols - set(smiles_df.columns)
    if missing:
        raise ValueError(f"smiles_lookup.csv missing columns: {missing}")

    for _, r in smiles_df.iterrows():
        role = norm(r.get("Role", ""))
        original = norm(r.get("Original", ""))
        status = norm(r.get("Status", ""))
        smi = norm(r.get("SMILES", ""))

        if not role or not original:
            continue

        if status.upper() == "OK" and smi:
            out[(role, original)] = smi
        else:
            out[(role, original)] = "Not Found"

    return out


def _smiles_for_cell(cell_value: str, role: str, smiles_map: dict[tuple[str, str], str]) -> str:
    """
    Convertit "Reactants" ou "Products" cell -> "SMILES1, SMILES2, ..."
    - conserve l'ordre des composés
    - si pas trouvé -> Not Found
    """
    names = smart_split_chem_list(cell_value)
    if not names:
        return ""

    smiles_list = []
    for name in names:
        key = (role, str(name).strip())
        smiles_list.append(smiles_map.get(key, "Not Found"))

    return ", ".join(smiles_list)


def build_index_title_map(input_json_path):
    import json
    from pathlib import Path

    with Path(input_json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    index_title_map = {}

    for reaction_key, payload in data.items():
        title = payload.get("Title", "")
        procedures = payload.get("Procedure", [])

        if isinstance(procedures, str):
            procedures = [procedures]

        # Step 1 output index = reaction_key_1
        for proc_idx in range(1, len(procedures) + 1):
            base_index = f"{reaction_key}_{proc_idx}"

            # Step 2 creates rows like base_index_1, base_index_2...
            # On ne sait pas combien il y aura de lignes,
            # donc on va mapper dynamiquement plus tard.
            index_title_map[base_index] = title

    return index_title_map


def merge_final(
    table_csv: Path,
    timetable_csv: Path,
    smiles_lookup_csv: Path,
    output_final_csv: Path, index_title_map
) -> Path:
    _log("Merging final outputs...")

    table_df = pd.read_csv(table_csv)
    time_df = pd.read_csv(timetable_csv)
    smiles_df = pd.read_csv(smiles_lookup_csv)

    # --- Merge time (minutes) ---
    if "Index" not in table_df.columns:
        raise ValueError("table.csv must contain 'Index' column.")
    if "Index" not in time_df.columns or "Reaction time" not in time_df.columns:
        raise ValueError("timetable.csv must contain ['Index', 'Reaction time'].")

    merged = table_df.merge(
        time_df[["Index", "Reaction time"]].rename(columns={"Reaction time": "Reaction time (minutes)"}),
        on="Index",
        how="left",
    )
    # --- Add Title column ---
    merged["Title"] = merged["Index"].apply(
    lambda x: index_title_map.get(
        "_".join(x.split("_")[:2]),  # pbfa_1 from pbfa_1_1
        ""
    )
)

    # --- Build smiles map and add columns ---
    smiles_map = _build_smiles_map(smiles_df)

    # Ensure columns exist in table
    if "Reactants" not in merged.columns:
        merged["Reactants"] = ""
    if "Products" not in merged.columns:
        merged["Products"] = ""

    merged["Reactants_SMILES"] = merged["Reactants"].apply(
        lambda x: _smiles_for_cell(x, "Reactant", smiles_map)
    )
    merged["Products_SMILES"] = merged["Products"].apply(
        lambda x: _smiles_for_cell(x, "Product", smiles_map)
    )

    # Optional: overwrite original Reaction time with standardized minutes if you want
    # Here we KEEP original + add standardized column.
    # If you prefer replacing:
    # merged["Reaction time"] = merged["Reaction time (minutes)"].fillna(merged.get("Reaction time"))
    cols = list(merged.columns)
    if "Title" in cols:
        cols.insert(1, cols.pop(cols.index("Title")))
        merged = merged[cols]
    output_final_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_final_csv, index=False)

    _log(f"Final CSV written: {output_final_csv}")
    return output_final_csv


def main():
    parser = argparse.ArgumentParser(description="Organic reaction extraction pipeline")
    parser.add_argument("input_json", help="Path to input JSON, e.g. data/input_test.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project_root>/outputs)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--extract-sleep", type=float, default=2.0, help="Sleep seconds between extract calls")
    parser.add_argument("--time-delay", type=float, default=2.0, help="Delay before time standardization call")

    args = parser.parse_args()

    # Project root assumed: src/ is alongside data/ and outputs/
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    input_json = Path(args.input_json).resolve()
    _ensure_exists(input_json, "Input JSON")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (project_root / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_json.stem  # input_test
    summary_csv = output_dir / f"{stem}_summary.csv"
    table_csv = output_dir / f"{stem}_table.csv"
    timetable_csv = output_dir / f"{stem}_timetable.csv"
    smiles_csv = output_dir / "smiles_lookup.csv"
    final_csv = output_dir / "final_output.csv"

    _log(f"Input: {input_json}")
    _log(f"Outputs dir: {output_dir}")
    _log(f"Model: {args.model}")

    # ---- Step 1: Extract ----
    _log("Step 1/5: Extract (LLM) -> summary.csv")
    run_extract(
        input_json_path=input_json,
        output_summary_csv_path=summary_csv,
        model=args.model,
        sleep_s=args.extract_sleep,
    )
    _ensure_exists(summary_csv, "Summary CSV")

    # ---- Step 2: Structure ----
    _log("Step 2/5: Structure -> table.csv")
    run_structure(
        input_summary_csv=str(summary_csv),
        output_table_csv=str(table_csv),
    )
    _ensure_exists(table_csv, "Table CSV")

    # ---- Step 3: Time standardize ----
    _log("Step 3/5: Time standardize -> timetable.csv")
    run_time_standardize(
        input_table_csv=str(table_csv),
        output_timetable_csv=str(timetable_csv),
        model=args.model,
        delay=args.time_delay,
    )
    _ensure_exists(timetable_csv, "Timetable CSV")

    # ---- Step 4: SMILES lookup ----
    _log("Step 4/5: SMILES lookup -> smiles_lookup.csv")
    run_smiles_lookup(
        input_table_csv=str(table_csv),
        output_smiles_csv=str(smiles_csv),
        model=args.model,
    )
    _ensure_exists(smiles_csv, "SMILES lookup CSV")
    index_title_map = build_index_title_map(input_json)
    # ---- Step 5: Merge final ----
    _log("Step 5/5: Merge final -> final_output.csv")
    merge_final(
    table_csv=table_csv,
    timetable_csv=timetable_csv,
    smiles_lookup_csv=smiles_csv,
    output_final_csv=final_csv,
    index_title_map=index_title_map
)

    _log("DONE ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[pipeline][ERROR] {e}", file=sys.stderr)
        raise