import numpy as np
import pandas as pd
import re
import csv

# map full state names to abbreviations
state_abbreviation_conv = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}


def extract_state_from_area(area_text):
    """
    Extract a two-letter state abbreviation from a BLS area_text string.

    Args:
        area_text: The human-readable area name from the BLS area lookup table.

    Returns:
        a string with two-letter abbreviation, "US" for national, or NaN if unresolved.
    """
    if pd.isna(area_text) or area_text == "National":
        return "US"
    match = re.search(r",\s*([A-Z]{2})(?:-[A-Z]{2})?$", area_text)

    if match:
        return match.group(1)
    
    resolved = next(
        (abbr for name, abbr in state_abbreviation_conv.items()
         if area_text.startswith(name)),
        np.nan,
    )
    return resolved

# Helper function to extract the state abbreviation from the BLS data
def parse_area_file(path):
    """
    Parse the BLS area lookup file, handling embedded commas in area names. 
    The wm.area.csv file contains commas inside area_text values
    (e.g., "Abilene, TX"), which breaks standard CSV parsers. This function
    reconstructs the correct fields by anchoring to the fixed trailing columns.

    Args:
        path: Path to wm.area.csv.

    Returns:
        a DataFrame with columns: area_code (int), area_text (str), display_level,
        selectable, sort_sequence.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if len(line) < 5:
                continue
            area_code = line[0]
            area_text = ",".join(line[1:-3]).strip()
            rows.append([area_code, area_text] + line[-3:])

    df = pd.DataFrame(rows, columns=["area_code", "area_text", "display_level", "selectable", "sort_sequence"])
    df["area_code"] = pd.to_numeric(df["area_code"], errors="coerce")
    return df.dropna(subset=["area_code"]).astype({"area_code": int})

# Loads and combines the different dataset of the BLS modeled wage series
def load_bls_wages(data_dir):
    """
    Load and merge BLS modeled wage series with area and occupation lookups.
    Reads series metadata from wm.series.csv, joins
    to actual wage values from wm.data.1.AllData.csv, and enriches with
    area names and occupation labels. Aggregates across work levels and
    filters to civilian all-workers subcell.

    Args:
        data_dir: Directory containing all wm.*.csv files.

    Returns:
        a dataframe with one row per (area, occupation) pair with mean/median hourly wage
        and estimated annual salary.
    """
    area_df = parse_area_file(data_dir / "wm.area.csv")
    area_df["state"] = area_df["area_text"].apply(extract_state_from_area)

    occ_df = pd.read_csv(data_dir / "wm.occupation.csv", on_bad_lines="skip", usecols=["occupation_code", "occupation_text"])

    sub_df = pd.read_csv(data_dir / "wm.subcell.csv", usecols=["subcell_code", "subcell_text"])
    sub_df["subcell_code"] = pd.to_numeric(sub_df["subcell_code"], errors="coerce")

    wm_data = pd.read_csv(data_dir / "wm.data.1.AllData.csv", on_bad_lines="skip", usecols=["series_id", "year", "value"])

    # parse metadata directly from the series_id string
    # series_id format: WMU AAAAAAA O EE IIIIII CCCCCC SS LL
    # positions:            3-10    10 12 16     22     28 30
    wm_data["area_code"]       = pd.to_numeric(wm_data["series_id"].str[3:10],  errors="coerce")
    wm_data["occupation_code"] = pd.to_numeric(wm_data["series_id"].str[16:22], errors="coerce")
    wm_data["subcell_code"]    = pd.to_numeric(wm_data["series_id"].str[22:24], errors="coerce")
    wm_data["level_code"]      = pd.to_numeric(wm_data["series_id"].str[24:26], errors="coerce")

    wm_data = wm_data.sort_values("year").drop_duplicates("series_id", keep="last")

    # merge area, occupation and subcell lookups
    bls = wm_data.merge(area_df[["area_code", "area_text", "state"]], on="area_code", how="left")
    bls = bls.merge(occ_df, on="occupation_code", how="left")
    bls = bls.merge(sub_df, on="subcell_code", how="left")
    bls = bls.rename(columns={"value": "hourly_wage_mean"})
 
    # subcell 0 = all workers
    filtered = bls[bls["subcell_code"] == 0].copy()

    group_cols = ["area_code", "area_text", "state", "occupation_code", "year"]

    df = filtered.groupby(group_cols, as_index=False).agg(
        hourly_wage_mean=("hourly_wage_mean", "mean"),
        hourly_wage_median=("hourly_wage_mean", "median"),
        n_levels=("level_code", "nunique"),
    )

    # add occupation_text back in after groupby
    df = df.merge(occ_df, on="occupation_code", how="left")

    df["annual_wage_est"] = (df["hourly_wage_mean"] * 2080).round(0)
    return df.reset_index(drop=True)

def export_bls_wages(df, output_dir):
    out_path = output_dir / "cleaned_bls_wages.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_bls_wages.csv — {len(df):,} rows")