import numpy as np
import pandas as pd
from pathlib import Path

pums_code_to_state_conv = {
    1: "AL",  2: "AK",  4: "AZ",  5: "AR",  6: "CA",  8: "CO",  9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

def load_irs_migration(data_dir):
    """
    Load and clean the IRS SOI state-to-state migration dataset.
    Filters out IRS aggregate rows (FIPS 96=total, 97=US, 98=foreign, 57=Foreign)
    and same-state rows, then computes average AGI per return as a proxy for
    the income of people who actually relocated.

    Args:
        data_dir: Directory containing stateinflow2122.csv and stateoutflow2122.csv.

    Returns:
        a dataframe with one row per interstate (origin_state → dest_state) pair with:
        n_returns, n_individuals, total_agi ($thousands), avg_agi, flow_direction.
        Returns an empty DataFrame with a message if files are not found.
    """
    inflow_path  = data_dir / "stateinflow2122.csv"
    outflow_path = data_dir / "stateoutflow2122.csv"
    
    if not inflow_path.exists() or not outflow_path.exists():
        raise AssertionError("IRS migration files not found in the specified directory.")

    # codes used by IRS to denote foreign, total, and US aggregate rows
    fips_codes = {57, 96, 97, 98} 

    def _read_inflow(path):
        """
        Read and clean the IRS inflow file, filtering out aggregate rows and same-state flows.
        
        Args:
            path: Path to the stateinflow2122.csv file
        
        Returns:
            a dataframe with cleaned inflow records, including origin_state, dest_state, n_returns,
            n_individuals, total_agi, and flow_direction="inflow".
        """
        df = pd.read_csv(path, encoding="latin-1")
        df.columns = df.columns.str.strip()
        df = df[
            ~df["y1_statefips"].isin(fips_codes) &
            ~df["y2_statefips"].isin(fips_codes) &
            (df["y1_statefips"] != df["y2_statefips"])
        ].copy()
        df["origin_state"] = df["y1_state"]
        df["dest_state"] = df["y2_statefips"].map(pums_code_to_state_conv)
        df["flow_direction"] = "inflow"
        return df.rename(columns={"n1": "n_returns", "n2": "n_individuals", "AGI": "total_agi"})

    def _read_outflow(path):
        """
        Read and clean the IRS outflow file, filtering out aggregate rows and same-state flows.
        
        Args:
            path: Path to the stateoutflow2122.csv file
        
        Returns:
            a dataframe with cleaned outflow records, including origin_state, dest_state, n_returns,
            n_individuals, total_agi, and flow_direction="outflow".
        """
        df = pd.read_csv(path, encoding="latin-1")
        df.columns = df.columns.str.strip()
        df = df[
            ~df["y1_statefips"].isin(fips_codes) &
            ~df["y2_statefips"].isin(fips_codes) &
            (df["y1_statefips"] != df["y2_statefips"])
        ].copy()
        df["origin_state"] = df["y1_statefips"].map(pums_code_to_state_conv)
        df["dest_state"] = df["y2_state"]
        df["flow_direction"] = "outflow"
        return df.rename(columns={"n1": "n_returns", "n2": "n_individuals", "AGI": "total_agi"})

    # Combines the inflow and outflow data
    df = pd.concat([_read_inflow(inflow_path), _read_outflow(outflow_path)], ignore_index=True)

    for col in ("n_returns", "n_individuals", "total_agi"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["n_returns"] > 0].copy()

    # IRS reports AGI in thousands, so multiply by 1000
    df["avg_agi"] = ((df["total_agi"] * 1000) / df["n_returns"]).round(0)

    keep_cols = ["origin_state", "dest_state", "n_returns", "n_individuals", "total_agi", "avg_agi", "flow_direction"]
    return df[keep_cols].dropna(subset=["origin_state", "dest_state"]).reset_index(drop=True)


def export_migration(df, output_dir):
    out_path = output_dir / "cleaned_migration.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_migration.csv — {len(df):,} rows")