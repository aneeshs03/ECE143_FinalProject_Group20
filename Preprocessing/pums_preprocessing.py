import numpy as np
import pandas as pd
from pathlib import Path    

# PUMS uses numeric FIPS codes for states, so we need this to convert them
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

# PUMS encodes education as numbers 1-24, map to readable tiers
pums_education_level_conv = {
    1: "Less than HS", 2: "Less than HS", 3: "Less than HS", 4: "Less than HS",
    5: "Less than HS", 6: "Less than HS", 7: "Less than HS", 8: "Less than HS",
    9: "Less than HS", 10: "Less than HS", 11: "Less than HS", 12: "Less than HS",
    13: "Less than HS", 14: "Less than HS", 15: "Less than HS", 
    16: "High School", 17: "High School", 18: "Some College", 19: "Some College",
    20: "Associate", 21: "Bachelors", 22: "Masters", 23: "Professional",
    24: "Doctorate",
}

def read_pums_chunked(path, cols, chunksize=50000):
    """ 
    Read the necessary PUMS data in chunks due to large file size and memory capacity.

    Args:
        path: Path to the PUMS CSV file
        cols: List of columns to read from the PUMS file
        chunksize: Number of rows per chunk to read

    Returns:
        a concatenated DataFrame of all chunks that meet the working-age adult criteria
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=cols, low_memory=False, chunksize=chunksize):
        chunk = chunk[chunk["AGEP"].between(22, 65) & chunk["WAGP"].notna() & (chunk["WAGP"] > 0)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def load_acs_pums(data_dir):
    """
    Load and clean the ACS Public Use Microdata Sample (PUMS).
    Keeps working-age adults (22–65) with a positive wage, maps FIPS codes to
    state abbreviations, translates SCHL codes to the same 7-tier education
    system used by df_adult, and flags interstate movers via the MIG column.

    Args:
    data_dir: Directory containing psam_pus*.csv files (any naming variant).

    Returns
        a dataframe with one row per working-age adult with state, education level, income,
        occupation, hours worked, sex, migration status, and age.
        Returns an empty DataFrame with a message if no files are found.
    """
    # Convert the PUMS codes to the desired column names in the final DataFrame
    puma_cols = {
        "ST": "fips",
        "AGEP": "age",
        "SCHL": "schl_code",
        "WAGP": "annual_income",
        "PINCP": "total_income",
        "OCCP": "occupation_code",
        "WKHP": "hours_per_week",
        "SEX": "sex_code",
        "PWGTP": "person_weight",
        "FOD1P": "field_of_degree",
        "MIG": "migration_code", 
        "MIGSP": "prev_state_fips",
    }

    # two PUMS files to go through
    paths = [data_dir / "psam_pusa.csv", data_dir / "psam_pusb.csv"]

    if not paths:
        raise AssertionError("PUMS files not found in the specified directory.")

    # Use the helper function to read file in chunks to the raw dataframe and rename the columns
    df_raw = pd.concat([read_pums_chunked(p, list(puma_cols)) for p in paths], ignore_index=True).rename(columns=puma_cols)

    # Filter to working age adult with a positive/valid wages
    df = df_raw[df_raw["age"].between(22, 65) & df_raw["annual_income"].notna() & (df_raw["annual_income"] > 0)].copy()

    
    df["state"] = df["fips"].map(pums_code_to_state_conv)
    df["education_level"] = df["schl_code"].map(pums_education_level_conv)
    df["is_graduate"] = df["education_level"].isin(["Bachelors", "Masters", "Professional", "Doctorate"])
    df["sex"] = df["sex_code"].map({1: "Male", 2: "Female"})
    df["is_recent_grad"] = df["is_graduate"] & df["age"].between(22, 32)
    df["is_interstate_mover"] = df["migration_code"] == 3
    df["prev_state"] = df["prev_state_fips"].map(pums_code_to_state_conv)

    return (df.drop(columns=["fips", "schl_code", "sex_code", "migration_code", "prev_state_fips"])
        .dropna(subset=["state", "education_level"]).reset_index(drop=True))

def export_pums(df, output_dir):
    out_path = output_dir / "cleaned_pums.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_pums.csv — {len(df):,} rows")