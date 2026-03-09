import numpy as np
import pandas as pd
import re

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

def extract_state_from_location(location):
    """
    Extract a US state abbreviation from a Monster.com location string.

    Args:
    location: Raw location string from the Monster dataset.

    Returns
        a string with two-letter state abbreviation, or NaN if not found / not a US state.
    """
    if pd.isna(location):
        return np.nan
    
    match = re.search(r"\b([A-Z]{2})\b", str(location))

    if match and match.group(1) in state_abbreviation_conv.values():
        return match.group(1)
    return np.nan

# helper function to parse the free-text salary strings in the Monster dataset
def parse_salary_string(salary):
    """
    Parse a Monster.com free-text salary string into annual low/high/mid.

    Hourly rates are annualised at 2,080 hours. Values outside the range
    $5,000–$500,000 are treated as implausible and discarded.

    Args:
    salary: Raw salary string from the Monster dataset.

    Returns:
        tuple of (low, high, mid) as floats, or (NaN, NaN, NaN) if unparseable.
    """
    nan_triple = (np.nan, np.nan, np.nan)

    if pd.isna(salary):
        return nan_triple

    # remove commas and extract all numbers (integers or decimals) from the string
    clean = str(salary).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", clean)

    if not numbers:
        return nan_triple

    # extract numbers from the cleaned string and convert to float
    low  = float(numbers[0])
    high = float(numbers[-1]) if len(numbers) > 1 else low
    mid  = (low + high) / 2

    # check if the salary is annual value
    is_annual = False
    if "/year" in clean.lower() or "/yr" in clean.lower() or "/annual" in clean.lower():
        is_annual = True

    # check if the salary is hourly value 
    is_hourly = False
    if "/hour" in clean.lower() or "/hr" in clean.lower():
        is_hourly = True

    # If it is an hourly rate, convert it to yearly
    if is_hourly or (not is_annual and mid < 500):
        low, high, mid = low * 2080, high * 2080, mid * 2080

    # ensures the salaries are reasonable
    if not (5_000 <= mid <= 500_000):
        return nan_triple
    
    return round(low, 0), round(high, 0), round(mid, 0)


def load_monster_jobs(data_dir):
    """
    Load and clean all Monster.com job posting sample files.

    Args:
    data_dir: Directory containing monster_comjob_sample_*.csv files.

    Returns:
        a dataframe with one row per job posting with state and parsed salary columns.
    """
    keep_cols = ["job_title", "job_type", "location", "organization", "salary", "sector", "date_added"]

    df = pd.read_csv(data_dir / "monster_com-job_sample.csv", usecols=keep_cols, on_bad_lines="skip")

    # Use the helper function above to extract state abbreviations
    df["state"] = df["location"].apply(extract_state_from_location)
    df = df[df["state"].isin(state_abbreviation_conv.values())].copy()

    # using the helper function above to create three columns for the salaries: low, high, and mid
    parsed = df["salary"].apply(parse_salary_string)
    df["salary_low"] = [t[0] for t in parsed]
    df["salary_high"] = [t[1] for t in parsed]
    df["salary_mid"] = [t[2] for t in parsed]
    df["salary_available"] = df["salary_mid"].notna()

    return (df.drop(columns=["salary"]).rename(columns={"location": "location_raw"}).reset_index(drop=True))

def export_monster(df, output_dir):
    out_path = output_dir / "cleaned_monster.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_monster.csv — {len(df):,} rows")
