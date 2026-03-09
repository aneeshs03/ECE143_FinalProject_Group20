import numpy as np
import pandas as pd

def compute_state_cost_of_living(path):
    """
    Convert the county-level cost-of-living data to the state level. 
    Uses the 1-adult, 0-children (1p0c) family type as a recent-graduate
    baseline, it computes a cost-of-living index relative to the national
    median and adds a metro/non-metro premium column per state.

    Args:
        path: Path to cost_of_living_us.csv

    Returns:
        a dataframe with one row per state (51 rows including DC) with cost breakdowns,
        a col_index, and metro_premium.
    """
    cost_cols = [
        "housing_cost", "food_cost", "transportation_cost", "healthcare_cost",
        "other_necessities_cost", "childcare_cost", "taxes", "total_cost",
    ]

    df_raw = pd.read_csv(path)

    # filter typical graduates are 1 person with 0 children
    typical_grad = df_raw[df_raw["family_member_count"] == "1p0c"].copy()

    df = typical_grad.groupby("state")[cost_cols].mean()

    df["median_family_income"] = typical_grad.groupby("state")["median_family_income"].first()
    df = df.reset_index()
    
    df = df.rename(columns={"total_cost": "annual_cost_of_living"})

    # col_index > 1 means more expensive than average
    national_median = df["annual_cost_of_living"].median()
    df["col_index"] = (df["annual_cost_of_living"] / national_median).round(4)

    # compute metro and nonmetro averages per state and add directly
    metro = typical_grad[typical_grad["isMetro"] == True].groupby("state")["total_cost"].mean()
    nonmetro = typical_grad[typical_grad["isMetro"] == False].groupby("state")["total_cost"].mean()

    df["metro_avg_cost"] = df["state"].map(metro)
    df["nonmetro_avg_cost"] = df["state"].map(nonmetro)
    df["metro_premium"] = df["metro_avg_cost"] - df["nonmetro_avg_cost"]

    return df

def export_cost_state(df, output_dir):
    out_path = output_dir / "cleaned_cost_state.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_cost_state.csv — {len(df):,} rows")