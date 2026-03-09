import numpy as np
import pandas as pd 

def build_state_summary(df_cost, df_bls, df_monster):
    """
    Join cost-of-living, BLS wages, and Monster salaries at the state level.

    Args:
    df_cost: Output of compute_state_cost_of_living().
    df_bls: Output of load_bls_wages().
    df_monster:Output of load_monster_jobs().

    Returns
        a dataframe with one row per state with wages, costs, real-wage gaps, and flags.
    """
    # filter out national and missing state rows
    bls_filtered = df_bls[df_bls["state"].notna() & (df_bls["state"] != "US")]

    # combine bls data to state level
    bls_state = bls_filtered.groupby("state", as_index=False).agg(
        avg_hourly_wage=("hourly_wage_mean", "mean"),
        median_hourly_wage=("hourly_wage_mean", "median"),
        n_occupations=("occupation_code", "nunique"),
        n_metro_areas=("area_code", "nunique"),
    )

    # convert hourly to annual
    bls_state["avg_annual_wage"] = (bls_state["avg_hourly_wage"] * 2080).round(0)

    # filter out monster rows with no salary
    monster_filtered = df_monster[df_monster["salary_mid"].notna()]

    # combine monster jobs data to state level
    monster_state = monster_filtered.groupby("state", as_index=False).agg(
        monster_median_salary=("salary_mid", "median"),
        monster_job_count=("salary_mid", "count"),
    )

    df = df_cost.merge(bls_state, on="state", how="left").merge(monster_state, on="state", how="left")

    # real wage gap = how much money is left after paying for cost of living
    df["real_wage_gap_bls"] = df["avg_annual_wage"] - df["annual_cost_of_living"]
    df["real_wage_gap_monster"] = df["monster_median_salary"] - df["annual_cost_of_living"]
    df["is_high_cost_state"] = df["col_index"] > df["col_index"].median()

    return df

def export_state_summary(df, output_dir):
    out_path = output_dir / "cleaned_state_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_state_summary.csv — {len(df):,} rows")