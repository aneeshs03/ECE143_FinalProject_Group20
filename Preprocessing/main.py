from pathlib import Path
import adult_csv_preprocessing
import cost_of_living_preprocessing
import bls_preprocessing
import monster_jobs_preprocessing
import state_summary_preprocessing
import pums_preprocessing
import irs_preprocessing

DATA_DIR   = Path("./project_datasets")
OUTPUT_DIR = DATA_DIR / "preprocessed_datasets"

def main():
    """
    Execute the full preprocessing pipeline, export CSVs, and return DataFrames.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_adult         = adult_csv_preprocessing.load_adult_census(DATA_DIR / "adult.csv")
    df_cost_state    = cost_of_living_preprocessing.compute_state_cost_of_living(DATA_DIR / "cost_of_living_us.csv")
    df_bls_wages     = bls_preprocessing.load_bls_wages(DATA_DIR)
    df_monster       = monster_jobs_preprocessing.load_monster_jobs(DATA_DIR)
    df_state_summary = state_summary_preprocessing.build_state_summary(df_cost_state, df_bls_wages, df_monster)
    df_pums          = pums_preprocessing.load_acs_pums(DATA_DIR)
    df_migration     = irs_preprocessing.load_irs_migration(DATA_DIR)

    adult_csv_preprocessing.export_adult(df_adult, OUTPUT_DIR)
    cost_of_living_preprocessing.export_cost_state(df_cost_state, OUTPUT_DIR)
    bls_preprocessing.export_bls_wages(df_bls_wages, OUTPUT_DIR)
    monster_jobs_preprocessing.export_monster(df_monster, OUTPUT_DIR)
    state_summary_preprocessing.export_state_summary(df_state_summary, OUTPUT_DIR)
    pums_preprocessing.export_pums(df_pums, OUTPUT_DIR)
    irs_preprocessing.export_migration(df_migration, OUTPUT_DIR)

    return {
        "adult":         df_adult,
        "cost_state":    df_cost_state,
        "bls_wages":     df_bls_wages,
        "monster":       df_monster,
        "state_summary": df_state_summary,
        "pums":          df_pums,
        "migration":     df_migration,
    }


def run():
    dataframes = main()

    df_adult         = dataframes["adult"]
    df_cost_state    = dataframes["cost_state"]
    df_bls_wages     = dataframes["bls_wages"]
    df_monster       = dataframes["monster"]
    df_state_summary = dataframes["state_summary"]
    df_pums          = dataframes["pums"]
    df_migration     = dataframes["migration"]


if __name__ == "__main__":
    run()