import numpy as np
import pandas as pd

# Load and Clean the Adult Census Income dataset
def load_adult_census(path):
    """
    Load and clean the Adult Census Income dataset. Filtering to US-born respondents, maps education to a clean 7-tier system,
    adds a graduate flag and a binary income target, and renames columns to
    snake_case for consistency.

    Args:
        path: Path to the adult.csv file

    Returns:
        a dataframe with the cleaned individual-level census records
    
    Raises:
        AssertionError: if the file path does not exist
    """
    if not path.exists():
        raise AssertionError(f"File not found: {path}")

    df = pd.read_csv(path)

    # get all string columns
    str_cols = df.select_dtypes("object").columns

    # strip whitespace from each string column
    for col in str_cols:
        df[col] = df[col].str.strip()

    # replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # rename all columns since they had "." in them
    df = df.rename(columns={"education.num": "education_num"})
    df = df.rename(columns={"marital.status": "marital_status"})
    df = df.rename(columns={"capital.gain": "capital_gain"})
    df = df.rename(columns={"capital.loss": "capital_loss"})
    df = df.rename(columns={"hours.per.week": "hours_per_week"})
    df = df.rename(columns={"native.country": "native_country"})

    # create two new boolean columns: is_graduate and income_over50k
    df["is_graduate"] = df["education"].isin(["Bachelors", "Masters", "Prof-school", "Doctorate"])
    df["income_over50k"] = (df["income"] == ">50K")
    
    return (df.drop(columns=["fnlwgt"]).reset_index(drop=True))

def export_adult(df, output_dir):
    out_path = output_dir / "cleaned_adult.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned_adult.csv — {len(df):,} rows")