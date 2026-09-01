import pandas as pd
from copy import deepcopy
import yaml
from importlib import resources
from typing import Dict, Union, List
import numpy as np


def convert_date_columns_to_datetime(
    df: pd.DataFrame,
):
    """
    Convert any date columns from the dataframe to pandas datetime

    __ PARAMS __
    :df: pd.DataFrame
        The input dataframe
    """

    # Copy
    tmp = deepcopy(df)

    # Convert columns with dbdate type to datetime columns
    for col in tmp.columns:
        if str(tmp[col].dtype) == "dbdate":
            tmp[col] = pd.to_datetime(tmp[col], errors="coerce")

    return tmp


def load_config(file_name: str) -> Dict:
    """Load a YAML config file from the installed package."""
    with resources.files("ds_maintenance_app").joinpath(f"config/{file_name}").open(
        "r"
    ) as f:
        return yaml.safe_load(f)


def check_valid_measurements(
    df: pd.DataFrame,
    measurement_columns: Union[List[str], np.ndarray],
    id_column: str,
) -> pd.DataFrame:
    """
    Checks the number and percentage of valid IDs per measurement.
    A valid ID is one where all values in the measurement column for that ID are not null.

    __PARAMETERS__
    df: pd.DataFrame
        Input DataFrame containing time-series measurements
    measurement_columns: Union[List[str], np.ndarray]
        List or array of measurement column names to check
    id_column: str
        Name of the column containing the series identifiers

    __RETURNS__
    pd.DataFrame
        A DataFrame with the following columns:
        - measurement: measurement column name
        - n_valid_ids: number of IDs with all non-null values
        - pct_valid_ids: percentage of valid IDs relative to total IDs
    """

    # Get the total ids in the data
    n_ids = df[id_column].nunique()
    records = []

    # Get current date
    current_date = pd.Timestamp.now()

    # Check counts of valid series for each measurement column
    for measurement_column in measurement_columns:
        df_tmp = df[[id_column, measurement_column]].dropna(how="any")
        n_valid_ids = df_tmp[id_column].nunique()
        pct_valid_ids = (n_valid_ids / n_ids) * 100
        records.append(
            {
                "measurement": measurement_column,
                "n_valid_ids": n_valid_ids,
                "pct_valid_ids": pct_valid_ids,
                "calculation_date": current_date,
            }
        )

    return pd.DataFrame(records)
