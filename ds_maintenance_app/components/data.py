import pandas as pd
from importlib.metadata import version
from copy import deepcopy

from ds_package_gcp.bigquery import DataLoader


def get_pipeline_version():
    """
    Get the app version from pyproject.toml.

    __RETURNS__
    str
        The version recorded in pyproject.toml as a string.
    """
    return version("ds-maintenance-app")


# Load features from BigQuery
def load_bigquery_features(database: str, features_table: str) -> pd.DataFrame:
    """
    Load feature data from a BigQuery table.

    __PARAMETERS__
    database: str
        The name of the BigQuery database
    features_table: str
        The name of the table containing feature data

    __RETURNS__
    pd.DataFrame
        A DataFrame containing the downloaded feature data
    """
    data_loader = DataLoader()
    features_query = f"SELECT * FROM {database}.{features_table}"
    df_features = data_loader.run_query(features_query)
    return df_features


def save_table_to_bigquery(
    df: pd.DataFrame,
    database: str,
    table: str,
    env: str,
):
    """
    Save preprocessed training data to a BigQuery table.

    The function overwrites the provided DataFrame to a BigQuery table whose
    name is prefixed with the environment string. This allows separating
    datasets by environment (e.g., dev, main).

    __PARAMETERS__
    df : pd.DataFrame
        DataFrame containing the preprocessed training data to save.
    database : str
        Name of the BigQuery database where the table resides.
    table : str
        Base name of the table to which the data will be appended.
    env : str
        Environment prefix to apply to the table name (e.g., 'dev', 'prod').

    __RETURNS__
    None
        The function performs an append operation; no value is returned.
    """

    # Add env to table name if not main
    if env != "main":
        table = f"{env}_{table}"

    # Overwrite table to BigQuery
    data_loader = DataLoader()
    data_loader.overwrite_table(
        database,
        table,
        df,
    )


def append_table_to_bigquery(
    df: pd.DataFrame,
    database: str,
    table: str,
    env: str,
    date_column: str = "cycle_date",
):
    """
    Save preprocessed training data to a BigQuery table using append.

    The function appends the provided DataFrame to a BigQuery table whose
    name is prefixed with the environment string. This allows separating
    datasets by environment (e.g., dev, main).

    __PARAMETERS__
    df : pd.DataFrame
        DataFrame containing the preprocessed training data to save.
    database : str
        Name of the BigQuery database where the table resides.
    table : str
        Base name of the table to which the data will be appended.
    env : str
        Environment prefix to apply to the table name (e.g., 'dev', 'prod').
    date_column : str
        Name of column to record the date of append.

    __RETURNS__
    None
        The function performs an append operation; no value is returned.
    """

    # Copy df
    tmp_df = deepcopy(df)

    # Create date column with current date
    tmp_df[date_column] = [pd.Timestamp.today()] * len(tmp_df)

    # Get pipeline app version
    version = get_pipeline_version()
    tmp_df["forecast_pipeline_version"] = [version] * len(tmp_df)

    # Add env to table name if not main
    if env != "main":
        table = f"{env}_{table}"

    # Append table to BigQuery
    data_loader = DataLoader()
    data_loader.append_table(
        database,
        table,
        tmp_df,
    )
